from __future__ import annotations
import argparse
from dataclasses import asdict
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from qtrm_mm.config import load_config
from qtrm_mm.diagnostics import next_token_diagnostics, repetition_stats, topk_token_report
from qtrm_mm.qtrm_model import QTRMMultimodalModel
from qtrm_mm.qwen_donor import QwenDonorAdapter
from qtrm_mm.losses import qtrm_smoke_loss
from qtrm_mm.training.synthetic_data import SyntheticTextVisionDataset, collate
from qtrm_mm.data.jsonl_dataset import JsonlTextVisionDataset, collate_jsonl


def load_initial_checkpoint(model, checkpoint_path: str, map_location):
    state = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    loaded = state.get("model", state)
    current = model.state_dict()
    compatible = {}
    skipped = []
    for name, value in loaded.items():
        if name in current and tuple(value.shape) != tuple(current[name].shape):
            skipped.append(name)
            continue
        compatible[name] = value
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    if skipped:
        print(f"[init] skipped shape-mismatched keys: {', '.join(skipped)}")
    return list(missing), list(unexpected)


def configure_trainable_parameters(model, policy: str = "all") -> list[str]:
    policy = (policy or "all").strip().lower()
    if policy == "all":
        for param in model.parameters():
            param.requires_grad_(True)
        return [name for name, param in model.named_parameters() if param.requires_grad]

    if policy == "controller_only":
        trainable_names = []
        for name, param in model.named_parameters():
            trainable = (
                name.startswith("ctrl.")
                or name.startswith("controller_signal_proj.")
                or name.startswith("controller_signal_head.")
            )
            param.requires_grad_(trainable)
            if trainable:
                trainable_names.append(name)
        if not trainable_names:
            raise ValueError("trainable_param_policy=controller_only requires model.ctrl heads")
        return trainable_names

    if policy == "core_halt_only":
        trainable_names = []
        for name, param in model.named_parameters():
            trainable = name.startswith("core.halt_head.")
            param.requires_grad_(trainable)
            if trainable:
                trainable_names.append(name)
        if not trainable_names:
            raise ValueError(
                "trainable_param_policy=core_halt_only requires model.core_halt_enabled=true"
            )
        return trainable_names

    if policy == "core_only":
        trainable_names = []
        for name, param in model.named_parameters():
            trainable = name.startswith("core.")
            param.requires_grad_(trainable)
            if trainable:
                trainable_names.append(name)
        if not trainable_names:
            raise ValueError("trainable_param_policy=core_only requires model.core")
        return trainable_names

    if policy == "core_and_loop_readout":
        trainable_names = []
        for name, param in model.named_parameters():
            trainable = name.startswith("core.") or name.startswith("core_loop_readout_")
            param.requires_grad_(trainable)
            if trainable:
                trainable_names.append(name)
        if not any(name.startswith("core_loop_readout_") for name in trainable_names):
            raise ValueError(
                "trainable_param_policy=core_and_loop_readout requires "
                "model.core_loop_readout_enabled=true"
            )
        return trainable_names

    if policy == "core_and_answer_state_loop":
        trainable_names = []
        for name, param in model.named_parameters():
            trainable = (
                name.startswith("core.")
                or name.startswith("answer_state_loop_")
                or name.startswith("transition_state_")
            )
            param.requires_grad_(trainable)
            if trainable:
                trainable_names.append(name)
        if not any(name.startswith("answer_state_loop_") for name in trainable_names):
            raise ValueError(
                "trainable_param_policy=core_and_answer_state_loop requires "
                "model.answer_state_loop_enabled=true"
            )
        return trainable_names

    if policy == "core_and_answer_state_loop_and_world_model":
        trainable_names = []
        for name, param in model.named_parameters():
            trainable = (
                name.startswith("core.")
                or name.startswith("answer_state_loop_")
                or name.startswith("core_world_model.")
            )
            param.requires_grad_(trainable)
            if trainable:
                trainable_names.append(name)
        if not any(name.startswith("answer_state_loop_") for name in trainable_names):
            raise ValueError(
                "trainable_param_policy=core_and_answer_state_loop_and_world_model requires "
                "model.answer_state_loop_enabled=true"
            )
        if not any(name.startswith("core_world_model.") for name in trainable_names):
            raise ValueError(
                "trainable_param_policy=core_and_answer_state_loop_and_world_model requires "
                "model.core_world_model_enabled=true"
            )
        return trainable_names

    if policy == "core_and_primitive_transition":
        trainable_names = []
        for name, param in model.named_parameters():
            trainable = (
                name.startswith("core.")
                or name.startswith("primitive_transition_")
            )
            param.requires_grad_(trainable)
            if trainable:
                trainable_names.append(name)
        if not any(name.startswith("primitive_transition_") for name in trainable_names):
            raise ValueError(
                "trainable_param_policy=core_and_primitive_transition requires "
                "model.primitive_transition_enabled=true"
            )
        if not any(name.startswith("core.") for name in trainable_names):
            raise ValueError(
                "trainable_param_policy=core_and_primitive_transition requires model.core"
            )
        return trainable_names

    if policy == "core_and_temporal_spatial_context":
        trainable_names = []
        prefixes = (
            "core.",
            "temporal_spatial_context_proj.",
            "temporal_spatial_context_norm.",
            "temporal_spatial_context_pos",
        )
        for name, param in model.named_parameters():
            trainable = name.startswith(prefixes)
            param.requires_grad_(trainable)
            if trainable:
                trainable_names.append(name)
        if not any(name.startswith("temporal_spatial_context_") for name in trainable_names):
            raise ValueError(
                "trainable_param_policy=core_and_temporal_spatial_context requires "
                "model.temporal_spatial_context_enabled=true"
            )
        if not any(name.startswith("core.") for name in trainable_names):
            raise ValueError(
                "trainable_param_policy=core_and_temporal_spatial_context requires model.core"
            )
        return trainable_names

    if policy == "workspace_gate_only":
        trainable_names = []
        gate_markers = (
            ".gate_norm_prev.",
            ".gate_norm_update.",
            ".update_gate.",
            ".reset_gate.",
            ".candidate.",
        )
        for name, param in model.named_parameters():
            trainable = name.startswith("workspace.layers.") and any(
                marker in name for marker in gate_markers
            )
            param.requires_grad_(trainable)
            if trainable:
                trainable_names.append(name)
        if not trainable_names:
            raise ValueError(
                "trainable_param_policy=workspace_gate_only requires "
                "model.workspace_memory_gate_enabled=true"
            )
        return trainable_names

    if policy == "generation_verifier_only":
        trainable_names = []
        prefixes = (
            "generation_repeat_head.",
            "generation_stop_head.",
            "generation_quality_head.",
        )
        for name, param in model.named_parameters():
            trainable = name.startswith(prefixes)
            param.requires_grad_(trainable)
            if trainable:
                trainable_names.append(name)
        if not trainable_names:
            raise ValueError(
                "trainable_param_policy=generation_verifier_only requires "
                "model.generation_verifier_enabled=true"
            )
        return trainable_names

    if policy == "controller_signal_head_only":
        trainable_names = []
        for name, param in model.named_parameters():
            trainable = name.startswith("controller_signal_head.")
            param.requires_grad_(trainable)
            if trainable:
                trainable_names.append(name)
        if not trainable_names:
            raise ValueError(
                "trainable_param_policy=controller_signal_head_only requires "
                "model.controller_signal_source to be a learned signal source"
            )
        return trainable_names

    if policy == "answer_decision_head_only":
        trainable_names = []
        prefixes = (
            "answer_decision_head.",
            "answer_decision_feature_norm.",
            "answer_decision_feature_proj.",
            "answer_decision_feature_head.",
        )
        for name, param in model.named_parameters():
            trainable = name.startswith(prefixes)
            param.requires_grad_(trainable)
            if trainable:
                trainable_names.append(name)
        if not trainable_names:
            raise ValueError(
                "trainable_param_policy=answer_decision_head_only requires "
                "model.answer_decision_head_enabled=true"
            )
        return trainable_names

    if policy == "answer_bottleneck_evidence_only":
        trainable_names = []
        answer_trainable_names = []
        prefixes = (
            "answer_bottleneck_query_norm.",
            "answer_bottleneck_workspace_norm.",
            "answer_bottleneck_cross.",
            "answer_bottleneck_output_norm.",
            "evidence_support_head.",
            "evidence_refute_head.",
            "evidence_missing_head.",
            "evidence_causal_gate_head.",
        )
        for name, param in model.named_parameters():
            trainable = name.startswith(prefixes)
            param.requires_grad_(trainable)
            if trainable:
                trainable_names.append(name)
                if name.startswith("answer_bottleneck_"):
                    answer_trainable_names.append(name)
        if not trainable_names or not answer_trainable_names:
            raise ValueError(
                "trainable_param_policy=answer_bottleneck_evidence_only requires "
                "model.answer_bottleneck_enabled=true and evidence heads"
            )
        return trainable_names

    if policy == "evidence_span_reader_only":
        trainable_names = []
        prefixes = (
            "evidence_span_",
            "evidence_support_head.",
            "evidence_refute_head.",
            "evidence_missing_head.",
            "evidence_causal_gate_head.",
            "projector.visual_proj.",
            "projector.norm.",
        )
        for name, param in model.named_parameters():
            trainable = name.startswith(prefixes)
            param.requires_grad_(trainable)
            if trainable:
                trainable_names.append(name)
        if not any(name.startswith("evidence_span_") for name in trainable_names):
            raise ValueError(
                "trainable_param_policy=evidence_span_reader_only requires "
                "model.evidence_span_reader_enabled=true"
            )
        return trainable_names

    raise ValueError(f"unknown trainable_param_policy: {policy}")


def trainable_parameters(model):
    return [param for param in model.parameters() if param.requires_grad]


def strip_training_only_batch_keys(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in batch.items()
        if key
        not in {
            "workspace_input_ids",
            "workspace_attention_mask",
            "workspace_counterfactual_input_ids",
            "workspace_counterfactual_attention_mask",
            "answer_decision_target",
            "answer_decision_sample_weight",
            "action_target",
            "action_sample_weight",
        }
    }


def build_core_world_model_actions(
    batch: dict[str, torch.Tensor],
    *,
    num_steps: int,
    num_actions: int,
    device,
) -> torch.Tensor:
    """Build simple LeWM-style action traces for recursive core states.

    Action ids are intentionally fixed for the first probe:
    0=OBSERVE, 1=RETRIEVE, 2=VERIFY, 3=ANSWER.
    """
    if num_steps < 0:
        raise ValueError("num_steps must be non-negative")
    if num_actions < 4:
        raise ValueError("num_actions must be at least 4 for core world-model actions")
    input_ids = batch["input_ids"]
    b = input_ids.shape[0]
    actions = torch.zeros((b, num_steps, num_actions), device=device, dtype=torch.float32)
    if num_steps == 0:
        return actions

    workspace_mask = batch.get("workspace_attention_mask")
    if workspace_mask is None:
        has_workspace = torch.zeros((b,), device=device, dtype=torch.bool)
    else:
        has_workspace = workspace_mask.to(device=device, dtype=torch.bool).any(dim=1)

    action_ids = torch.full((b, num_steps), 2, device=device, dtype=torch.long)
    action_ids[:, 0] = torch.where(
        has_workspace,
        torch.full((b,), 1, device=device, dtype=torch.long),
        torch.zeros((b,), device=device, dtype=torch.long),
    )
    if num_steps >= 2:
        action_ids[:, -1] = 3
    return actions.scatter_(dim=-1, index=action_ids.unsqueeze(-1), value=1.0)


def scheduled_donor_logits_scale(
    *,
    config_scale: float,
    start: float | None,
    end: float | None,
    step: int,
    total_steps: int,
) -> float:
    start_scale = float(config_scale if start is None else start)
    end_scale = float(start_scale if end is None else end)
    total_steps = max(1, int(total_steps))
    step = min(max(0, int(step)), total_steps - 1)
    if total_steps == 1:
        return end_scale
    ratio = step / float(total_steps - 1)
    return start_scale + (end_scale - start_scale) * ratio


def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--multimodal", action="store_true")
    ap.add_argument("--use-donor", action="store_true")
    ap.add_argument("--tokenizer-model-id", default=None)
    ap.add_argument("--data-jsonl", nargs="*", default=None, help="Optional downloaded JSONL files. If omitted, synthetic data is used.")
    ap.add_argument("--diag-every", type=int, default=0, help="Run prompt diagnostics every N steps. Disabled when 0.")
    ap.add_argument("--diag-prompt", action="append", default=None, help="Prompt for live logit/repetition diagnostics. Can be repeated.")
    ap.add_argument("--diag-max-new-tokens", type=int, default=8)
    ap.add_argument("--diag-top-k", type=int, default=5)
    ap.add_argument("--init-checkpoint", default=None, help="Optional checkpoint to load before training.")
    ap.add_argument("--save-every", type=int, default=0, help="Save step checkpoints every N steps. Disabled when 0.")
    return ap


def prepare_donor_batch(
    donor,
    batch: dict[str, torch.Tensor],
    *,
    return_logits: bool = False,
) -> dict[str, torch.Tensor]:
    donor_out = donor.encode_inputs(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
        return_logits=return_logits,
    )
    out = {"text_states": donor_out["text_states"].detach()}
    if donor_out.get("visual_features") is not None:
        out["visual_features"] = donor_out["visual_features"].detach()
    if return_logits and donor_out.get("logits") is not None:
        out["donor_logits"] = donor_out["logits"].detach()
    if "preference_rejected_input_ids" in batch:
        rejected_out = donor.encode_inputs(
            input_ids=batch["preference_rejected_input_ids"],
            attention_mask=batch.get("preference_rejected_attention_mask"),
            return_logits=return_logits,
        )
        out["preference_rejected_text_states"] = rejected_out["text_states"].detach()
        if return_logits and rejected_out.get("logits") is not None:
            out["preference_rejected_donor_logits"] = rejected_out["logits"].detach()
    workspace_attention_mask = batch.get("workspace_attention_mask")
    has_workspace_tokens = (
        workspace_attention_mask is None
        or bool(workspace_attention_mask.to(torch.bool).any().detach().cpu().item())
    )
    if "workspace_input_ids" in batch and has_workspace_tokens:
        workspace_out = donor.encode_inputs(
            input_ids=batch["workspace_input_ids"],
            attention_mask=workspace_attention_mask,
            return_logits=False,
        )
        out["workspace_text_states"] = workspace_out["text_states"].detach()
        if workspace_attention_mask is not None:
            out["workspace_attention_mask"] = workspace_attention_mask.detach()
    counterfactual_attention_mask = batch.get("workspace_counterfactual_attention_mask")
    has_counterfactual_tokens = (
        counterfactual_attention_mask is not None
        and bool(counterfactual_attention_mask.to(torch.bool).any().detach().cpu().item())
    )
    if "workspace_counterfactual_input_ids" in batch and has_counterfactual_tokens:
        counterfactual_out = donor.encode_inputs(
            input_ids=batch["workspace_counterfactual_input_ids"],
            attention_mask=counterfactual_attention_mask,
            return_logits=False,
        )
        out["workspace_counterfactual_text_states"] = counterfactual_out["text_states"].detach()
        out["workspace_counterfactual_attention_mask"] = counterfactual_attention_mask.detach()
    return out


@torch.no_grad()
def _donor_diag_kwargs(
    donor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: str,
    *,
    return_logits: bool = False,
):
    if donor is None:
        return {}
    donor_out = donor.encode_inputs(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_logits=return_logits,
    )
    out = {"text_states": donor_out["text_states"].detach().to(device)}
    if donor_out.get("visual_features") is not None:
        out["visual_features"] = donor_out["visual_features"].detach().to(device)
    if return_logits and donor_out.get("logits") is not None:
        out["donor_logits"] = donor_out["logits"].detach().to(device)
    return out


@torch.no_grad()
def run_prompt_diagnostics(
    model,
    donor,
    tokenizer,
    prompts: list[str],
    *,
    step: int,
    device: str,
    max_new_tokens: int,
    top_k: int,
    amp: bool,
) -> None:
    was_training = model.training
    use_donor_logits = bool(getattr(model.cfg, "donor_logits_scale", 0.0) != 0.0)
    model.eval()
    for idx, prompt in enumerate(prompts):
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128, add_special_tokens=True)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)
        extra = _donor_diag_kwargs(
            donor,
            input_ids,
            attention_mask,
            device,
            return_logits=use_donor_logits,
        )
        with torch.amp.autocast("cuda", enabled=(amp and device == "cuda"), dtype=torch.bfloat16):
            outputs = model(input_ids, attention_mask=attention_mask, **extra)
        offset = outputs["logits"].shape[1] - input_ids.shape[1]
        metrics = next_token_diagnostics(
            outputs["logits"],
            input_ids,
            offset=offset,
            attention_mask=attention_mask,
        )
        next_report = topk_token_report(outputs["logits"][0, -1].float(), tokenizer=tokenizer, k=top_k)

        generated = input_ids[0].detach().cpu().tolist()
        for _ in range(max_new_tokens):
            cur_ids = torch.tensor([generated], device=device, dtype=torch.long)
            cur_mask = torch.ones_like(cur_ids)
            cur_extra = _donor_diag_kwargs(
                donor,
                cur_ids,
                cur_mask,
                device,
                return_logits=use_donor_logits,
            )
            with torch.amp.autocast("cuda", enabled=(amp and device == "cuda"), dtype=torch.bfloat16):
                gen_outputs = model(cur_ids, attention_mask=cur_mask, **cur_extra)
            next_id = int(gen_outputs["logits"][0, -1].float().argmax(dim=-1).detach().cpu().item())
            if tokenizer.eos_token_id is not None and next_id == tokenizer.eos_token_id:
                break
            generated.append(next_id)
        rep = repetition_stats(generated, prompt_len=input_ids.shape[1])
        decoded = tokenizer.decode(generated, skip_special_tokens=True).replace("\n", "\\n")
        top = next_report[0] if next_report else {"token": "", "prob": 0.0, "token_id": -1}
        print(
            "[diag "
            f"step={step} prompt={idx} "
            f"loss={metrics['loss']:.3f} rank={metrics['target_rank_mean']:.1f} "
            f"top1={metrics['target_top1_acc']:.2f} ent={metrics['entropy_mean']:.2f} "
            f"next={top['token']!r}/{top['prob']:.3f} "
            f"run={rep['max_token_run']} rep2={rep['repeated_2gram_rate']:.2f}] "
            f"{decoded[:220]}",
            flush=True,
        )
    if was_training:
        model.train()


def main():
    args = build_arg_parser().parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() and cfg.train.device in {"auto", "cuda"} else "cpu"
    model = QTRMMultimodalModel(cfg.model).to(device)
    if args.init_checkpoint:
        missing, unexpected = load_initial_checkpoint(model, args.init_checkpoint, map_location=device)
        if missing:
            print(f"[init] missing keys: {len(missing)}")
        if unexpected:
            print(f"[init] unexpected keys: {len(unexpected)}")
        print(f"[init] loaded {args.init_checkpoint}")
    model.train()
    donor = None
    if args.use_donor:
        if not cfg.donor.model_id:
            raise ValueError("--use-donor requires donor.model_id in the config")
        donor = QwenDonorAdapter(cfg.donor)

    if args.data_jsonl:
        files = [str(x) for x in args.data_jsonl]
        print(f"using downloaded JSONL data: {files}")
        tokenizer_model_id = args.tokenizer_model_id or (cfg.donor.model_id if args.use_donor else None)
        if tokenizer_model_id:
            print(f"using tokenizer: {tokenizer_model_id}")
        ds = JsonlTextVisionDataset(
            files=files,
            vocab_size=cfg.model.vocab_size,
            seq_len=cfg.train.seq_len,
            visual_dim=cfg.model.visual_dim,
            max_visual_tokens=min(cfg.model.max_visual_tokens, 256),
            multimodal=args.multimodal,
            tokenizer_model_id=tokenizer_model_id,
            workspace_evidence_injection=cfg.train.workspace_evidence_injection,
            workspace_evidence_injection_mode=cfg.train.workspace_evidence_injection_mode,
        )
        loader = DataLoader(ds, batch_size=cfg.train.batch_size, collate_fn=collate_jsonl)
    else:
        ds = SyntheticTextVisionDataset(
            vocab_size=cfg.model.vocab_size,
            seq_len=cfg.train.seq_len,
            visual_dim=cfg.model.visual_dim,
            max_visual_tokens=min(cfg.model.max_visual_tokens, 16),
            multimodal=args.multimodal,
        )
        loader = DataLoader(ds, batch_size=cfg.train.batch_size, collate_fn=collate)

    trainable_names = configure_trainable_parameters(model, cfg.train.trainable_param_policy)
    trainable_params = trainable_parameters(model)
    if not trainable_params:
        raise ValueError("no trainable parameters selected")
    print(
        f"[trainable] policy={cfg.train.trainable_param_policy} "
        f"params={sum(p.numel() for p in trainable_params):,} "
        f"tensors={len(trainable_names)}"
    )
    opt = torch.optim.AdamW(trainable_params, lr=cfg.train.lr, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.train.use_amp and device == "cuda"))
    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    diag_prompts = args.diag_prompt or []
    diag_tokenizer = None
    if diag_prompts:
        tokenizer_model_id = args.tokenizer_model_id or (cfg.donor.model_id if args.use_donor else None)
        if tokenizer_model_id is None:
            raise ValueError("--diag-prompt requires --tokenizer-model-id or donor.model_id")
        from transformers import AutoTokenizer
        diag_tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_id, trust_remote_code=True)
        if diag_tokenizer.pad_token_id is None:
            diag_tokenizer.pad_token = diag_tokenizer.eos_token

    pbar = tqdm(range(cfg.train.steps))
    it = iter(loader)
    use_donor_logits = bool(cfg.model.donor_logits_scale != 0.0)
    for step in pbar:
        model.cfg.donor_logits_scale = scheduled_donor_logits_scale(
            config_scale=cfg.model.donor_logits_scale,
            start=cfg.train.donor_logits_scale_start,
            end=cfg.train.donor_logits_scale_end,
            step=step,
            total_steps=cfg.train.steps,
        )
        use_donor_logits = bool(model.cfg.donor_logits_scale != 0.0)
        batch = next(it)
        batch = {k: v.to(device) for k, v in batch.items()}
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(cfg.train.use_amp and device == "cuda"), dtype=torch.bfloat16):
            model_batch = strip_training_only_batch_keys(batch)
            if donor is not None:
                model_batch.update(
                    prepare_donor_batch(
                        donor,
                        batch,
                        return_logits=use_donor_logits,
                    )
                )
            if cfg.model.core_world_model_enabled or cfg.train.loss_core_world_model_weight != 0.0:
                model_batch["core_world_model_actions"] = build_core_world_model_actions(
                    batch,
                    num_steps=cfg.model.outer_steps,
                    num_actions=cfg.model.num_actions,
                    device=device,
                )
            if (
                cfg.train.workspace_evidence_injection
                and cfg.train.workspace_evidence_injection_mode == "ssot"
            ):
                model_batch["evidence_span_reader_context"] = "input"
            if "action_target" in batch:
                model_batch["action_targets"] = batch["action_target"]
            if "action_sample_weight" in batch:
                model_batch["action_sample_weight"] = batch["action_sample_weight"]
            if "answer_decision_target" in batch:
                model_batch["answer_decision_target"] = batch["answer_decision_target"]
            if "answer_decision_sample_weight" in batch:
                model_batch["answer_decision_sample_weight"] = batch[
                    "answer_decision_sample_weight"
                ]
            if (
                "controller_signal" in batch
                and str(cfg.model.controller_signal_source).lower() != "external"
            ):
                model_batch["controller_signal_target"] = batch["controller_signal"]
                model_batch.pop("controller_signal", None)
            loss, metrics, _ = qtrm_smoke_loss(
                model,
                **model_batch,
                jepa_weight=cfg.train.loss_jepa_weight,
                lm_weight=cfg.train.loss_lm_weight,
                aux_weight=cfg.train.loss_aux_weight,
                core_halt_weight=cfg.train.loss_core_halt_weight,
                core_halt_auto_targets=cfg.train.core_halt_auto_targets,
                core_halt_target_mode=cfg.train.core_halt_target_mode,
                core_halt_donor_kl_threshold=cfg.train.core_halt_donor_kl_threshold,
                core_halt_teacher_depth_threshold=cfg.train.core_halt_teacher_depth_threshold,
                core_halt_teacher_depth_logit_kl_threshold=cfg.train.core_halt_teacher_depth_logit_kl_threshold,
                core_halt_teacher_depth_min_step=cfg.train.core_halt_teacher_depth_min_step,
                student_lm_weight=cfg.train.loss_student_lm_weight,
                donor_kl_weight=cfg.train.loss_donor_kl_weight,
                donor_kl_beta=cfg.train.donor_kl_beta,
                donor_kl_temperature=cfg.train.donor_kl_temperature,
                repeat_unlikelihood_weight=cfg.train.loss_repeat_unlikelihood_weight,
                greedy_token_margin_weight=cfg.train.loss_greedy_token_margin_weight,
                greedy_token_margin=cfg.train.greedy_token_margin,
                greedy_token_margin_only_donor_errors=(
                    cfg.train.greedy_token_margin_only_donor_errors
                ),
                donor_correct_margin_weight=cfg.train.loss_donor_correct_margin_weight,
                donor_correct_margin=cfg.train.donor_correct_margin,
                preference_weight=cfg.train.loss_preference_weight,
                preference_beta=cfg.train.preference_beta,
                preference_margin=cfg.train.preference_margin,
                workspace_contrastive_weight=cfg.train.loss_workspace_contrastive_weight,
                workspace_contrastive_beta=cfg.train.workspace_contrastive_beta,
                workspace_contrastive_margin=cfg.train.workspace_contrastive_margin,
                logical_evidence_weight=cfg.train.loss_logical_evidence_weight,
                causal_evidence_gate_weight=cfg.train.loss_causal_evidence_gate_weight,
                core_world_model_weight=cfg.train.loss_core_world_model_weight,
                generation_verifier_weight=cfg.train.loss_generation_verifier_weight,
                evidence_span_reader_weight=cfg.train.loss_evidence_span_reader_weight,
                evidence_span_no_answer_span_suppression_weight=(
                    cfg.train.loss_evidence_span_no_answer_span_suppression_weight
                ),
                answer_decision_weight=cfg.train.loss_answer_decision_weight,
                answer_residual_governor_weight=(
                    cfg.train.loss_answer_residual_governor_weight
                ),
                canonical_causal_weight=cfg.train.loss_canonical_causal_weight,
                canonical_causal_beta=cfg.train.canonical_causal_beta,
                canonical_causal_margin=cfg.train.canonical_causal_margin,
                canonical_causal_ablation_modes=cfg.train.canonical_causal_ablation_modes,
                action_policy_weight=cfg.train.loss_action_policy_weight,
                controller_signal_weight=cfg.train.loss_controller_signal_weight,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        step_idx = step + 1
        if step % cfg.train.log_every == 0:
            metrics["donor_scale"] = torch.tensor(
                model.cfg.donor_logits_scale,
                device=loss.device,
            )
            pbar.set_description(" ".join(f"{k}={float(v):.4f}" for k, v in metrics.items()))
        if diag_prompts and args.diag_every > 0 and step_idx % args.diag_every == 0:
            run_prompt_diagnostics(
                model,
                donor,
                diag_tokenizer,
                diag_prompts,
                step=step_idx,
                device=device,
                max_new_tokens=args.diag_max_new_tokens,
                top_k=args.diag_top_k,
                amp=cfg.train.use_amp,
            )
        if args.save_every > 0 and step_idx % args.save_every == 0:
            step_path = out_dir / f"step_{step_idx:06d}.pt"
            torch.save({"model": model.state_dict(), "config": asdict(cfg), "step": step_idx}, step_path)
            print(f"saved {step_path}", flush=True)
    torch.save({"model": model.state_dict(), "config": asdict(cfg)}, out_dir / "last.pt")
    print(f"saved {out_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
