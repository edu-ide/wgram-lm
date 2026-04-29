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
    missing, unexpected = model.load_state_dict(state.get("model", state), strict=False)
    return list(missing), list(unexpected)


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
            max_visual_tokens=min(cfg.model.max_visual_tokens, 64),
            multimodal=args.multimodal,
            tokenizer_model_id=tokenizer_model_id,
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

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, betas=(0.9, 0.95), weight_decay=0.1)
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
        batch = next(it)
        batch = {k: v.to(device) for k, v in batch.items()}
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(cfg.train.use_amp and device == "cuda"), dtype=torch.bfloat16):
            model_batch = dict(batch)
            if donor is not None:
                model_batch.update(
                    prepare_donor_batch(
                        donor,
                        batch,
                        return_logits=use_donor_logits,
                    )
                )
            loss, metrics, _ = qtrm_smoke_loss(
                model,
                **model_batch,
                jepa_weight=cfg.train.loss_jepa_weight,
                aux_weight=cfg.train.loss_aux_weight,
                core_halt_weight=cfg.train.loss_core_halt_weight,
                core_halt_auto_targets=cfg.train.core_halt_auto_targets,
                core_halt_target_mode=cfg.train.core_halt_target_mode,
                core_halt_donor_kl_threshold=cfg.train.core_halt_donor_kl_threshold,
                core_halt_teacher_depth_threshold=cfg.train.core_halt_teacher_depth_threshold,
                core_halt_teacher_depth_logit_kl_threshold=cfg.train.core_halt_teacher_depth_logit_kl_threshold,
                core_halt_teacher_depth_min_step=cfg.train.core_halt_teacher_depth_min_step,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        step_idx = step + 1
        if step % cfg.train.log_every == 0:
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
