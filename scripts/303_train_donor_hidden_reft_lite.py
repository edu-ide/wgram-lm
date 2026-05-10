#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
import torch.nn.functional as F


def _load_raw_eval_module():
    path = Path(__file__).with_name("192_eval_raw_intelligence.py")
    spec = importlib.util.spec_from_file_location("qtrm_raw_eval_192", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DonorHiddenReftLite(nn.Module):
    """Low-rank intervention from QTRM core state into donor final hidden state."""

    def __init__(
        self,
        *,
        core_dim: int,
        donor_dim: int,
        rank: int = 64,
        scale: float = 1.0,
        gate_init_bias: float = -1.0,
    ) -> None:
        super().__init__()
        rank = max(1, int(rank))
        self.scale = float(scale)
        self.norm = nn.LayerNorm(int(core_dim))
        self.down = nn.Linear(int(core_dim), rank, bias=False)
        self.up = nn.Linear(rank, int(donor_dim), bias=False)
        self.gate = nn.Linear(int(core_dim), 1)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.up.weight, mean=0.0, std=0.002)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, float(gate_init_bias))

    def forward(self, core_hidden: torch.Tensor) -> torch.Tensor:
        x = self.norm(core_hidden.float())
        delta = self.up(F.gelu(self.down(x)))
        gate = torch.sigmoid(self.gate(x))
        return delta * gate * self.scale


def find_output_embeddings(model: nn.Module) -> nn.Module:
    getter = getattr(model, "get_output_embeddings", None)
    if callable(getter):
        head = getter()
        if head is not None:
            return head
    for path in ("lm_head", "language_model.lm_head", "model.lm_head", "base_model.lm_head"):
        cur: Any = model
        ok = True
        for part in path.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok and isinstance(cur, nn.Module):
            return cur
    raise RuntimeError("could not locate donor output embeddings / lm_head")


def _gold_answer(case: dict[str, Any]) -> str:
    for key in ("answer", "chosen", "canonical_answer"):
        value = str(case.get(key) or "").strip()
        if value:
            return value
    for alias in case.get("answer_aliases") or []:
        value = str(alias).strip()
        if value:
            return value
    raise ValueError(f"case has no answer: {case.get('id')}")


def _target_token_ids(tokenizer, answer: str, *, max_target_tokens: int) -> list[int]:
    ids = tokenizer.encode(f" {answer}", add_special_tokens=False)
    if not ids:
        ids = tokenizer.encode(str(answer), add_special_tokens=False)
    ids = [int(token_id) for token_id in ids]
    if max_target_tokens > 0:
        ids = ids[: int(max_target_tokens)]
    if not ids:
        raise ValueError(f"answer has no token ids: {answer!r}")
    return ids


def _prepare_prefix_ids(
    tokenizer,
    prompt: str,
    target_ids: list[int],
    *,
    pos: int,
    max_length: int,
) -> list[int]:
    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=True,
    )["input_ids"][0].detach().cpu().tolist()
    prefix = [int(token_id) for token_id in prompt_ids + target_ids[:pos]]
    return prefix[-int(max_length) :]


def _runtime_kwargs(runtime: dict[str, Any]) -> dict[str, Any]:
    raw_eval = _load_raw_eval_module()
    return {
        "disable_core": bool(runtime.get("disable_core", False)),
        "enable_core_halt": raw_eval._runtime_enable_core_halt(runtime),
        "disable_qtrm_residual": bool(runtime.get("disable_qtrm_residual", False)),
        "disable_qtrm_residual_gate": bool(runtime.get("disable_qtrm_residual_gate", False)),
        "disable_transition_state": bool(runtime.get("disable_transition_state", False)),
        "disable_core_role_value_answer_bridge": bool(
            runtime.get("disable_core_role_value_answer_bridge", False)
        ),
        "disable_answer_state_loop_recurrent": bool(
            runtime.get("disable_answer_state_loop_recurrent", False)
        ),
        "disable_answer_state_loop_selective_context": bool(
            runtime.get("disable_answer_state_loop_selective_context", False)
        ),
        "disable_answer_state_loop_finality_selector": bool(
            runtime.get("disable_answer_state_loop_finality_selector", False)
        ),
        "disable_answer_state_loop_finality_gate": bool(
            runtime.get("disable_answer_state_loop_finality_gate", False)
        ),
        "disable_answer_state_loop_halt_gate": bool(
            runtime.get("disable_answer_state_loop_halt_gate", False)
        ),
        "disable_answer_state_loop_hidden_bridge": bool(
            runtime.get("disable_answer_state_loop_hidden_bridge", False)
        ),
        "disable_answer_state_loop_next_token_decoder": bool(
            runtime.get("disable_answer_state_loop_next_token_decoder", False)
        ),
        "disable_answer_state_loop_talker": bool(
            runtime.get("disable_answer_state_loop_talker", False)
        ),
    }


def _last_core_hidden(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    for key in (
        "core_loop_readout_hidden",
        "answer_state_loop_hidden",
        "answer_bottleneck_hidden",
    ):
        value = outputs.get(key)
        if value is not None and value.numel() != 0:
            return value[:, -1, :].detach()
    raise RuntimeError("QTRM output has no usable core/readout hidden state")


def _feature_summary(features: list[dict[str, Any]]) -> dict[str, Any]:
    if not features:
        return {"count": 0}
    return {
        "count": len(features),
        "case_count": len({str(item["case_id"]) for item in features}),
        "target_positions": max(int(item["pos"]) for item in features) + 1,
        "core_dim": int(features[0]["core_hidden"].numel()),
        "donor_dim": int(features[0]["donor_hidden"].numel()),
    }


@torch.no_grad()
def precompute_features(
    *,
    cases: list[dict[str, Any]],
    tokenizer,
    donor,
    model,
    device: str,
    core_steps: int,
    max_length: int,
    max_target_tokens: int,
) -> list[dict[str, Any]]:
    raw_eval = _load_raw_eval_module()
    old_outer_steps = int(model.cfg.outer_steps)
    model.cfg.outer_steps = int(core_steps)
    runtime = raw_eval.mode_runtime(f"qtrm_core_steps_{int(core_steps)}_no_evidence")
    features: list[dict[str, Any]] = []
    try:
        for case in cases:
            prompt = str(case.get("prompt") or case.get("question") or "")
            answer = _gold_answer(case)
            target_ids = _target_token_ids(
                tokenizer,
                answer,
                max_target_tokens=max_target_tokens,
            )
            for pos, target_id in enumerate(target_ids):
                prefix_ids = _prepare_prefix_ids(
                    tokenizer,
                    prompt,
                    target_ids,
                    pos=pos,
                    max_length=max_length,
                )
                input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
                attention_mask = torch.ones_like(input_ids)
                donor_out = donor.encode_inputs(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_logits=False,
                )
                text_states = donor_out["text_states"].to(device)
                with torch.amp.autocast(
                    "cuda",
                    enabled=(device == "cuda"),
                    dtype=torch.bfloat16,
                ):
                    outputs = model(
                        input_ids,
                        attention_mask=attention_mask,
                        text_states=text_states,
                        **_runtime_kwargs(runtime),
                    )
                features.append(
                    {
                        "case_id": str(case.get("id")),
                        "prompt": prompt,
                        "answer": answer,
                        "answer_aliases": list(case.get("answer_aliases") or [answer]),
                        "target_id": int(target_id),
                        "pos": int(pos),
                        "target_token_count": len(target_ids),
                        "core_hidden": _last_core_hidden(outputs)[0].float().cpu(),
                        "donor_hidden": donor_out["text_states"][0, -1, :].float().cpu(),
                    }
                )
    finally:
        model.cfg.outer_steps = old_outer_steps
    return features


def train_adapter(
    *,
    adapter: DonorHiddenReftLite,
    lm_head: nn.Module,
    features: list[dict[str, Any]],
    device: torch.device,
    steps: int,
    batch_size: int,
    lr: float,
    log_every: int,
    seed: int,
) -> list[dict[str, Any]]:
    if not features:
        raise ValueError("no training features")
    adapter.train()
    for param in lm_head.parameters():
        param.requires_grad_(False)
    lm_head.eval()
    core = torch.stack([item["core_hidden"] for item in features]).to(device)
    donor = torch.stack([item["donor_hidden"] for item in features]).to(device)
    target = torch.tensor([int(item["target_id"]) for item in features], dtype=torch.long, device=device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=float(lr), weight_decay=0.01)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    history: list[dict[str, Any]] = []
    for step in range(1, int(steps) + 1):
        idx = torch.randint(0, core.shape[0], (int(batch_size),), device=device, generator=generator)
        with torch.amp.autocast(
            "cuda",
            enabled=(device.type == "cuda"),
            dtype=torch.bfloat16,
        ):
            hidden = donor[idx] + adapter(core[idx])
            logits = lm_head(hidden)
            loss = F.cross_entropy(logits.float(), target[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
        opt.step()
        if step == 1 or step % int(log_every) == 0 or step == int(steps):
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                acc = (pred == target[idx]).float().mean().item()
            item = {"step": step, "loss": float(loss.detach().cpu()), "batch_acc": float(acc)}
            history.append(item)
            print(f"step={step} loss={item['loss']:.4f} batch_acc={item['batch_acc']:.3f}")
    return history


@torch.no_grad()
def teacher_forced_metrics(
    *,
    adapter: DonorHiddenReftLite,
    lm_head: nn.Module,
    features: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    if not features:
        return {"count": 0}
    adapter.eval()
    core = torch.stack([item["core_hidden"] for item in features]).to(device)
    donor = torch.stack([item["donor_hidden"] for item in features]).to(device)
    target = torch.tensor([int(item["target_id"]) for item in features], dtype=torch.long, device=device)
    zero_core = torch.zeros_like(core)
    with torch.amp.autocast(
        "cuda",
        enabled=(device.type == "cuda"),
        dtype=torch.bfloat16,
    ):
        donor_logits = lm_head(donor)
        full_logits = lm_head(donor + adapter(core))
        core_off_logits = lm_head(donor + adapter(zero_core))
    out: dict[str, Any] = {"count": int(target.numel())}
    for name, logits in (
        ("donor", donor_logits),
        ("reft_full", full_logits),
        ("reft_core_off", core_off_logits),
    ):
        pred = logits.argmax(dim=-1)
        ranks = []
        for row, target_id in zip(logits.float(), target):
            target_logit = row[int(target_id)]
            ranks.append(int((row > target_logit).sum().detach().cpu().item()) + 1)
        out[f"{name}_top1"] = float((pred == target).float().mean().item())
        out[f"{name}_mean_rank"] = float(sum(ranks) / max(1, len(ranks)))
        out[f"{name}_rank_le_10"] = float(sum(rank <= 10 for rank in ranks) / max(1, len(ranks)))
    return out


@torch.no_grad()
def generate_one(
    *,
    case: dict[str, Any],
    tokenizer,
    donor,
    model,
    adapter: DonorHiddenReftLite,
    lm_head: nn.Module,
    device: str,
    mode: str,
    core_steps: int,
    max_length: int,
    max_new_tokens: int,
    suppressed_token_ids: Iterable[int] = (),
) -> tuple[str, int]:
    raw_eval = _load_raw_eval_module()
    runtime = raw_eval.mode_runtime(mode)
    old_outer_steps = int(model.cfg.outer_steps)
    if runtime.get("core_steps_override") is not None:
        model.cfg.outer_steps = int(runtime["core_steps_override"])
    else:
        model.cfg.outer_steps = int(core_steps)
    prompt = str(case.get("prompt") or case.get("question") or "")
    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=True,
    )["input_ids"][0].detach().cpu().tolist()
    generated = [int(token_id) for token_id in prompt_ids]
    try:
        for _ in range(int(max_new_tokens)):
            cur_ids = torch.tensor([generated[-int(max_length) :]], dtype=torch.long, device=device)
            cur_mask = torch.ones_like(cur_ids)
            donor_out = donor.encode_inputs(
                input_ids=cur_ids,
                attention_mask=cur_mask,
                return_logits=False,
            )
            donor_hidden = donor_out["text_states"][:, -1, :].to(next(adapter.parameters()).device)
            if mode == "donor_only_no_evidence" or "delta_off" in mode:
                with torch.amp.autocast(
                    "cuda",
                    enabled=(next(adapter.parameters()).device.type == "cuda"),
                    dtype=torch.bfloat16,
                ):
                    logits = lm_head(donor_hidden)
            else:
                with torch.amp.autocast(
                    "cuda",
                    enabled=(device == "cuda"),
                    dtype=torch.bfloat16,
                ):
                    outputs = model(
                        cur_ids,
                        attention_mask=cur_mask,
                        text_states=donor_out["text_states"].to(device),
                        **_runtime_kwargs(runtime),
                    )
                if bool(runtime.get("disable_core", False)):
                    core_hidden = torch.zeros(
                        (1, model.cfg.d_model),
                        dtype=torch.float32,
                        device=next(adapter.parameters()).device,
                    )
                else:
                    core_hidden = _last_core_hidden(outputs).to(next(adapter.parameters()).device)
                with torch.amp.autocast(
                    "cuda",
                    enabled=(next(adapter.parameters()).device.type == "cuda"),
                    dtype=torch.bfloat16,
                ):
                    logits = lm_head(donor_hidden + adapter(core_hidden))
            if suppressed_token_ids:
                valid_ids = [
                    int(token_id)
                    for token_id in suppressed_token_ids
                    if 0 <= int(token_id) < int(logits.shape[-1])
                ]
                if valid_ids:
                    if logits.ndim == 3:
                        logits[:, -1, valid_ids] = -1.0e9
                    else:
                        logits[:, valid_ids] = -1.0e9
            next_id = int(logits[:, -1, :].argmax(dim=-1).detach().cpu().item()) if logits.ndim == 3 else int(logits.argmax(dim=-1).detach().cpu().item())
            generated.append(next_id)
            if tokenizer.eos_token_id is not None and next_id == int(tokenizer.eos_token_id):
                break
    finally:
        model.cfg.outer_steps = old_outer_steps
    completion = tokenizer.decode(generated[len(prompt_ids) :], skip_special_tokens=True).strip()
    return completion, len(generated) - len(prompt_ids)


@torch.no_grad()
def generation_metrics(
    *,
    cases: list[dict[str, Any]],
    tokenizer,
    donor,
    model,
    adapter: DonorHiddenReftLite,
    lm_head: nn.Module,
    device: str,
    modes: Iterable[str],
    core_steps: int,
    max_length: int,
    max_new_tokens: int,
    suppress_visible_reasoning_tokens: bool,
    out_jsonl: Path,
) -> dict[str, Any]:
    raw_eval = _load_raw_eval_module()
    suppressed_token_ids = raw_eval._visible_reasoning_token_ids(
        tokenizer,
        enabled=bool(suppress_visible_reasoning_tokens),
    )
    summary: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    for mode in modes:
        runtime = raw_eval.mode_runtime(mode)
        hits = 0
        exact = 0
        for case in cases:
            completion, generated_tokens = generate_one(
                case=case,
                tokenizer=tokenizer,
                donor=donor,
                model=model,
                adapter=adapter,
                lm_head=lm_head,
                device=device,
                mode=mode,
                core_steps=core_steps,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                suppressed_token_ids=suppressed_token_ids,
            )
            record = raw_eval.score_case_record(
                case,
                mode=mode,
                completion=completion,
                runtime=runtime,
                generated_tokens=generated_tokens,
            )
            records.append(record)
            hits += int(bool(record["hit"]))
            exact += int(bool(record["exact_match"] or record["normalized_exact"]))
        summary[mode] = {
            "hits": hits,
            "exact": exact,
            "total": len(cases),
            "accuracy": float(hits / max(1, len(cases))),
            "exact_accuracy": float(exact / max(1, len(cases))),
        }
    with out_jsonl.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a small ReFT-style bridge from QTRM recursive core hidden "
            "states into the frozen donor LM head. This is a renderer bottleneck "
            "falsifier, not a canonical promotion script."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-cases", required=True)
    parser.add_argument("--eval-cases", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--adapter-checkpoint", default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--max-train-cases", type=int, default=64)
    parser.add_argument("--max-eval-cases", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-target-tokens", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--core-steps", type=int, default=8)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--gate-init-bias", type=float, default=-1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--suppress-visible-reasoning-tokens", action="store_true")
    return parser


def main() -> None:
    from transformers import AutoTokenizer

    from qtrm_mm.config import load_config
    from qtrm_mm.qtrm_model import QTRMMultimodalModel
    from qtrm_mm.qwen_donor import QwenDonorAdapter

    args = build_arg_parser().parse_args()
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    raw_eval = _load_raw_eval_module()
    cfg = load_config(args.config)
    device = raw_eval._select_device(cfg.train.device, args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.donor.model_id,
        trust_remote_code=cfg.donor.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = QTRMMultimodalModel(cfg.model).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state.get("model", state), strict=False)
    if missing:
        print(f"[qtrm] missing keys: {len(missing)}")
    if unexpected:
        print(f"[qtrm] unexpected keys: {len(unexpected)}")
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    donor = QwenDonorAdapter(cfg.donor)
    lm_head = find_output_embeddings(donor.model)
    lm_head.eval()
    for param in lm_head.parameters():
        param.requires_grad_(False)
    head_param = next(lm_head.parameters(), None)
    adapter_device = head_param.device if head_param is not None else torch.device(device)

    train_cases = (
        []
        if bool(args.eval_only)
        else raw_eval.load_cases(args.train_cases, max_cases=args.max_train_cases)
    )
    eval_cases = raw_eval.load_cases(args.eval_cases, max_cases=args.max_eval_cases)
    print(
        "[gate] target=L0/L1 renderer falsifier "
        "bottleneck=latent-core-to-autoregressive-text "
        "baseline=donor_only"
    )
    train_features: list[dict[str, Any]] = []
    if not bool(args.eval_only):
        print("[features] precompute train")
        train_features = precompute_features(
            cases=train_cases,
            tokenizer=tokenizer,
            donor=donor,
            model=model,
            device=device,
            core_steps=int(args.core_steps),
            max_length=int(args.max_length),
            max_target_tokens=int(args.max_target_tokens),
        )
        print(f"[features] train {_feature_summary(train_features)}")
    print("[features] precompute eval")
    eval_features = precompute_features(
        cases=eval_cases,
        tokenizer=tokenizer,
        donor=donor,
        model=model,
        device=device,
        core_steps=int(args.core_steps),
        max_length=int(args.max_length),
        max_target_tokens=int(args.max_target_tokens),
    )
    print(f"[features] eval {_feature_summary(eval_features)}")

    dim_source = train_features or eval_features
    if not dim_source:
        raise ValueError("no features")
    core_dim = int(dim_source[0]["core_hidden"].numel())
    donor_dim = int(dim_source[0]["donor_hidden"].numel())
    adapter = DonorHiddenReftLite(
        core_dim=core_dim,
        donor_dim=donor_dim,
        rank=int(args.rank),
        scale=float(args.scale),
        gate_init_bias=float(args.gate_init_bias),
    ).to(adapter_device)
    loaded_adapter_report = None
    if args.adapter_checkpoint:
        adapter_state = torch.load(
            args.adapter_checkpoint,
            map_location=adapter_device,
            weights_only=False,
        )
        adapter.load_state_dict(adapter_state["adapter"], strict=True)
        loaded_adapter_report = adapter_state.get("report")

    history: list[dict[str, Any]] = []
    if not bool(args.eval_only):
        history = train_adapter(
            adapter=adapter,
            lm_head=lm_head,
            features=train_features,
            device=adapter_device,
            steps=int(args.steps),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            log_every=int(args.log_every),
            seed=int(args.seed),
        )
    tf_metrics = teacher_forced_metrics(
        adapter=adapter,
        lm_head=lm_head,
        features=eval_features,
        device=adapter_device,
    )
    modes = [
        "donor_only_no_evidence",
        "qtrm_core_off_no_evidence",
        f"qtrm_core_steps_{int(args.core_steps)}_no_evidence",
        f"qtrm_core_steps_{int(args.core_steps)}_delta_off_no_evidence",
    ]
    generation = generation_metrics(
        cases=eval_cases,
        tokenizer=tokenizer,
        donor=donor,
        model=model,
        adapter=adapter,
        lm_head=lm_head,
        device=device,
        modes=modes,
        core_steps=int(args.core_steps),
        max_length=int(args.max_length),
        max_new_tokens=int(args.max_new_tokens),
        suppress_visible_reasoning_tokens=bool(args.suppress_visible_reasoning_tokens),
        out_jsonl=out_dir / "generation.jsonl",
    )
    full_mode = f"qtrm_core_steps_{int(args.core_steps)}_no_evidence"
    donor_acc = generation["donor_only_no_evidence"]["accuracy"]
    full_acc = generation[full_mode]["accuracy"]
    core_off_acc = generation["qtrm_core_off_no_evidence"]["accuracy"]
    accepted = bool(full_acc > donor_acc and full_acc > core_off_acc)
    decision = "accepted_l1_renderer_bridge" if accepted else "rejected"
    report = {
        "decision": decision,
        "accepted": accepted,
        "target_level": "L0/L1 renderer falsifier",
        "major_bottleneck": "latent-core-to-autoregressive-text",
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "adapter": {
            "core_dim": core_dim,
            "donor_dim": donor_dim,
            "rank": int(args.rank),
            "scale": float(args.scale),
            "gate_init_bias": float(args.gate_init_bias),
        },
        "feature_summary": {
            "train": _feature_summary(train_features),
            "eval": _feature_summary(eval_features),
        },
        "teacher_forced": tf_metrics,
        "generation": generation,
        "history": history,
        "eval_only": bool(args.eval_only),
        "adapter_checkpoint": str(args.adapter_checkpoint) if args.adapter_checkpoint else None,
        "loaded_adapter_report_decision": (
            loaded_adapter_report.get("decision")
            if isinstance(loaded_adapter_report, dict)
            else None
        ),
        "suppress_visible_reasoning_tokens": bool(args.suppress_visible_reasoning_tokens),
        "next_action": (
            "promote to integrated donor-hidden intervention only after held-out generation "
            "beats donor and core_off"
            if accepted
            else "do not tune local logits further; try true internal donor-layer ReFT hook or reset renderer task"
        ),
    }
    torch.save(
        {
            "adapter": adapter.state_dict(),
            "report": report,
        },
        out_dir / "adapter.pt",
    )
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
