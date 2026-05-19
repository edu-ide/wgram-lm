#!/usr/bin/env python3
"""Evaluate a Qwen-integrated QTRM checkpoint on public MCQ suites.

This is the M4 public-benchmark recheck for the integrated native path:
Qwen tokenizer/full vocab/backbone/LM head stay in one graph, the QTRM core is
mandatory on the normal path, and scoring compares core_on against core_off
using the same next-token option-letter probability-mass scorer.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Sequence

import torch

from qtrm_mm.qwen_backbone_qtrm import QwenBackboneQTRM


OPTION_LETTERS = "ABCDEFGHIJ"


def parse_int_list(value: str) -> tuple[int, ...] | None:
    text = str(value).strip()
    if text == "":
        return None
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def dtype_from_name(name: str) -> torch.dtype:
    value = str(name).lower()
    if value in {"float16", "fp16"}:
        return torch.float16
    if value in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if value in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def normalize_mcq_answer(text: str) -> str:
    upper = str(text).strip().upper()
    if upper in OPTION_LETTERS:
        return upper
    match = re.search(r"(?:ANSWER\s*[:：]?\s*)?\(?\b([A-J])\b\)?", upper)
    if match:
        return match.group(1)
    return ""


def load_suite(path: str | Path, *, max_cases: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"suite row must be an object at {path}:{line_no}")
        for key in ("benchmark_id", "case_id", "qtrm_prompt", "answer"):
            if key not in row:
                raise ValueError(f"suite row missing {key} at {path}:{line_no}")
        rows.append(row)
        if int(max_cases) > 0 and len(rows) >= int(max_cases):
            break
    if not rows:
        raise ValueError(f"suite is empty: {path}")
    return rows


def option_count(row: dict[str, Any]) -> int:
    options = row.get("options", [])
    if isinstance(options, list) and options:
        return min(len(options), len(OPTION_LETTERS))
    return len(OPTION_LETTERS)


def single_token_option_ids(tokenizer, letter: str) -> list[int]:
    token_ids: list[int] = []
    for variant in (letter, f" {letter}", f"\n{letter}"):
        encoded = tokenizer.encode(variant, add_special_tokens=False)
        if len(encoded) == 1:
            token_ids.append(int(encoded[0]))
    return sorted(set(token_ids))


def option_token_map(tokenizer, count: int) -> dict[str, list[int]]:
    return {
        OPTION_LETTERS[index]: single_token_option_ids(tokenizer, OPTION_LETTERS[index])
        for index in range(int(count))
    }


def option_score_from_log_probs(log_probs: torch.Tensor, token_ids: list[int]) -> torch.Tensor:
    """Score all acceptable one-token renderings of the same option letter."""
    if not token_ids:
        return torch.tensor(-math.inf, device=log_probs.device, dtype=log_probs.dtype)
    target = torch.tensor(token_ids, device=log_probs.device, dtype=torch.long)
    return torch.logsumexp(log_probs.index_select(dim=0, index=target), dim=0)


@torch.no_grad()
def score_prompt_next_token(
    model,
    tokenizer,
    *,
    prompt: str,
    choices: Sequence[str],
    force_core_off: bool,
    max_seq_len: int,
    device: torch.device,
) -> dict[str, Any]:
    encoded = tokenizer(
        str(prompt),
        return_tensors="pt",
        truncation=True,
        max_length=int(max_seq_len),
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    logits = model(
        input_ids,
        attention_mask=attention_mask,
        force_core_off=bool(force_core_off),
    ).logits[:, -1, :]
    finite_logits = bool(torch.isfinite(logits).all().item())
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    scores: dict[str, float] = {}
    token_map = option_token_map(tokenizer, len(choices))
    for letter in choices:
        ids = token_map.get(letter, [])
        if ids:
            scores[letter] = float(
                option_score_from_log_probs(log_probs[0], ids).detach().cpu()
            )
        else:
            scores[letter] = -math.inf
    pred = max(scores.items(), key=lambda item: item[1])[0] if scores else ""
    return {
        "pred_answer": pred,
        "scores": scores,
        "finite_logits": finite_logits,
        "prompt_tokens": int(input_ids.shape[-1]),
    }


def score_rows(rows: list[dict[str, Any]], *, pred_key: str) -> dict[str, Any]:
    hits = sum(1 for row in rows if row.get(pred_key) == row.get("gold_answer"))
    by_category: dict[str, dict[str, int]] = {}
    for row in rows:
        category = str(row.get("category", "unknown"))
        bucket = by_category.setdefault(category, {"hits": 0, "total": 0})
        bucket["hits"] += int(row.get(pred_key) == row.get("gold_answer"))
        bucket["total"] += 1
    return {
        "hits": hits,
        "cases": len(rows),
        "accuracy": float(hits / max(1, len(rows))),
        "by_category": {
            key: {
                "hits": value["hits"],
                "total": value["total"],
                "accuracy": float(value["hits"] / max(1, value["total"])),
            }
            for key, value in sorted(by_category.items())
        },
    }


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> dict[str, object]:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    incompatible = model.load_state_dict(state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"unexpected checkpoint keys: {unexpected[:8]}")
    return {
        "missing_key_count": len(incompatible.missing_keys),
        "unexpected_key_count": len(unexpected),
        "checkpoint_report": checkpoint.get("report", {}),
    }


def load_model(args: argparse.Namespace):
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("transformers is required") from exc

    device = torch.device(str(args.device))
    dtype = dtype_from_name(str(args.dtype))
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_id), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = QwenBackboneQTRM.from_pretrained(
        str(args.model_id),
        dtype=dtype,
        device=device,
        max_seq_len=int(args.max_seq_len),
        freeze_qwen=True,
        core_gate_init=float(args.core_gate_init),
        residual_scale=float(args.residual_scale),
        core_impl=str(args.core_impl),
        core_insertion_mode=str(args.core_insertion_mode),
        core_insert_after_layer=int(args.core_insert_after_layer),
        qwen_core_layer_indices=parse_int_list(str(args.qwen_core_layer_indices)),
        core_adapter_dim=int(args.core_adapter_dim),
        core_delta_adapter_mode=str(args.core_delta_adapter_mode),
        core_residual_gate_mode=str(args.core_residual_gate_mode),
        core_residual_gate_dim=int(args.core_residual_gate_dim),
        core_residual_gate_init=float(args.core_residual_gate_init),
        mandatory_core=bool(args.mandatory_core),
        n_core_layers=int(args.n_core_layers),
        h_cycles=int(args.h_cycles),
        l_cycles=int(args.l_cycles),
        outer_steps=int(args.outer_steps),
        core_convergence_halt_enabled=bool(args.core_convergence_halt_enabled),
        core_convergence_halt_threshold=float(args.core_convergence_halt_threshold),
        core_convergence_halt_min_outer=int(args.core_convergence_halt_min_outer),
        core_step_conditioning_enabled=bool(args.core_step_conditioning_enabled),
        core_step_conditioning_max_steps=int(args.core_step_conditioning_max_steps),
        core_step_conditioning_scale=float(args.core_step_conditioning_scale),
        delta_backend="fla_gated_delta",
        strict_backends=False,
        core_causal=True,
    ).to(device)
    checkpoint_info = _load_checkpoint(model, str(args.checkpoint))
    model.eval()
    return tokenizer, model, device, checkpoint_info


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    suite = load_suite(args.suite_jsonl, max_cases=int(args.max_cases))
    tokenizer, model, device, checkpoint_info = load_model(args)
    scored_rows: list[dict[str, Any]] = []
    finite_logits = True
    for index, row in enumerate(suite, start=1):
        count = option_count(row)
        choices = list(OPTION_LETTERS[:count])
        gold = normalize_mcq_answer(str(row["answer"]))
        base = score_prompt_next_token(
            model,
            tokenizer,
            prompt=str(row["qtrm_prompt"]),
            choices=choices,
            force_core_off=True,
            max_seq_len=int(args.max_seq_len),
            device=device,
        )
        core = score_prompt_next_token(
            model,
            tokenizer,
            prompt=str(row["qtrm_prompt"]),
            choices=choices,
            force_core_off=False,
            max_seq_len=int(args.max_seq_len),
            device=device,
        )
        finite_logits = bool(finite_logits and base["finite_logits"] and core["finite_logits"])
        scored = dict(row)
        scored.update(
            {
                "gold_answer": gold,
                "base_pred_answer": base["pred_answer"],
                "core_pred_answer": core["pred_answer"],
                "base_scores": base["scores"],
                "core_scores": core["scores"],
                "base_exact": bool(base["pred_answer"] == gold),
                "core_exact": bool(core["pred_answer"] == gold),
                "finite_logits": bool(base["finite_logits"] and core["finite_logits"]),
            }
        )
        scored_rows.append(scored)
        if int(args.log_every) > 0 and index % int(args.log_every) == 0:
            core_metrics = score_rows(scored_rows, pred_key="core_pred_answer")
            base_metrics = score_rows(scored_rows, pred_key="base_pred_answer")
            print(
                json.dumps(
                    {
                        "progress": index,
                        "cases": len(suite),
                        "base_accuracy": base_metrics["accuracy"],
                        "core_accuracy": core_metrics["accuracy"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    base_metrics = score_rows(scored_rows, pred_key="base_pred_answer")
    core_metrics = score_rows(scored_rows, pred_key="core_pred_answer")
    gain = float(core_metrics["accuracy"] - base_metrics["accuracy"])
    accepted_core_gain = bool(
        finite_logits
        and len(scored_rows) >= int(args.min_cases)
        and gain >= float(args.min_core_gain)
        and core_metrics["accuracy"] >= float(args.min_core_accuracy)
    )
    target_score = float(args.qwen36_target_percent) / 100.0
    parity_floor = target_score - float(args.parity_tolerance)
    accepted_parity = bool(
        finite_logits
        and len(scored_rows) >= int(args.min_cases_for_parity)
        and core_metrics["accuracy"] >= parity_floor
    )
    report = {
        "status": "complete",
        "decision": (
            "accepted_m4_public_mcq_core_gain"
            if accepted_core_gain
            else "rejected_m4_public_mcq_core_gain"
        ),
        "accepted": accepted_core_gain,
        "accepted_core_gain": accepted_core_gain,
        "accepted_parity": accepted_parity,
        "accepted_finite_logits": bool(finite_logits),
        "target_level": "M4 public MCQ recheck",
        "benchmark_id": str(args.benchmark_id),
        "benchmark_name": str(args.benchmark_name),
        "suite_jsonl": str(args.suite_jsonl),
        "checkpoint": str(args.checkpoint),
        "model_id": str(args.model_id),
        "core_insertion_mode": str(args.core_insertion_mode),
        "core_insert_after_layer": int(args.core_insert_after_layer),
        "core_adapter_dim": int(args.core_adapter_dim),
        "core_delta_adapter_mode": str(args.core_delta_adapter_mode),
        "core_residual_gate_mode": str(args.core_residual_gate_mode),
        "core_residual_gate_dim": int(args.core_residual_gate_dim),
        "core_residual_gate_init": float(args.core_residual_gate_init),
        "residual_scale": float(args.residual_scale),
        "h_cycles": int(args.h_cycles),
        "l_cycles": int(args.l_cycles),
        "outer_steps": int(args.outer_steps),
        "core_convergence_halt_enabled": bool(args.core_convergence_halt_enabled),
        "core_step_conditioning_enabled": bool(args.core_step_conditioning_enabled),
        "model_report": model.report().__dict__,
        "checkpoint_info": checkpoint_info,
        "base_metrics": base_metrics,
        "core_metrics": core_metrics,
        "core_gain_over_base": gain,
        "qwen36_target_percent": float(args.qwen36_target_percent),
        "qwen36_target_score": target_score,
        "parity_tolerance": float(args.parity_tolerance),
        "parity_floor": parity_floor,
        "scorer": "next-token option-letter probability mass over acceptable one-token renderings",
        "thresholds": {
            "min_cases": int(args.min_cases),
            "min_core_gain": float(args.min_core_gain),
            "min_core_accuracy": float(args.min_core_accuracy),
            "min_cases_for_parity": int(args.min_cases_for_parity),
        },
        "limitations": [
            "This is a public subset recheck unless the supplied suite contains the full benchmark.",
            "Core gain is not a 27B parity claim; parity requires accepted_parity=true on enough cases.",
        ],
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.out_jsonl).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in scored_rows),
        encoding="utf-8",
    )
    Path(args.out_json).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-jsonl", default="local_eval/m7_public_reasoning_suite/mmlu_pro_validation_64.jsonl")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-2B-Base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--benchmark-id", default="mmlu_pro")
    parser.add_argument("--benchmark-name", default="MMLU-Pro")
    parser.add_argument(
        "--core-impl",
        choices=["qwen_layer_wrapped", "qwen_shared_layer_wrapped"],
        default="qwen_layer_wrapped",
    )
    parser.add_argument(
        "--core-insertion-mode",
        choices=["final_residual", "mid_layer_suffix"],
        default="final_residual",
    )
    parser.add_argument("--core-insert-after-layer", type=int, default=-1)
    parser.add_argument("--qwen-core-layer-indices", default="3")
    parser.add_argument("--core-adapter-dim", type=int, default=128)
    parser.add_argument("--core-delta-adapter-mode", choices=["add", "adapter_only"], default="add")
    parser.add_argument("--core-residual-gate-mode", choices=["constant", "token_mlp"], default="constant")
    parser.add_argument("--core-residual-gate-dim", type=int, default=128)
    parser.add_argument("--core-residual-gate-init", type=float, default=-2.0)
    parser.add_argument("--n-core-layers", type=int, default=1)
    parser.add_argument("--h-cycles", type=int, default=1)
    parser.add_argument("--l-cycles", type=int, default=1)
    parser.add_argument("--outer-steps", type=int, default=1)
    parser.add_argument(
        "--core-convergence-halt-enabled",
        dest="core_convergence_halt_enabled",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--no-core-convergence-halt",
        dest="core_convergence_halt_enabled",
        action="store_false",
    )
    parser.add_argument("--core-convergence-halt-threshold", type=float, default=0.001)
    parser.add_argument("--core-convergence-halt-min-outer", type=int, default=1)
    parser.add_argument(
        "--core-step-conditioning-enabled",
        dest="core_step_conditioning_enabled",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--no-core-step-conditioning",
        dest="core_step_conditioning_enabled",
        action="store_false",
    )
    parser.add_argument("--core-step-conditioning-max-steps", type=int, default=64)
    parser.add_argument("--core-step-conditioning-scale", type=float, default=1.0)
    parser.add_argument("--mandatory-core", action="store_true")
    parser.add_argument("--core-gate-init", type=float, default=-4.0)
    parser.add_argument("--residual-scale", type=float, default=0.05)
    parser.add_argument("--min-cases", type=int, default=64)
    parser.add_argument("--min-core-gain", type=float, default=0.01)
    parser.add_argument("--min-core-accuracy", type=float, default=0.0)
    parser.add_argument("--qwen36-target-percent", type=float, default=86.2)
    parser.add_argument("--parity-tolerance", type=float, default=0.02)
    parser.add_argument("--min-cases-for-parity", type=int, default=256)
    parser.add_argument("--log-every", type=int, default=16)
    parser.add_argument("--out-dir", default="local_eval/qwen35_integrated_m4_mmlu_pro64")
    parser.add_argument("--out-json", default="local_eval/qwen35_integrated_m4_mmlu_pro64/report.json")
    parser.add_argument("--out-jsonl", default="local_eval/qwen35_integrated_m4_mmlu_pro64/predictions.jsonl")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = evaluate(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["accepted"] else 1)


if __name__ == "__main__":
    main()
