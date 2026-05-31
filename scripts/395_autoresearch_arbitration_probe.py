#!/usr/bin/env python3
"""Autoresearch-style arbitration probe for Qwen-integrated QTRM.

This is a fixed-budget diagnostic, not a promotion benchmark. It tests whether
the current mandatory QTRM core's useful option flips are separable from harmful
flips using base/core score geometry. The operating pattern follows
karpathy/autoresearch: one small editable hypothesis, one decisive metric, and a
keep/discard ledger row.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import torch

from wgram_lm.qwen_backbone_wgram import QwenBackboneQTRM


def load_healing_module():
    path = Path(__file__).with_name("394_train_qwen35_integrated_language_knowledge_healing.py")
    spec = importlib.util.spec_from_file_location("qwen35_integrated_language_knowledge_healing", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclasses.dataclass(frozen=True)
class ArbitrationRule:
    base_margin_max: float
    core_margin_min: float
    switch_adv_min: float


@dataclasses.dataclass(frozen=True)
class LinearPolicy:
    feature_names: tuple[str, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    weights: tuple[float, ...]
    bias: float
    threshold: float


def parse_float_grid(value: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in str(value).split(",") if part.strip())
    if not values:
        raise ValueError("threshold grid cannot be empty")
    return values


def batched(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), int(batch_size)):
        yield items[start : start + int(batch_size)]


def option_margin(scores: dict[str, torch.Tensor]) -> float:
    values = sorted((float(score.detach().cpu()) for score in scores.values()), reverse=True)
    if len(values) < 2:
        return 0.0
    return float(values[0] - values[1])


def option_stats(scores: dict[str, torch.Tensor], pred: str) -> dict[str, float]:
    letters = sorted(scores)
    if not letters:
        return {"confidence": 0.0, "entropy": 0.0}
    vector = torch.stack([scores[letter].detach().float().cpu() for letter in letters])
    probs = torch.softmax(vector, dim=-1)
    entropy = float(-(probs * probs.clamp_min(1e-12).log()).sum().item())
    confidence = float(probs[letters.index(pred)].item()) if pred in letters else 0.0
    return {"confidence": confidence, "entropy": entropy}


def apply_rule(case: dict[str, Any], rule: ArbitrationRule) -> tuple[str, bool]:
    switched = bool(
        case["base_pred"] != case["core_pred"]
        and float(case["base_margin"]) <= float(rule.base_margin_max)
        and float(case["core_margin"]) >= float(rule.core_margin_min)
        and float(case["switch_adv"]) >= float(rule.switch_adv_min)
    )
    return (str(case["core_pred"]) if switched else str(case["base_pred"])), switched


def linear_features(case: dict[str, Any]) -> list[float]:
    return [
        float(case.get("base_margin", 0.0)),
        float(case.get("core_margin", 0.0)),
        float(case.get("core_margin", 0.0)) - float(case.get("base_margin", 0.0)),
        float(case.get("switch_adv", 0.0)),
        float(case.get("base_confidence", 0.0)),
        float(case.get("core_confidence", 0.0)),
        float(case.get("core_confidence", 0.0)) - float(case.get("base_confidence", 0.0)),
        float(case.get("base_entropy", 0.0)),
        float(case.get("core_entropy", 0.0)),
        float(case.get("base_entropy", 0.0)) - float(case.get("core_entropy", 0.0)),
        1.0 if str(case.get("base_pred")) != str(case.get("core_pred")) else 0.0,
        float(len(case.get("choices", ""))),
    ]


def linear_feature_names() -> tuple[str, ...]:
    return (
        "base_margin",
        "core_margin",
        "margin_delta",
        "switch_adv",
        "base_confidence",
        "core_confidence",
        "confidence_delta",
        "base_entropy",
        "core_entropy",
        "entropy_drop",
        "switch_candidate",
        "option_count",
    )


def apply_linear_policy(case: dict[str, Any], policy: LinearPolicy) -> tuple[str, bool, float]:
    if str(case["base_pred"]) == str(case["core_pred"]):
        return str(case["base_pred"]), False, 0.0
    raw = torch.tensor(linear_features(case), dtype=torch.float32)
    mean = torch.tensor(policy.mean, dtype=torch.float32)
    std = torch.tensor(policy.std, dtype=torch.float32).clamp_min(1e-6)
    weights = torch.tensor(policy.weights, dtype=torch.float32)
    logit = ((raw - mean) / std * weights).sum() + float(policy.bias)
    prob = float(torch.sigmoid(logit).item())
    switched = bool(prob >= float(policy.threshold))
    return (str(case["core_pred"]) if switched else str(case["base_pred"])), switched, prob


def summarize_cases(cases: list[dict[str, Any]], rule: ArbitrationRule | None = None) -> dict[str, Any]:
    base_hits = 0
    core_hits = 0
    arb_hits = 0
    corrections = 0
    regressions = 0
    switches = 0
    by_category: dict[str, dict[str, int]] = {}
    for case in cases:
        gold = str(case["gold"])
        base_ok = str(case["base_pred"]) == gold
        core_ok = str(case["core_pred"]) == gold
        if rule is None:
            arb_pred = str(case["core_pred"])
            switched = str(case["base_pred"]) != str(case["core_pred"])
        else:
            arb_pred, switched = apply_rule(case, rule)
        arb_ok = arb_pred == gold
        base_hits += int(base_ok)
        core_hits += int(core_ok)
        arb_hits += int(arb_ok)
        switches += int(switched)
        corrections += int(switched and (not base_ok) and arb_ok)
        regressions += int(switched and base_ok and (not arb_ok))
        category = str(case.get("category", "unknown"))
        bucket = by_category.setdefault(
            category,
            {"total": 0, "base_hits": 0, "core_hits": 0, "arb_hits": 0},
        )
        bucket["total"] += 1
        bucket["base_hits"] += int(base_ok)
        bucket["core_hits"] += int(core_ok)
        bucket["arb_hits"] += int(arb_ok)
    for bucket in by_category.values():
        total = max(1, int(bucket["total"]))
        bucket["base_accuracy"] = float(bucket["base_hits"] / total)
        bucket["core_accuracy"] = float(bucket["core_hits"] / total)
        bucket["arb_accuracy"] = float(bucket["arb_hits"] / total)
        bucket["arb_minus_base"] = int(bucket["arb_hits"] - bucket["base_hits"])
    total = max(1, len(cases))
    return {
        "cases": len(cases),
        "base_hits": int(base_hits),
        "core_hits": int(core_hits),
        "arb_hits": int(arb_hits),
        "base_accuracy": float(base_hits / total),
        "core_accuracy": float(core_hits / total),
        "arb_accuracy": float(arb_hits / total),
        "core_gain": float((core_hits - base_hits) / total),
        "arb_gain": float((arb_hits - base_hits) / total),
        "corrections": int(corrections),
        "regressions": int(regressions),
        "switches": int(switches),
        "by_category": by_category,
    }


def summarize_linear_policy(cases: list[dict[str, Any]], policy: LinearPolicy) -> dict[str, Any]:
    base_hits = 0
    core_hits = 0
    arb_hits = 0
    corrections = 0
    regressions = 0
    switches = 0
    probabilities: list[float] = []
    for case in cases:
        gold = str(case["gold"])
        base_ok = str(case["base_pred"]) == gold
        core_ok = str(case["core_pred"]) == gold
        arb_pred, switched, prob = apply_linear_policy(case, policy)
        arb_ok = arb_pred == gold
        base_hits += int(base_ok)
        core_hits += int(core_ok)
        arb_hits += int(arb_ok)
        switches += int(switched)
        corrections += int(switched and (not base_ok) and arb_ok)
        regressions += int(switched and base_ok and (not arb_ok))
        probabilities.append(prob)
    total = max(1, len(cases))
    return {
        "cases": len(cases),
        "base_hits": int(base_hits),
        "core_hits": int(core_hits),
        "arb_hits": int(arb_hits),
        "base_accuracy": float(base_hits / total),
        "core_accuracy": float(core_hits / total),
        "arb_accuracy": float(arb_hits / total),
        "core_gain": float((core_hits - base_hits) / total),
        "arb_gain": float((arb_hits - base_hits) / total),
        "corrections": int(corrections),
        "regressions": int(regressions),
        "switches": int(switches),
        "mean_switch_probability": float(sum(probabilities) / max(1, len(probabilities))),
    }


def fit_best_rule(
    cases: list[dict[str, Any]],
    *,
    base_margin_grid: tuple[float, ...],
    core_margin_grid: tuple[float, ...],
    switch_adv_grid: tuple[float, ...],
) -> tuple[ArbitrationRule, dict[str, Any]]:
    best_rule: ArbitrationRule | None = None
    best_summary: dict[str, Any] | None = None
    best_key: tuple[Any, ...] | None = None
    for base_margin_max in base_margin_grid:
        for core_margin_min in core_margin_grid:
            for switch_adv_min in switch_adv_grid:
                rule = ArbitrationRule(
                    base_margin_max=base_margin_max,
                    core_margin_min=core_margin_min,
                    switch_adv_min=switch_adv_min,
                )
                summary = summarize_cases(cases, rule)
                key = (
                    int(summary["arb_hits"]),
                    int(summary["corrections"]) - int(summary["regressions"]),
                    -int(summary["regressions"]),
                    int(summary["corrections"]),
                    -int(summary["switches"]),
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_rule = rule
                    best_summary = summary
    assert best_rule is not None and best_summary is not None
    return best_rule, best_summary


def fit_linear_policy(
    cases: list[dict[str, Any]],
    *,
    steps: int,
    lr: float,
    weight_decay: float,
    threshold_grid: tuple[float, ...],
) -> tuple[LinearPolicy, dict[str, Any]]:
    feature_names = linear_feature_names()
    features = torch.tensor([linear_features(case) for case in cases], dtype=torch.float32)
    if features.numel() == 0:
        raise ValueError("cannot fit linear policy with no cases")
    labels = torch.tensor(
        [
            1.0
            if str(case["core_pred"]) == str(case["gold"])
            and str(case["base_pred"]) != str(case["gold"])
            else 0.0
            for case in cases
        ],
        dtype=torch.float32,
    )
    candidate = torch.tensor(
        [1.0 if str(case["base_pred"]) != str(case["core_pred"]) else 0.0 for case in cases],
        dtype=torch.float32,
    )
    mean = features.mean(dim=0)
    std = features.std(dim=0).clamp_min(1e-6)
    x = (features - mean) / std
    weights = torch.zeros(x.shape[1], dtype=torch.float32, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float32, requires_grad=True)
    pos = float(labels.sum().item())
    neg = float(labels.numel() - pos)
    pos_weight = torch.tensor(max(1.0, neg / max(1.0, pos)), dtype=torch.float32)
    optimizer = torch.optim.AdamW([weights, bias], lr=float(lr), weight_decay=float(weight_decay))
    for _ in range(int(steps)):
        logits = x @ weights + bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=pos_weight,
        )
        loss = loss + 0.05 * torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            candidate,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    best_policy: LinearPolicy | None = None
    best_summary: dict[str, Any] | None = None
    best_key: tuple[Any, ...] | None = None
    for threshold in threshold_grid:
        policy = LinearPolicy(
            feature_names=feature_names,
            mean=tuple(float(v) for v in mean.tolist()),
            std=tuple(float(v) for v in std.tolist()),
            weights=tuple(float(v) for v in weights.detach().tolist()),
            bias=float(bias.detach().item()),
            threshold=float(threshold),
        )
        summary = summarize_linear_policy(cases, policy)
        key = (
            int(summary["arb_hits"]),
            int(summary["corrections"]) - int(summary["regressions"]),
            -int(summary["regressions"]),
            int(summary["corrections"]),
            -int(summary["switches"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_policy = policy
            best_summary = summary
    assert best_policy is not None and best_summary is not None
    return best_policy, best_summary


def score_cases(model, tokenizer, rows: list[dict[str, Any]], args) -> list[dict[str, Any]]:
    healing = load_healing_module()
    device = next(model.parameters()).device
    cases: list[dict[str, Any]] = []
    with torch.no_grad():
        for chunk in batched(rows, int(args.batch_size)):
            input_ids, attention_mask = healing._encode(
                tokenizer,
                [str(row["qtrm_prompt"]) for row in chunk],
                max_seq_len=int(args.max_seq_len),
                device=device,
            )
            base_logits = healing.last_nonpad_logits(
                model(input_ids, attention_mask=attention_mask, force_core_off=True).logits,
                attention_mask,
            )
            core_logits = healing.last_nonpad_logits(
                model(input_ids, attention_mask=attention_mask).logits,
                attention_mask,
            )
            for index, row in enumerate(chunk):
                choices = healing.OPTION_LETTERS[: healing.option_count(row)]
                gold = healing.normalize_mcq_answer(str(row["answer"]))
                base_scores = healing.option_letter_scores(tokenizer, base_logits[index], row)
                core_scores = healing.option_letter_scores(tokenizer, core_logits[index], row)
                if not base_scores or not core_scores:
                    continue
                base_pred = max(base_scores.items(), key=lambda item: float(item[1].detach().cpu()))[0]
                core_pred = max(core_scores.items(), key=lambda item: float(item[1].detach().cpu()))[0]
                base_stats = option_stats(base_scores, base_pred)
                core_stats = option_stats(core_scores, core_pred)
                switch_adv = (
                    float((core_scores[core_pred] - core_scores[base_pred]).detach().cpu())
                    if core_pred in core_scores and base_pred in core_scores
                    else -1.0e9
                )
                cases.append(
                    {
                        "case_id": row.get("case_id", row.get("benchmark_id", len(cases))),
                        "category": row.get("category", "unknown"),
                        "gold": gold,
                        "choices": choices,
                        "base_pred": base_pred,
                        "core_pred": core_pred,
                        "base_margin": option_margin(base_scores),
                        "core_margin": option_margin(core_scores),
                        "base_confidence": base_stats["confidence"],
                        "core_confidence": core_stats["confidence"],
                        "base_entropy": base_stats["entropy"],
                        "core_entropy": core_stats["entropy"],
                        "switch_adv": switch_adv,
                    }
                )
    return cases


def load_model(args):
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("transformers is required") from exc

    healing = load_healing_module()
    device = torch.device(str(args.device))
    dtype = healing.dtype_from_name(str(args.dtype))
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
        qwen_core_layer_indices=healing.parse_int_list(str(args.qwen_core_layer_indices)),
        core_adapter_dim=int(args.core_adapter_dim),
        core_delta_adapter_mode=str(args.core_delta_adapter_mode),
        core_residual_gate_mode=str(args.core_residual_gate_mode),
        core_residual_gate_dim=int(args.core_residual_gate_dim),
        core_residual_gate_init=float(args.core_residual_gate_init),
        clone_qwen_core_layers=bool(args.clone_qwen_core_layers),
        mandatory_core=True,
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
    checkpoint_info = healing.load_checkpoint(
        model,
        str(args.checkpoint),
        load_mode=str(args.checkpoint_load_mode),
    )
    model.eval()
    return tokenizer, model, checkpoint_info


def append_ledger(path: str | Path, report: dict[str, Any]) -> None:
    ledger = Path(path)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "timestamp\tprobe\tpolicy\tdecision\tstatus\tfit_base\tfit_core\tfit_arb\t"
        "eval_base\teval_core\teval_arb\tpolicy_detail\treport_path\tnext_action\n"
    )
    if not ledger.exists() or ledger.stat().st_size == 0:
        ledger.write_text(header, encoding="utf-8")
    detail = str(report.get("policy_detail", ""))
    row = [
        str(report["timestamp"]),
        str(report["probe"]),
        str(report.get("policy", "")),
        str(report["decision"]),
        "keep" if bool(report["accepted"]) else "discard",
        str(report["fit_summary"]["base_hits"]),
        str(report["fit_summary"]["core_hits"]),
        str(report["fit_summary"]["arb_hits"]),
        str(report["eval_summary"]["base_hits"]),
        str(report["eval_summary"]["core_hits"]),
        str(report["eval_summary"]["arb_hits"]),
        detail,
        str(report["report_path"]),
        str(report["next_action"]).replace("\t", " ").replace("\n", " "),
    ]
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write("\t".join(row) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    healing = load_healing_module()
    rows = healing.load_mcq_rows(
        args.suite_jsonl,
        max_rows=int(args.max_cases),
        seed=int(args.seed),
    )
    if len(rows) < 4:
        raise ValueError("at least four rows are required for fit/eval arbitration")
    rng = random.Random(int(args.seed) + 101)
    rng.shuffle(rows)
    split = max(1, min(len(rows) - 1, int(round(len(rows) * float(args.fit_fraction)))))
    fit_rows = rows[:split]
    eval_rows = rows[split:]
    tokenizer, model, checkpoint_info = load_model(args)
    fit_cases = score_cases(model, tokenizer, fit_rows, args)
    eval_cases = score_cases(model, tokenizer, eval_rows, args)
    best_rule: ArbitrationRule | None = None
    best_policy: LinearPolicy | None = None
    if str(args.policy) == "threshold":
        best_rule, fit_summary = fit_best_rule(
            fit_cases,
            base_margin_grid=parse_float_grid(str(args.base_margin_grid)),
            core_margin_grid=parse_float_grid(str(args.core_margin_grid)),
            switch_adv_grid=parse_float_grid(str(args.switch_adv_grid)),
        )
        eval_summary = summarize_cases(eval_cases, best_rule)
        policy_detail = (
            f"bm<={best_rule.base_margin_max},"
            f"cm>={best_rule.core_margin_min},"
            f"adv>={best_rule.switch_adv_min}"
        )
    else:
        best_policy, fit_summary = fit_linear_policy(
            fit_cases,
            steps=int(args.linear_steps),
            lr=float(args.linear_lr),
            weight_decay=float(args.linear_weight_decay),
            threshold_grid=parse_float_grid(str(args.linear_threshold_grid)),
        )
        eval_summary = summarize_linear_policy(eval_cases, best_policy)
        policy_detail = f"linear_threshold>={best_policy.threshold}"
    raw_fit_summary = summarize_cases(fit_cases)
    raw_eval_summary = summarize_cases(eval_cases)
    accepted = bool(
        int(eval_summary["arb_hits"]) > int(eval_summary["base_hits"])
        and int(eval_summary["corrections"]) > int(eval_summary["regressions"])
    )
    out_dir = Path(args.out_dir)
    report_path = out_dir / "report.json"
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "probe": "autoresearch_qtrm_score_geometry_arbitration",
        "decision": "accepted_arbitration_probe" if accepted else "rejected_arbitration_probe",
        "accepted": accepted,
        "status": "complete",
        "reference": {
            "repo": "https://github.com/karpathy/autoresearch",
            "local_path": "references/official/autoresearch",
            "commit": str(args.autoresearch_commit),
            "adopted_pattern": "fixed-budget run, one decisive metric, keep/discard ledger",
        },
        "checkpoint": str(args.checkpoint),
        "checkpoint_info": checkpoint_info,
        "suite_jsonl": str(args.suite_jsonl),
        "fit_cases": len(fit_cases),
        "eval_cases": len(eval_cases),
        "policy": str(args.policy),
        "policy_detail": policy_detail,
        "best_rule": dataclasses.asdict(best_rule) if best_rule is not None else None,
        "best_linear_policy": dataclasses.asdict(best_policy) if best_policy is not None else None,
        "fit_summary": fit_summary,
        "eval_summary": eval_summary,
        "raw_fit_summary": raw_fit_summary,
        "raw_eval_summary": raw_eval_summary,
        "report_path": str(report_path),
        "ledger_path": str(args.ledger_path),
        "next_action": (
            "promote this arbitration policy into the QTRM integrated eval path"
            if accepted
            else "do not scale blind MCQ repair; improve core signal or add richer non-label separability features"
        ),
        "limitations": [
            "Probe-only: threshold fitting uses labels on the fit split.",
            "Acceptance here is not a public benchmark promotion.",
            "A learned arbitration head must be validated on a larger held-out set before becoming canonical.",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if str(args.ledger_path):
        append_ledger(args.ledger_path, report)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-2B-Base")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-load-mode", choices=["strict_shapes", "skip_mismatch"], default="strict_shapes")
    parser.add_argument("--suite-jsonl", required=True)
    parser.add_argument("--out-dir", default="local_eval/qwen35_integrated_autoresearch_arbitration_probe")
    parser.add_argument("--ledger-path", default="local_eval/qwen35_integrated_autoresearch_arbitration_probe/results.tsv")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-seq-len", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-cases", type=int, default=64)
    parser.add_argument("--fit-fraction", type=float, default=0.5)
    parser.add_argument("--core-impl", choices=["qwen_layer_wrapped"], default="qwen_layer_wrapped")
    parser.add_argument("--core-insertion-mode", choices=["final_residual", "mid_layer_suffix"], default="mid_layer_suffix")
    parser.add_argument("--core-insert-after-layer", type=int, default=11)
    parser.add_argument("--qwen-core-layer-indices", default="3")
    parser.add_argument("--core-adapter-dim", type=int, default=512)
    parser.add_argument("--core-delta-adapter-mode", choices=["add", "adapter_only"], default="adapter_only")
    parser.add_argument("--core-residual-gate-mode", choices=["constant", "token_mlp"], default="constant")
    parser.add_argument("--core-residual-gate-dim", type=int, default=128)
    parser.add_argument("--core-residual-gate-init", type=float, default=-2.0)
    parser.add_argument("--clone-qwen-core-layers", action="store_true")
    parser.add_argument("--n-core-layers", type=int, default=1)
    parser.add_argument("--h-cycles", type=int, default=3)
    parser.add_argument("--l-cycles", type=int, default=6)
    parser.add_argument("--outer-steps", type=int, default=3)
    parser.add_argument("--core-convergence-halt-enabled", action="store_true", default=True)
    parser.add_argument("--core-convergence-halt-threshold", type=float, default=0.2)
    parser.add_argument("--core-convergence-halt-min-outer", type=int, default=1)
    parser.add_argument("--core-step-conditioning-enabled", action="store_true", default=True)
    parser.add_argument("--core-step-conditioning-max-steps", type=int, default=64)
    parser.add_argument("--core-step-conditioning-scale", type=float, default=1.0)
    parser.add_argument("--core-gate-init", type=float, default=-4.0)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--policy", choices=["threshold", "linear"], default="threshold")
    parser.add_argument("--base-margin-grid", default="0,0.1,0.25,0.5,0.75,1,1.5,2,3,5")
    parser.add_argument("--core-margin-grid", default="0,0.1,0.25,0.5,0.75,1,1.5,2,3,5")
    parser.add_argument("--switch-adv-grid", default="-1,-0.5,-0.25,0,0.25,0.5,0.75,1,1.5,2")
    parser.add_argument("--linear-steps", type=int, default=300)
    parser.add_argument("--linear-lr", type=float, default=0.05)
    parser.add_argument("--linear-weight-decay", type=float, default=0.01)
    parser.add_argument("--linear-threshold-grid", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--autoresearch-commit", default="")
    parser.add_argument("--seed", type=int, default=20260522)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if not str(args.autoresearch_commit):
        ref = Path("references/official/autoresearch/.git/HEAD")
        args.autoresearch_commit = ref.read_text(encoding="utf-8").strip() if ref.exists() else ""
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if bool(report["accepted"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
