#!/usr/bin/env python3
"""Build the QTRM-Native-vs-Qwen3.6-27B milestone status report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


QWEN36_27B_TARGETS: dict[str, float] = {
    "swe_bench_verified": 77.2,
    "swe_bench_pro": 53.5,
    "swe_bench_multilingual": 71.3,
    "terminal_bench_2_0": 59.3,
    "skillsbench_avg5": 48.2,
    "qwenwebbench": 1487.0,
    "nl2repo": 36.2,
    "claw_eval_avg": 72.4,
    "mmlu_pro": 86.2,
    "gpqa_diamond": 87.8,
    "aime_2026": 94.1,
    "hmmt_feb_2026": 84.3,
    "hle": 24.0,
}


MILESTONE_ORDER = [
    "M0_TARGET_CONTRACT",
    "M1_BRIDGE_CAUSAL_SIGNAL",
    "M2_NATIVE_TINY_LM",
    "M3_NATIVE_CORE_CAUSALITY",
    "M4_NATIVE_LANGUAGE_BOOTSTRAP",
    "M5_QWEN36_EVAL_HARNESS",
    "M6_NATIVE_SMALL_BEATS_27B_ON_SCOPED_RAW_REASONING",
    "M7_NATIVE_2B_3B_PUBLIC_BENCH_PARITY",
    "M8_NATIVE_2B_3B_PUBLIC_BENCH_WIN",
]

DEPENDENCY_ORDER = [
    "M_A_RECURSIVE_CORE_REASONING_PROOF",
    "M_B_CORE_TO_LM_ATTACHMENT",
    "M_C_LANGUAGE_HEALING_AFTER_CORE",
]


MILESTONE_SCHEDULE: dict[str, dict[str, str]] = {
    "M0_TARGET_CONTRACT": {
        "fast_path_estimate": "<= 0.5 day",
        "max_time_before_pivot": "none; already accepted",
        "actual_duration": "same day on 2026-05-15",
    },
    "M1_BRIDGE_CAUSAL_SIGNAL": {
        "fast_path_estimate": "0.5 day",
        "max_time_before_pivot": "1 day; then stop bridge tuning and transfer only the useful pattern",
        "actual_duration": "same day on 2026-05-15; rejected by 3-seed stability gate",
    },
    "M2_NATIVE_TINY_LM": {
        "fast_path_estimate": "0.5-1 day",
        "max_time_before_pivot": "1 day; switch to Qwen-config native initialization if random init stalls",
        "actual_duration": "same day on 2026-05-15; accepted from existing native language bootstrap report",
    },
    "M3_NATIVE_CORE_CAUSALITY": {
        "fast_path_estimate": "1 day",
        "max_time_before_pivot": "2 days; simplify to one mandatory recurrent block if ablations are noisy",
        "actual_duration": "same day on 2026-05-15; accepted by state-reset ablation",
    },
    "M4_NATIVE_LANGUAGE_BOOTSTRAP": {
        "fast_path_estimate": "2-4 days on local 4090, 1-2 days on DGX",
        "max_time_before_pivot": "4 days; move to stronger pretrained initialization or DGX run",
        "actual_duration": "same day on 2026-05-15 for small bootstrap; scale run still pending",
    },
    "M5_QWEN36_EVAL_HARNESS": {
        "fast_path_estimate": "0.5-1 day",
        "max_time_before_pivot": "1 day; use published Qwen3.6 target scores first, direct rerun only if needed",
        "actual_duration": "not started",
    },
    "M6_NATIVE_SMALL_BEATS_27B_ON_SCOPED_RAW_REASONING": {
        "fast_path_estimate": "3-7 days after M3/M4",
        "max_time_before_pivot": "7 days; revise task mix or core interface",
        "actual_duration": "not started",
    },
    "M7_NATIVE_2B_3B_PUBLIC_BENCH_PARITY": {
        "fast_path_estimate": "2-4 weeks after stable native language bootstrap",
        "max_time_before_pivot": "4 weeks; scale data/compute or narrow benchmark target",
        "actual_duration": "not started",
    },
    "M8_NATIVE_2B_3B_PUBLIC_BENCH_WIN": {
        "fast_path_estimate": "1-3 months after parity evidence",
        "max_time_before_pivot": "3 months; reassess architecture claim",
        "actual_duration": "not started",
    },
}


COMPUTE_PLAN: dict[str, dict[str, str]] = {
    "M0_TARGET_CONTRACT": {
        "4090_feasibility": "yes",
        "preferred_compute": "local 4090 or CPU",
        "fastest_path": "documentation plus executable status gate",
    },
    "M1_BRIDGE_CAUSAL_SIGNAL": {
        "4090_feasibility": "yes",
        "preferred_compute": "local 4090",
        "fastest_path": "short bridge gates only; pivot after stability rejection",
    },
    "M2_NATIVE_TINY_LM": {
        "4090_feasibility": "yes",
        "preferred_compute": "local 4090",
        "fastest_path": "Qwen tokenizer/config shape, tiny native path, no random architecture shopping",
    },
    "M3_NATIVE_CORE_CAUSALITY": {
        "4090_feasibility": "yes",
        "preferred_compute": "local 4090",
        "fastest_path": "mandatory core-on/core-off/depth/reset ablations on small native LM",
    },
    "M4_NATIVE_LANGUAGE_BOOTSTRAP": {
        "4090_feasibility": "partial",
        "preferred_compute": "4090 for smoke, DGX for real bilingual bootstrap",
        "fastest_path": "pretrained Qwen-compatible initialization, short context first, then DGX scale",
    },
    "M5_QWEN36_EVAL_HARNESS": {
        "4090_feasibility": "yes for public-target mode; direct Qwen rerun is optional",
        "preferred_compute": "local 4090 for QTRM plus published Qwen3.6 benchmark targets",
        "fastest_path": "use official/public Qwen3.6 scores as fixed targets; run QTRM on matching public tasks/scorers",
    },
    "M6_NATIVE_SMALL_BEATS_27B_ON_SCOPED_RAW_REASONING": {
        "4090_feasibility": "possible for scoped gates",
        "preferred_compute": "local 4090 first, DGX for confirmation",
        "fastest_path": "small held-out raw-reasoning suite with strict ablations",
    },
    "M7_NATIVE_2B_3B_PUBLIC_BENCH_PARITY": {
        "4090_feasibility": "no for serious 2B/3B training",
        "preferred_compute": "DGX",
        "fastest_path": "scale only the architecture that passed M2-M6",
    },
    "M8_NATIVE_2B_3B_PUBLIC_BENCH_WIN": {
        "4090_feasibility": "no for serious 2B/3B training",
        "preferred_compute": "DGX plus reproducible eval harness",
        "fastest_path": "focus one benchmark win first, not all targets at once",
    },
}


def _load_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None or str(path).strip() == "":
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _family_summary(report: dict[str, Any]) -> dict[str, Any]:
    return (
        report.get("accepted_family_summary")
        or report.get("after_family_summary")
        or {}
    )


def _metric(report: dict[str, Any], name: str, default: float = 0.0) -> float:
    for container_name in ("decisive_metrics", "eval_metrics"):
        container = report.get(container_name)
        if isinstance(container, dict) and isinstance(container.get(name), (int, float)):
            return float(container[name])
    if isinstance(report.get(name), (int, float)):
        return float(report[name])
    return float(default)


def recursive_core_reasoning_proof_status(report: dict[str, Any] | None) -> dict[str, Any]:
    claim = (
        "prove the recursive core actually improves held-out reasoning before "
        "language healing or benchmark scaling"
    )
    if report is None:
        return {
            "status": "pending",
            "accepted": False,
            "claim": claim,
            "reason": "no recursive-core reasoning report supplied",
        }
    full = _metric(report, "full_generation_exact")
    min_family = _metric(report, "min_family_generation_exact")
    full_minus_think0 = _metric(report, "full_minus_think0")
    full_minus_worst = _metric(report, "full_minus_worst_ablation")
    accepted = bool(
        report.get("accepted", False)
        and full >= 0.50
        and min_family >= 0.30
        and full_minus_think0 >= 0.05
        and full_minus_worst >= 0.05
    )
    return {
        "status": "accepted" if accepted else "rejected",
        "accepted": accepted,
        "claim": claim,
        "metrics": {
            "decision": str(report.get("decision", "")),
            "full_generation_exact": full,
            "min_family_generation_exact": min_family,
            "full_minus_think0": full_minus_think0,
            "full_minus_worst_ablation": full_minus_worst,
        },
        "checks": {
            "report_accepted": bool(report.get("accepted", False)),
            "full_generation_exact_ge_0_50": full >= 0.50,
            "min_family_generation_exact_ge_0_30": min_family >= 0.30,
            "full_minus_think0_ge_0_05": full_minus_think0 >= 0.05,
            "full_minus_worst_ablation_ge_0_05": full_minus_worst >= 0.05,
        },
        "reject_reasons": report.get("reject_reasons", []),
    }


def core_to_lm_attachment_status(
    core_reasoning: dict[str, Any],
    report: dict[str, Any] | None,
) -> dict[str, Any]:
    claim = (
        "attach the proven recursive core to the normal LM path so answers are "
        "generated by LM logits, not a side channel"
    )
    if not bool(core_reasoning.get("accepted", False)):
        return {
            "status": "blocked",
            "accepted": False,
            "claim": claim,
            "reason": "M-A recursive-core reasoning proof is not accepted",
        }
    if report is None:
        return {
            "status": "pending",
            "accepted": False,
            "claim": claim,
            "reason": "no core-to-LM generation report supplied",
        }
    full = _metric(report, "full_generation_exact")
    full_minus_worst = _metric(report, "full_minus_worst_ablation")
    decision = str(report.get("decision", ""))
    generation_evidence = full > 0.0 or "generation" in decision or "multifamily" in decision
    accepted = bool(
        core_reasoning.get("accepted", False)
        and bool(report.get("accepted", False))
        and generation_evidence
        and full_minus_worst >= 0.05
    )
    return {
        "status": "accepted" if accepted else "rejected",
        "accepted": accepted,
        "claim": claim,
        "metrics": {
            "decision": decision,
            "full_generation_exact": full,
            "full_minus_worst_ablation": full_minus_worst,
            "normal_lm_generation_evidence": bool(generation_evidence),
        },
        "checks": {
            "m_a_accepted": bool(core_reasoning.get("accepted", False)),
            "report_accepted": bool(report.get("accepted", False)),
            "has_generation_metric_or_decision": bool(generation_evidence),
            "ablation_drop_ge_0_05": full_minus_worst >= 0.05,
        },
        "reject_reasons": report.get("reject_reasons", []),
    }


def language_healing_after_core_status(
    language: dict[str, Any],
    core_reasoning: dict[str, Any],
    core_to_lm: dict[str, Any],
) -> dict[str, Any]:
    claim = (
        "heal language only after the recursive core is proven and attached to "
        "the LM-logit path"
    )
    accepted = bool(
        language.get("accepted", False)
        and core_reasoning.get("accepted", False)
        and core_to_lm.get("accepted", False)
    )
    if accepted:
        status = "accepted"
        reason = ""
    elif not bool(core_reasoning.get("accepted", False)):
        status = "blocked"
        reason = "M-A recursive-core reasoning proof is not accepted"
    elif not bool(core_to_lm.get("accepted", False)):
        status = "blocked"
        reason = "M-B core-to-LM attachment is not accepted"
    else:
        status = str(language.get("status", "pending"))
        reason = str(language.get("reason", "language healing report is not accepted"))
    result = dict(language)
    result.update(
        {
            "status": status,
            "accepted": accepted,
            "claim": claim,
            "reason": reason,
            "dependency_checks": {
                "m_a_recursive_core_reasoning_proof": bool(core_reasoning.get("accepted", False)),
                "m_b_core_to_lm_attachment": bool(core_to_lm.get("accepted", False)),
                "language_report_accepted": bool(language.get("accepted", False)),
            },
        }
    )
    return result


def bridge_causal_signal(
    report: dict[str, Any] | None,
    stability_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if stability_report is not None:
        summary = stability_report.get("summary", {})
        accepted = bool(stability_report.get("accepted", False))
        return {
            "status": "accepted" if accepted else "rejected",
            "accepted": accepted,
            "diagnostic_only": True,
            "stability_required": True,
            "reason": (
                "bridge causal signal passed all held-out seed stability gates"
                if accepted
                else "bridge signal is not stable across held-out eval seeds; do not keep tuning this path beyond the pivot cap"
            ),
            "metrics": {
                "num_seeds": int(summary.get("num_seeds", 0)),
                "num_accepted": int(summary.get("num_accepted", 0)),
                "min_gain": float(summary.get("min_gain", 0.0)),
                "mean_gain": float(summary.get("mean_gain", 0.0)),
                "min_family_gain": float(summary.get("min_family_gain", 0.0)),
                "mean_family_gain": float(summary.get("mean_family_gain", 0.0)),
                "min_family_core_accuracy": float(
                    summary.get("min_family_core_accuracy", 0.0)
                ),
                "mean_family_core_accuracy": float(
                    summary.get("mean_family_core_accuracy", 0.0)
                ),
                "min_language_top1_agreement": float(
                    summary.get("min_language_top1_agreement", 0.0)
                ),
            },
            "checks": {
                "all_seed_reports_accepted": accepted,
                "min_gain_ge_0_05": float(summary.get("min_gain", 0.0)) >= 0.05,
                "min_family_gain_ge_0_01": float(summary.get("min_family_gain", 0.0)) >= 0.01,
                "min_family_core_accuracy_ge_0_10": float(
                    summary.get("min_family_core_accuracy", 0.0)
                )
                >= 0.10,
                "min_language_top1_ge_0_50": float(
                    summary.get("min_language_top1_agreement", 0.0)
                )
                >= 0.50,
            },
        }
    if report is None:
        return {
            "status": "pending",
            "accepted": False,
            "reason": "no bridge report supplied",
        }
    after = report.get("after_eval", {})
    family = _family_summary(report)
    gain = float(after.get("gain", 0.0))
    min_family_gain = float(family.get("min_gain", 0.0))
    min_family_core_accuracy = float(family.get("min_core_accuracy", 0.0))
    language_top1 = float(report.get("after_language", {}).get("top1_agreement", 0.0))
    reasoning = gain >= 0.05
    family_gain = min_family_gain >= 0.01
    language = language_top1 >= 0.50
    family_accuracy = min_family_core_accuracy >= 0.10
    accepted = bool(reasoning and family_gain and language and family_accuracy)
    return {
        "status": "accepted" if accepted else "partial" if reasoning and family_gain and language else "rejected",
        "accepted": accepted,
        "diagnostic_only": True,
        "reason": (
            "bridge report is diagnostic only; native reproduction is required"
            if accepted
            else "bridge has a useful signal but has not met all strict family floors"
            if reasoning and language
            else "bridge did not meet the reasoning signal threshold"
        ),
        "metrics": {
            "gain": gain,
            "base_accuracy": float(after.get("base_accuracy", 0.0)),
            "core_accuracy": float(after.get("core_accuracy", 0.0)),
            "min_family_gain": min_family_gain,
            "min_family_core_accuracy": min_family_core_accuracy,
            "language_top1_agreement": language_top1,
        },
        "checks": {
            "gain_ge_0_05": reasoning,
            "min_family_gain_ge_0_01": family_gain,
            "min_family_core_accuracy_ge_0_10": family_accuracy,
            "language_top1_ge_0_50": language,
        },
    }


def native_language_bootstrap_status(report: dict[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {
            "status": "pending",
            "accepted": False,
            "claim": "English and Korean language bootstrap passes non-degeneration/on-policy gates.",
        }
    decision = str(report.get("decision", ""))
    is_language_bootstrap = "language_bootstrap" in decision
    accepted = bool(report.get("accepted", False)) and is_language_bootstrap
    metrics = report.get("eval_metrics", {})
    degeneracy = metrics.get("sample_degeneracy", {})
    return {
        "status": "accepted" if accepted else "manual_review" if bool(report.get("accepted", False)) else "rejected",
        "accepted": accepted,
        "claim": "English and Korean language bootstrap passes non-degeneration/on-policy gates.",
        "reason": "" if accepted else "native report is not an accepted language-bootstrap report",
        "metrics": {
            "decision": decision,
            "vocab_size": int(report.get("vocab_size", 0)),
            "think_eval_loss": float(metrics.get("think_eval_loss", 0.0)),
            "unique_chars": float(degeneracy.get("unique_chars", 0.0)),
            "max_run_fraction": float(degeneracy.get("max_run_fraction", 1.0)),
            "on_policy_count": int(report.get("on_policy_candidates", {}).get("count", 0)),
        },
        "reject_reasons": report.get("reject_reasons", []),
    }


def native_tiny_lm_status(report: dict[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {
            "status": "pending",
            "accepted": False,
            "claim": "donorless native LM learns non-degenerate English/Korean next-token behavior.",
        }
    metrics = report.get("eval_metrics", {})
    degeneracy = metrics.get("sample_degeneracy", {})
    pretrained = report.get("pretrained_init") or {}
    runtime_donor = bool(pretrained.get("runtime_donor", False)) if isinstance(pretrained, dict) else False
    accepted = bool(report.get("accepted", False)) and not runtime_donor
    return {
        "status": "accepted" if accepted else "rejected",
        "accepted": accepted,
        "claim": "donorless native LM learns non-degenerate English/Korean next-token behavior.",
        "metrics": {
            "runtime_donor": runtime_donor,
            "vocab_size": int(report.get("vocab_size", 0)),
            "think_eval_loss": float(metrics.get("think_eval_loss", 0.0)),
            "think0_loss": float(metrics.get("think0_loss", 0.0)),
            "thinking_block_off_loss": float(metrics.get("thinking_block_off_loss", 0.0)),
            "unique_chars": float(degeneracy.get("unique_chars", 0.0)),
            "max_run_fraction": float(degeneracy.get("max_run_fraction", 1.0)),
        },
        "reject_reasons": report.get("reject_reasons", []),
    }


def native_core_causality_status(report: dict[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {
            "status": "pending",
            "accepted": False,
            "claim": "native mandatory recursive core beats core_off/shallow on the same held-out LM/reasoning metric.",
        }
    metrics = report.get("eval_metrics", {})
    ratios = metrics.get("loss_ratios", {})
    full_loss = float(metrics.get("think_eval_loss", 0.0))
    think0_loss = float(metrics.get("think0_loss", 0.0))
    off_loss = float(metrics.get("thinking_block_off_loss", 0.0))
    full_vs_best_shallow = ratios.get("full_vs_best_shallow_depth")
    beats_think0 = full_loss > 0.0 and think0_loss > full_loss
    beats_off = full_loss > 0.0 and off_loss > full_loss
    beats_shallow = (
        isinstance(full_vs_best_shallow, (int, float))
        and float(full_vs_best_shallow) < 1.0
    )
    has_reset_or_corruption = bool(
        metrics.get("state_reset_ablation")
        or metrics.get("corruption_ablation")
        or report.get("state_reset_ablation")
        or report.get("corruption_ablation")
    )
    state_reset_ablation = metrics.get("state_reset_ablation") or report.get("state_reset_ablation") or {}
    corruption_ablation = metrics.get("corruption_ablation") or report.get("corruption_ablation") or {}
    state_reset_loss = (
        float(state_reset_ablation.get("loss"))
        if isinstance(state_reset_ablation, dict)
        and isinstance(state_reset_ablation.get("loss"), (int, float))
        else None
    )
    corruption_loss = (
        float(corruption_ablation.get("loss"))
        if isinstance(corruption_ablation, dict)
        and isinstance(corruption_ablation.get("loss"), (int, float))
        else None
    )
    reset_or_corruption_degrades = bool(
        (state_reset_loss is not None and state_reset_loss > full_loss)
        or (corruption_loss is not None and corruption_loss > full_loss)
    )
    accepted = bool(
        beats_think0
        and beats_off
        and beats_shallow
        and has_reset_or_corruption
        and reset_or_corruption_degrades
    )
    return {
        "status": "accepted" if accepted else "partial" if beats_think0 and beats_off and beats_shallow else "rejected",
        "accepted": accepted,
        "claim": "native mandatory recursive core beats core_off/shallow on the same held-out LM/reasoning metric.",
        "reason": (
            "core improves loss but reset/corruption ablation is still missing"
            if beats_think0 and beats_off and beats_shallow and not has_reset_or_corruption
            else ""
        ),
        "metrics": {
            "think_eval_loss": full_loss,
            "think0_loss": think0_loss,
            "thinking_block_off_loss": off_loss,
            "full_vs_best_shallow_depth": (
                float(full_vs_best_shallow)
                if isinstance(full_vs_best_shallow, (int, float))
                else None
            ),
            "beats_think0": bool(beats_think0),
            "beats_off": bool(beats_off),
            "beats_best_shallow": bool(beats_shallow),
            "has_reset_or_corruption_ablation": bool(has_reset_or_corruption),
            "reset_or_corruption_degrades": bool(reset_or_corruption_degrades),
            "state_reset_loss": state_reset_loss,
            "corruption_loss": corruption_loss,
        },
    }


def qwen36_eval_harness_status(report: dict[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {
            "status": "pending",
            "accepted": False,
            "claim": "published Qwen3.6 targets, public task/scorer mapping, and QTRM-Native outputs are saved under one comparison manifest.",
        }
    checks = report.get("acceptance_checks", {})
    accepted = bool(report.get("accepted", False))
    return {
        "status": "accepted" if accepted else "rejected",
        "accepted": accepted,
        "claim": "published Qwen3.6 targets, public task/scorer mapping, and QTRM-Native outputs are saved under one comparison manifest.",
        "metrics": {
            "comparison_mode": str(report.get("comparison_mode", "")),
            "direct_qwen36_rerun_required": bool(
                report.get("direct_qwen36_rerun_required", True)
            ),
            "benchmark_count": len(report.get("benchmark_map", {})),
            "artifact_count": len(report.get("qtrm_native", {}).get("artifacts", [])),
        },
        "checks": checks,
        "limitations": report.get("limitations", []),
    }


def scoped_raw_reasoning_status(report: dict[str, Any] | None) -> dict[str, Any]:
    claim = "QTRM-Native beats Qwen3.6-27B on a scoped raw reasoning/memory gate with ablations."
    if report is None:
        return {
            "status": "pending",
            "accepted": False,
            "claim": claim,
        }
    accepted = bool(report.get("accepted", False))
    best = report.get("best_qtrm_native", {}) or {}
    baseline = report.get("qwen36_baseline", {}) or {}
    return {
        "status": "accepted" if accepted else "rejected",
        "accepted": accepted,
        "claim": claim,
        "metrics": {
            "suite_id": str(report.get("suite_id", "")),
            "qtrm_score": best.get("full_generation_exact"),
            "qwen36_score": baseline.get("score"),
            "core_gain": best.get("core_gain"),
            "ablation_drop": best.get("ablation_drop"),
            "min_family_generation_exact": best.get("min_family_generation_exact"),
        },
        "checks": report.get("acceptance_checks", {}),
        "reject_reasons": report.get("reject_reasons", []),
        "limitations": report.get("limitations", []),
    }


def public_benchmark_parity_status(report: dict[str, Any] | None) -> dict[str, Any]:
    claim = "QTRM-Native reaches parity band on selected public benchmarks."
    if report is None:
        return {
            "status": "pending",
            "accepted": False,
            "claim": claim,
        }
    accepted = bool(report.get("accepted", False))
    metrics = report.get("metrics", {}) if isinstance(report.get("metrics", {}), dict) else {}
    score = metrics.get("accuracy")
    target = report.get("qwen36_target_score")
    parity_floor = report.get("parity_floor")
    cases = metrics.get("cases")
    min_cases = report.get("min_cases_for_parity")
    return {
        "status": "accepted" if accepted else "rejected",
        "accepted": accepted,
        "claim": claim,
        "metrics": {
            "benchmark_id": str(report.get("benchmark_id", "")),
            "benchmark_name": str(report.get("benchmark_name", "")),
            "qtrm_score": score,
            "qwen36_target_score": target,
            "parity_floor": parity_floor,
            "cases": cases,
            "min_cases_for_parity": min_cases,
            "invalid_pred_rate": metrics.get("invalid_pred_rate"),
            "prompt_echo_rate": metrics.get("prompt_echo_rate"),
            "pred_answer_histogram": metrics.get("pred_answer_histogram", {}),
            "by_category": metrics.get("by_category", {}),
        },
        "checks": {
            "public_benchmark_report_present": True,
            "has_accuracy": isinstance(score, (int, float)),
            "has_qwen36_target": isinstance(target, (int, float)),
            "score_ge_parity_floor": (
                isinstance(score, (int, float))
                and isinstance(parity_floor, (int, float))
                and float(score) >= float(parity_floor)
            ),
            "cases_ge_min": (
                isinstance(cases, int)
                and isinstance(min_cases, int)
                and int(cases) >= int(min_cases)
            ),
        },
        "reject_reasons": [] if accepted else [
            key
            for key, ok in {
                "has_accuracy": isinstance(score, (int, float)),
                "has_qwen36_target": isinstance(target, (int, float)),
                "score_ge_parity_floor": (
                    isinstance(score, (int, float))
                    and isinstance(parity_floor, (int, float))
                    and float(score) >= float(parity_floor)
                ),
                "cases_ge_min": (
                    isinstance(cases, int)
                    and isinstance(min_cases, int)
                    and int(cases) >= int(min_cases)
                ),
            }.items()
            if not ok
        ],
        "limitations": report.get("limitations", []),
    }


def build_status(
    *,
    bridge_report: dict[str, Any] | None = None,
    bridge_stability_report: dict[str, Any] | None = None,
    native_report: dict[str, Any] | None = None,
    native_core_report: dict[str, Any] | None = None,
    core_reasoning_report: dict[str, Any] | None = None,
    eval_manifest: dict[str, Any] | None = None,
    m6_report: dict[str, Any] | None = None,
    m7_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bridge = bridge_causal_signal(bridge_report, bridge_stability_report)
    native = native_language_bootstrap_status(native_report)
    native_tiny = native_tiny_lm_status(native_report)
    native_core = native_core_causality_status(native_core_report or native_report)
    recursive_reasoning = recursive_core_reasoning_proof_status(core_reasoning_report)
    core_to_lm = core_to_lm_attachment_status(recursive_reasoning, core_reasoning_report)
    language_healing = language_healing_after_core_status(native, recursive_reasoning, core_to_lm)
    eval_harness = qwen36_eval_harness_status(eval_manifest)
    scoped_raw_reasoning = scoped_raw_reasoning_status(m6_report)
    public_parity = public_benchmark_parity_status(m7_report)
    dependency_chain = {
        "M_A_RECURSIVE_CORE_REASONING_PROOF": recursive_reasoning,
        "M_B_CORE_TO_LM_ATTACHMENT": core_to_lm,
        "M_C_LANGUAGE_HEALING_AFTER_CORE": language_healing,
    }
    milestones = {
        "M0_TARGET_CONTRACT": {
            "status": "accepted",
            "accepted": True,
            "claim": "QTRM-Native-2B/3B must beat Qwen3.6-27B only through native token-to-logit causal path.",
        },
        "M1_BRIDGE_CAUSAL_SIGNAL": bridge,
        "M2_NATIVE_TINY_LM": native_tiny,
        "M3_NATIVE_CORE_CAUSALITY": native_core,
        "M4_NATIVE_LANGUAGE_BOOTSTRAP": language_healing,
        "M5_QWEN36_EVAL_HARNESS": eval_harness,
        "M6_NATIVE_SMALL_BEATS_27B_ON_SCOPED_RAW_REASONING": scoped_raw_reasoning,
        "M7_NATIVE_2B_3B_PUBLIC_BENCH_PARITY": public_parity,
        "M8_NATIVE_2B_3B_PUBLIC_BENCH_WIN": {
            "status": "pending",
            "accepted": False,
            "claim": "QTRM-Native-2B/3B exceeds Qwen3.6-27B target scores on selected public benchmarks.",
        },
    }
    milestone_schedule = {key: value.copy() for key, value in MILESTONE_SCHEDULE.items()}
    if eval_harness.get("accepted"):
        milestone_schedule["M5_QWEN36_EVAL_HARNESS"][
            "actual_duration"
        ] = "same day on 2026-05-15; accepted public-target manifest"
    if m6_report is not None:
        milestone_schedule["M6_NATIVE_SMALL_BEATS_27B_ON_SCOPED_RAW_REASONING"][
            "actual_duration"
        ] = (
            "same day on 2026-05-15; rejected until matched Qwen3.6 scoped baseline exists"
            if not scoped_raw_reasoning.get("accepted")
            else "same day on 2026-05-15; accepted scoped raw-reasoning win"
        )
    if m7_report is not None:
        milestone_schedule["M7_NATIVE_2B_3B_PUBLIC_BENCH_PARITY"][
            "actual_duration"
        ] = (
            "started on 2026-05-16; public benchmark parity rejected by current checkpoint"
            if not public_parity.get("accepted")
            else "started on 2026-05-16; accepted public benchmark parity"
        )
    return {
        "project_goal": "QTRM-Native-2B/3B beats Qwen3.6-27B on reasoning/memory-relevant benchmarks with about 10x fewer parameters.",
        "qwen36_27b_targets": QWEN36_27B_TARGETS,
        "milestone_order": MILESTONE_ORDER,
        "fast_path_strategy": [
            "Do not claim bridge results as native wins.",
            "Prove M-A recursive-core reasoning before M-B core-to-LM attachment or M-C language healing.",
            "After the M1 stability rejection, pivot to native instead of spending more time tuning the bridge.",
            "Use the bridge only to identify causal patterns, then move quickly to native.",
            "Run automatic gates with fixed thresholds; discard runs that miss family floors.",
            "Prefer Qwen tokenizer/config/pretrained initialization over random-init novelty.",
            "Use published Qwen3.6 benchmark scores as the fastest baseline; direct DGX rerun is optional for custom suites.",
        ],
        "milestone_schedule": milestone_schedule,
        "compute_plan": COMPUTE_PLAN,
        "core_to_lm_to_healing_dependency_order": DEPENDENCY_ORDER,
        "core_to_lm_to_healing_dependencies": dependency_chain,
        "milestones": milestones,
        "next_action": _next_action(milestones, dependency_chain),
    }


def _next_action(
    milestones: dict[str, Any],
    dependency_chain: dict[str, Any] | None = None,
) -> str:
    if dependency_chain is not None:
        for dependency_id in DEPENDENCY_ORDER:
            item = dependency_chain[dependency_id]
            if not bool(item.get("accepted", False)):
                return f"work on {dependency_id}: {item.get('claim') or item.get('reason')}"
    for milestone_id in MILESTONE_ORDER:
        item = milestones[milestone_id]
        if (
            milestone_id == "M1_BRIDGE_CAUSAL_SIGNAL"
            and item.get("status") == "rejected"
            and bool(item.get("diagnostic_only", False))
        ):
            continue
        if not bool(item.get("accepted", False)):
            return f"work on {milestone_id}: {item.get('claim') or item.get('reason')}"
    return "all milestones accepted; prepare paper-grade replication package"


def render_markdown(status: dict[str, Any]) -> str:
    lines = [
        "# QTRM-Native 27B Target Status",
        "",
        f"Goal: {status['project_goal']}",
        "",
        "## Qwen3.6-27B Targets",
        "",
        "| Benchmark | Target |",
        "|---|---:|",
    ]
    for key, value in status["qwen36_27b_targets"].items():
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "## Milestones", "", "| ID | Status | Accepted | Claim/Reason |", "|---|---|---:|---|"])
    for milestone_id in status["milestone_order"]:
        item = status["milestones"][milestone_id]
        claim = item.get("claim") or item.get("reason") or ""
        lines.append(
            f"| `{milestone_id}` | {item.get('status')} | {bool(item.get('accepted'))} | {claim} |"
        )
    lines.extend(
        [
            "",
            "## Core-to-LM-to-Healing Dependency",
            "",
            "| ID | Status | Accepted | Claim/Reason |",
            "|---|---|---:|---|",
        ]
    )
    for dependency_id in status["core_to_lm_to_healing_dependency_order"]:
        item = status["core_to_lm_to_healing_dependencies"][dependency_id]
        claim = item.get("claim") or item.get("reason") or ""
        lines.append(
            f"| `{dependency_id}` | {item.get('status')} | {bool(item.get('accepted'))} | {claim} |"
        )
    lines.extend(
        [
            "",
            "## Fast-Path Schedule",
            "",
            "| ID | Estimate | Max Before Pivot | Actual |",
            "|---|---|---|---|",
        ]
    )
    for milestone_id in status["milestone_order"]:
        schedule = status["milestone_schedule"][milestone_id]
        lines.append(
            "| "
            f"`{milestone_id}` | "
            f"{schedule['fast_path_estimate']} | "
            f"{schedule['max_time_before_pivot']} | "
            f"{schedule['actual_duration']} |"
        )
    lines.extend(["", "## Fast-Path Strategy", ""])
    for item in status["fast_path_strategy"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Compute Plan",
            "",
            "| ID | 4090 Feasibility | Preferred Compute | Fastest Path |",
            "|---|---|---|---|",
        ]
    )
    for milestone_id in status["milestone_order"]:
        compute = status["compute_plan"][milestone_id]
        lines.append(
            "| "
            f"`{milestone_id}` | "
            f"{compute['4090_feasibility']} | "
            f"{compute['preferred_compute']} | "
            f"{compute['fastest_path']} |"
        )
    lines.extend(["", "## Next Action", "", status["next_action"], ""])
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-report", default="")
    parser.add_argument("--bridge-stability-report", default="")
    parser.add_argument("--native-report", default="")
    parser.add_argument("--native-core-report", default="")
    parser.add_argument("--core-reasoning-report", default="")
    parser.add_argument("--eval-manifest", default="")
    parser.add_argument("--m6-report", default="")
    parser.add_argument("--m7-report", default="")
    parser.add_argument("--out-json", default="local_eval/qtrm_native_27b_milestone_status/report.json")
    parser.add_argument("--out-md", default="local_eval/qtrm_native_27b_milestone_status/report.md")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    status = build_status(
        bridge_report=_load_json(args.bridge_report),
        bridge_stability_report=_load_json(args.bridge_stability_report),
        native_report=_load_json(args.native_report),
        native_core_report=_load_json(args.native_core_report),
        core_reasoning_report=_load_json(args.core_reasoning_report),
        eval_manifest=_load_json(args.eval_manifest),
        m6_report=_load_json(args.m6_report),
        m7_report=_load_json(args.m7_report),
    )
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(render_markdown(status), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
