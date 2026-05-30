from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

DEFAULT_DONOR_MODE = "donor_only_no_evidence"
DEFAULT_CORE_OFF_MODE = "qtrm_core_off_no_evidence"
DEFAULT_CORE_DEPTH_MODES = [
    "qtrm_core_steps_1_no_evidence",
    "qtrm_core_steps_2_no_evidence",
    "qtrm_core_steps_4_no_evidence",
    "qtrm_core_steps_8_no_evidence",
]
DEFAULT_MEMORY_ON_MODE = "qtrm_memory_on_no_evidence"
DEFAULT_MEMORY_OFF_MODE = "qtrm_memory_off_no_evidence"
DEFAULT_COMPOSITION_FULL_MODE = "qtrm_core_memory_on_no_evidence"
DEFAULT_COMPOSITION_CORE_OFF_MODE = "qtrm_core_off_memory_on_no_evidence"
DEFAULT_COMPOSITION_MEMORY_OFF_MODE = "qtrm_core_on_memory_off_no_evidence"
DEFAULT_TEMPORAL_SPATIAL_CONTEXT_ON_MODE = "qtrm_core_steps_8_no_evidence"
DEFAULT_TEMPORAL_SPATIAL_CONTEXT_OFF_MODE = (
    "qtrm_core_steps_8_temporal_spatial_off_no_evidence"
)
DEFAULT_TRANSITION_STATE_OFF_MODE = "qtrm_core_steps_8_transition_state_off_no_evidence"

# === RI-4: Sparse Persistent Memory Ablations (new for 2026-06) ===
# These are the highest-value new modes for proving causal contribution of
# MSA/Raven-style sparse memory inside the One-Body recurrence.
DEFAULT_HYBRID_SLOTS_ON_MODE = "hybrid_sparse_slots_on_no_evidence"
DEFAULT_HYBRID_SLOTS_OFF_MODE = "hybrid_sparse_slots_off_no_evidence"
DEFAULT_HYBRID_PERSISTENCE_ABLATION_MODE = "hybrid_persistent_memory_ablation_no_evidence"
DEFAULT_HYBRID_ROUTER_ABLATION_MODE = "hybrid_sparse_router_ablation_no_evidence"
DEFAULT_HYBRID_RECURRENCE_OFF_MODE = "hybrid_recurrence_off_no_evidence"
DEFAULT_HYBRID_DEPTH_MODES = [
    "hybrid_recurrence_depth_1_no_evidence",
    "hybrid_recurrence_depth_4_no_evidence",
    "hybrid_recurrence_depth_8_no_evidence",
    "hybrid_recurrence_depth_12_no_evidence",
]
DEFAULT_HYBRID_STOCHASTIC_OFF_MODE = "hybrid_stochastic_breadth_off_no_evidence"
DEFAULT_HYBRID_556_FULL_MODE = "hybrid_556_full_no_evidence"
DEFAULT_HYBRID_556_STOCH_ZERO_MODE = "hybrid_556_stoch_zero_no_evidence"
DEFAULT_HYBRID_556_GOLD_OFF_MODE = "hybrid_556_gold_off_no_evidence"
DEFAULT_HYBRID_556_PROTECTION_OFF_MODE = "hybrid_556_protection_off_no_evidence"
DEFAULT_HYBRID_556_DECAY_DISABLED_MODE = "hybrid_556_decay_disabled_no_evidence"
DEFAULT_HYBRID_556_ABLATION_MODES = [
    DEFAULT_HYBRID_556_STOCH_ZERO_MODE,
    DEFAULT_HYBRID_556_GOLD_OFF_MODE,
    DEFAULT_HYBRID_556_PROTECTION_OFF_MODE,
    DEFAULT_HYBRID_556_DECAY_DISABLED_MODE,
]


PURE_RECURSIVE_MODE_SEMANTICS: dict[str, str] = {
    "donor": (
        "Donor baseline. QTRM residual logits are forced off and donor logits are "
        "used as the scoring policy; this is the real donor-only comparison."
    ),
    "core_off": (
        "Internal QTRM ablation. The model still runs through the QTRM forward path "
        "with disable_core=True; donor fallback is not forced, so this is not "
        "equivalent to donor_only."
    ),
    "deepest_core": (
        "QTRM candidate with recursive core enabled at the deepest evaluated "
        "core_steps value."
    ),
    "transition_state_off": (
        "QTRM candidate with the recursive core still enabled but the explicit "
        "transition-state/code path disabled; this tests whether that state path "
        "is answer-causal."
    ),
    # RI-4 new semantics
    "hybrid_sparse_slots_on": (
        "RI-4 candidate: OneBodyParallelHybridBlock with persistent sparse memory slots enabled. "
        "Slots are carried across recurrence steps and actively read/injected into the recurrent computation."
    ),
    "hybrid_sparse_slots_off": (
        "RI-4 ablation: Same hybrid architecture but sparse slot router is disabled (dense behavior). "
        "Used to prove causal contribution of the persistent sparse memory mechanism."
    ),
    "hybrid_persistent_memory_ablation": (
        "RI-4 strong ablation: Persistent slots are carried but receive no selective update / persistence protection "
        "(effectively dense rehearsal on all slots). Tests the value of selective write + strong persistence."
    ),
    "hybrid_recurrence_depth": (
        "RI-1 candidate: OneBodyParallelHybridBlock used as the answer-state recurrent engine at a named "
        "test-time depth budget. Used for causal depth-scaling curves."
    ),
    "hybrid_recurrence_off": (
        "RI-1 ablation: answer-state recurrence is disabled while the same no-retrieval scoring contract is kept."
    ),
    "hybrid_stochastic_breadth_off": (
        "RI-3 ablation: hybrid recurrent engine remains active but stochastic breadth is zeroed."
    ),
    "hybrid_556_full": (
        "RI-3 candidate: full 5.56-style hybrid recipe with stochastic breadth, gold structural injection, "
        "attractor protection, and scheduled binding decay active."
    ),
    "hybrid_556_stoch_zero": (
        "RI-3 ablation: full hybrid recipe except stochastic breadth is zeroed."
    ),
    "hybrid_556_gold_off": (
        "RI-3 ablation: full hybrid recipe except gold structural injection is disabled."
    ),
    "hybrid_556_protection_off": (
        "RI-3 ablation: full hybrid recipe except attractor protection during rehearsal is disabled."
    ),
    "hybrid_556_decay_disabled": (
        "RI-3 ablation: full hybrid recipe except scheduled binding decay is disabled."
    ),
}


def _records_by_mode(records: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record.get("mode", "unknown")), []).append(record)
    return grouped


def _summary(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    count = len(rows)
    hits = sum(1 for row in rows if bool(row.get("hit")))
    return {
        "count": count,
        "hits": hits,
        "accuracy": hits / count if count else 0.0,
    }


def _mode_summary(by_mode: dict[str, list[dict[str, Any]]], mode: str) -> dict[str, Any]:
    summary = _summary(by_mode.get(mode, []))
    summary["mode"] = mode
    return summary


def _summaries_by_field(
    records: Iterable[dict[str, Any]],
    field: str,
) -> dict[str, dict[str, dict[str, Any]]]:
    by_bucket_mode: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for record in records:
        bucket = str(record.get(field) or "unknown")
        mode = str(record.get("mode", "unknown"))
        by_bucket_mode.setdefault(bucket, {}).setdefault(mode, []).append(record)

    out: dict[str, dict[str, dict[str, Any]]] = {}
    for bucket, rows_by_mode in sorted(by_bucket_mode.items()):
        out[bucket] = {}
        for mode, rows in sorted(rows_by_mode.items()):
            summary = _summary(rows)
            summary["mode"] = mode
            out[bucket][mode] = summary
    return out


def _compare(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    hit_advantage = int(candidate["hits"]) - int(baseline["hits"])
    accuracy_advantage = float(candidate["accuracy"]) - float(baseline["accuracy"])
    return {
        "candidate_mode": candidate["mode"],
        "baseline_mode": baseline["mode"],
        "candidate_hits": int(candidate["hits"]),
        "baseline_hits": int(baseline["hits"]),
        "candidate_count": int(candidate["count"]),
        "baseline_count": int(baseline["count"]),
        "hit_advantage": hit_advantage,
        "accuracy_advantage": accuracy_advantage,
        "candidate_beats_baseline": hit_advantage > 0 or accuracy_advantage > 0.0,
    }


def _core_step_from_mode(mode: str) -> int | None:
    match = re.search(r"core_steps_(\d+)", mode)
    return int(match.group(1)) if match else None


def _hybrid_depth_from_mode(mode: str) -> int | None:
    match = re.fullmatch(r"hybrid_recurrence_depth_(\d+)_no_evidence", mode)
    return int(match.group(1)) if match else None


def _canonical_completion(record: dict[str, Any]) -> str:
    completion = str(record.get("completion", "")).strip().casefold()
    return re.sub(r"\s+", " ", completion)


def _depth_output_diversity(
    by_mode: dict[str, list[dict[str, Any]]],
    depth_modes: Iterable[str],
) -> dict[str, Any]:
    modes = list(depth_modes)
    if len(modes) < 2:
        return {
            "measured": False,
            "reason": "fewer_than_two_depth_modes",
            "case_count": 0,
            "identical_case_count": 0,
            "changed_case_count": 0,
            "all_depth_outputs_identical": False,
            "examples": [],
        }

    by_case: dict[Any, dict[str, dict[str, Any]]] = {}
    for mode in modes:
        for record in by_mode.get(mode, []):
            if "completion" not in record:
                continue
            by_case.setdefault(record.get("id"), {})[mode] = record

    comparable: list[dict[str, Any]] = []
    for case_id, rows_by_mode in by_case.items():
        if all(mode in rows_by_mode for mode in modes):
            outputs = {
                mode: _canonical_completion(rows_by_mode[mode])
                for mode in modes
            }
            if all(output == "" for output in outputs.values()):
                continue
            comparable.append(
                {
                    "id": case_id,
                    "outputs": outputs,
                    "all_identical": len(set(outputs.values())) == 1,
                }
            )

    if not comparable:
        return {
            "measured": False,
            "reason": "missing_completion_records",
            "case_count": 0,
            "identical_case_count": 0,
            "changed_case_count": 0,
            "all_depth_outputs_identical": False,
            "examples": [],
        }

    identical = [row for row in comparable if bool(row["all_identical"])]
    changed = [row for row in comparable if not bool(row["all_identical"])]
    return {
        "measured": True,
        "reason": "ok",
        "case_count": len(comparable),
        "identical_case_count": len(identical),
        "changed_case_count": len(changed),
        "all_depth_outputs_identical": len(changed) == 0,
        "examples": comparable[:5],
    }


def _shortcut_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    shortcuts: list[dict[str, Any]] = []
    for record in records:
        prompt = str(record.get("prompt", ""))
        completion = str(record.get("completion", ""))
        has_memoryos_text = "MemoryOS evidence" in prompt or "MemoryOS evidence" in completion
        has_hidden_evidence = (
            bool(record.get("memoryos_used"))
            or bool(record.get("retrieval_used"))
            or int(record.get("evidence_token_count", 0) or 0) > 0
            or int(record.get("workspace_memory_token_count", 0) or 0) > 0
            or has_memoryos_text
        )
        if has_hidden_evidence:
            shortcuts.append(
                {
                    "id": record.get("id"),
                    "mode": record.get("mode"),
                    "memoryos_used": bool(record.get("memoryos_used")),
                    "retrieval_used": bool(record.get("retrieval_used")),
                    "evidence_token_count": int(record.get("evidence_token_count", 0) or 0),
                    "workspace_memory_token_count": int(
                        record.get("workspace_memory_token_count", 0) or 0
                    ),
                }
            )
    return shortcuts


def _unique_field_values(records: Iterable[dict[str, Any]], field: str) -> list[str]:
    values = {
        str(record[field])
        for record in records
        if field in record and record[field] is not None and str(record[field]).strip()
    }
    return sorted(values)


def _status_from_checks(
    *,
    missing_modes: list[str],
    failed_checks: list[str],
    mode_summaries: dict[str, dict[str, Any]],
) -> str:
    if missing_modes:
        return "inconclusive"
    if any(int(summary["count"]) == 0 for summary in mode_summaries.values()):
        return "inconclusive"
    return "rejected" if failed_checks else "accepted"


def build_pure_recursive_reasoning_gate(
    records: Iterable[dict[str, Any]],
    *,
    donor_mode: str = DEFAULT_DONOR_MODE,
    core_off_mode: str = DEFAULT_CORE_OFF_MODE,
    core_depth_modes: Iterable[str] = DEFAULT_CORE_DEPTH_MODES,
    transition_state_off_mode: str = DEFAULT_TRANSITION_STATE_OFF_MODE,
    min_hit_advantage: int = 1,
) -> dict[str, Any]:
    """Gate the TRM-like raw reasoning claim without retrieval/evidence shortcuts."""
    record_list = list(records)
    by_mode = _records_by_mode(record_list)
    depth_modes = [mode for mode in core_depth_modes if mode in by_mode]
    missing_modes = [mode for mode in [donor_mode, core_off_mode] if mode not in by_mode]
    if not depth_modes:
        missing_modes.append("qtrm_core_steps_N_no_evidence")

    donor = _mode_summary(by_mode, donor_mode)
    core_off = _mode_summary(by_mode, core_off_mode)
    depth_ladder = [
        _mode_summary(by_mode, mode)
        for mode in sorted(depth_modes, key=lambda item: _core_step_from_mode(item) or 0)
    ]
    deepest = depth_ladder[-1] if depth_ladder else {
        "mode": None,
        "count": 0,
        "hits": 0,
        "accuracy": 0.0,
    }

    donor_comparison = _compare(deepest, donor)
    core_off_comparison = _compare(deepest, core_off)
    transition_state_off = (
        _mode_summary(by_mode, transition_state_off_mode)
        if transition_state_off_mode in by_mode
        else None
    )
    transition_state_off_comparison = (
        _compare(deepest, transition_state_off)
        if transition_state_off is not None
        else None
    )
    depth_scaling_gain = any(
        int(later["hits"]) - int(earlier["hits"]) >= min_hit_advantage
        for earlier, later in zip(depth_ladder, depth_ladder[1:])
    )
    depth_output_diversity = _depth_output_diversity(
        by_mode,
        [str(row["mode"]) for row in depth_ladder],
    )
    shortcuts = _shortcut_records(record_list)

    failed_checks: list[str] = []
    passed_checks: list[str] = []

    if int(deepest["hits"]) - int(core_off["hits"]) >= min_hit_advantage:
        passed_checks.append("deep_core_beats_core_off")
    else:
        failed_checks.append("deep_core_does_not_beat_core_off")
    if int(deepest["hits"]) - int(donor["hits"]) >= min_hit_advantage:
        passed_checks.append("deep_core_beats_donor")
    else:
        failed_checks.append("deep_core_does_not_beat_donor")
    if len(depth_ladder) >= 2 and depth_scaling_gain:
        passed_checks.append("depth_scaling_gain_present")
    else:
        failed_checks.append("no_depth_scaling_gain")
    if transition_state_off is not None:
        if int(deepest["hits"]) - int(transition_state_off["hits"]) >= min_hit_advantage:
            passed_checks.append("deep_core_beats_transition_state_off")
        else:
            failed_checks.append("deep_core_does_not_beat_transition_state_off")
    if depth_output_diversity["measured"]:
        if depth_output_diversity["all_depth_outputs_identical"]:
            failed_checks.append("depth_outputs_identical_across_steps")
        else:
            passed_checks.append("depth_outputs_not_all_identical")
    if shortcuts:
        failed_checks.append("non_raw_shortcut_present")
    else:
        passed_checks.append("no_retrieval_or_memoryos_shortcut")

    mode_summaries = {
        donor_mode: donor,
        core_off_mode: core_off,
        **{str(row["mode"]): row for row in depth_ladder},
    }
    if transition_state_off is not None:
        mode_summaries[transition_state_off_mode] = transition_state_off
    status = _status_from_checks(
        missing_modes=missing_modes,
        failed_checks=failed_checks,
        mode_summaries=mode_summaries,
    )

    return {
        "gate_type": "pure_recursive_reasoning",
        "claim": (
            "QTRM recursive core depth should improve held-out reasoning without "
            "retrieval, MemoryOS, or hidden evidence shortcuts."
        ),
        "status": status,
        "donor_mode": donor_mode,
        "core_off_mode": core_off_mode,
        "deepest_core_mode": deepest.get("mode"),
        "donor": donor,
        "core_off": core_off,
        "deepest_core": deepest,
        "transition_state_off": transition_state_off or {},
        "depth_ladder": depth_ladder,
        "depth_output_diversity": depth_output_diversity,
        "by_task_family": _summaries_by_field(record_list, "task_family"),
        "by_reasoning_family": _summaries_by_field(record_list, "reasoning_family"),
        "by_expected_paradigm": _summaries_by_field(record_list, "expected_paradigm"),
        "donor_comparison": donor_comparison,
        "core_off_comparison": core_off_comparison,
        "transition_state_off_comparison": transition_state_off_comparison or {},
        "mode_semantics": PURE_RECURSIVE_MODE_SEMANTICS,
        "eval_contract": {
            "scoring": _unique_field_values(record_list, "scoring"),
            "choice_score_normalization": _unique_field_values(
                record_list,
                "choice_score_normalization",
            ),
        },
        "shortcut_records": shortcuts,
        "missing_modes": missing_modes,
        "failed_checks": failed_checks,
        "passed_checks": passed_checks,
        "recommendation": _recommendation_for_gate("pure_recursive_reasoning", status, failed_checks),
    }


def build_ri4_sparse_memory_gate(
    records: Iterable[dict[str, Any]],
    *,
    slots_on_mode: str = DEFAULT_HYBRID_SLOTS_ON_MODE,
    slots_off_mode: str = DEFAULT_HYBRID_SLOTS_OFF_MODE,
    persistence_ablation_mode: str = DEFAULT_HYBRID_PERSISTENCE_ABLATION_MODE,
    min_hit_advantage: int = 1,
) -> dict[str, Any]:
    """
    RI-4 Gate: Proves that MSA/Raven-style sparse persistent memory inside the
    One-Body recurrence causally improves raw intelligence (no retrieval).

    This is currently the highest-value missing measurement infrastructure.
    """
    record_list = list(records)
    by_mode = _records_by_mode(record_list)

    slots_on = _mode_summary(by_mode, slots_on_mode)
    slots_off = _mode_summary(by_mode, slots_off_mode)

    persistence_ablation = (
        _mode_summary(by_mode, persistence_ablation_mode)
        if persistence_ablation_mode in by_mode
        else None
    )

    failed_checks: list[str] = []
    passed_checks: list[str] = []

    # Core RI-4 claim: sparse persistent memory on beats off
    if int(slots_on["hits"]) - int(slots_off["hits"]) >= min_hit_advantage:
        passed_checks.append("sparse_persistent_slots_beats_disabled")
    else:
        failed_checks.append("sparse_persistent_slots_do_not_beat_disabled")

    if persistence_ablation is not None:
        if int(slots_on["hits"]) - int(persistence_ablation["hits"]) >= min_hit_advantage:
            passed_checks.append("selective_persistence_beats_dense_rehearsal")
        else:
            failed_checks.append("selective_persistence_does_not_beat_dense_rehearsal")

    status = "accepted" if not failed_checks else "rejected"

    return {
        "gate_type": "ri4_sparse_persistent_memory",
        "claim": (
            "MSA/Raven-style sparse persistent memory slots inside OneBodyParallelHybridBlock "
            "causally improve held-out raw reasoning when carried across steps with selective "
            "5.56-style rehearsal and strong persistence on non-selected slots."
        ),
        "status": status,
        "slots_on_mode": slots_on_mode,
        "slots_off_mode": slots_off_mode,
        "persistence_ablation_mode": persistence_ablation_mode,
        "slots_on": slots_on,
        "slots_off": slots_off,
        "persistence_ablation": persistence_ablation or {},
        "slots_on_vs_off": _compare(slots_on, slots_off),
        "failed_checks": failed_checks,
        "passed_checks": passed_checks,
        "mode_semantics": PURE_RECURSIVE_MODE_SEMANTICS,
        "recommendation": "Run with real gold + clean probes. This is the primary gate for RI-4.",
    }


def build_hybrid_recurrence_depth_gate(
    records: Iterable[dict[str, Any]],
    *,
    recurrence_off_mode: str = DEFAULT_HYBRID_RECURRENCE_OFF_MODE,
    depth_modes: Iterable[str] = DEFAULT_HYBRID_DEPTH_MODES,
    stochastic_off_mode: str = DEFAULT_HYBRID_STOCHASTIC_OFF_MODE,
    min_hit_advantage: int = 1,
) -> dict[str, Any]:
    """RI-1/RI-3 gate for hybrid recurrent-depth and stochastic breadth scaling."""
    record_list = list(records)
    by_mode = _records_by_mode(record_list)
    requested_depth_modes = list(depth_modes)
    present_depth_modes = [mode for mode in requested_depth_modes if mode in by_mode]
    depth_ladder = [
        _mode_summary(by_mode, mode)
        for mode in sorted(present_depth_modes, key=lambda item: _hybrid_depth_from_mode(item) or 0)
    ]
    recurrence_off = _mode_summary(by_mode, recurrence_off_mode)
    stochastic_off = (
        _mode_summary(by_mode, stochastic_off_mode)
        if stochastic_off_mode in by_mode
        else None
    )

    missing_modes = [
        mode for mode in [recurrence_off_mode, *requested_depth_modes] if mode not in by_mode
    ]
    failed_checks: list[str] = []
    passed_checks: list[str] = []

    if len(depth_ladder) >= 2:
        hit_pairs = [
            (int(earlier["hits"]), int(later["hits"]))
            for earlier, later in zip(depth_ladder, depth_ladder[1:])
        ]
        if any(later > earlier for earlier, later in hit_pairs):
            passed_checks.append("depth_scaling_gain_present")
        else:
            failed_checks.append("no_depth_scaling_gain")
        if all(later >= earlier for earlier, later in hit_pairs):
            passed_checks.append("depth_scaling_monotonic")
        else:
            failed_checks.append("depth_scaling_not_monotonic")
    else:
        failed_checks.append("no_depth_scaling_gain")
        failed_checks.append("depth_scaling_not_monotonic")

    if depth_ladder:
        deepest = depth_ladder[-1]
        if int(deepest["hits"]) - int(recurrence_off["hits"]) >= min_hit_advantage:
            passed_checks.append("deepest_hybrid_beats_recurrence_off")
        else:
            failed_checks.append("deepest_hybrid_does_not_beat_recurrence_off")
    else:
        deepest = {"mode": "", "count": 0, "hits": 0, "accuracy": 0.0}
        failed_checks.append("no_hybrid_depth_modes")

    if stochastic_off is not None and depth_ladder:
        comparable_depth = next(
            (row for row in depth_ladder if row["mode"] == "hybrid_recurrence_depth_4_no_evidence"),
            deepest,
        )
        if int(comparable_depth["hits"]) - int(stochastic_off["hits"]) >= min_hit_advantage:
            passed_checks.append("stochastic_breadth_beats_zero_ablation")
        else:
            failed_checks.append("stochastic_breadth_does_not_beat_zero_ablation")

    shortcuts = _shortcut_records(record_list)
    if shortcuts:
        failed_checks.append("non_raw_shortcut_present")
    else:
        passed_checks.append("no_memoryos_shortcut")

    mode_summaries = {recurrence_off_mode: recurrence_off}
    mode_summaries.update({row["mode"]: row for row in depth_ladder})
    if stochastic_off is not None:
        mode_summaries[stochastic_off_mode] = stochastic_off
    status = _status_from_checks(
        missing_modes=missing_modes,
        failed_checks=failed_checks,
        mode_summaries=mode_summaries,
    )

    return {
        "gate_type": "hybrid_recurrence_depth_scaling",
        "claim": (
            "OneBodyParallelHybridBlock must show causal no-retrieval test-time depth scaling "
            "and beat the recurrence-off baseline; stochastic breadth should add measurable lift "
            "when the matching ablation is present."
        ),
        "status": status,
        "recurrence_off_mode": recurrence_off_mode,
        "depth_modes": requested_depth_modes,
        "stochastic_off_mode": stochastic_off_mode,
        "recurrence_off": recurrence_off,
        "stochastic_off": stochastic_off or {},
        "deepest_hybrid": deepest,
        "depth_ladder": depth_ladder,
        "depth_output_diversity": _depth_output_diversity(by_mode, present_depth_modes),
        "failed_checks": failed_checks,
        "passed_checks": passed_checks,
        "missing_modes": missing_modes,
        "shortcut_records": shortcuts,
        "mode_semantics": PURE_RECURSIVE_MODE_SEMANTICS,
        "recommendation": "Run the hybrid recurrence depth sweep plus recurrence-off and stochastic-zero ablations on no-retrieval heldouts.",
    }


def build_trainable_memory_gate(
    records: Iterable[dict[str, Any]],
    *,
    memory_on_mode: str = DEFAULT_MEMORY_ON_MODE,
    memory_off_mode: str = DEFAULT_MEMORY_OFF_MODE,
    donor_mode: str = DEFAULT_DONOR_MODE,
    min_hit_advantage: int = 1,
) -> dict[str, Any]:
    record_list = list(records)
    by_mode = _records_by_mode(record_list)
    memory_on = _mode_summary(by_mode, memory_on_mode)
    memory_off = _mode_summary(by_mode, memory_off_mode)
    donor = _mode_summary(by_mode, donor_mode)
    missing_modes = [
        mode
        for mode in [memory_on_mode, memory_off_mode]
        if mode not in by_mode
    ]
    donor_missing = donor_mode not in by_mode

    memory_off_comparison = _compare(memory_on, memory_off)
    donor_comparison = _compare(memory_on, donor)
    shortcuts = _shortcut_records(record_list)

    failed_checks: list[str] = []
    passed_checks: list[str] = []
    if int(memory_on["hits"]) - int(memory_off["hits"]) >= min_hit_advantage:
        passed_checks.append("memory_on_beats_memory_off")
    else:
        failed_checks.append("memory_on_does_not_beat_memory_off")
    if donor_missing or int(memory_on["hits"]) - int(donor["hits"]) >= min_hit_advantage:
        passed_checks.append("memory_on_beats_or_uncompared_to_donor")
    else:
        failed_checks.append("memory_on_does_not_beat_donor")
    if shortcuts:
        failed_checks.append("non_raw_shortcut_present")
    else:
        passed_checks.append("no_memoryos_shortcut")

    mode_summaries = {
        memory_on_mode: memory_on,
        memory_off_mode: memory_off,
    }
    if not donor_missing:
        mode_summaries[donor_mode] = donor
    status = _status_from_checks(
        missing_modes=missing_modes,
        failed_checks=failed_checks,
        mode_summaries=mode_summaries,
    )

    return {
        "gate_type": "trainable_memory_intelligence",
        "claim": (
            "Trainable memory should improve long-memory recall/use over memory-off "
            "and simpler baselines without MemoryOS retrieval shortcuts."
        ),
        "status": status,
        "memory_on_mode": memory_on_mode,
        "memory_off_mode": memory_off_mode,
        "donor_mode": donor_mode,
        "memory_on": memory_on,
        "memory_off": memory_off,
        "donor": donor,
        "memory_off_comparison": memory_off_comparison,
        "donor_comparison": donor_comparison,
        "shortcut_records": shortcuts,
        "missing_modes": missing_modes,
        "failed_checks": failed_checks,
        "passed_checks": passed_checks,
        "recommendation": _recommendation_for_gate(
            "trainable_memory_intelligence", status, failed_checks
        ),
    }


def build_composition_gate(
    records: Iterable[dict[str, Any]],
    *,
    full_mode: str = DEFAULT_COMPOSITION_FULL_MODE,
    core_off_mode: str = DEFAULT_COMPOSITION_CORE_OFF_MODE,
    memory_off_mode: str = DEFAULT_COMPOSITION_MEMORY_OFF_MODE,
    donor_mode: str = DEFAULT_DONOR_MODE,
    min_hit_advantage: int = 1,
) -> dict[str, Any]:
    record_list = list(records)
    by_mode = _records_by_mode(record_list)
    full = _mode_summary(by_mode, full_mode)
    core_off = _mode_summary(by_mode, core_off_mode)
    memory_off = _mode_summary(by_mode, memory_off_mode)
    donor = _mode_summary(by_mode, donor_mode)
    missing_modes = [
        mode
        for mode in [full_mode, core_off_mode, memory_off_mode]
        if mode not in by_mode
    ]

    core_off_comparison = _compare(full, core_off)
    memory_off_comparison = _compare(full, memory_off)
    donor_comparison = _compare(full, donor)
    shortcuts = _shortcut_records(record_list)

    failed_checks: list[str] = []
    passed_checks: list[str] = []
    if int(full["hits"]) - int(core_off["hits"]) >= min_hit_advantage:
        passed_checks.append("full_beats_core_off")
    else:
        failed_checks.append("full_does_not_beat_core_off")
    if int(full["hits"]) - int(memory_off["hits"]) >= min_hit_advantage:
        passed_checks.append("full_beats_memory_off")
    else:
        failed_checks.append("full_does_not_beat_memory_off")
    if donor_mode not in by_mode or int(full["hits"]) - int(donor["hits"]) >= min_hit_advantage:
        passed_checks.append("full_beats_or_uncompared_to_donor")
    else:
        failed_checks.append("full_does_not_beat_donor")
    if shortcuts:
        failed_checks.append("non_raw_shortcut_present")
    else:
        passed_checks.append("no_memoryos_shortcut")

    mode_summaries = {
        full_mode: full,
        core_off_mode: core_off,
        memory_off_mode: memory_off,
    }
    if donor_mode in by_mode:
        mode_summaries[donor_mode] = donor
    status = _status_from_checks(
        missing_modes=missing_modes,
        failed_checks=failed_checks,
        mode_summaries=mode_summaries,
    )

    return {
        "gate_type": "reasoning_memory_composition",
        "claim": (
            "Recursive core and trainable memory should both be causally needed "
            "when memory retention and reasoning composition are required together."
        ),
        "status": status,
        "full_mode": full_mode,
        "core_off_mode": core_off_mode,
        "memory_off_mode": memory_off_mode,
        "donor_mode": donor_mode,
        "full": full,
        "core_off": core_off,
        "memory_off": memory_off,
        "donor": donor,
        "core_off_comparison": core_off_comparison,
        "memory_off_comparison": memory_off_comparison,
        "donor_comparison": donor_comparison,
        "shortcut_records": shortcuts,
        "missing_modes": missing_modes,
        "failed_checks": failed_checks,
        "passed_checks": passed_checks,
        "recommendation": _recommendation_for_gate(
            "reasoning_memory_composition", status, failed_checks
        ),
    }


def build_temporal_spatial_context_gate(
    records: Iterable[dict[str, Any]],
    *,
    context_on_mode: str = DEFAULT_TEMPORAL_SPATIAL_CONTEXT_ON_MODE,
    context_off_mode: str = DEFAULT_TEMPORAL_SPATIAL_CONTEXT_OFF_MODE,
    min_hit_advantage: int = 1,
) -> dict[str, Any]:
    record_list = list(records)
    by_mode = _records_by_mode(record_list)
    context_on = _mode_summary(by_mode, context_on_mode)
    context_off = _mode_summary(by_mode, context_off_mode)
    missing_modes = [
        mode for mode in [context_on_mode, context_off_mode] if mode not in by_mode
    ]

    context_off_comparison = _compare(context_on, context_off)
    shortcuts = _shortcut_records(record_list)
    full_rows = by_mode.get(context_on_mode, [])
    off_rows = by_mode.get(context_off_mode, [])

    failed_checks: list[str] = []
    passed_checks: list[str] = []
    if int(context_on["hits"]) - int(context_off["hits"]) >= min_hit_advantage:
        passed_checks.append("context_on_beats_context_off")
    else:
        failed_checks.append("context_on_does_not_beat_context_off")
    if full_rows and all(
        bool(row.get("temporal_spatial_context_available"))
        and int(row.get("temporal_spatial_context_token_count", 0) or 0) > 0
        for row in full_rows
    ):
        passed_checks.append("context_available_on_full_mode")
    else:
        failed_checks.append("context_not_available_on_full_mode")
    if off_rows and all(bool(row.get("disable_temporal_spatial_context")) for row in off_rows):
        passed_checks.append("context_disabled_on_ablation_mode")
    else:
        failed_checks.append("context_not_disabled_on_ablation_mode")
    if shortcuts:
        failed_checks.append("non_raw_shortcut_present")
    else:
        passed_checks.append("no_retrieval_or_memoryos_shortcut")

    mode_summaries = {
        context_on_mode: context_on,
        context_off_mode: context_off,
    }
    status = _status_from_checks(
        missing_modes=missing_modes,
        failed_checks=failed_checks,
        mode_summaries=mode_summaries,
    )

    return {
        "gate_type": "temporal_spatial_context",
        "claim": (
            "SSOT-derived temporal/spatial context tokens should causally improve "
            "held-out temporal and spatial reasoning over the same model with "
            "the context path ablated."
        ),
        "status": status,
        "context_on_mode": context_on_mode,
        "context_off_mode": context_off_mode,
        "context_on": context_on,
        "context_off": context_off,
        "context_off_comparison": context_off_comparison,
        "by_task_family": _summaries_by_field(record_list, "task_family"),
        "by_reasoning_family": _summaries_by_field(record_list, "reasoning_family"),
        "by_expected_paradigm": _summaries_by_field(record_list, "expected_paradigm"),
        "shortcut_records": shortcuts,
        "missing_modes": missing_modes,
        "failed_checks": failed_checks,
        "passed_checks": passed_checks,
        "recommendation": _recommendation_for_gate(
            "temporal_spatial_context", status, failed_checks
        ),
    }


def build_hybrid_556_causal_matrix_gate(
    records: Iterable[dict[str, Any]],
    *,
    full_mode: str = DEFAULT_HYBRID_556_FULL_MODE,
    ablation_modes: Iterable[str] = DEFAULT_HYBRID_556_ABLATION_MODES,
    min_hit_drop: int = 1,
) -> dict[str, Any]:
    """RI-3 gate for full 5.56 recipe causal ablation matrix on the hybrid."""
    record_list = list(records)
    by_mode = _records_by_mode(record_list)
    requested_ablation_modes = list(ablation_modes)
    full = _mode_summary(by_mode, full_mode)
    ablations = {mode: _mode_summary(by_mode, mode) for mode in requested_ablation_modes}
    missing_modes = [
        mode for mode in [full_mode, *requested_ablation_modes] if mode not in by_mode
    ]
    failed_checks: list[str] = []
    passed_checks: list[str] = []

    for mode, ablation in ablations.items():
        drop = int(full["hits"]) - int(ablation["hits"])
        if drop >= min_hit_drop:
            passed_checks.append(f"full_beats_{mode}")
        else:
            failed_checks.append(f"full_does_not_beat_{mode}")

    shortcuts = _shortcut_records(record_list)
    if shortcuts:
        failed_checks.append("non_raw_shortcut_present")
    else:
        passed_checks.append("no_memoryos_shortcut")

    mode_summaries = {full_mode: full, **ablations}
    status = _status_from_checks(
        missing_modes=missing_modes,
        failed_checks=failed_checks,
        mode_summaries=mode_summaries,
    )

    return {
        "gate_type": "hybrid_556_causal_matrix",
        "claim": (
            "The full 5.56-style hybrid recipe must causally outperform clean ablations "
            "of stochastic breadth, gold structural injection, attractor protection, and "
            "scheduled binding decay on no-retrieval heldouts."
        ),
        "status": status,
        "full_mode": full_mode,
        "ablation_modes": requested_ablation_modes,
        "full": full,
        "ablations": ablations,
        "full_vs_ablations": {
            mode: _compare(full, ablation) for mode, ablation in ablations.items()
        },
        "failed_checks": failed_checks,
        "passed_checks": passed_checks,
        "missing_modes": missing_modes,
        "shortcut_records": shortcuts,
        "mode_semantics": PURE_RECURSIVE_MODE_SEMANTICS,
        "recommendation": "Run the full 5.56 matrix on the same checkpoint, seed, heldout split, and no-retrieval contract.",
    }


def build_raw_intelligence_gate(
    records: Iterable[dict[str, Any]],
    *,
    gate_type: str,
) -> dict[str, Any]:
    if gate_type == "pure_recursive_reasoning":
        return build_pure_recursive_reasoning_gate(records)
    if gate_type == "trainable_memory_intelligence":
        return build_trainable_memory_gate(records)
    if gate_type == "reasoning_memory_composition":
        return build_composition_gate(records)
    if gate_type == "temporal_spatial_context":
        return build_temporal_spatial_context_gate(records)
    if gate_type == "ri4_sparse_persistent_memory":
        return build_ri4_sparse_memory_gate(records)
    if gate_type == "hybrid_recurrence_depth_scaling":
        return build_hybrid_recurrence_depth_gate(records)
    if gate_type == "hybrid_556_causal_matrix":
        return build_hybrid_556_causal_matrix_gate(records)
    raise ValueError(f"unknown raw intelligence gate_type: {gate_type}")


def load_records(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "summary" in row and "mode" not in row:
                    continue
                records.append(row)
    return records


def render_markdown(gate: dict[str, Any]) -> str:
    lines = [
        "# Raw Intelligence Gate",
        "",
        "## Verdict",
        "",
        f"Gate type: `{gate.get('gate_type', 'unknown')}`",
        "",
        f"Status: `{gate.get('status', 'unknown')}`",
        "",
        f"Claim: {gate.get('claim', '')}",
        "",
        f"Recommendation: {gate.get('recommendation', '')}",
        "",
        "## Checks",
        "",
        f"- Passed: `{', '.join(gate.get('passed_checks', [])) or 'none'}`",
        f"- Failed: `{', '.join(gate.get('failed_checks', [])) or 'none'}`",
        f"- Missing modes: `{', '.join(gate.get('missing_modes', [])) or 'none'}`",
        f"- Shortcut records: `{len(gate.get('shortcut_records', []))}`",
        "",
    ]
    mode_semantics = gate.get("mode_semantics")
    if isinstance(mode_semantics, dict) and mode_semantics:
        lines.extend(
            [
                "## Mode Semantics",
                "",
                "| Label | Meaning |",
                "| --- | --- |",
            ]
        )
        for label, meaning in mode_semantics.items():
            lines.append(f"| {label} | {meaning} |")
        lines.append("")
    eval_contract = gate.get("eval_contract")
    if isinstance(eval_contract, dict) and any(eval_contract.values()):
        scoring = ", ".join(eval_contract.get("scoring", [])) or "unknown"
        choice_norm = (
            ", ".join(eval_contract.get("choice_score_normalization", []))
            or "unknown"
        )
        lines.extend(
            [
                "## Eval Contract",
                "",
                f"- Scoring: `{scoring}`",
                f"- Choice score normalization: `{choice_norm}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Mode Metrics",
            "",
            "| Label | Mode | Hits | Accuracy |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for label in (
        "donor",
        "core_off",
        "deepest_core",
        "transition_state_off",
        "memory_on",
        "memory_off",
        "full",
        "context_on",
        "context_off",
    ):
        value = gate.get(label)
        if not isinstance(value, dict) or not value.get("mode"):
            continue
        lines.append(
            "| {label} | {mode} | {hits}/{count} | {accuracy:.3f} |".format(
                label=label,
                mode=value.get("mode"),
                hits=int(value.get("hits", 0)),
                count=int(value.get("count", 0)),
                accuracy=float(value.get("accuracy", 0.0)),
            )
        )

    if gate.get("depth_ladder"):
        lines.extend(
            [
                "",
                "## Depth Ladder",
                "",
                "| Mode | Hits | Accuracy |",
                "| --- | ---: | ---: |",
            ]
        )
        for row in gate["depth_ladder"]:
            lines.append(
                "| {mode} | {hits}/{count} | {accuracy:.3f} |".format(
                    mode=row.get("mode"),
                    hits=int(row.get("hits", 0)),
                    count=int(row.get("count", 0)),
                    accuracy=float(row.get("accuracy", 0.0)),
                )
            )

    diversity = gate.get("depth_output_diversity")
    if isinstance(diversity, dict) and diversity.get("measured"):
        lines.extend(
            [
                "",
                "## Depth Output Diversity",
                "",
                f"- Comparable cases: `{int(diversity.get('case_count', 0))}`",
                f"- Identical across all depth modes: `{int(diversity.get('identical_case_count', 0))}`",
                f"- Changed by depth: `{int(diversity.get('changed_case_count', 0))}`",
            f"- All depth outputs identical: `{bool(diversity.get('all_depth_outputs_identical', False))}`",
            ]
        )

    by_expected_paradigm = gate.get("by_expected_paradigm")
    if isinstance(by_expected_paradigm, dict) and by_expected_paradigm:
        lines.extend(
            [
                "",
                "## Expected-Paradigm Metrics",
                "",
                "| Expected paradigm | Mode | Hits | Accuracy |",
                "| --- | --- | ---: | ---: |",
            ]
        )
        for paradigm, rows_by_mode in sorted(by_expected_paradigm.items()):
            if not isinstance(rows_by_mode, dict):
                continue
            for mode, row in sorted(rows_by_mode.items()):
                if not isinstance(row, dict):
                    continue
                lines.append(
                    "| {paradigm} | {mode} | {hits}/{count} | {accuracy:.3f} |".format(
                        paradigm=paradigm,
                        mode=mode,
                        hits=int(row.get("hits", 0)),
                        count=int(row.get("count", 0)),
                        accuracy=float(row.get("accuracy", 0.0)),
                    )
                )

    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "- `accepted` means the tested raw-intelligence component was causally useful on this eval.",
            "- `rejected` means the component did not beat the simpler ablation/baseline or a shortcut contaminated the run.",
            "- `inconclusive` means required modes are missing or empty.",
            "- This gate is not a RAG, MemoryOS, answer-format, or SSOT score.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_gate(
    gate: dict[str, Any],
    *,
    markdown_out: str | Path,
    json_out: str | Path,
) -> None:
    markdown_path = Path(markdown_out)
    json_path = Path(json_out)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(gate), encoding="utf-8")
    json_path.write_text(json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _recommendation_for_gate(gate_type: str, status: str, failed_checks: list[str]) -> str:
    if status == "accepted":
        return (
            "Promote this result as raw-intelligence evidence only for the tested "
            "axis, then rerun on harder held-out cases before changing architecture."
        )
    if status == "inconclusive":
        return "Run the missing baseline/ablation modes before making an architecture claim."
    if "non_raw_shortcut_present" in failed_checks:
        return (
            "Discard this raw-intelligence run. Re-run with MemoryOS, retrieval, "
            "hidden evidence, and workspace-memory evidence disabled."
        )
    if gate_type == "pure_recursive_reasoning":
        return (
            "Do not tune answer formatting. Redesign or retrain the recursive core "
            "so deeper latent steps beat donor-only and core-off on no-evidence tasks."
        )
    if gate_type == "trainable_memory_intelligence":
        return (
            "Do not promote MSA/LM2 memory yet. Train or redesign memory writes/reads "
            "until memory-on beats memory-off on length and distractor sweeps."
        )
    if gate_type == "temporal_spatial_context":
        return (
            "Do not claim temporal/spatial conditioning yet. The context-on path "
            "must beat the context-off ablation with no retrieval or MemoryOS shortcut."
        )
    return (
        "Do not claim reasoning-memory composition yet. Full path must beat both "
        "core-off and memory-off ablations on the same held-out cases."
    )
