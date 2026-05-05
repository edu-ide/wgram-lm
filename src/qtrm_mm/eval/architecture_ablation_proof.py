from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from qtrm_mm.eval.memory_retrieval import case_task_family, summarize_records
from qtrm_mm.eval.residual_adapter_proof import load_eval_records

BASELINE_MODE = "qtrm_residual_with_evidence"
DEFAULT_MODES = [
    "qtrm_residual_with_evidence",
    "qtrm_workspace_off_with_evidence",
    "qtrm_core_off_with_evidence",
    "qtrm_coda_off_with_evidence",
    "qtrm_residual_head_off_with_evidence",
    "qtrm_donor_hidden_off_with_evidence",
    "qtrm_workspace_only_with_evidence",
    "qtrm_workspace_gate_off_with_evidence",
    "qtrm_workspace_memory_off_with_evidence",
    "qtrm_core_context_off_with_evidence",
    "qtrm_evidence_bottleneck_off_with_evidence",
]


def _records_by_mode(records: Iterable[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    return [record for record in records if record.get("mode") == mode]


def _overall(records: list[dict[str, Any]]) -> dict[str, Any]:
    return summarize_records(records)["overall"]


def _drop_from_baseline(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "hit_drop": int(baseline["hits"]) - int(candidate["hits"]),
        "accuracy_drop": float(baseline["accuracy"]) - float(candidate["accuracy"]),
    }


def _records_by_id(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(record.get("id")): record for record in records}


def _completion_identity(
    baseline_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_by_id = _records_by_id(baseline_records)
    candidate_by_id = _records_by_id(candidate_records)
    paired_ids = sorted(set(baseline_by_id) & set(candidate_by_id))
    same_count = sum(
        str(baseline_by_id[case_id].get("completion", ""))
        == str(candidate_by_id[case_id].get("completion", ""))
        for case_id in paired_ids
    )
    paired_count = len(paired_ids)
    return {
        "paired_completion_count": paired_count,
        "same_completion_count": same_count,
        "same_completion_rate": same_count / paired_count if paired_count else 0.0,
    }


def _family_records(records: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(case_task_family(record), []).append(record)
    return grouped


def _load_spec_records(spec: dict[str, Any]) -> list[dict[str, Any]]:
    if "records" in spec:
        return list(spec["records"])
    return load_eval_records(str(spec["path"]))


def build_ablation_summary(
    eval_specs: Iterable[dict[str, Any]],
    *,
    baseline_mode: str = BASELINE_MODE,
    modes: Iterable[str] = DEFAULT_MODES,
) -> dict[str, Any]:
    mode_list = list(modes)
    records: list[dict[str, Any]] = []
    sources = []
    for spec in eval_specs:
        path = str(spec["path"])
        sources.append({"name": spec["name"], "path": path})
        records.extend(_load_spec_records(spec))

    by_mode_records = {mode: _records_by_mode(records, mode) for mode in mode_list}
    mode_metrics = {mode: _overall(mode_records) for mode, mode_records in by_mode_records.items()}
    baseline = mode_metrics[baseline_mode]
    drops = {}
    for mode, metrics in mode_metrics.items():
        if mode == baseline_mode:
            continue
        drop = _drop_from_baseline(baseline, metrics)
        drop.update(_completion_identity(by_mode_records[baseline_mode], by_mode_records[mode]))
        drops[mode] = drop

    families = sorted(
        {
            case_task_family(record)
            for record in records
            if record.get("mode") in set(mode_list)
        }
    )
    by_family_records = {
        mode: _family_records(mode_records)
        for mode, mode_records in by_mode_records.items()
    }
    by_task_family: dict[str, dict[str, Any]] = {}
    for family in families:
        family_baseline = _overall(by_family_records[baseline_mode].get(family, []))
        family_summary: dict[str, Any] = {}
        for mode in mode_list:
            metrics = _overall(by_family_records[mode].get(family, []))
            row = dict(metrics)
            if mode != baseline_mode:
                row.update(_drop_from_baseline(family_baseline, metrics))
            else:
                row.update({"hit_drop": 0, "accuracy_drop": 0.0})
            family_summary[mode] = row
        by_task_family[family] = family_summary

    return {
        "claim": "Expanded MemoryOS ablation measures which QTRM components are causally responsible for residual behavior.",
        "baseline_mode": baseline_mode,
        "modes": mode_metrics,
        "drop_from_residual": drops,
        "by_task_family": by_task_family,
        "sources": sources,
    }


def _hit_text(metrics: dict[str, Any]) -> str:
    return f"{int(metrics['hits'])}/{int(metrics['count'])}"


def _signed_int(value: int) -> str:
    return f"{value:+d}"


def _signed_float(value: float) -> str:
    return f"{value:+.3f}"


def render_markdown(proof: dict[str, Any]) -> str:
    baseline_mode = proof["baseline_mode"]
    lines = [
        "# Expanded Workspace/Core Ablation Proof",
        "",
        "Claim: this expanded 72-case MemoryOS gate measures whether residual behavior is localized to workspace, core, coda, residual-head, donor-hidden, or workspace-gate paths.",
        "",
        "Positive drop means the ablated mode is worse than full `qtrm_residual_with_evidence` on the same expanded gate.",
        "",
        "## Sources",
        "",
    ]
    for source in proof["sources"]:
        lines.append(f"- {source['name']}: `{source['path']}`")

    lines.extend(
        [
            "",
            "## Overall",
            "",
            "| Mode | Hits | Hit drop vs residual | Accuracy drop vs residual |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for mode, metrics in proof["modes"].items():
        drop = {"hit_drop": 0, "accuracy_drop": 0.0}
        if mode != baseline_mode:
            drop = proof["drop_from_residual"][mode]
        lines.append(
            "| {mode} | {hits} | {hit_drop} | {accuracy_drop} |".format(
                mode=mode,
                hits=_hit_text(metrics),
                hit_drop=_signed_int(int(drop["hit_drop"])),
                accuracy_drop=_signed_float(float(drop["accuracy_drop"])),
            )
        )

    if proof["drop_from_residual"] and all(
        int(drop["hit_drop"]) == 0 for drop in proof["drop_from_residual"].values()
    ):
        lines.extend(
            [
                "",
                "Current result: workspace-off and core-off match the full residual score, so this run does not localize the residual gain to the latent workspace or recursive core.",
            ]
        )

    if proof["drop_from_residual"]:
        lines.extend(
            [
                "",
                "## Completion Identity",
                "",
                "| Mode | Same completions vs residual | Same rate |",
                "| --- | ---: | ---: |",
            ]
        )
        for mode, drop in proof["drop_from_residual"].items():
            lines.append(
                "| {mode} | {same}/{paired} | {rate:.3f} |".format(
                    mode=mode,
                    same=int(drop["same_completion_count"]),
                    paired=int(drop["paired_completion_count"]),
                    rate=float(drop["same_completion_rate"]),
                )
            )

    drops = proof.get("drop_from_residual", {})
    if drops:
        lines.extend(["", "## Current Interpretation", ""])
        residual_head = drops.get("qtrm_residual_head_off_with_evidence")
        if residual_head is not None and int(residual_head["hit_drop"]) > 0:
            lines.append(
                "- Turning off QTRM residual logits causes a large drop, so the measured gain is genuinely in the residual head rather than donor-only generation."
            )
        coda = drops.get("qtrm_coda_off_with_evidence")
        if coda is not None and int(coda["hit_drop"]) > 0:
            lines.append(
                "- Turning off coda causes a smaller but real drop, so coda contributes to the residual behavior."
            )
        donor_hidden = drops.get("qtrm_donor_hidden_off_with_evidence")
        if donor_hidden is not None and int(donor_hidden["hit_drop"]) == 0:
            lines.append(
                "- Removing projected donor hidden states does not change this gate, so the current gain is not caused by direct donor-hidden prefix tokens."
            )
        workspace_only = drops.get("qtrm_workspace_only_with_evidence")
        if workspace_only is not None and int(workspace_only["hit_drop"]) == 0:
            lines.append(
                "- Workspace-only context matches full residual, but workspace-off also matches full residual; this gate still does not prove latent-workspace causality."
            )
        workspace_gate = drops.get("qtrm_workspace_gate_off_with_evidence")
        if workspace_gate is not None and int(workspace_gate["hit_drop"]) > 0:
            lines.append(
                "- Turning off the workspace memory gate causes a drop, so the gated latent memory path contributes on this gate."
            )
        elif workspace_gate is not None and int(workspace_gate["hit_drop"]) == 0:
            lines.append(
                "- Turning off the workspace memory gate does not change score or completions, so this run does not prove gated latent-memory causality."
            )
        workspace_memory = drops.get("qtrm_workspace_memory_off_with_evidence")
        if workspace_memory is not None and int(workspace_memory["hit_drop"]) > 0:
            lines.append(
                "- Removing workspace-side evidence memory causes a drop, so retrieved evidence is flowing through the workspace-memory path on this gate."
            )
        elif workspace_memory is not None and int(workspace_memory["hit_drop"]) == 0:
            lines.append(
                "- Removing workspace-side evidence memory does not change score or completions, so this run does not prove workspace-memory evidence causality."
            )
        core_context = drops.get("qtrm_core_context_off_with_evidence")
        if core_context is not None and int(core_context["hit_drop"]) > 0:
            lines.append(
                "- Turning off gated core context injection causes a drop, so the recursive core is using direct prompt/evidence context on this gate."
            )
        elif core_context is not None and int(core_context["hit_drop"]) == 0:
            lines.append(
                "- Turning off gated core context injection does not change score or completions, so this run does not prove direct core-context causality."
            )

    lines.extend(
        [
            "",
            "## Task-Family Drops",
            "",
            "| Task family | Mode | Hits | Hit drop vs residual | Accuracy drop vs residual |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for family, modes in proof["by_task_family"].items():
        for mode, metrics in modes.items():
            lines.append(
                "| {family} | {mode} | {hits} | {hit_drop} | {accuracy_drop} |".format(
                    family=family,
                    mode=mode,
                    hits=_hit_text(metrics),
                    hit_drop=_signed_int(int(metrics["hit_drop"])),
                    accuracy_drop=_signed_float(float(metrics["accuracy_drop"])),
                )
            )

    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "- If workspace-off or core-off matches full residual, the current gain is not yet localized to that component.",
            "- If an ablation drops below full residual, that component is contributing to the measured behavior.",
            "- This is still a MemoryOS evidence-task proof, not a broad standalone-LM proof.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_ablation_summary(proof: dict[str, Any], *, markdown_out: str, json_out: str) -> None:
    markdown_path = Path(markdown_out)
    json_path = Path(json_out)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(proof), encoding="utf-8")
    json_path.write_text(json.dumps(proof, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
