from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from qtrm_mm.eval.memory_retrieval import case_task_family, summarize_records

DEFAULT_DONOR_MODE = "donor_only_with_evidence"
DEFAULT_RESIDUAL_MODE = "qtrm_residual_with_evidence"


def load_eval_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
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


def _overall(records: list[dict[str, Any]]) -> dict[str, Any]:
    return summarize_records(records)["overall"]


def _compare_bucket(
    donor_records: list[dict[str, Any]],
    residual_records: list[dict[str, Any]],
) -> dict[str, Any]:
    donor = _overall(donor_records)
    residual = _overall(residual_records)
    return {
        "donor": donor,
        "residual": residual,
        "delta_hits": int(residual["hits"]) - int(donor["hits"]),
        "delta_accuracy": float(residual["accuracy"]) - float(donor["accuracy"]),
    }


def _records_by_mode(records: Iterable[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    return [record for record in records if record.get("mode") == mode]


def _compare_by_task_family(
    donor_records: list[dict[str, Any]],
    residual_records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    donor_by_family: dict[str, list[dict[str, Any]]] = {}
    residual_by_family: dict[str, list[dict[str, Any]]] = {}
    for record in donor_records:
        donor_by_family.setdefault(case_task_family(record), []).append(record)
    for record in residual_records:
        residual_by_family.setdefault(case_task_family(record), []).append(record)

    families = sorted(set(donor_by_family) | set(residual_by_family))
    return {
        family: _compare_bucket(donor_by_family.get(family, []), residual_by_family.get(family, []))
        for family in families
    }


def _changed_cases(
    donor_records: list[dict[str, Any]],
    residual_records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    donor_by_id = {str(record.get("id")): record for record in donor_records}
    residual_by_id = {str(record.get("id")): record for record in residual_records}
    improvements = []
    regressions = []
    for case_id in sorted(set(donor_by_id) & set(residual_by_id)):
        donor = donor_by_id[case_id]
        residual = residual_by_id[case_id]
        donor_hit = bool(donor.get("hit"))
        residual_hit = bool(residual.get("hit"))
        if donor_hit == residual_hit:
            continue
        item = {
            "id": case_id,
            "task_family": case_task_family(residual),
            "donor_completion": donor.get("completion", ""),
            "residual_completion": residual.get("completion", ""),
        }
        if residual_hit:
            improvements.append(item)
        else:
            regressions.append(item)
    return {"improvements": improvements, "regressions": regressions}


def build_proof_summary(
    eval_specs: Iterable[dict[str, Any]],
    *,
    donor_mode: str = DEFAULT_DONOR_MODE,
    residual_mode: str = DEFAULT_RESIDUAL_MODE,
) -> dict[str, Any]:
    proof_evals = []
    aggregate_donor_records: list[dict[str, Any]] = []
    aggregate_residual_records: list[dict[str, Any]] = []

    for spec in eval_specs:
        path = str(spec["path"])
        records = list(spec.get("records") or load_eval_records(path))
        donor_records = _records_by_mode(records, donor_mode)
        residual_records = _records_by_mode(records, residual_mode)
        comparison = _compare_bucket(donor_records, residual_records)
        comparison.update(
            {
                "name": spec["name"],
                "path": path,
                "donor_mode": donor_mode,
                "residual_mode": residual_mode,
                "by_task_family": _compare_by_task_family(donor_records, residual_records),
                "changed_cases": _changed_cases(donor_records, residual_records),
            }
        )
        proof_evals.append(comparison)
        aggregate_donor_records.extend(donor_records)
        aggregate_residual_records.extend(residual_records)

    aggregate = _compare_bucket(aggregate_donor_records, aggregate_residual_records)
    aggregate["by_task_family"] = _compare_by_task_family(
        aggregate_donor_records,
        aggregate_residual_records,
    )
    return {
        "claim": "QTRM is a donor-backed residual adapter that improves selected MemoryOS evidence tasks over donor-only.",
        "donor_mode": donor_mode,
        "residual_mode": residual_mode,
        "evals": proof_evals,
        "aggregate": aggregate,
    }


def _hit_text(metrics: dict[str, Any]) -> str:
    return f"{int(metrics['hits'])}/{int(metrics['count'])}"


def _signed_int(value: int) -> str:
    return f"{value:+d}"


def _signed_float(value: float) -> str:
    return f"{value:+.3f}"


def render_markdown(proof: dict[str, Any]) -> str:
    lines = [
        "# Residual Adapter Proof",
        "",
        "Claim: QTRM is currently a donor-backed residual adapter, not a standalone donor-free language model.",
        "",
        "This package fixes the current proof target: with donor logits intact, QTRM residual generation should improve evidence-sensitive MemoryOS tasks over donor-only generation while preserving the Qwen base language policy.",
        "",
        "This is not a donor-free standalone-LM claim.",
        "",
        "## Summary",
        "",
        "| Eval | Source | Donor-only | QTRM residual | Delta hits | Delta accuracy |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in proof["evals"]:
        lines.append(
            "| {name} | {path} | {donor} | {residual} | {delta_hits} | {delta_accuracy} |".format(
                name=item["name"],
                path=item["path"],
                donor=_hit_text(item["donor"]),
                residual=_hit_text(item["residual"]),
                delta_hits=_signed_int(int(item["delta_hits"])),
                delta_accuracy=_signed_float(float(item["delta_accuracy"])),
            )
        )

    aggregate = proof["aggregate"]
    lines.extend(
        [
            "| **aggregate** | all listed evals | {donor} | {residual} | {delta_hits} | {delta_accuracy} |".format(
                donor=_hit_text(aggregate["donor"]),
                residual=_hit_text(aggregate["residual"]),
                delta_hits=_signed_int(int(aggregate["delta_hits"])),
                delta_accuracy=_signed_float(float(aggregate["delta_accuracy"])),
            ),
            "",
            "## Task-Family Delta",
            "",
            "| Task family | Donor-only | QTRM residual | Delta hits | Delta accuracy |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for family, item in aggregate["by_task_family"].items():
        lines.append(
            "| {family} | {donor} | {residual} | {delta_hits} | {delta_accuracy} |".format(
                family=family,
                donor=_hit_text(item["donor"]),
                residual=_hit_text(item["residual"]),
                delta_hits=_signed_int(int(item["delta_hits"])),
                delta_accuracy=_signed_float(float(item["delta_accuracy"])),
            )
        )

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- These evals prove residual-adapter usefulness on small MemoryOS probes, not broad language-model replacement.",
            "- Donor-free `donor_logits_scale=0.0` generation remains a later standalone-student gate.",
            "- The held-out set is still small; the next gate should expand to 50-100 balanced cases.",
            "- Retrieval success is tracked separately from answer accuracy; evidence recall alone is not enough.",
            "",
            "## Next Gates",
            "",
            "1. Expand held-out MemoryOS cases across conflict, multi-hop, abstention, and Korean authority conflict.",
            "2. Keep donor-only and QTRM residual modes paired in every eval.",
            "3. Add workspace/core ablations when claiming latent-memory or recursive-core causality.",
            "4. Resume donor-free work only after OPD/GKD/DistiLLM-style rollout training is implemented.",
        ]
    )
    return "\n".join(lines) + "\n"
