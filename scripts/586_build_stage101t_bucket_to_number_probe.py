#!/usr/bin/env python3
"""Build Stage101T bucket-to-number belief probes.

Stage101S showed that direct numeric calibration strengthened the easy numeric
habit (`0.90`) without making low/neutral states speakable.  Stage101T separates
the lesson:

  1. semantic bucket: low / neutral / high style labels
  2. numeric readback: convert the bucket into the numeric answer string

The final answer still goes through the normal LM head.  No side scalar head is
introduced.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SOURCE = "stage101t_bucket_to_number_probe"

SUPPORT_BUCKET_CHOICES = [" contradicts", " neutral", " supports"]
RELIABILITY_BUCKET_CHOICES = [" low", " unknown", " high"]
SUFFICIENCY_BUCKET_CHOICES = [" insufficient", " partial", " sufficient"]

SUPPORT_NUMBER_CHOICES = [" -0.80", " +0.00", " +0.80"]
RELIABILITY_NUMBER_CHOICES = [" 0.10", " 0.50", " 0.90"]
SUFFICIENCY_NUMBER_CHOICES = [" 0.10", " 0.50", " 0.90"]

BUCKET_TO_NUMBER: dict[tuple[str, str], str] = {
    ("support", " contradicts"): " -0.80",
    ("support", " neutral"): " +0.00",
    ("support", " supports"): " +0.80",
    ("reliability", " low"): " 0.10",
    ("reliability", " unknown"): " 0.50",
    ("reliability", " high"): " 0.90",
    ("sufficiency", " insufficient"): " 0.10",
    ("sufficiency", " partial"): " 0.50",
    ("sufficiency", " sufficient"): " 0.90",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not str(path):
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def negative_answers(answer: str, choices: list[str]) -> list[str]:
    return [choice for choice in choices if choice != answer]


def bucket_choices(axis: str) -> list[str]:
    if axis == "support":
        return list(SUPPORT_BUCKET_CHOICES)
    if axis == "reliability":
        return list(RELIABILITY_BUCKET_CHOICES)
    if axis == "sufficiency":
        return list(SUFFICIENCY_BUCKET_CHOICES)
    raise ValueError(f"unknown bucket axis: {axis}")


def number_choices(axis: str) -> list[str]:
    if axis == "support":
        return list(SUPPORT_NUMBER_CHOICES)
    if axis == "reliability":
        return list(RELIABILITY_NUMBER_CHOICES)
    if axis == "sufficiency":
        return list(SUFFICIENCY_NUMBER_CHOICES)
    raise ValueError(f"unknown number axis: {axis}")


def bucket_question(axis: str) -> str:
    if axis == "support":
        return "Choose support bucket: contradicts, neutral, or supports."
    if axis == "reliability":
        return "Choose reliability bucket: low, unknown, or high."
    if axis == "sufficiency":
        return "Choose sufficiency bucket: insufficient, partial, or sufficient."
    raise ValueError(f"unknown bucket axis: {axis}")


def number_question(axis: str) -> str:
    if axis == "support":
        return "Write numeric support. Choose -0.80, +0.00, or +0.80."
    if axis == "reliability":
        return "Write numeric reliability. Choose 0.10, 0.50, or 0.90."
    if axis == "sufficiency":
        return "Write numeric sufficiency. Choose 0.10, 0.50, or 0.90."
    raise ValueError(f"unknown number axis: {axis}")


def make_source_prompt(*, claim: str, evidence: str) -> str:
    return f"Claim: {claim}\nEvidence: {evidence}"


def make_row(
    row_id: str,
    *,
    split: str,
    lesson_type: str,
    belief_axis: str,
    prompt: str,
    answer: str,
    choices: list[str],
    case_type: str,
    bucket_case_id: str,
    axis: str,
    bucket_answer: str,
    numeric_answer: str,
    support_score: float,
    reliability_score: float,
    sufficiency_score: float,
    concept: str = "",
    claim: str = "",
    source_template: str = "claim_first",
) -> dict[str, Any]:
    if answer not in choices:
        raise ValueError(f"answer {answer!r} not in choices: {choices!r}")
    return {
        "id": row_id,
        "source": SOURCE,
        "task": f"stage101t_{lesson_type}_{belief_axis}_icl",
        "prompt": prompt,
        "intelligence_answer": answer,
        "parrot_answer": negative_answers(answer, choices)[0],
        "candidate_answers": choices,
        "negative_answers": negative_answers(answer, choices),
        "plain_language_axis": (
            "Stage101T first names the belief bucket, then reads that bucket as "
            "a numeric belief value through the same LM head."
        ),
        "source_case_type": case_type,
        "source_template": source_template,
        "source_concept": concept,
        "source_claim": claim,
        "belief_axis": belief_axis,
        "bucket_axis": f"{axis}_bucket",
        "readback_axis": axis,
        "bucket_case_id": bucket_case_id,
        "bucket_answer": bucket_answer,
        "readback_bucket_answer": bucket_answer,
        "readback_numeric_answer": numeric_answer,
        "belief_support_score": float(support_score),
        "source_reliability_score": float(reliability_score),
        "evidence_sufficiency_score": float(sufficiency_score),
        "stage101t_lesson_type": lesson_type,
        "stage101t_bucket_to_number_required": True,
        "factorized_numeric_belief_required": True,
        "split": split,
    }


def make_bucket_row(
    row_id: str,
    *,
    split: str,
    prompt_body: str,
    axis: str,
    bucket_answer: str,
    case_type: str,
    bucket_case_id: str,
    support_score: float,
    reliability_score: float,
    sufficiency_score: float,
    concept: str,
    claim: str,
) -> dict[str, Any]:
    return make_row(
        row_id,
        split=split,
        lesson_type="semantic_bucket",
        belief_axis=f"{axis}_bucket",
        prompt=f"{prompt_body}\nQ: {bucket_question(axis)}\nA:",
        answer=bucket_answer,
        choices=bucket_choices(axis),
        case_type=case_type,
        bucket_case_id=bucket_case_id,
        axis=axis,
        bucket_answer=bucket_answer,
        numeric_answer=BUCKET_TO_NUMBER[(axis, bucket_answer)],
        support_score=support_score,
        reliability_score=reliability_score,
        sufficiency_score=sufficiency_score,
        concept=concept,
        claim=claim,
    )


def make_readback_row(
    row_id: str,
    *,
    split: str,
    prompt_body: str,
    axis: str,
    bucket_answer: str,
    case_type: str,
    bucket_case_id: str,
    support_score: float,
    reliability_score: float,
    sufficiency_score: float,
    concept: str,
    claim: str,
    source_template: str = "bucket_readback",
) -> dict[str, Any]:
    number_answer = BUCKET_TO_NUMBER[(axis, bucket_answer)]
    return make_row(
        row_id,
        split=split,
        lesson_type="numeric_readback",
        belief_axis=axis,
        prompt=f"{prompt_body}\nBucket: {axis} = {bucket_answer.strip()}\nQ: {number_question(axis)}\nA:",
        answer=number_answer,
        choices=number_choices(axis),
        case_type=case_type,
        bucket_case_id=bucket_case_id,
        axis=axis,
        bucket_answer=bucket_answer,
        numeric_answer=number_answer,
        support_score=support_score,
        reliability_score=reliability_score,
        sufficiency_score=sufficiency_score,
        concept=concept,
        claim=claim,
        source_template=source_template,
    )


def make_case_rows(
    row_id: str,
    *,
    split: str,
    concept: str,
    claim: str,
    evidence: str,
    case_type: str,
    support_bucket: str,
    reliability_bucket: str,
    sufficiency_bucket: str,
) -> list[dict[str, Any]]:
    prompt_body = make_source_prompt(claim=claim, evidence=evidence)
    buckets = {
        "support": support_bucket,
        "reliability": reliability_bucket,
        "sufficiency": sufficiency_bucket,
    }
    scores = {
        "support": float(BUCKET_TO_NUMBER[("support", support_bucket)]),
        "reliability": float(BUCKET_TO_NUMBER[("reliability", reliability_bucket)]),
        "sufficiency": float(BUCKET_TO_NUMBER[("sufficiency", sufficiency_bucket)]),
    }
    rows: list[dict[str, Any]] = []
    for axis, bucket in buckets.items():
        rows.append(
            make_bucket_row(
                f"{row_id}_{axis}_bucket",
                split=split,
                prompt_body=prompt_body,
                axis=axis,
                bucket_answer=bucket,
                case_type=case_type,
                bucket_case_id=row_id,
                support_score=scores["support"],
                reliability_score=scores["reliability"],
                sufficiency_score=scores["sufficiency"],
                concept=concept,
                claim=claim,
            )
        )
        rows.append(
            make_readback_row(
                f"{row_id}_{axis}_number",
                split=split,
                prompt_body=prompt_body,
                axis=axis,
                bucket_answer=bucket,
                case_type=case_type,
                bucket_case_id=row_id,
                support_score=scores["support"],
                reliability_score=scores["reliability"],
                sufficiency_score=scores["sufficiency"],
                concept=concept,
                claim=claim,
            )
        )
    return rows


def dictionary_readback_rows(*, split: str) -> list[dict[str, Any]]:
    prefix = "stage101t_dictionary_train" if split == "train" else "stage101t_dictionary_heldout"
    pairs = [
        ("support", " contradicts"),
        ("support", " neutral"),
        ("support", " supports"),
        ("reliability", " low"),
        ("reliability", " unknown"),
        ("reliability", " high"),
        ("sufficiency", " insufficient"),
        ("sufficiency", " partial"),
        ("sufficiency", " sufficient"),
    ]
    rows: list[dict[str, Any]] = []
    for index, (axis, bucket_answer) in enumerate(pairs):
        rows.append(
            make_readback_row(
                f"{prefix}_{index:02d}_{axis}_{bucket_answer.strip()}",
                split=split,
                prompt_body="Bucket dictionary card.",
                axis=axis,
                bucket_answer=bucket_answer,
                case_type="bucket_dictionary_readback",
                bucket_case_id=f"{prefix}_{index:02d}",
                support_score=0.0,
                reliability_score=0.0,
                sufficiency_score=0.0,
                concept="bucket_dictionary",
                claim="",
                source_template="dictionary",
            )
        )
    return rows


def base_specs() -> list[dict[str, str]]:
    return [
        {
            "concept": "battery_charge",
            "claim": "the battery is fully charged.",
            "trusted_true": "Reliable note: exact claim supported; battery reads 100 percent.",
            "trusted_false": "Reliable note: exact claim contradicted; battery reads 12 percent.",
            "irrelevant": "Reliable note: packet discusses case color, not battery charge.",
            "untrusted_true": "Untrusted note: exact claim supported; battery reads 100 percent.",
            "partial": "Reliable note: battery is charging, but no final percentage is given.",
            "unknown_source": "Unsigned note: battery may be full, but source quality is unknown.",
        },
        {
            "concept": "bridge_status",
            "claim": "the bridge is open to traffic.",
            "trusted_true": "Reliable note: exact claim supported; official status says open.",
            "trusted_false": "Reliable note: exact claim contradicted; official status says closed.",
            "irrelevant": "Reliable note: packet discusses river name, not bridge status.",
            "untrusted_true": "Untrusted note: exact claim supported; bridge is open to traffic.",
            "partial": "Reliable note: crews inspected the bridge, but no open or closed status is given.",
            "unknown_source": "Unsigned note: bridge may be open, but source quality is unknown.",
        },
        {
            "concept": "medicine_label",
            "claim": "the label says take two tablets.",
            "trusted_true": "Reliable note: exact claim supported; label says take two tablets.",
            "trusted_false": "Reliable note: exact claim contradicted; label says take one tablet.",
            "irrelevant": "Reliable note: packet discusses bottle size, not dosage directions.",
            "untrusted_true": "Untrusted note: exact claim supported; label says take two tablets.",
            "partial": "Reliable note: label mentions tablets, but dosage count is unreadable.",
            "unknown_source": "Unsigned note: label may say two tablets, but source quality is unknown.",
        },
        {
            "concept": "meeting_time",
            "claim": "the meeting starts at 9 AM.",
            "trusted_true": "Reliable note: exact claim supported; calendar says 9 AM.",
            "trusted_false": "Reliable note: exact claim contradicted; calendar says 11 AM.",
            "irrelevant": "Reliable note: packet discusses room color, not meeting time.",
            "untrusted_true": "Untrusted note: exact claim supported; meeting starts at 9 AM.",
            "partial": "Reliable note: meeting is in the morning, but exact start time is missing.",
            "unknown_source": "Unsigned note: meeting may start at 9 AM, but source quality is unknown.",
        },
    ]


def heldout_specs() -> list[dict[str, str]]:
    return [
        {
            "concept": "train_platform",
            "claim": "the train leaves from platform 4.",
            "trusted_true": "Reliable note: exact claim supported; official board says platform 4.",
            "trusted_false": "Reliable note: exact claim contradicted; official board says platform 2.",
            "irrelevant": "Reliable note: packet discusses ticket color, not platform number.",
            "untrusted_true": "Untrusted note: exact claim supported; train leaves from platform 4.",
            "partial": "Reliable note: train leaves soon, but platform number is not shown.",
            "unknown_source": "Unsigned note: train may leave from platform 4, but source quality is unknown.",
        },
        {
            "concept": "weather_alert",
            "claim": "there is a flood warning today.",
            "trusted_true": "Reliable note: exact claim supported; official alert says flood warning today.",
            "trusted_false": "Reliable note: exact claim contradicted; official alert says no flood warning today.",
            "irrelevant": "Reliable note: packet discusses yesterday's temperature, not today's warning.",
            "untrusted_true": "Untrusted note: exact claim supported; flood warning today.",
            "partial": "Reliable note: heavy rain is possible, but no warning status is given.",
            "unknown_source": "Unsigned note: there may be a flood warning, but source quality is unknown.",
        },
    ]


def rows_from_spec(spec: dict[str, str], *, index: int, split: str) -> list[dict[str, Any]]:
    prefix = "stage101t_train" if split == "train" else "stage101t_heldout"
    claim = str(spec["claim"])
    concept = str(spec["concept"])
    cases = [
        ("direct_true", str(spec["trusted_true"]), "direct_reliable_bucket_belief", " supports", " high", " sufficient"),
        ("direct_false", str(spec["trusted_false"]), "direct_reliable_bucket_belief", " contradicts", " high", " sufficient"),
        ("insufficient", str(spec["irrelevant"]), "insufficient_bucket_belief", " neutral", " high", " insufficient"),
        ("untrusted_only", str(spec["untrusted_true"]), "untrusted_only_bucket_belief", " neutral", " low", " insufficient"),
        ("partial", str(spec["partial"]), "partial_bucket_belief", " neutral", " high", " partial"),
        ("unknown_source", str(spec["unknown_source"]), "unknown_source_bucket_belief", " neutral", " unknown", " insufficient"),
    ]
    rows: list[dict[str, Any]] = []
    for suffix, evidence, case_type, support_bucket, reliability_bucket, sufficiency_bucket in cases:
        rows.extend(
            make_case_rows(
                f"{prefix}_{index:02d}_{suffix}",
                split=split,
                concept=concept,
                claim=claim,
                evidence=evidence,
                case_type=case_type,
                support_bucket=support_bucket,
                reliability_bucket=reliability_bucket,
                sufficiency_bucket=sufficiency_bucket,
            )
        )
    return rows


def bucket_to_number_rows() -> list[dict[str, Any]]:
    rows = dictionary_readback_rows(split="train")
    for index, spec in enumerate(base_specs()):
        rows.extend(rows_from_spec(spec, index=index, split="train"))
    return rows


def bucket_to_number_heldout_rows() -> list[dict[str, Any]]:
    rows = dictionary_readback_rows(split="heldout")
    for index, spec in enumerate(heldout_specs()):
        rows.extend(rows_from_spec(spec, index=index, split="heldout"))
    return rows


def clone_rows_for_replay(rows: list[dict[str, Any]], factor: int, suffix: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for replay_index in range(max(1, int(factor))):
        for item in rows:
            cloned = dict(item)
            cloned["id"] = f"{item['id']}_{suffix}_replay{replay_index:02d}"
            out.append(cloned)
    return out


def anchor_rows(paths: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in paths:
        if not path:
            continue
        for item in load_jsonl(Path(path)):
            row_id = str(item.get("id", ""))
            if row_id.startswith("gd_lite_") or row_id.startswith("stage101b_"):
                out.append(item)
    return out


def dedupe_by_id(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for item in rows:
        row_id = str(item.get("id", ""))
        if row_id in seen:
            continue
        seen.add(row_id)
        out.append(item)
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    rows = bucket_to_number_rows()
    heldout = bucket_to_number_heldout_rows()
    anchors = anchor_rows([str(args.anchor_jsonl), str(args.extra_anchor_jsonl)])
    train_rows = dedupe_by_id(anchors + clone_rows_for_replay(rows, int(args.bucket_replay_factor), "bucket"))
    eval_rows = dedupe_by_id(heldout)
    write_jsonl(Path(args.train_out), train_rows)
    write_jsonl(Path(args.eval_out), eval_rows)
    report = {
        "decision": "built_stage101t_bucket_to_number_probe",
        "train_out": str(args.train_out),
        "eval_out": str(args.eval_out),
        "anchor_rows": int(len(anchors)),
        "bucket_to_number_rows": int(len(rows)),
        "heldout_rows": int(len(heldout)),
        "bucket_replay_factor": int(args.bucket_replay_factor),
        "train_rows": int(len(train_rows)),
        "eval_rows": int(len(eval_rows)),
        "plain_language_read": (
            "Stage101T separates belief meaning from numeric readback. The "
            "model first chooses a semantic bucket, then reads the bucket as a "
            "number through the same LM head."
        ),
    }
    if str(args.report_out):
        report_out = Path(args.report_out)
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-jsonl", default="data/eval/generalization_dynamics_lite_probe.jsonl")
    parser.add_argument("--extra-anchor-jsonl", default="data/eval/stage101b_solution_attractor_heldout_probe.jsonl")
    parser.add_argument("--train-out", default="data/eval/stage101t_bucket_to_number_train_probe.jsonl")
    parser.add_argument("--eval-out", default="data/eval/stage101t_bucket_to_number_heldout_probe.jsonl")
    parser.add_argument("--bucket-replay-factor", type=int, default=1)
    parser.add_argument("--report-out", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = build(args)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
