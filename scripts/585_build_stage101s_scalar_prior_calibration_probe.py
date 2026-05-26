#!/usr/bin/env python3
"""Build Stage101S scalar-prior calibration probes.

Stage101R showed that factorized numeric belief is the right route, but the
remaining failures are low/neutral scalar states:

  support +0.00 -> predicted -0.80
  reliability 0.10 -> predicted 0.90
  sufficiency 0.10 -> predicted 0.90

Stage101S therefore teaches the number prior before asking for more source
semantics.  The model still answers through the normal LM head; no side scalar
head or external evaluator is introduced.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SOURCE = "stage101s_scalar_prior_calibration_probe"
SUPPORT_CHOICES = [" -0.80", " +0.00", " +0.80"]
RELIABILITY_CHOICES = [" 0.10", " 0.50", " 0.90"]
SUFFICIENCY_CHOICES = [" 0.10", " 0.50", " 0.90"]


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


def axis_question(axis: str) -> str:
    if axis == "support":
        return "Estimate usable support only. Choose -0.80, +0.00, or +0.80."
    if axis == "reliability":
        return "Estimate reliability of usable evidence only. Choose 0.10, 0.50, or 0.90."
    if axis == "sufficiency":
        return "Estimate whether usable evidence decides the claim. Choose 0.10, 0.50, or 0.90."
    raise ValueError(f"unknown axis: {axis}")


def choices_for_axis(axis: str) -> list[str]:
    if axis == "support":
        return list(SUPPORT_CHOICES)
    if axis == "reliability":
        return list(RELIABILITY_CHOICES)
    if axis == "sufficiency":
        return list(SUFFICIENCY_CHOICES)
    raise ValueError(f"unknown axis: {axis}")


def make_scalar_row(
    row_id: str,
    *,
    split: str,
    axis: str,
    answer: str,
    prompt_body: str,
    case_type: str,
    support_score: float,
    reliability_score: float,
    sufficiency_score: float,
    concept: str = "",
    claim: str = "",
    source_template: str = "scalar_prior",
) -> dict[str, Any]:
    choices = choices_for_axis(axis)
    if answer not in choices:
        raise ValueError(f"answer {answer!r} is not in choices for {axis}: {choices!r}")
    return {
        "id": row_id,
        "source": SOURCE,
        "task": f"stage101s_scalar_prior_{axis}_icl",
        "prompt": f"{prompt_body}\nQ: {axis_question(axis)}\nA:",
        "intelligence_answer": answer,
        "parrot_answer": negative_answers(answer, choices)[0],
        "candidate_answers": choices,
        "negative_answers": negative_answers(answer, choices),
        "plain_language_axis": (
            "The model must learn low and neutral numeric belief states before "
            "combining them with source semantics."
        ),
        "source_case_type": case_type,
        "source_template": source_template,
        "source_concept": concept,
        "source_claim": claim,
        "belief_axis": axis,
        "belief_support_score": float(support_score),
        "source_reliability_score": float(reliability_score),
        "evidence_sufficiency_score": float(sufficiency_score),
        "stage101s_scalar_prior_calibration_required": True,
        "factorized_numeric_belief_required": True,
        "split": split,
    }


def make_source_prompt(*, claim: str, evidence: str) -> str:
    return f"Claim: {claim}\nEvidence: {evidence}"


def make_case_rows(
    row_id: str,
    *,
    split: str,
    concept: str,
    claim: str,
    evidence: str,
    case_type: str,
    support: str,
    reliability: str,
    sufficiency: str,
) -> list[dict[str, Any]]:
    prompt_body = make_source_prompt(claim=claim, evidence=evidence)
    return [
        make_scalar_row(
            f"{row_id}_support",
            split=split,
            axis="support",
            answer=f" {support}",
            prompt_body=prompt_body,
            case_type=case_type,
            support_score=float(support),
            reliability_score=float(reliability),
            sufficiency_score=float(sufficiency),
            concept=concept,
            claim=claim,
            source_template="claim_first",
        ),
        make_scalar_row(
            f"{row_id}_reliability",
            split=split,
            axis="reliability",
            answer=f" {reliability}",
            prompt_body=prompt_body,
            case_type=case_type,
            support_score=float(support),
            reliability_score=float(reliability),
            sufficiency_score=float(sufficiency),
            concept=concept,
            claim=claim,
            source_template="claim_first",
        ),
        make_scalar_row(
            f"{row_id}_sufficiency",
            split=split,
            axis="sufficiency",
            answer=f" {sufficiency}",
            prompt_body=prompt_body,
            case_type=case_type,
            support_score=float(support),
            reliability_score=float(reliability),
            sufficiency_score=float(sufficiency),
            concept=concept,
            claim=claim,
            source_template="claim_first",
        ),
    ]


def source_free_rows(*, split: str) -> list[dict[str, Any]]:
    specs = [
        ("support_strong_positive", "support", " +0.80", "Calibration card: usable evidence strongly supports the claim.", 0.8, 0.9, 0.9),
        ("support_strong_negative", "support", " -0.80", "Calibration card: usable evidence strongly contradicts the claim.", -0.8, 0.9, 0.9),
        ("support_absent_neutral", "support", " +0.00", "Calibration card: usable evidence is absent, so support stays neutral.", 0.0, 0.1, 0.1),
        ("support_offtopic_neutral", "support", " +0.00", "Calibration card: available text is off topic, so support stays neutral.", 0.0, 0.9, 0.1),
        ("support_untrusted_neutral", "support", " +0.00", "Calibration card: only an untrusted note supports the claim, so usable support stays neutral.", 0.0, 0.1, 0.1),
        ("reliability_high", "reliability", " 0.90", "Calibration card: usable evidence comes from a reliable source.", 0.8, 0.9, 0.9),
        ("reliability_unknown", "reliability", " 0.50", "Calibration card: source quality is unclear.", 0.0, 0.5, 0.1),
        ("reliability_low_untrusted", "reliability", " 0.10", "Calibration card: the only source is explicitly untrusted.", 0.0, 0.1, 0.1),
        ("reliability_low_rumor", "reliability", " 0.10", "Calibration card: the note is a rumor and should be treated as unreliable.", 0.0, 0.1, 0.1),
        ("reliability_low_unsigned", "reliability", " 0.10", "Calibration card: the note has no author and no source trail.", 0.0, 0.1, 0.1),
        ("reliability_low_guess", "reliability", " 0.10", "Calibration card: the statement is labeled as a guess.", 0.0, 0.1, 0.1),
        ("reliability_low_ad", "reliability", " 0.10", "Calibration card: the note is promotional text, not evidence.", 0.0, 0.1, 0.1),
        ("reliability_low_contradicted", "reliability", " 0.10", "Calibration card: the source is marked unreliable by the packet.", 0.0, 0.1, 0.1),
        ("reliability_low_fiction", "reliability", " 0.10", "Calibration card: the line is from a fictional example.", 0.0, 0.1, 0.1),
        ("reliability_low_stale", "reliability", " 0.10", "Calibration card: the source is stale and explicitly not trusted.", 0.0, 0.1, 0.1),
        ("reliability_low_hearsay", "reliability", " 0.10", "Calibration card: the note says someone heard it from someone else.", 0.0, 0.1, 0.1),
        ("reliability_low_unknown_origin", "reliability", " 0.10", "Calibration card: the origin is unknown and the packet says do not rely on it.", 0.0, 0.1, 0.1),
        ("reliability_low_corrupt", "reliability", " 0.10", "Calibration card: the source line is marked corrupted.", 0.0, 0.1, 0.1),
        ("reliability_low_unverified", "reliability", " 0.10", "Calibration card: the note is explicitly unverified.", 0.0, 0.1, 0.1),
        ("reliability_low_decoy", "reliability", " 0.10", "Calibration card: the source is labeled decoy evidence.", 0.0, 0.1, 0.1),
        ("sufficiency_high", "sufficiency", " 0.90", "Calibration card: usable evidence directly decides the claim.", 0.8, 0.9, 0.9),
        ("sufficiency_partial", "sufficiency", " 0.50", "Calibration card: usable evidence is related but incomplete.", 0.0, 0.9, 0.5),
        ("sufficiency_low_missing", "sufficiency", " 0.10", "Calibration card: usable evidence is missing.", 0.0, 0.9, 0.1),
        ("sufficiency_low_offtopic", "sufficiency", " 0.10", "Calibration card: usable evidence is off topic and cannot decide the claim.", 0.0, 0.9, 0.1),
        ("sufficiency_low_background", "sufficiency", " 0.10", "Calibration card: usable evidence gives background but not the needed fact.", 0.0, 0.9, 0.1),
        ("sufficiency_low_wrong_field", "sufficiency", " 0.10", "Calibration card: usable evidence names a different field than the claim asks for.", 0.0, 0.9, 0.1),
        ("sufficiency_low_no_measure", "sufficiency", " 0.10", "Calibration card: usable evidence lacks the measurement needed to decide.", 0.0, 0.9, 0.1),
        ("sufficiency_low_ambiguous", "sufficiency", " 0.10", "Calibration card: usable evidence is ambiguous and cannot decide the claim.", 0.0, 0.9, 0.1),
    ]
    prefix = "stage101s_prior_train" if split == "train" else "stage101s_prior_heldout"
    return [
        make_scalar_row(
            f"{prefix}_{index:02d}_{suffix}",
            split=split,
            axis=axis,
            answer=answer,
            prompt_body=prompt_body,
            case_type="source_free_scalar_prior_calibration",
            support_score=support_score,
            reliability_score=reliability_score,
            sufficiency_score=sufficiency_score,
            concept="scalar_prior",
            claim="",
            source_template="source_free",
        )
        for index, (suffix, axis, answer, prompt_body, support_score, reliability_score, sufficiency_score)
        in enumerate(specs)
    ]


def base_specs() -> list[dict[str, str]]:
    return [
        {
            "concept": "battery_charge",
            "claim": "the battery is fully charged.",
            "trusted_true": "Reliable note: exact claim supported; the battery reads 100 percent.",
            "trusted_false": "Reliable note: exact claim contradicted; the battery reads 12 percent.",
            "irrelevant": "Reliable note: packet discusses the case color, not battery charge.",
            "untrusted_true": "Untrusted note: exact claim supported; the battery reads 100 percent.",
        },
        {
            "concept": "bridge_status",
            "claim": "the bridge is open to traffic.",
            "trusted_true": "Reliable note: exact claim supported; official status says open.",
            "trusted_false": "Reliable note: exact claim contradicted; official status says closed.",
            "irrelevant": "Reliable note: packet discusses the river name, not bridge status.",
            "untrusted_true": "Untrusted note: exact claim supported; bridge is open to traffic.",
        },
        {
            "concept": "medicine_label",
            "claim": "the label says take two tablets.",
            "trusted_true": "Reliable note: exact claim supported; label says take two tablets.",
            "trusted_false": "Reliable note: exact claim contradicted; label says take one tablet.",
            "irrelevant": "Reliable note: packet discusses bottle size, not dosage directions.",
            "untrusted_true": "Untrusted note: exact claim supported; label says take two tablets.",
        },
        {
            "concept": "meeting_time",
            "claim": "the meeting starts at 9 AM.",
            "trusted_true": "Reliable note: exact claim supported; calendar says 9 AM.",
            "trusted_false": "Reliable note: exact claim contradicted; calendar says 11 AM.",
            "irrelevant": "Reliable note: packet discusses the room color, not meeting time.",
            "untrusted_true": "Untrusted note: exact claim supported; meeting starts at 9 AM.",
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
        },
        {
            "concept": "weather_alert",
            "claim": "there is a flood warning today.",
            "trusted_true": "Reliable note: exact claim supported; official alert says flood warning today.",
            "trusted_false": "Reliable note: exact claim contradicted; official alert says no flood warning today.",
            "irrelevant": "Reliable note: packet discusses yesterday's temperature, not today's warning.",
            "untrusted_true": "Untrusted note: exact claim supported; flood warning today.",
        },
    ]


def rows_from_spec(spec: dict[str, str], *, index: int, split: str) -> list[dict[str, Any]]:
    prefix = "stage101s_semantic_train" if split == "train" else "stage101s_semantic_heldout"
    claim = str(spec["claim"])
    concept = str(spec["concept"])
    rows: list[dict[str, Any]] = []
    cases = [
        ("direct_true", str(spec["trusted_true"]), "direct_reliable_factorized_belief", "+0.80", "0.90", "0.90"),
        ("direct_false", str(spec["trusted_false"]), "direct_reliable_factorized_belief", "-0.80", "0.90", "0.90"),
        ("insufficient", str(spec["irrelevant"]), "insufficient_factorized_belief", "+0.00", "0.90", "0.10"),
        ("untrusted_only", str(spec["untrusted_true"]), "untrusted_only_factorized_belief", "+0.00", "0.10", "0.10"),
        (
            "untrusted_override",
            f"Untrusted says the claim is true. Reliable note says: {spec['trusted_false']}",
            "untrusted_override_factorized_belief",
            "-0.80",
            "0.90",
            "0.90",
        ),
    ]
    for suffix, evidence, case_type, support, reliability, sufficiency in cases:
        rows.extend(
            make_case_rows(
                f"{prefix}_{index:02d}_{suffix}",
                split=split,
                concept=concept,
                claim=claim,
                evidence=evidence,
                case_type=case_type,
                support=support,
                reliability=reliability,
                sufficiency=sufficiency,
            )
        )
    return rows


def scalar_prior_calibration_rows() -> list[dict[str, Any]]:
    rows = source_free_rows(split="train")
    for index, spec in enumerate(base_specs()):
        rows.extend(rows_from_spec(spec, index=index, split="train"))
    return rows


def scalar_prior_calibration_heldout_rows() -> list[dict[str, Any]]:
    rows = source_free_rows(split="heldout")
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
    calibration_rows = scalar_prior_calibration_rows()
    heldout_rows = scalar_prior_calibration_heldout_rows()
    anchors = anchor_rows([str(args.anchor_jsonl), str(args.extra_anchor_jsonl)])
    base_rows = load_jsonl(Path(str(args.base_factorized_jsonl))) if str(args.base_factorized_jsonl) else []
    train_rows = dedupe_by_id(
        anchors
        + clone_rows_for_replay(calibration_rows, int(args.calibration_replay_factor), "calib")
        + clone_rows_for_replay(base_rows, int(args.base_replay_factor), "base")
    )
    eval_rows = dedupe_by_id(heldout_rows)
    write_jsonl(Path(args.train_out), train_rows)
    write_jsonl(Path(args.eval_out), eval_rows)
    report = {
        "decision": "built_stage101s_scalar_prior_calibration_probe",
        "train_out": str(args.train_out),
        "eval_out": str(args.eval_out),
        "anchor_rows": int(len(anchors)),
        "base_factorized_rows": int(len(base_rows)),
        "calibration_rows": int(len(calibration_rows)),
        "heldout_rows": int(len(heldout_rows)),
        "base_replay_factor": int(args.base_replay_factor),
        "calibration_replay_factor": int(args.calibration_replay_factor),
        "train_rows": int(len(train_rows)),
        "eval_rows": int(len(eval_rows)),
        "plain_language_read": (
            "Stage101S teaches the model the numeric meaning of neutral support "
            "and low reliability/sufficiency before scaling source-belief rows."
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
    parser.add_argument("--base-factorized-jsonl", default="data/eval/stage101r_factorized_numeric_belief_train_probe.jsonl")
    parser.add_argument("--train-out", default="data/eval/stage101s_scalar_prior_calibration_train_probe.jsonl")
    parser.add_argument("--eval-out", default="data/eval/stage101s_scalar_prior_calibration_heldout_probe.jsonl")
    parser.add_argument("--base-replay-factor", type=int, default=1)
    parser.add_argument("--calibration-replay-factor", type=int, default=2)
    parser.add_argument("--report-out", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = build(args)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
