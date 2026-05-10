#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = "configs/qwen35_2b_4090_source_copy_pointer_renderer_scaffold.yaml"
DEFAULT_CHECKPOINT = (
    "/mnt/nvme1n1p2/qtrm-runs/research_gate_runner/"
    "qtrm_source_position_l3_hard_batch_s240_b8_eval/accepted_l3_last.pt"
)
DEFAULT_CASES = "data/eval/qtrm_source_copy_lexicalization_eval128.jsonl"
DEFAULT_REPORT = (
    "/mnt/nvme0n1p2/qtrm-runs/research_gate_runner/"
    "qtrm_source_copy_alignment_probe/report.json"
)


def load_jsonl(path: str | Path, *, max_cases: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if int(max_cases) > 0 and len(rows) >= int(max_cases):
                break
    return rows


def target_source_position_classes(
    row: dict[str, Any],
    *,
    num_roles: int,
) -> list[int]:
    values = [int(value) for value in row.get("input_list") or []]
    classes = [index + 1 for index, value in enumerate(values) if value % 2 == 0]
    classes = classes[: int(num_roles)]
    classes.extend([0] * max(0, int(num_roles) - len(classes)))
    return classes


def score_source_position_logits(
    logits: Any,
    *,
    target_classes: list[int],
) -> dict[str, Any]:
    import torch

    values = logits.detach().float()
    if values.ndim == 4:
        values = values[0, 0]
    elif values.ndim == 3:
        values = values[0]
    if values.ndim != 2 or int(values.shape[0]) == 0 or int(values.shape[1]) == 0:
        target = [int(value) for value in target_classes]
        return {
            "predicted_classes": [],
            "target_classes": target,
            "content_positions": sum(1 for value in target if int(value) > 0),
            "correct_content_positions": 0,
            "null_positions": sum(1 for value in target if int(value) == 0),
            "correct_null_positions": 0,
            "row_content_exact": False,
            "row_full_exact": False,
        }
    predicted = values.argmax(dim=-1).to(dtype=torch.long).cpu().tolist()
    roles = min(len(predicted), len(target_classes))
    predicted = [int(value) for value in predicted[:roles]]
    target = [int(value) for value in target_classes[:roles]]
    content_indices = [idx for idx, value in enumerate(target) if int(value) > 0]
    null_indices = [idx for idx, value in enumerate(target) if int(value) == 0]
    correct_content = sum(
        1 for idx in content_indices if int(predicted[idx]) == int(target[idx])
    )
    correct_null = sum(
        1 for idx in null_indices if int(predicted[idx]) == int(target[idx])
    )
    return {
        "predicted_classes": predicted,
        "target_classes": target,
        "content_positions": len(content_indices),
        "correct_content_positions": correct_content,
        "null_positions": len(null_indices),
        "correct_null_positions": correct_null,
        "row_content_exact": bool(content_indices)
        and correct_content == len(content_indices),
        "row_full_exact": predicted == target,
    }


def select_alignment_logits(
    outputs: dict[str, Any],
    *,
    prefer_primitive: bool = False,
) -> Any:
    primitive = outputs.get("core_primitive_role_value_state_logits")
    if (
        bool(prefer_primitive)
        and primitive is not None
        and getattr(primitive, "ndim", 0) == 4
        and int(primitive.shape[1]) > 0
    ):
        return primitive[:, -1:, :, :]
    recurrent = outputs.get("core_role_value_state_logits")
    if (
        recurrent is not None
        and getattr(recurrent, "ndim", 0) == 4
        and int(recurrent.shape[1]) > 0
    ):
        return recurrent[:, -1:, :, :]
    if (
        primitive is not None
        and getattr(primitive, "ndim", 0) == 4
        and int(primitive.shape[1]) > 0
    ):
        return primitive[:, -1:, :, :]
    return outputs.get("core_source_position_prompt_logits")


def mask_alignment_logits_to_answer_roles(logits: Any, *, num_roles: int) -> Any:
    if (
        logits is None
        or getattr(logits, "ndim", 0) != 4
        or int(logits.shape[2]) == 0
        or int(logits.shape[-1]) == 0
    ):
        return logits
    answer_roles = max(1, (int(num_roles) - 2) // 2)
    if answer_roles >= int(logits.shape[2]):
        return logits
    masked = logits.clone()
    masked[:, :, answer_roles:, :] = -1.0e4
    masked[:, :, answer_roles:, 0] = 1.0e4
    return masked


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = len(records)
    content_total = sum(int(record["content_positions"]) for record in records)
    content_correct = sum(
        int(record["correct_content_positions"]) for record in records
    )
    null_total = sum(int(record["null_positions"]) for record in records)
    null_correct = sum(int(record["correct_null_positions"]) for record in records)
    row_content_exact = sum(int(bool(record["row_content_exact"])) for record in records)
    row_full_exact = sum(int(bool(record["row_full_exact"])) for record in records)
    return {
        "rows": rows,
        "content_positions": content_total,
        "correct_content_positions": content_correct,
        "content_position_accuracy": (
            float(content_correct) / float(content_total) if content_total else 0.0
        ),
        "null_positions": null_total,
        "correct_null_positions": null_correct,
        "null_position_accuracy": (
            float(null_correct) / float(null_total) if null_total else 0.0
        ),
        "row_content_exact": row_content_exact,
        "row_content_exact_accuracy": (
            float(row_content_exact) / float(rows) if rows else 0.0
        ),
        "row_full_exact": row_full_exact,
        "row_full_exact_accuracy": float(row_full_exact) / float(rows)
        if rows
        else 0.0,
    }


def _select_device(train_device: str, requested: str) -> str:
    import torch

    if requested:
        return str(requested)
    if str(train_device) == "cuda":
        return "cuda"
    if str(train_device) == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _prepare_prompt_and_source_slots(
    tokenizer: Any,
    row: dict[str, Any],
    *,
    max_length: int,
    device: str,
    value_vocab_size: int,
    max_slots: int,
):
    import torch
    from qtrm_mm.algorithmic_value_state import (
        token_numeric_source_slot_ids,
        token_numeric_source_slot_token_ids,
    )

    enc = tokenizer(
        str(row["prompt"]),
        return_tensors="pt",
        truncation=True,
        max_length=int(max_length),
        padding=False,
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)
    offsets = enc["offset_mapping"][0].tolist()
    ids, mask = token_numeric_source_slot_ids(
        row,
        offsets=offsets,
        max_list_len=int(max_slots),
        value_vocab_size=int(value_vocab_size),
    )
    token_ids = token_numeric_source_slot_token_ids(
        row,
        offsets=offsets,
        input_ids=enc["input_ids"][0].tolist(),
        max_list_len=int(max_slots),
        value_vocab_size=int(value_vocab_size),
    )
    return (
        input_ids,
        attention_mask,
        torch.tensor([ids], dtype=torch.long, device=device),
        torch.tensor([token_ids], dtype=torch.long, device=device),
        torch.tensor([mask], dtype=torch.long, device=device),
    )


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    from qtrm_mm.config import load_config
    from qtrm_mm.qtrm_model import QTRMMultimodalModel
    from qtrm_mm.qwen_donor import QwenDonorAdapter
    from qtrm_mm.training.train import load_initial_checkpoint

    cfg = load_config(args.config)
    cfg.model.token_numeric_source_slot_embedding_enabled = True
    cfg.model.token_numeric_source_slot_vocab_size = int(
        args.token_numeric_source_slot_vocab_size
    )
    cfg.model.token_numeric_source_slot_max_slots = int(
        args.token_numeric_source_slot_max_slots
    )
    cfg.model.token_numeric_source_slot_gate_min = float(
        args.token_numeric_source_slot_gate_min
    )
    cfg.model.token_numeric_source_slot_predicate_feedback_enabled = True
    cfg.model.token_numeric_source_slot_predicate_gate_min = float(
        args.token_numeric_source_slot_predicate_gate_min
    )
    cfg.model.core_source_position_binder_enabled = True
    cfg.model.core_source_position_binder_gate_min = float(
        args.core_source_position_binder_gate_min
    )
    cfg.model.core_source_position_binder_state_gate_min = float(
        args.core_source_position_binder_state_gate_min
    )
    cfg.model.core_source_position_binder_state_straight_through = True
    cfg.model.core_source_position_binder_source_slots_only = True
    cfg.model.core_source_position_binder_raw_source_slots_enabled = True

    device = _select_device(cfg.train.device, args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_model_id or cfg.donor.model_id,
        trust_remote_code=cfg.donor.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = QTRMMultimodalModel(cfg.model)
    missing, unexpected = load_initial_checkpoint(
        model,
        args.checkpoint,
        map_location=device,
    )
    model = model.to(device).eval()
    donor = QwenDonorAdapter(cfg.donor)
    rows = load_jsonl(args.cases, max_cases=int(args.max_cases))
    old_outer_steps = int(model.cfg.outer_steps)
    model.cfg.outer_steps = int(args.core_steps)
    records: list[dict[str, Any]] = []
    max_length = int(args.max_length or cfg.train.seq_len)
    try:
        with torch.no_grad():
            for row in rows:
                (
                    input_ids,
                    attention_mask,
                    source_slot_ids,
                    source_slot_token_ids,
                    source_slot_mask,
                ) = _prepare_prompt_and_source_slots(
                    tokenizer,
                    row,
                    max_length=max_length,
                    device=device,
                    value_vocab_size=int(args.token_numeric_source_slot_vocab_size),
                    max_slots=int(args.token_numeric_source_slot_max_slots),
                )
                donor_out = donor.encode_inputs(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_logits=False,
                )
                with torch.amp.autocast(
                    "cuda",
                    enabled=(bool(cfg.train.use_amp) and device == "cuda"),
                    dtype=torch.bfloat16,
                ):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_numeric_source_slot_ids=source_slot_ids,
                        token_numeric_source_slot_token_ids=source_slot_token_ids,
                        token_numeric_source_slot_mask=source_slot_mask,
                        text_states=donor_out["text_states"].detach().to(device),
                        disable_core_source_position_binder=bool(
                            args.disable_core_source_position_binder
                        ),
                    )
                logits = mask_alignment_logits_to_answer_roles(
                    select_alignment_logits(
                        outputs,
                        prefer_primitive=bool(
                            cfg.model.core_role_value_state_vocab_renderer_source_copy_from_primitive_enabled
                        ),
                    ),
                    num_roles=int(cfg.model.core_role_value_state_num_roles),
                )
                target = target_source_position_classes(
                    row,
                    num_roles=int(
                        logits.shape[2]
                        if logits is not None and getattr(logits, "ndim", 0) == 4
                        else cfg.model.core_role_value_state_num_roles
                    ),
                )
                score = score_source_position_logits(logits, target_classes=target)
                score.update(
                    {
                        "id": row.get("id"),
                        "input_list": row.get("input_list"),
                        "answer": row.get("answer"),
                    }
                )
                records.append(score)
    finally:
        model.cfg.outer_steps = old_outer_steps

    summary = summarize_records(records)
    content_exact = float(summary["row_content_exact_accuracy"])
    report = {
        "decision": (
            "accepted_l2_source_copy_alignment_probe"
            if content_exact >= float(args.min_row_content_exact_accuracy)
            else "rejected_l2_source_copy_alignment_probe"
        ),
        "accepted": content_exact >= float(args.min_row_content_exact_accuracy),
        "target_level": "L2 prerequisite repair diagnostic",
        "method_class": "diagnostic probe",
        "major_bottleneck": "source-position binder to pointer/copy lexicalizer",
        "prior_family": "pointer-generator / copy attention",
        "canonical_path": (
            "prompt tokens -> token source slots -> source-position logits -> "
            "pointer/copy LM-token rendering"
        ),
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "cases": str(args.cases),
        "core_steps": int(args.core_steps),
        "disable_core_source_position_binder": bool(
            args.disable_core_source_position_binder
        ),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "summary": summary,
        "examples": records[: int(args.example_count)],
        "next_action": (
            "repair answer-step renderer/query path if generation still fails"
            if content_exact >= float(args.min_row_content_exact_accuracy)
            else "repair source-position binder supervision/interface before L4 training"
        ),
    }
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Probe whether QTRM source-position logits match the source-copy "
            "oracle positions required by the pointer/copy renderer."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--tokenizer-model-id", default="Qwen/Qwen3.5-2B-Base")
    parser.add_argument("--device", default="")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--core-steps", type=int, default=8)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--example-count", type=int, default=8)
    parser.add_argument("--token-numeric-source-slot-vocab-size", type=int, default=128)
    parser.add_argument("--token-numeric-source-slot-max-slots", type=int, default=5)
    parser.add_argument("--token-numeric-source-slot-gate-min", type=float, default=1.0)
    parser.add_argument(
        "--token-numeric-source-slot-predicate-gate-min",
        type=float,
        default=1.0,
    )
    parser.add_argument("--core-source-position-binder-gate-min", type=float, default=1.0)
    parser.add_argument(
        "--core-source-position-binder-state-gate-min",
        type=float,
        default=1.0,
    )
    parser.add_argument("--disable-core-source-position-binder", action="store_true")
    parser.add_argument("--min-row-content-exact-accuracy", type=float, default=0.95)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    report = run_probe(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if bool(report.get("accepted")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
