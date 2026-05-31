#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PRIMITIVE_TRANSITION_OPERATION_ORDER = (
    "add_operands",
    "multiply_sum",
    "subtract_offset",
    "hold_final",
    "filter_even",
    "double_filtered",
    "first_mapping",
    "second_mapping",
    "not_q",
    "and_with_p",
    "or_with_r",
    "filter_above_threshold",
)

PRIMITIVE_OPERATION_TO_DYNAMIC_HALT_CODE = {
    "add_operands": 0,
    "first_mapping": 0,
    "not_q": 0,
    "filter_even": 0,
    "filter_above_threshold": 1,
    "second_mapping": 1,
    "double_filtered": 1,
    "multiply_sum": 2,
    "and_with_p": 2,
    "subtract_offset": 3,
    "or_with_r": 3,
    "hold_final": 4,
}


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get("prompt"):
                raise ValueError(f"{path}:{line_no}: missing prompt")
            if not isinstance(row.get("transition_state_codes"), dict):
                raise ValueError(f"{path}:{line_no}: missing transition_state_codes")
            rows.append(row)
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def code_targets_from_row(row: dict[str, Any], *, num_steps: int) -> list[int]:
    codes = row.get("transition_state_codes")
    if not isinstance(codes, dict):
        raise ValueError("row must contain transition_state_codes")
    targets: list[int] = []
    for depth in range(1, int(num_steps) + 1):
        raw_code = codes.get(str(depth))
        targets.append(-100 if raw_code is None else int(raw_code))
    return targets


def finality_targets_from_row(row: dict[str, Any], *, num_steps: int) -> list[float]:
    targets = row.get("transition_finality_targets")
    if not isinstance(targets, dict):
        return [-100.0] * int(num_steps)
    out: list[float] = []
    for depth in range(1, int(num_steps) + 1):
        raw_value = targets.get(str(depth))
        out.append(-100.0 if raw_value is None else float(raw_value))
    return out


def score_code_predictions(
    *,
    predicted_codes: list[int],
    target_codes: list[int],
) -> dict[str, Any]:
    if len(predicted_codes) != len(target_codes):
        raise ValueError("predicted_codes and target_codes must have the same length")
    correct = 0
    total = 0
    labelled_records = []
    for index, (predicted, target) in enumerate(zip(predicted_codes, target_codes)):
        if int(target) < 0:
            continue
        hit = int(predicted) == int(target)
        correct += int(hit)
        total += 1
        labelled_records.append(
            {
                "step_index": index,
                "depth": index + 1,
                "predicted": int(predicted),
                "target": int(target),
                "correct": bool(hit),
            }
        )
    return {
        "correct_steps": correct,
        "total_steps": total,
        "step_accuracy": (float(correct) / float(total)) if total else 0.0,
        "trace_exact": bool(total and correct == total),
        "labelled_steps": labelled_records,
    }


def score_finality_predictions(
    *,
    finality_logits: list[float],
    target_values: list[float],
) -> dict[str, Any]:
    if len(finality_logits) != len(target_values):
        raise ValueError("finality_logits and target_values must have the same length")
    correct = 0
    total = 0
    records = []
    for index, (logit, target) in enumerate(zip(finality_logits, target_values)):
        if float(target) < 0.0:
            continue
        pred = 1.0 if float(logit) > 0.0 else 0.0
        hit = pred == float(target)
        correct += int(hit)
        total += 1
        records.append(
            {
                "step_index": index,
                "depth": index + 1,
                "predicted": pred,
                "target": float(target),
                "correct": bool(hit),
            }
        )
    return {
        "finality_correct_steps": correct,
        "finality_total_steps": total,
        "finality_step_accuracy": float(correct) / float(total) if total else 0.0,
        "finality_trace_exact": bool(total and correct == total),
        "finality_labelled_steps": records,
    }


def score_halted_transition_predictions(
    *,
    predicted_codes: list[int],
    target_codes: list[int],
    finality_logits: list[float],
    target_finality: list[float],
) -> dict[str, Any]:
    if len(predicted_codes) != len(target_codes):
        raise ValueError("predicted_codes and target_codes must have the same length")
    if len(finality_logits) != len(target_finality):
        raise ValueError("finality_logits and target_finality must have the same length")
    if len(predicted_codes) != len(finality_logits):
        raise ValueError("code and finality predictions must have the same length")
    predicted_finality = [1.0 if float(logit) > 0.0 else 0.0 for logit in finality_logits]
    halt_index = next(
        (index for index, value in enumerate(predicted_finality) if value == 1.0),
        None,
    )
    if halt_index is None:
        return {
            "halted_trace_exact": False,
            "halted_depth": None,
            "halted_prefix_steps": 0,
        }
    if halt_index >= len(target_finality) or float(target_finality[halt_index]) != 1.0:
        return {
            "halted_trace_exact": False,
            "halted_depth": halt_index + 1,
            "halted_prefix_steps": 0,
        }
    checked = 0
    for index in range(halt_index + 1):
        if int(target_codes[index]) >= 0:
            checked += 1
            if int(predicted_codes[index]) != int(target_codes[index]):
                return {
                    "halted_trace_exact": False,
                    "halted_depth": halt_index + 1,
                    "halted_prefix_steps": checked,
                }
        if float(target_finality[index]) >= 0.0 and predicted_finality[index] != float(
            target_finality[index]
        ):
            return {
                "halted_trace_exact": False,
                "halted_depth": halt_index + 1,
                "halted_prefix_steps": checked,
            }
    return {
        "halted_trace_exact": bool(checked > 0),
        "halted_depth": halt_index + 1,
        "halted_prefix_steps": checked,
    }


def predicted_codes_from_logits(logits: Any) -> list[int]:
    if logits.ndim != 3 or int(logits.shape[-1]) == 0:
        return []
    return logits.detach().float().argmax(dim=-1)[0].cpu().tolist()


def predicted_joint_states_from_logits(logits: Any) -> list[int]:
    if logits.ndim != 3 or int(logits.shape[-1]) == 0:
        return []
    return logits.detach().float().argmax(dim=-1)[0].cpu().tolist()


def predicted_operation_ids_from_logits(logits: Any) -> list[int]:
    if logits.ndim != 3 or int(logits.shape[-1]) == 0:
        return []
    return logits.detach().float().argmax(dim=-1)[0].cpu().tolist()


def transition_codes_from_primitive_operations(operation_ids: list[int]) -> list[int]:
    codes: list[int] = []
    for operation_id in operation_ids:
        if int(operation_id) < 0 or int(operation_id) >= len(
            PRIMITIVE_TRANSITION_OPERATION_ORDER
        ):
            raise ValueError(f"unknown primitive operation id: {operation_id}")
        operation = PRIMITIVE_TRANSITION_OPERATION_ORDER[int(operation_id)]
        codes.append(int(PRIMITIVE_OPERATION_TO_DYNAMIC_HALT_CODE[operation]))
    return codes


def predicted_source_ids_from_logits(logits: Any) -> list[int]:
    if logits.ndim != 3 or int(logits.shape[-1]) == 0:
        return []
    return logits.detach().float().argmax(dim=-1)[0].cpu().tolist()


def parse_code_permutation(value: str) -> dict[int, int] | None:
    text = str(value or "").strip()
    if not text:
        return None
    out: dict[int, int] = {}
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"invalid code permutation item: {item!r}")
        src, dst = item.split(":", 1)
        out[int(src.strip())] = int(dst.strip())
    if not out:
        return None
    expected = set(range(max(out) + 1))
    if set(out) != expected:
        raise ValueError(
            f"code permutation keys must be contiguous 0..N, got {sorted(out)}"
        )
    return out


def apply_predicted_code_ablation(
    predicted_codes: list[int],
    *,
    code_permutation: dict[int, int] | None = None,
    drop_codes_to: int | None = None,
) -> list[int]:
    if code_permutation is not None and drop_codes_to is not None:
        raise ValueError("code permutation and code dropout cannot both be enabled")
    if drop_codes_to is not None:
        return [int(drop_codes_to)] * len(predicted_codes)
    if code_permutation is None:
        return [int(value) for value in predicted_codes]
    out: list[int] = []
    for value in predicted_codes:
        code = int(value)
        if code not in code_permutation:
            raise ValueError(f"predicted code {code} is missing from permutation")
        out.append(int(code_permutation[code]))
    return out


def _prepare_prompt(tokenizer: Any, prompt: str, *, max_length: int, device: str):
    import torch

    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=True,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)
    return input_ids, attention_mask


def _family(row: dict[str, Any]) -> str:
    return str(row.get("task_family") or row.get("category") or "unknown")


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    total_steps = sum(int(record["total_steps"]) for record in records)
    correct_steps = sum(int(record["correct_steps"]) for record in records)
    exact_rows = sum(int(bool(record["trace_exact"])) for record in records)
    finality_total_steps = sum(int(record.get("finality_total_steps", 0)) for record in records)
    finality_correct_steps = sum(int(record.get("finality_correct_steps", 0)) for record in records)
    finality_exact_rows = sum(int(bool(record.get("finality_trace_exact", False))) for record in records)
    halted_exact_rows = sum(int(bool(record.get("halted_trace_exact", False))) for record in records)
    by_family: dict[str, dict[str, Any]] = {}
    for record in records:
        family = str(record["task_family"])
        bucket = by_family.setdefault(
            family,
            {"rows": 0, "exact_rows": 0, "correct_steps": 0, "total_steps": 0},
        )
        bucket["rows"] += 1
        bucket["exact_rows"] += int(bool(record["trace_exact"]))
        bucket["correct_steps"] += int(record["correct_steps"])
        bucket["total_steps"] += int(record["total_steps"])
        bucket["finality_correct_steps"] = bucket.get("finality_correct_steps", 0) + int(
            record.get("finality_correct_steps", 0)
        )
        bucket["finality_total_steps"] = bucket.get("finality_total_steps", 0) + int(
            record.get("finality_total_steps", 0)
        )
        bucket["finality_exact_rows"] = bucket.get("finality_exact_rows", 0) + int(
            bool(record.get("finality_trace_exact", False))
        )
        bucket["halted_exact_rows"] = bucket.get("halted_exact_rows", 0) + int(
            bool(record.get("halted_trace_exact", False))
        )
    for bucket in by_family.values():
        bucket["step_accuracy"] = (
            float(bucket["correct_steps"]) / float(bucket["total_steps"])
            if int(bucket["total_steps"])
            else 0.0
        )
        bucket["trace_exact_accuracy"] = (
            float(bucket["exact_rows"]) / float(bucket["rows"])
            if int(bucket["rows"])
            else 0.0
        )
        bucket["finality_step_accuracy"] = (
            float(bucket["finality_correct_steps"]) / float(bucket["finality_total_steps"])
            if int(bucket["finality_total_steps"])
            else 0.0
        )
        bucket["finality_trace_exact_accuracy"] = (
            float(bucket["finality_exact_rows"]) / float(bucket["rows"])
            if int(bucket["rows"])
            else 0.0
        )
        bucket["halted_trace_exact_accuracy"] = (
            float(bucket["halted_exact_rows"]) / float(bucket["rows"])
            if int(bucket["rows"])
            else 0.0
        )
    return {
        "rows": len(records),
        "exact_rows": exact_rows,
        "trace_exact_accuracy": float(exact_rows) / float(len(records)) if records else 0.0,
        "correct_steps": correct_steps,
        "total_steps": total_steps,
        "step_accuracy": (
            float(correct_steps) / float(total_steps) if total_steps else 0.0
        ),
        "finality_exact_rows": finality_exact_rows,
        "finality_trace_exact_accuracy": (
            float(finality_exact_rows) / float(len(records)) if records else 0.0
        ),
        "halted_exact_rows": halted_exact_rows,
        "halted_trace_exact_accuracy": (
            float(halted_exact_rows) / float(len(records)) if records else 0.0
        ),
        "finality_correct_steps": finality_correct_steps,
        "finality_total_steps": finality_total_steps,
        "finality_step_accuracy": (
            float(finality_correct_steps) / float(finality_total_steps)
            if finality_total_steps
            else 0.0
        ),
        "by_family": by_family,
    }


def evaluate_rows(
    *,
    config: str,
    checkpoint: str,
    data_jsonl: str,
    out_json: str | None = None,
    tokenizer_model_id: str = "Qwen/Qwen3.5-2B-Base",
    max_length: int | None = None,
    core_steps: int = 8,
    max_cases: int = 0,
    disable_transition_state: bool = False,
    predicted_code_permutation: str = "",
    drop_predicted_codes_to: int | None = None,
    prediction_source: str = "joint",
) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    from wgram_lm.config import load_config
    from wgram_lm.wgram_model import QTRMMultimodalModel
    from wgram_lm.qwen_donor import QwenDonorAdapter
    from wgram_lm.training.train import load_initial_checkpoint

    cfg = load_config(config)
    device = "cuda" if torch.cuda.is_available() and cfg.train.device in {"auto", "cuda"} else "cpu"
    rows = load_rows(data_jsonl)
    if int(max_cases) > 0:
        rows = rows[: int(max_cases)]
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_model_id,
        trust_remote_code=cfg.donor.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = QTRMMultimodalModel(cfg.model).to(device)
    missing, unexpected = load_initial_checkpoint(model, checkpoint, map_location=device)
    if missing:
        print(f"[init] missing keys: {len(missing)}")
    if unexpected:
        print(f"[init] unexpected keys: {len(unexpected)}")
    donor = QwenDonorAdapter(cfg.donor)
    model.eval()
    max_len = int(max_length or cfg.train.seq_len)
    records: list[dict[str, Any]] = []
    old_outer_steps = int(model.cfg.outer_steps)
    code_permutation = parse_code_permutation(predicted_code_permutation)
    prediction_source = str(prediction_source or "joint").lower()
    if prediction_source not in {"joint", "primitive", "routed", "core_feedback"}:
        raise ValueError(
            "prediction_source must be 'joint', 'primitive', 'routed', or 'core_feedback'"
        )
    model.cfg.outer_steps = int(core_steps)
    try:
        with torch.no_grad():
            for row in rows:
                input_ids, attention_mask = _prepare_prompt(
                    tokenizer,
                    str(row["prompt"]),
                    max_length=max_len,
                    device=device,
                )
                donor_out = donor.encode_inputs(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_logits=False,
                )
                with torch.amp.autocast(
                    "cuda",
                    enabled=(cfg.train.use_amp and device == "cuda"),
                    dtype=torch.bfloat16,
                ):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        text_states=donor_out["text_states"].detach().to(device),
                        disable_transition_state=bool(disable_transition_state),
                    )
                logits = outputs["transition_state_code_logits"]
                predicted = predicted_codes_from_logits(logits)
                joint_logits = outputs.get("transition_state_joint_logits")
                predicted_joint = (
                    predicted_joint_states_from_logits(joint_logits)
                    if joint_logits is not None
                    else []
                )
                if not predicted and predicted_joint:
                    predicted = [int(value) // 2 for value in predicted_joint]
                predicted_operation_ids = []
                primitive_predicted = []
                if prediction_source in {"primitive", "routed", "core_feedback"}:
                    primitive_logits = outputs.get(
                        "core_transition_feedback_operation_logits"
                        if prediction_source == "core_feedback"
                        else "primitive_transition_operation_logits"
                    )
                    if primitive_logits is None:
                        raise ValueError(
                            f"prediction_source={prediction_source} requires "
                            + (
                                "core_transition_feedback_operation_logits"
                                if prediction_source == "core_feedback"
                                else "primitive_transition_operation_logits"
                            )
                        )
                    predicted_operation_ids = predicted_operation_ids_from_logits(
                        primitive_logits
                    )
                    primitive_predicted = transition_codes_from_primitive_operations(
                        predicted_operation_ids
                    )
                    if prediction_source in {"primitive", "core_feedback"}:
                        predicted = primitive_predicted
                predicted_source_ids = []
                if prediction_source == "routed":
                    source_logits = outputs.get("transition_source_router_logits")
                    if source_logits is None:
                        raise ValueError(
                            "prediction_source=routed requires "
                            "transition_source_router_logits"
                        )
                    predicted_source_ids = predicted_source_ids_from_logits(source_logits)
                    joint_predicted = [int(value) for value in predicted]
                    if len(joint_predicted) != len(primitive_predicted):
                        raise ValueError("joint and primitive predictions differ in length")
                    predicted = [
                        primitive_predicted[index]
                        if int(predicted_source_ids[index]) == 1
                        else joint_predicted[index]
                        for index in range(len(joint_predicted))
                    ]
                predicted = apply_predicted_code_ablation(
                    [int(value) for value in predicted],
                    code_permutation=code_permutation,
                    drop_codes_to=drop_predicted_codes_to,
                )
                targets = code_targets_from_row(row, num_steps=len(predicted))
                score = score_code_predictions(
                    predicted_codes=[int(value) for value in predicted],
                    target_codes=targets,
                )
                finality_logits = (
                    outputs["transition_state_finality_logits"]
                    .detach()
                    .float()[0]
                    .cpu()
                    .tolist()
                    if "transition_state_finality_logits" in outputs
                    else []
                )
                if not finality_logits and predicted_joint:
                    finality_logits = [
                        1.0 if int(value) % 2 == 1 else -1.0
                        for value in predicted_joint
                    ]
                finality_targets = finality_targets_from_row(
                    row,
                    num_steps=len(finality_logits),
                )
                finality_score = score_finality_predictions(
                    finality_logits=[float(value) for value in finality_logits],
                    target_values=finality_targets,
                )
                halted_score = score_halted_transition_predictions(
                    predicted_codes=[int(value) for value in predicted],
                    target_codes=targets,
                    finality_logits=[float(value) for value in finality_logits],
                    target_finality=finality_targets,
                )
                records.append(
                    {
                        "id": row.get("id", ""),
                        "task_family": _family(row),
                        "predicted_codes": [int(value) for value in predicted],
                        "target_codes": targets,
                        "predicted_joint_states": [
                            int(value) for value in predicted_joint
                        ],
                        "predicted_operation_ids": [
                            int(value) for value in predicted_operation_ids
                        ],
                        "predicted_source_ids": [
                            int(value) for value in predicted_source_ids
                        ],
                        "finality_logits": [float(value) for value in finality_logits],
                        "finality_targets": finality_targets,
                        **score,
                        **finality_score,
                        **halted_score,
                    }
                )
    finally:
        model.cfg.outer_steps = old_outer_steps

    summary = _summarize_records(records)
    result = {
        "config": config,
        "checkpoint": checkpoint,
        "data_jsonl": data_jsonl,
        "core_steps": int(core_steps),
        "disable_transition_state": bool(disable_transition_state),
        "prediction_source": prediction_source,
        "predicted_code_permutation": dict(code_permutation or {}),
        "drop_predicted_codes_to": drop_predicted_codes_to,
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "summary": summary,
        "records": records,
    }
    if out_json:
        out = Path(out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate QTRM transition_state_code latent-action predictions."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-jsonl", required=True)
    parser.add_argument("--out-json", default="")
    parser.add_argument("--tokenizer-model-id", default="Qwen/Qwen3.5-2B-Base")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--core-steps", type=int, default=8)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--disable-transition-state", action="store_true")
    parser.add_argument(
        "--prediction-source",
        choices=["joint", "primitive", "routed", "core_feedback"],
        default="joint",
    )
    parser.add_argument("--predicted-code-permutation", default="")
    parser.add_argument("--drop-predicted-codes-to", type=int, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = evaluate_rows(
        config=args.config,
        checkpoint=args.checkpoint,
        data_jsonl=args.data_jsonl,
        out_json=args.out_json or None,
        tokenizer_model_id=args.tokenizer_model_id,
        max_length=args.max_length,
        core_steps=args.core_steps,
        max_cases=args.max_cases,
        disable_transition_state=args.disable_transition_state,
        prediction_source=args.prediction_source,
        predicted_code_permutation=args.predicted_code_permutation,
        drop_predicted_codes_to=args.drop_predicted_codes_to,
    )
    summary = result["summary"]
    print(
        "latent action codebook eval: "
        f"rows={summary['rows']} exact={summary['exact_rows']}/{summary['rows']} "
        f"step_acc={summary['step_accuracy']:.4f} "
        f"finality_acc={summary['finality_step_accuracy']:.4f} "
        f"halted_exact={summary['halted_exact_rows']}/{summary['rows']}"
    )


if __name__ == "__main__":
    main()
