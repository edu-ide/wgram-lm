#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Iterable


def _load_raw_eval_module():
    path = Path(__file__).resolve().with_name("192_eval_raw_intelligence.py")
    spec = importlib.util.spec_from_file_location("raw_intelligence_eval_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_core_steps(value: str | Iterable[int]) -> list[int]:
    if isinstance(value, str):
        steps = [int(part.strip()) for part in value.split(",") if part.strip()]
    else:
        steps = [int(step) for step in value]
    if not steps:
        raise ValueError("core steps must not be empty")
    if any(step <= 0 for step in steps):
        raise ValueError("core steps must be positive")
    return steps


def _final_answer(row: dict[str, Any]) -> str:
    aliases = row.get("answer_aliases") or []
    if aliases:
        return str(aliases[0])
    return str(row.get("chosen") or row.get("answer") or "")


def transition_target_for_step(row: dict[str, Any], core_steps: int) -> str:
    depth_targets = row.get("depth_targets")
    if isinstance(depth_targets, dict):
        target = depth_targets.get(str(int(core_steps)))
        if target:
            return str(target)
    final = _final_answer(row)
    if not final:
        raise ValueError(f"row has no final answer: {row.get('id')}")
    return final


def _unique(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def transition_choices_for_step(row: dict[str, Any], core_steps: int) -> list[str]:
    target = transition_target_for_step(row, core_steps)
    depth_targets = row.get("depth_targets") or {}
    candidates = [target]
    if isinstance(depth_targets, dict):
        candidates.extend(str(value) for value in depth_targets.values())
    candidates.extend(str(value) for value in row.get("answer_aliases", []))
    candidates.extend(str(value) for value in row.get("choices", []))
    choices = _unique(candidates)
    if len(choices) < 2:
        raise ValueError(f"not enough transition choices for row: {row.get('id')}")
    return choices


def transition_case_for_step(row: dict[str, Any], core_steps: int) -> dict[str, Any]:
    target = transition_target_for_step(row, core_steps)
    choices = transition_choices_for_step(row, core_steps)
    out = dict(row)
    out["answer_aliases"] = [target]
    out["choices"] = choices
    out["symbolic_transition_target"] = target
    out["symbolic_transition_core_steps"] = int(core_steps)
    return out


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    hit_count = sum(1 for record in records if bool(record.get("hit")))
    by_core_steps: dict[str, dict[str, Any]] = {}
    by_category: dict[str, dict[str, Any]] = {}
    for record in records:
        for key, name in (
            ("by_core_steps", str(record.get("core_steps"))),
            ("by_category", str(record.get("category", "uncategorized"))),
        ):
            bucket_map = by_core_steps if key == "by_core_steps" else by_category
            bucket = bucket_map.setdefault(name, {"total": 0, "hit_count": 0})
            bucket["total"] += 1
            bucket["hit_count"] += int(bool(record.get("hit")))
    for bucket_map in (by_core_steps, by_category):
        for bucket in bucket_map.values():
            bucket["accuracy"] = (
                float(bucket["hit_count"]) / float(bucket["total"])
                if bucket["total"]
                else 0.0
            )
    return {
        "total_records": total,
        "hit_count": hit_count,
        "accuracy": float(hit_count) / float(total) if total else 0.0,
        "by_core_steps": by_core_steps,
        "by_category": by_category,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate symbolic intermediate-state targets at each QTRM core depth."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--core-steps", default="1,2,4,8")
    parser.add_argument(
        "--scoring",
        choices=["forced_choice", "causal_forced_choice"],
        default="causal_forced_choice",
    )
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser


def _load_cases(path: str | Path, *, max_cases: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row.setdefault("id", f"case-{line_no}")
            if row.get("evidence"):
                raise ValueError(f"{path}:{line_no}: symbolic transition gate forbids evidence")
            if not row.get("depth_targets"):
                raise ValueError(f"{path}:{line_no}: missing depth_targets")
            rows.append(row)
            if max_cases is not None and len(rows) >= int(max_cases):
                break
    return rows


def run_eval(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch
    from transformers import AutoTokenizer

    from wgram_lm.config import load_config
    from wgram_lm.wgram_model import QTRMMultimodalModel
    from wgram_lm.qwen_donor import QwenDonorAdapter

    raw_eval = _load_raw_eval_module()
    cfg = load_config(args.config)
    device = raw_eval._select_device(cfg.train.device, args.device)
    max_length = int(args.max_length or cfg.train.seq_len)
    core_steps_list = parse_core_steps(args.core_steps)
    cases = _load_cases(args.cases, max_cases=args.max_cases)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.donor.model_id,
        trust_remote_code=cfg.donor.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = QTRMMultimodalModel(cfg.model)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state.get("model", state), strict=False)
    if missing:
        print(f"[checkpoint] missing keys: {len(missing)}", file=sys.stderr)
    if unexpected:
        print(f"[checkpoint] unexpected keys: {len(unexpected)}", file=sys.stderr)
    model = model.to(device).eval()
    donor = QwenDonorAdapter(cfg.donor)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with out.open("w", encoding="utf-8") as f, torch.no_grad(), redirect_stdout(sys.stderr):
        for core_steps in core_steps_list:
            mode = f"qtrm_core_steps_{core_steps}_no_evidence"
            runtime = raw_eval.mode_runtime(mode)
            for case in cases:
                transition_case = transition_case_for_step(case, core_steps)
                if args.scoring == "forced_choice":
                    completion, choice_scores = raw_eval._forced_choice_case(
                        model,
                        donor,
                        tokenizer,
                        transition_case,
                        runtime=runtime,
                        max_length=max_length,
                        device=device,
                    )
                else:
                    completion, choice_scores = raw_eval._causal_forced_choice_case(
                        model,
                        donor,
                        tokenizer,
                        transition_case,
                        runtime=runtime,
                        max_length=max_length,
                        device=device,
                    )
                record = raw_eval.score_case_record(
                    transition_case,
                    mode=mode,
                    completion=completion,
                    runtime=runtime,
                    generated_tokens=0,
                )
                record["scoring"] = args.scoring
                record["core_steps"] = int(core_steps)
                record["symbolic_transition_target"] = transition_case["symbolic_transition_target"]
                record["final_answer"] = _final_answer(case)
                record["choice_scores"] = choice_scores
                record["choice_tied"] = (
                    sum(1 for row in choice_scores if bool(row.get("tied_for_best"))) > 1
                )
                records.append(record)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()

    summary = summarize_records(records)
    summary["checkpoint"] = args.checkpoint
    summary["config"] = args.config
    summary["cases"] = args.cases
    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return records, summary


def main() -> None:
    args = build_arg_parser().parse_args()
    records, summary = run_eval(args)
    print(f"wrote {len(records)} records to {args.out}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
