#!/usr/bin/env python3
"""Evaluate QTRM-native checkpoint interpolation without saving full checkpoints."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch


def load_module(script_name: str, module_name: str):
    path = Path(__file__).with_name(script_name)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_candidate(value: str) -> tuple[str, str]:
    if "=" in str(value):
        name, path = str(value).split("=", 1)
        return name.strip(), path.strip()
    path = str(value).strip()
    return Path(path).parent.name or Path(path).stem, path


def checkpoint_state(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    state = checkpoint.get("model_state", checkpoint.get("model"))
    if not isinstance(state, dict):
        raise ValueError("checkpoint missing model_state/model")
    return state


@torch.no_grad()
def evaluate_model(eval_module, eval_args, tokenizer, model, device, rows, args) -> dict[str, Any]:
    scored_rows: list[dict[str, Any]] = []
    for row in rows:
        completion, answer_text, pred, prompt_echo = eval_module.generate_answer(
            eval_module,
            eval_args,
            tokenizer,
            model,
            device,
            prompt=str(row["qtrm_prompt"]),
            think_steps=int(args.think_steps),
            max_new_chars=int(args.max_new_chars),
        )
        gold = eval_module.normalize_mcq_answer(str(row["answer"]))
        scored = dict(row)
        scored.update(
            {
                "raw_completion": completion,
                "answer_text": answer_text,
                "pred_answer": pred,
                "gold_answer": gold,
                "exact": bool(pred == gold),
                "prompt_echo": bool(prompt_echo),
            }
        )
        scored_rows.append(scored)
    return {
        "metrics": eval_module.score_rows(scored_rows),
        "predictions": scored_rows if bool(args.write_predictions) else [],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    eval_module = load_module("384_eval_qtrm_native_public_mcq.py", "qtrm_native_public_mcq_eval")
    avg_module = load_module("349_average_qtrm_native_checkpoints.py", "qtrm_native_checkpoint_average")

    load_args = argparse.Namespace(
        checkpoint=str(args.base_checkpoint),
        device=str(args.device),
        out_dir=str(args.out_dir),
        think_steps=int(args.think_steps),
        max_new_chars=int(args.max_new_chars),
    )
    _loaded_eval_module, eval_args, tokenizer, model, device = eval_module.load_model_bundle(load_args)
    base_checkpoint = torch.load(str(args.base_checkpoint), map_location="cpu")
    base_state = checkpoint_state(base_checkpoint)
    rows = eval_module.load_suite(str(args.suite_jsonl), max_cases=int(args.max_cases))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []

    for candidate_arg in args.candidate:
        candidate_name, candidate_path = parse_candidate(candidate_arg)
        candidate_checkpoint = torch.load(str(candidate_path), map_location="cpu")
        candidate_state = checkpoint_state(candidate_checkpoint)
        for alpha in [float(value) for value in args.alpha]:
            averaged_state = avg_module.average_model_states(
                base_state,
                candidate_state,
                alpha=float(alpha),
            )
            model.load_state_dict(averaged_state)
            model.eval()
            result = evaluate_model(eval_module, eval_args, tokenizer, model, device, rows, args)
            metrics = result["metrics"]
            row = {
                "candidate": candidate_name,
                "candidate_checkpoint": str(candidate_path),
                "alpha": float(alpha),
                "think_steps": int(args.think_steps),
                "metrics": metrics,
            }
            summaries.append(row)
            run_slug = f"{candidate_name}_a{str(alpha).replace('.', 'p')}"
            run_dir = out_dir / run_slug
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "report.json").write_text(
                json.dumps(row, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            if bool(args.write_predictions):
                (run_dir / "predictions.jsonl").write_text(
                    "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in result["predictions"]),
                    encoding="utf-8",
                )

    summaries.sort(
        key=lambda item: (
            int(item["metrics"].get("hits", 0)),
            -max(int(value) for value in item["metrics"].get("pred_answer_histogram", {"": 1}).values()),
        ),
        reverse=True,
    )
    report = {
        "status": "complete",
        "decision": "m7c_checkpoint_soup_in_memory_triage",
        "accepted": False,
        "base_checkpoint": str(args.base_checkpoint),
        "suite_jsonl": str(args.suite_jsonl),
        "max_cases": int(args.max_cases),
        "note": "diagnostic only; no full checkpoint was saved",
        "best": summaries[0] if summaries else None,
        "rows": summaries,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--alpha", action="append", required=True)
    parser.add_argument("--suite-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--think-steps", type=int, default=4)
    parser.add_argument("--max-new-chars", type=int, default=1)
    parser.add_argument("--max-cases", type=int, default=64)
    parser.add_argument("--write-predictions", action="store_true")
    return parser


def main() -> None:
    report = run(build_arg_parser().parse_args())
    print(json.dumps({"best": report["best"], "out_dir": report["out_dir"] if "out_dir" in report else ""}, ensure_ascii=False))


if __name__ == "__main__":
    main()
