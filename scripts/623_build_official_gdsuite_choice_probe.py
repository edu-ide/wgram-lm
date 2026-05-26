#!/usr/bin/env python3
"""Convert official GDsuite logprob families into this repo's choice probe.

The official GDsuite runner uses vLLM over HuggingFace models.  Our BLT/Data-IO
checkpoints are custom PyTorch modules, so we reuse the official data and prompt
assembly rules but emit the JSONL schema consumed by
``567_eval_blt_generalization_dynamics_probe.py``:

  {prompt, intelligence_answer, parrot_answer, task, source, ...}

This covers the five official logprob families.  The sixth GDsuite family,
multi-hop persona QA, is generative + regex-based and is intentionally reported
as a remaining gap until a custom BLT generation evaluator mirrors it.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_GDSUITE = ROOT / "references" / "official" / "GDsuite"
DEFAULT_HF_DATASET = "jiaxin-wen/generalization-dynamics-evals"
SOURCE = "https://github.com/Jiaxin-Wen/GDsuite"
LOGPROB_FAMILIES = (
    "flipped_answer",
    "repetitive_answer",
    "successive_answer",
    "truthy_answer",
    "intuitive_answer",
)


def load_official_gdsuite(path: Path = OFFICIAL_GDSUITE) -> Any:
    run_eval = path / "run_eval.py"
    if not run_eval.exists():
        raise FileNotFoundError(
            f"official GDsuite not found at {run_eval}; clone it under {path}"
        )
    spec = importlib.util.spec_from_file_location("official_gdsuite_run_eval", run_eval)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {run_eval}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sanitize(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return cleaned.strip("_") or "unknown"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}") from exc
            if isinstance(value, dict):
                rows.append(value)
    return rows


def official_answer(value: Any) -> str:
    """Match GDsuite score_logprobs, which scores prompt + ' ' + answer."""
    text = str(value).strip()
    return " " + text


def resolve_config(
    official: Any,
    *,
    config: str,
    hf_dataset: str,
    local_data_dir: str,
) -> dict[str, Any]:
    namespace = argparse.Namespace(
        config=config or None,
        hf_dataset=hf_dataset,
        local_data_dir=local_data_dir or None,
        gpu_memory_utilization=None,
        tensor_parallel_size=None,
        max_model_len=None,
        max_num_seqs=None,
    )
    return official._resolve_config(str(OFFICIAL_GDSUITE), namespace)


def build_logprob_rows(
    official: Any,
    config: dict[str, Any],
    *,
    families: list[str],
    n_seeds_override: int,
    max_eval_per_task: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out: list[dict[str, Any]] = []
    family_counts: dict[str, int] = {}
    task_counts: dict[str, int] = {}

    for family in families:
        if family not in LOGPROB_FAMILIES:
            raise ValueError(
                f"unsupported choice family={family!r}; expected one of {LOGPROB_FAMILIES}"
            )
        cfg = config[family]
        strategy = cfg.get("demo_strategy", "random")
        joiner = cfg.get("joiner", "\n")
        sampler = official.DEMO_STRATEGIES[strategy]
        n_seeds_cfg = int(n_seeds_override if n_seeds_override > 0 else cfg.get("n_seeds", 1))

        for task in cfg["tasks"]:
            name = str(task["name"])
            k = int(task.get("k", 0))
            path = Path(cfg["data_dir"]) / f"{name}.jsonl"
            rows = read_jsonl(path)
            demos_all = [row for row in rows if row.get("split") == "demo"]
            tests = [row for row in rows if row.get("split") == "test"]
            if int(max_eval_per_task) > 0:
                tests = tests[: int(max_eval_per_task)]
            n_seeds = 1 if strategy == "none" or not demos_all else n_seeds_cfg
            resample_per_item = strategy == "random"
            task_key = f"{family}/{name}"

            for seed in range(n_seeds):
                rng = random.Random(seed)
                pool = official._select_demo_set(demos_all, seed)
                if not resample_per_item:
                    blocks = [
                        f"{demo['prompt']} {demo['answer']}"
                        for demo in sampler(pool, k, rng)
                    ]
                for test_index, test in enumerate(tests):
                    if resample_per_item:
                        blocks = [
                            f"{demo['prompt']} {demo['answer']}"
                            for demo in sampler(pool, k, rng)
                        ]
                    prompt = official.build_icl_prompt(blocks, str(test["prompt"]), joiner)
                    row_id = (
                        f"gdsuite_{sanitize(family)}_{sanitize(name)}"
                        f"_seed{seed:02d}_test{test_index:05d}"
                    )
                    out.append(
                        {
                            "id": row_id,
                            "source": SOURCE,
                            "source_dataset": DEFAULT_HF_DATASET,
                            "family": family,
                            "task": task_key,
                            "official_task": name,
                            "seed": int(seed),
                            "k": int(k),
                            "demo_strategy": str(strategy),
                            "prompt": prompt,
                            "intelligence_answer": official_answer(test["correct_answer"]),
                            "parrot_answer": official_answer(test["incorrect_answer"]),
                            "plain_language_axis": (
                                "Prefers the official GDsuite correct answer over the "
                                "tempting memorized/parrot answer."
                            ),
                        }
                    )
            family_counts[family] = family_counts.get(family, 0) + int(n_seeds * len(tests))
            task_counts[task_key] = int(n_seeds * len(tests))

    report = {
        "probe_type": "official_gdsuite_choice_logprob_families",
        "source": SOURCE,
        "source_dataset": DEFAULT_HF_DATASET,
        "families": families,
        "family_counts": family_counts,
        "task_counts": task_counts,
        "rows": int(len(out)),
        "skipped_official_families": {
            "multihop_persona_qa": (
                "Official family 6 is generative + regex-based; this script emits "
                "the five official logprob families for the BLT choice evaluator."
            )
        },
        "plain_language_read": (
            "This is no longer the hand-written 6-row GD-lite smoke. It uses the "
            "official GDsuite data and prompt sampling for the five logprob "
            "families, then asks our custom checkpoint whether the correct answer "
            "has higher probability than the parrot answer."
        ),
    }
    return out, report


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/eval/official_gdsuite_choice_probe.jsonl")
    parser.add_argument("--report-out", default="")
    parser.add_argument("--config", default="")
    parser.add_argument("--hf-dataset", default=DEFAULT_HF_DATASET)
    parser.add_argument("--local-data-dir", default="")
    parser.add_argument("--families", nargs="*", default=list(LOGPROB_FAMILIES))
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=0,
        help="Override official per-family seed count; <=0 keeps config.yaml.",
    )
    parser.add_argument(
        "--max-eval-per-task",
        type=int,
        default=0,
        help="Smoke cap per task; <=0 emits full official choice set.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    official = load_official_gdsuite()
    config = resolve_config(
        official,
        config=str(args.config),
        hf_dataset=str(args.hf_dataset),
        local_data_dir=str(args.local_data_dir),
    )
    rows, report = build_logprob_rows(
        official,
        config,
        families=[str(family) for family in args.families],
        n_seeds_override=int(args.n_seeds),
        max_eval_per_task=int(args.max_eval_per_task),
    )
    out_path = Path(args.out)
    write_jsonl(out_path, rows)
    report["out"] = str(out_path)
    report_path = Path(args.report_out) if str(args.report_out) else out_path.with_suffix(out_path.suffix + ".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
