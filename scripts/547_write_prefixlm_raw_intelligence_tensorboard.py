#!/usr/bin/env python3
"""Write PrefixLM raw-intelligence eval JSON scalars to TensorBoard."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


def sanitize_tag_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return cleaned.strip("_") or "unknown"


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(int(value))
    if not isinstance(value, (int, float)):
        return None
    result = float(value)
    if not math.isfinite(result):
        return None
    return result


def add_scalars(writer: Any, prefix: str, values: dict[str, Any], step: int) -> int:
    written = 0
    for key in (
        "cases",
        "target_tokens",
        "tokens_seen",
        "target_tokens_seen",
        "loss",
        "perplexity",
        "token_accuracy",
        "hits",
        "accuracy",
        "nonempty_rate",
        "degenerate_repetition_rate",
        "avg_cleaned_chars",
        "special_tokens_per_case",
        "generation_hits",
        "generation_accuracy",
    ):
        value = finite_number(values.get(key))
        if value is None:
            continue
        writer.add_scalar(f"{prefix}/{key}", value, int(step))
        written += 1
    return written


def write_report_to_tensorboard(
    *,
    report: dict[str, Any],
    tensorboard_dir: str | Path,
    prefix: str = "eval/raw_intelligence",
    step: int | None = None,
) -> int:
    from torch.utils.tensorboard import SummaryWriter

    global_step = int(report.get("step") or 0) if step is None else int(step)
    written = 0
    with SummaryWriter(log_dir=str(tensorboard_dir)) as writer:
        written += add_scalars(writer, prefix, report, global_step)
        for section_name in ("by_primitive", "by_family", "by_language"):
            section = report.get(section_name) or {}
            if not isinstance(section, dict):
                continue
            group = section_name.removeprefix("by_")
            for name, values in sorted(section.items()):
                if not isinstance(values, dict):
                    continue
                tag_name = sanitize_tag_part(str(name))
                written += add_scalars(writer, f"{prefix}/{group}/{tag_name}", values, global_step)
        writer.flush()
    return written


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-json", required=True)
    parser.add_argument("--tensorboard-dir", required=True)
    parser.add_argument("--prefix", default="eval/raw_intelligence")
    parser.add_argument("--step", type=int, default=-1)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = json.loads(Path(args.raw_json).read_text(encoding="utf-8"))
    step = None if int(args.step) < 0 else int(args.step)
    written = write_report_to_tensorboard(
        report=report,
        tensorboard_dir=args.tensorboard_dir,
        prefix=str(args.prefix),
        step=step,
    )
    print(
        json.dumps(
            {
                "event": "raw_intelligence_tensorboard_written",
                "raw_json": str(args.raw_json),
                "tensorboard_dir": str(args.tensorboard_dir),
                "scalars_written": int(written),
                "step": int(report.get("step") or 0) if step is None else int(step),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
