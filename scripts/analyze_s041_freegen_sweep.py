#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Any


MODE_RE = re.compile(
    r"qtrm_core_steps_(?P<depth>\d+)"
    r"(?:_qtrm_scale_(?P<qtrm>\d+(?:p\d+)?)_donor_scale_(?P<donor>\d+(?:p\d+)?))?"
    r"_no_evidence"
)


def _scale_text(token: str | None) -> str:
    if token is None:
        return "-"
    return token.replace("p", ".")


def _mode_key(mode: str) -> tuple[int, float, float, str]:
    if mode == "donor_only_no_evidence":
        return (-2, -1.0, -1.0, mode)
    if mode == "qtrm_core_off_no_evidence":
        return (-1, -1.0, -1.0, mode)
    match = MODE_RE.fullmatch(mode)
    if not match:
        return (999, 999.0, 999.0, mode)
    depth = int(match.group("depth"))
    qtrm = float(_scale_text(match.group("qtrm"))) if match.group("qtrm") else 1.0
    donor = float(_scale_text(match.group("donor"))) if match.group("donor") else 0.0
    return (depth, qtrm, donor, mode)


def _mode_descriptor(mode: str) -> dict[str, Any]:
    if mode == "donor_only_no_evidence":
        return {"family": "donor_only", "depth": None, "qtrm_scale": 0.0, "donor_scale": 1.0}
    if mode == "qtrm_core_off_no_evidence":
        return {"family": "core_off", "depth": None, "qtrm_scale": None, "donor_scale": None}
    match = MODE_RE.fullmatch(mode)
    if not match:
        return {"family": "other", "depth": None, "qtrm_scale": None, "donor_scale": None}
    return {
        "family": "qtrm_core",
        "depth": int(match.group("depth")),
        "qtrm_scale": float(_scale_text(match.group("qtrm"))) if match.group("qtrm") else 1.0,
        "donor_scale": float(_scale_text(match.group("donor"))) if match.group("donor") else 0.0,
    }


def _collapse_flags(completion: str) -> list[str]:
    text = str(completion or "")
    compact = text.replace(" ", "")
    flags: list[str] = []
    if "Answer:Answer" in compact:
        flags.append("answer_loop")
    if "!!!!!!" in compact:
        flags.append("bang_loop")
    if "1600000" in compact:
        flags.append("numeric_attractor_1600000")
    if len(compact) >= 6 and len(set(compact)) <= 2:
        flags.append("low_diversity")
    if "," in compact and any(token.startswith("1000") for token in compact.split(",")):
        flags.append("intermediate_list")
    return sorted(set(flags))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_mode[str(row.get("mode", ""))].append(row)

    mode_summaries: list[dict[str, Any]] = []
    for mode, mode_rows in sorted(by_mode.items(), key=lambda item: _mode_key(item[0])):
        hits = sum(1 for row in mode_rows if bool(row.get("hit")))
        exact = sum(1 for row in mode_rows if bool(row.get("exact_match") or row.get("normalized_exact")))
        generated_tokens = [
            int(row.get("generated_tokens", 0))
            for row in mode_rows
            if row.get("generated_tokens") is not None
        ]
        completions = Counter(str(row.get("completion", "")) for row in mode_rows)
        collapse_counter: Counter[str] = Counter()
        for row in mode_rows:
            collapse_counter.update(_collapse_flags(str(row.get("completion", ""))))
        descriptor = _mode_descriptor(mode)
        mode_summaries.append(
            {
                "mode": mode,
                **descriptor,
                "cases": len(mode_rows),
                "hits": hits,
                "hit_rate": hits / max(1, len(mode_rows)),
                "exact": exact,
                "exact_rate": exact / max(1, len(mode_rows)),
                "avg_generated_tokens": (
                    sum(generated_tokens) / max(1, len(generated_tokens))
                ),
                "top_completion": completions.most_common(1)[0][0] if completions else "",
                "top_completion_count": completions.most_common(1)[0][1] if completions else 0,
                "collapse_flags": dict(sorted(collapse_counter.items())),
            }
        )

    best = sorted(
        mode_summaries,
        key=lambda row: (row["hits"], row["exact"], -row["avg_generated_tokens"]),
        reverse=True,
    )[:5]
    return {
        "records": len(rows),
        "modes": len(mode_summaries),
        "total_hits": sum(int(row["hits"]) for row in mode_summaries),
        "total_cases": sum(int(row["cases"]) for row in mode_summaries),
        "mode_summaries": mode_summaries,
        "best_modes": best,
    }


def _markdown(summary: dict[str, Any], title: str, source: Path) -> str:
    lines = [
        f"# {title}",
        "",
        f"Source: `{source}`",
        "",
        "## Aggregate",
        "",
        f"- Records: {summary['records']}",
        f"- Modes: {summary['modes']}",
        f"- Hits: {summary['total_hits']}/{summary['total_cases']}",
        "",
        "## Per-Mode Results",
        "",
        "| Mode | Depth | QTRM scale | Donor scale | Hits | Exact | Avg tokens | Top completion | Collapse flags |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in summary["mode_summaries"]:
        flags = ", ".join(f"{k}:{v}" for k, v in row["collapse_flags"].items()) or "-"
        depth = "-" if row["depth"] is None else str(row["depth"])
        qtrm = "-" if row["qtrm_scale"] is None else str(row["qtrm_scale"])
        donor = "-" if row["donor_scale"] is None else str(row["donor_scale"])
        completion = str(row["top_completion"]).replace("\n", "\\n")
        lines.append(
            "| {mode} | {depth} | {qtrm} | {donor} | {hits}/{cases} | {exact}/{cases} | {avg:.2f} | `{completion}` | {flags} |".format(
                mode=row["mode"],
                depth=depth,
                qtrm=qtrm,
                donor=donor,
                hits=row["hits"],
                exact=row["exact"],
                cases=row["cases"],
                avg=row["avg_generated_tokens"],
                completion=completion,
                flags=flags,
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Contract",
            "",
            "This report is a smoke diagnostic, not a promotion gate. A mode can only be promoted if it improves free generation over donor-only, keeps delta/core ablations causal, and does not replace the donor language path with a private renderer.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize S041 free-generation sweeps.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--title", default="S041 Donor-Preserving Free Generation Sweep")
    args = parser.parse_args()

    source = Path(args.input)
    rows = _read_jsonl(source)
    summary = summarize(rows)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_markdown(summary, args.title, source) + "\n", encoding="utf-8")

    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
