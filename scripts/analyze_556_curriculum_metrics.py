#!/usr/bin/env python3
"""
analyze_556_curriculum_metrics.py

Post-run analyzer for 5.56 Adaptive Rehearsal Curriculum experiments.

Takes one or more metrics.json (or directories containing them) produced by
train_556_full_curriculum_minimal.py and produces a human-readable comparison
focused on the historical 5.56 gold recipe signals:

- Scheduled binding decay fidelity
- Stochastic trajectory diversity (the Reverse I→G→A piece)
- Gold basin adherence (gold_dist reduction)
- Attractor protection effect
- Overall state stability

Usage:
    python scripts/analyze_556_curriculum_metrics.py \
        local_556_ablation_matrix_*/**/metrics.json \
        --output 556_ablation_summary.md

    python scripts/analyze_556_curriculum_metrics.py \
        run1/metrics.json run2/metrics.json --labels "full_556" "no_stoch"
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import statistics

def load_run(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        data = json.load(f)
    return {
        "path": str(path),
        "steps": len(data),
        "metrics": data,
        "label": path.parent.name if path.parent.name else path.stem,
    }

def summarize_run(run: Dict[str, Any]) -> Dict[str, Any]:
    m = run["metrics"]
    if not m:
        return {"label": run["label"], "error": "empty"}

    binds = [x.get("bind_weight", 0.0) for x in m]
    divs = [x.get("stochastic_diversity", 0.0) for x in m]
    golds = [x.get("gold_dist", 0.0) for x in m]
    drifts = [x.get("drift", 0.0) for x in m]
    prots = [x.get("attractor_protection_active", False) for x in m]

    return {
        "label": run["label"],
        "steps": len(m),
        "bind_start": round(binds[0], 4),
        "bind_end": round(binds[-1], 4),
        "decay_range": round(binds[0] - binds[-1], 4),
        "stoch_div_max": round(max(divs), 4),
        "stoch_div_mean": round(statistics.mean(divs), 4),
        "gold_dist_start": round(golds[0], 4),
        "gold_dist_end": round(golds[-1], 4),
        "gold_dist_reduction": round(golds[0] - golds[-1], 4) if golds[0] > golds[-1] else 0.0,
        "mean_drift": round(statistics.mean(drifts), 5),
        "protection_active": all(prots),
        "protection_fraction": round(sum(prots) / len(prots), 2),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="metrics.json files or directories")
    parser.add_argument("--output", type=str, default=None, help="Write markdown summary to this file")
    parser.add_argument("--labels", type=str, default=None, help="Comma-separated labels (same order as paths)")
    args = parser.parse_args()

    runs = []
    for p in args.paths:
        p = Path(p)
        if p.is_dir():
            for mj in p.rglob("metrics.json"):
                runs.append(load_run(mj))
        elif p.suffix == ".json":
            runs.append(load_run(p))

    if args.labels:
        labels = [x.strip() for x in args.labels.split(",")]
        for i, lab in enumerate(labels[:len(runs)]):
            runs[i]["label"] = lab
    else:
        # Auto-generate reasonable labels from paths if not provided
        for i, r in enumerate(runs):
            if "label" not in r or not r["label"]:
                p = Path(r["path"])
                r["label"] = p.parent.name or p.stem

    summaries = [summarize_run(r) for r in runs]

    # Simple text report
    lines = []
    lines.append("# 5.56 Curriculum Ablation Analysis")
    lines.append(f"Generated from {len(runs)} runs\n")

    # Table header
    header = "| run | steps | bind_start→end | decay | stoch_div_max | gold_dist_red | mean_drift | prot |"
    lines.append(header)
    lines.append("|" + "---|" * 8)

    for s in summaries:
        row = (f"| {s['label']} | {s['steps']} | "
               f"{s.get('bind_start',0)}→{s.get('bind_end',0)} | "
               f"{s.get('decay_range',0)} | "
               f"{s.get('stoch_div_max',0)} | "
               f"{s.get('gold_dist_reduction',0)} | "
               f"{s.get('mean_drift',0)} | "
               f"{'yes' if s.get('protection_active') else 'no'} |")
        lines.append(row)

    lines.append("\n## Key Observations (for Promotion Gate)")

    # Very simple heuristic notes
    for s in summaries:
        note = f"- **{s['label']}**: "
        if s.get("stoch_div_max", 0) > 1.0:
            note += "strong stochastic breadth signal; "
        if s.get("decay_range", 0) > 0.2:
            note += "clear scheduled decay; "
        if s.get("protection_active"):
            note += "attractor protection was on throughout; "
        lines.append(note)

    report = "\n".join(lines)
    print(report)

    if args.output:
        Path(args.output).write_text(report)
        print(f"\nReport written to {args.output}")

if __name__ == "__main__":
    main()
