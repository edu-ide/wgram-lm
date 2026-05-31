#!/usr/bin/env python3
"""
Analyzer for RI-1 depth scaling results on memory requirement buckets.

It infers bucket and depth from the filename or mode when not present in the JSON.

Usage:
    python scripts/analyze_ri1_bucket_depth_results.py /tmp/ri1_*.jsonl
"""

import argparse
import json
import re
from collections import defaultdict
from typing import Dict, List, Tuple

def infer_bucket_and_depth(path: str, rec: dict) -> Tuple[str, int, str]:
    """Try to figure out (bucket, depth, mem_state) from filename or record."""
    filename = path.lower()

    # Try to detect bucket from filename
    bucket = "unknown"
    if "low_bucket" in filename or "_low_" in filename:
        bucket = "low"
    elif "high_bucket" in filename or "_high_" in filename:
        bucket = "high"
    elif "medium_bucket" in filename or "_medium_" in filename:
        bucket = "medium"

    # Try to detect depth from mode or filename
    depth = rec.get("core_steps_requested") or rec.get("depth")
    if depth is None:
        mode = rec.get("mode", "")
        m = re.search(r"depth_(\d+)", mode)
        if m:
            depth = int(m.group(1))
        else:
            m = re.search(r"_d(\d+)_", filename)
            if m:
                depth = int(m.group(1))

    # Detect memory state
    mode = rec.get("mode", "").lower()
    filename = filename.lower()
    if "on" in mode or "_on_" in filename:
        mem = "on"
    elif "off" in mode or "_off_" in filename:
        mem = "off"
    else:
        mem = "unknown"

    return bucket, int(depth) if depth else 0, mem

def load_results(paths: List[str]) -> Dict:
    results = defaultdict(list)
    for p in paths:
        with open(p) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except:
                    continue

                bucket, depth, mem = infer_bucket_and_depth(p, rec)
                hits = rec.get("hits", 0)
                total = rec.get("total", rec.get("cases", 0))
                if total > 0:
                    results[(bucket, depth, mem)].append((hits, total))
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()

    data = load_results(args.files)

    print("RI-1 Depth Scaling on Memory Buckets — Analysis")
    print("=" * 65)

    for bucket in ["low", "high", "medium"]:
        print(f"\n=== Bucket: {bucket.upper()} ===")
        depths = sorted({k[1] for k in data if k[0] == bucket})
        if not depths:
            print("  (no data)")
            continue

        for depth in depths:
            on_key = (bucket, depth, "on")
            off_key = (bucket, depth, "off")

            def avg_acc(key):
                runs = data.get(key, [])
                if not runs:
                    return None
                return sum(h / t for h, t in runs if t > 0) / len(runs)

            on_acc = avg_acc(on_key)
            off_acc = avg_acc(off_key)

            on_str = f"{on_acc*100:5.1f}%" if on_acc is not None else "  -  "
            off_str = f"{off_acc*100:5.1f}%" if off_acc is not None else "  -  "

            margin = ""
            if on_acc is not None and off_acc is not None:
                margin = f"  | margin: +{(on_acc - off_acc)*100:4.1f}pp"

            print(f"  Depth {depth:2d} | ON: {on_str}   OFF: {off_str}{margin}")

    print("\n" + "=" * 65)
    print("Tip: Look for whether the gap (ON - OFF) increases with depth,")
    print("especially in the HIGH bucket. That is the signal we care about.")

    # === RI-1 specific: Scaling slope + Monotonicity per bucket ===
    print("\n=== RI-1 Depth Scaling Summary (Monotonicity & ON/OFF Delta) ===")
    for bucket in ["low", "high", "medium"]:
        print(f"\n--- {bucket.upper()} bucket ---")
        depths_sorted = sorted({k[1] for k in data if k[0] == bucket})
        if len(depths_sorted) < 2:
            print("  (insufficient depths for scaling analysis)")
            continue

        for mem in ["on", "off"]:
            accs = []
            for d in depths_sorted:
                key = (bucket, d, mem)
                a = avg_acc(key)
                if a is not None:
                    accs.append((d, a))
            if len(accs) >= 2:
                print(f"  Memory {mem.upper()}: ", end="")
                for i, (d, a) in enumerate(accs):
                    print(f"d={d} {a*100:5.1f}%", end="")
                    if i < len(accs)-1:
                        delta = (accs[i+1][1] - a) * 100
                        arrow = "↑" if delta > 0.5 else ("↓" if delta < -0.5 else "→")
                        print(f" {arrow}{delta:+4.1f}pp  ", end="")
                    else:
                        print()
                # Simple monotonic score
                mono = all(accs[i][1] <= accs[i+1][1] + 0.005 for i in range(len(accs)-1))
                print(f"    monotonic? {'YES' if mono else 'NO (peaks or drops)'}")

if __name__ == "__main__":
    main()