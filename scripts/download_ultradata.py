#!/usr/bin/env python3
"""
Memory-optimized downloader for openbmb/UltraData-SFT-2605.

Key features for low-memory environments:
- Uses streaming=True (never loads full dataset into RAM)
- Writes JSONL line-by-line with frequent flushing
- Downloads one config at a time (or selected configs)
- Good resume support (skips if output file already has enough lines)
- Optional psutil memory logging

Usage:
    # Download all 4 configs (safest for memory)
    HF_TOKEN=xxx python scripts/download_ultradata.py

    # Download only Math
    HF_TOKEN=xxx python scripts/download_ultradata.py --config Math
"""

import os
import sys
import json
import argparse
import time

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x   # fallback if tqdm not installed


def get_memory_mb():
    """Return current RSS memory usage in MB (best effort)."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def download_config(config: str, out_dir: str, min_lines_to_skip: int = 100):
    from datasets import load_dataset

    print(f"\n=== Downloading config: {config} (streaming, memory-safe) ===")
    print(f"Current memory usage: {get_memory_mb():.1f} MB")

    try:
        ds = load_dataset("openbmb/UltraData-SFT-2605", config, streaming=True)
    except Exception as e:
        print(f"  ERROR loading config {config}: {e}")
        return False

    available_splits = list(ds.keys())
    print(f"  Available splits: {available_splits}")

    success = True

    for split in available_splits:
        safe_config = config.lower().replace("-", "_")
        out_path = os.path.join(out_dir, f"ultradata_sft_2605_{safe_config}_{split}.jsonl")

        # Resume check
        existing_lines = 0
        if os.path.exists(out_path):
            try:
                with open(out_path, "r", encoding="utf-8") as f:
                    existing_lines = sum(1 for _ in f)
                if existing_lines >= min_lines_to_skip:
                    print(f"  Skipping {config}/{split} (already has {existing_lines:,} lines)")
                    continue
                else:
                    print(f"  {config}/{split} has only {existing_lines} lines → re-downloading")
                    os.remove(out_path)
            except Exception:
                print(f"  Could not check existing file for {split}, re-downloading...")

        print(f"  Streaming {config}/{split} → {out_path}")
        print(f"  Memory before streaming: {get_memory_mb():.1f} MB")

        try:
            split_ds = ds[split]
            count = existing_lines
            start_time = time.time()

            with open(out_path, "a", encoding="utf-8") as f:
                pbar = tqdm(split_ds, desc=f"{config}/{split}", unit="ex", leave=False)
                for example in pbar:
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
                    count += 1

                    # Flush every 2000 examples + print memory occasionally
                    if count % 2000 == 0:
                        f.flush()
                        mem = get_memory_mb()
                        elapsed = time.time() - start_time
                        rate = count / elapsed if elapsed > 0 else 0
                        pbar.set_postfix({
                            "lines": f"{count:,}",
                            "mem": f"{mem:.0f}MB",
                            "rate": f"{rate:.0f}/s"
                        })

                pbar.close()

            print(f"  ✓ Saved {config}/{split} ({count:,} lines) | Final mem: {get_memory_mb():.1f} MB")

        except Exception as e:
            print(f"  ERROR on {config}/{split}: {e}")
            success = False

    return success


def main():
    parser = argparse.ArgumentParser(description="Memory-efficient streaming downloader for UltraData-SFT-2605")
    parser.add_argument(
        "--config",
        type=str,
        choices=["all", "Math", "IF", "Code", "Knowledge"],
        default="all",
        help="Which config(s) to download"
    )
    parser.add_argument("--min-lines", type=int, default=100, help="Skip if file already has this many lines")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.abspath(os.path.join(script_dir, "..", "data", "raw"))
    os.makedirs(out_dir, exist_ok=True)

    print("=== UltraData-SFT-2605 Memory-Efficient Downloader ===")
    print(f"Output directory: {out_dir}")
    print(f"Mode: streaming + line-by-line JSONL + frequent flush")
    print(f"Memory before start: {get_memory_mb():.1f} MB")

    configs = ["Math", "IF", "Code", "Knowledge"] if args.config == "all" else [args.config]

    overall_success = True
    for cfg in configs:
        ok = download_config(cfg, out_dir, args.min_lines)
        if not ok:
            overall_success = False

    print(f"\n=== Finished. Final memory: {get_memory_mb():.1f} MB ===")
    return 0 if overall_success else 1


if __name__ == "__main__":
    sys.exit(main())