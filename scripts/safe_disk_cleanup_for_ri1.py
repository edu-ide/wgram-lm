#!/usr/bin/env python3
"""
Safe Disk Cleanup Script for RI-1 Experiments

Protects the current promising M1 Curriculum v2 run and only targets known old/expendable directories.

Usage:
    # Dry run (recommended first)
    python scripts/safe_disk_cleanup_for_ri1.py --dry-run

    # Real execution
    python scripts/safe_disk_cleanup_for_ri1.py --execute

This script will NEVER touch:
- The latest RI-1 M1 v2 curriculum run (20260529_1932)
- Any directory containing "curriculum_v2" in the name
"""

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple

ROOT = Path("checkpoints")
PROTECTED_SUBSTRINGS = [
    "curriculum_v2",
    "20260529_1932",   # explicit protection of the promising run
]

# Categories of directories that are safe to delete in bulk (old experiments)
CLEANUP_PATTERNS = [
    "diag_trainer_prep_v*",
    "diag_attractor_climb_v*",
    "diag_wiring_feedback",
    "diag_item*",
    "diag_real_hybrid*",
    "diag_noisy*",
    "diag_densing*",
    "diag_internalization*",
    "brain_push_*",
    "brain_stable_push*",
    "brain_long_*",
    "hybrid_ri4_ri1_m1_50step_*",
    "hybrid_ri4_ri1_m1_long_*",
    "hybrid_ri4_ri1_m1_progress_*",
    "hybrid_ri4_pilot_sigreg",
    "hybrid_ri4_scaleout",
    # We are more conservative with the giant hybrid_ri4_cont — handled separately
]

def is_protected(path: Path) -> bool:
    name = path.name
    for substr in PROTECTED_SUBSTRINGS:
        if substr in name:
            return True
    return False

def find_candidates() -> List[Tuple[Path, int]]:
    candidates = []
    if not ROOT.exists():
        return candidates

    for pattern in CLEANUP_PATTERNS:
        for p in ROOT.glob(pattern):
            if p.is_dir() and not is_protected(p):
                try:
                    size = sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
                    candidates.append((p, size))
                except Exception:
                    pass

    # Special handling suggestion for the very large hybrid_ri4_cont
    giant = ROOT / "hybrid_ri4_cont"
    if giant.exists() and not is_protected(giant):
        try:
            size = sum(f.stat().st_size for f in giant.rglob('*') if f.is_file())
            candidates.append((giant, size))
        except Exception:
            pass

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates

def human_size(num_bytes: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f}TB"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Actually delete files (DANGEROUS)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted (default)")
    parser.add_argument("--keep-latest-n", type=int, default=50,
                        help="When cleaning hybrid_ri4_cont, keep the N most recent checkpoints (default 50)")
    args = parser.parse_args()

    if not args.execute:
        print("=== DRY RUN MODE (no files will be deleted) ===\n")

    candidates = find_candidates()

    if not candidates:
        print("No cleanup candidates found.")
        return

    total_to_free = 0
    to_delete = []

    print("Candidates for cleanup (sorted by size, descending):\n")
    for path, size in candidates:
        protected = is_protected(path)
        status = "PROTECTED (will skip)" if protected else "DELETE CANDIDATE"
        print(f"  {human_size(size):>10}  {path.name:<50}  {status}")
        if not protected:
            total_to_free += size
            to_delete.append(path)

    print(f"\nPotential space to recover: {human_size(total_to_free)}")

    if not args.execute:
        print("\nThis was a dry run. Re-run with --execute to actually delete.")
        print("Recommendation: First clean the giant hybrid_ri4_cont manually or with extra caution.")
        return

    # Real execution
    print("\n=== EXECUTING DELETIONS ===\n")
    freed = 0
    for path in to_delete:
        if is_protected(path):
            print(f"Skipping protected: {path}")
            continue
        try:
            size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
            shutil.rmtree(path)
            freed += size
            print(f"Deleted: {path}  ({human_size(size)})")
        except Exception as e:
            print(f"Failed to delete {path}: {e}")

    print(f"\nTotal space freed: {human_size(freed)}")

    # Optional: trim the giant hybrid_ri4_cont
    giant = ROOT / "hybrid_ri4_cont"
    if giant.exists() and args.execute:
        print("\n--- Special handling for hybrid_ri4_cont ---")
        ckpts = sorted(giant.glob("hybrid_ri4_cont_step*.pt"), key=lambda p: int(p.stem.split("step")[-1]))
        if len(ckpts) > args.keep_latest_n:
            to_trim = ckpts[:-args.keep_latest_n]
            print(f"Keeping latest {args.keep_latest_n} checkpoints, trimming {len(to_trim)} older ones...")
            for p in to_trim:
                try:
                    p.unlink()
                    print(f"  Removed old ckpt: {p.name}")
                except Exception as e:
                    print(f"  Failed: {p} - {e}")
            # Also clean associated logs if they exist
            for log in giant.glob("*.log"):
                try:
                    log.unlink()
                except:
                    pass

    print("\nCleanup complete. Run 'df -h /' to verify.")

if __name__ == "__main__":
    main()
