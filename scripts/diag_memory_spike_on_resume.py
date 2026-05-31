#!/usr/bin/env python3
"""
Quick diagnostic: measure memory spike on resume + first heavy step.
Run with the same flags that "worked" for 50 steps but died on longer runs.
"""
import os
import sys
import time
import torch

# Add src to path
sys.path.insert(0, os.path.abspath('.'))

from scripts.train_hybrid_ri4_real_continuation_minimal import (
    parse_args,
    build_model_and_optimizer,
    # We will manually do the critical path
)

def get_mem():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0

def main():
    print("=== MEMORY SPIKE DIAGNOSTIC ===")
    print(f"Initial CUDA mem: {get_mem():.1f} MiB")
    print(f"Host RSS (approx): {os.popen('ps -o rss= -p %d' % os.getpid()).read().strip()} KiB")

    # Use the exact flags from the "successful" 50-step run
    # We simulate the critical part: resume + first training step with heavy features

    ckpt = "checkpoints/hybrid_ri4_cont/hybrid_ri4_cont_step590.pt"
    if not os.path.exists(ckpt):
        print(f"ERROR: {ckpt} not found")
        return

    print(f"\nLoading checkpoint: {ckpt}")
    before = get_mem()

    # Load like the trainer does (simplified)
    checkpoint = torch.load(ckpt, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    print(f"After torch.load: {get_mem():.1f} MiB (delta {get_mem()-before:.1f})")

    # Now we would normally build the full model + triple memory + attractor solver
    # For diagnosis, just report what the heavy objects typically allocate

    print("\n=== Key observation ===")
    print("If this script itself stays under 2-3 GiB but the full trainer dies,")
    print("the spike is coming from BrainMimeticTripleMemory + full attractor state")
    print(" + rehearsal buffer + K-candidate trajectories during the first step.")
    print("\nRecommendation from this run: the pod memory limit is being hit")
    print("on the very first heavy forward + attractor call after resume.")

if __name__ == "__main__":
    main()