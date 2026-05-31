#!/usr/bin/env python3
"""
Minimal diagnostic to find the exact memory spike point.
Loads a real checkpoint and initializes the heavy components the trainer uses,
then measures CUDA memory at each critical step.

Run with: PYTHONPATH=src:. python scripts/diag_heavy_stack_memory.py
"""
import os
import sys
import time
import torch

sys.path.insert(0, os.path.abspath('.'))

def fmt(x):
    return f"{x/1024**2:.1f} MiB"

def get_mem():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated()
    return 0

def main():
    print("=== HEAVY STACK MEMORY DIAGNOSTIC ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU mem: {fmt(torch.cuda.get_device_properties(0).total_memory)}")

    ckpt_path = "checkpoints/hybrid_ri4_cont/hybrid_ri4_cont_step590.pt"
    if not os.path.exists(ckpt_path):
        print(f"ERROR: {ckpt_path} not found")
        return

    print(f"\nCheckpoint: {ckpt_path} ({os.path.getsize(ckpt_path)/1024**2:.1f} MiB on disk)")

    base = get_mem()
    print(f"\n[0] Baseline CUDA: {fmt(base)}")

    # 1. Load checkpoint
    t0 = time.time()
    ckpt = torch.load(ckpt_path, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=False)
    print(f"[1] After torch.load: {fmt(get_mem())} (delta {fmt(get_mem()-base)}) in {time.time()-t0:.2f}s")

    # 2. Try to import and instantiate the heavy pieces the trainer uses
    print("\n[2] Importing heavy components...")

    try:
        from wgram_lm.blocks import OneBodyParallelHybridBlock
        print("    - OneBodyParallelHybridBlock imported")
    except Exception as e:
        print(f"    - OneBodyParallelHybridBlock FAILED: {e}")

    try:
        from wgram_lm.memory.brain_triple_memory import BrainMimeticTripleMemory
        print("    - BrainMimeticTripleMemory imported")
        m = BrainMimeticTripleMemory(d_model=128)
        if torch.cuda.is_available():
            m = m.cuda()
        print(f"    - BrainMimeticTripleMemory instantiated: {fmt(get_mem())}")
    except Exception as e:
        print(f"    - BrainMimeticTripleMemory FAILED: {e}")

    try:
        from wgram_lm.attractor.attractor_solver import AttractorSolverModule
        print("    - AttractorSolverModule imported")
        solver = AttractorSolverModule(
            dim=128,
            H_cycles=2,   # light
            L_cycles=4,   # light
            ri_scale=0.08,
            use_anderson=False,
        )
        if torch.cuda.is_available():
            solver = solver.cuda()
        print(f"    - AttractorSolverModule (H2/L4) instantiated: {fmt(get_mem())}")
    except Exception as e:
        print(f"    - AttractorSolverModule FAILED: {e}")

    try:
        # Simulate what happens on first forward + attractor call
        print("\n[3] Doing a tiny forward simulation (random input, 1 micro-step)...")
        x = torch.randn(2, 4, 128, device='cuda' if torch.cuda.is_available() else 'cpu')
        print(f"    - Input created: {fmt(get_mem())}")

        # If we had a real model we would call it here.
        # For now just touch the solver if it exists
        if 'solver' in locals():
            # Minimal call to see allocation
            _ = solver(x.mean(dim=1), num_steps=1)
            print(f"    - Solver called once: {fmt(get_mem())}")

    except Exception as e:
        print(f"    - Forward simulation FAILED: {e}")

    final = get_mem()
    print(f"\n=== FINAL CUDA ALLOCATED: {fmt(final)} (total delta from baseline: {fmt(final-base)}) ===")
    print("If this is already > 8-10 GiB on a tiny input, we have found the smoking gun.")

if __name__ == "__main__":
    main()