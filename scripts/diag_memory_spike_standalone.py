#!/usr/bin/env python3
"""
Standalone memory spike diagnostic.
No complex imports. Just load the checkpoint and report sizes.
This tells us if the model weights themselves are the problem or the runtime state.
"""
import os
import torch
import time

def get_cuda_mem():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0.0

def main():
    print("=== STANDALONE MEMORY SPIKE DIAGNOSTIC ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Initial CUDA allocated: {get_cuda_mem():.1f} MiB")

    ckpt_path = "checkpoints/hybrid_ri4_cont/hybrid_ri4_cont_step590.pt"
    if not os.path.exists(ckpt_path):
        print(f"ERROR: {ckpt_path} not found")
        return

    size_mb = os.path.getsize(ckpt_path) / (1024 ** 2)
    print(f"\nCheckpoint file size: {size_mb:.1f} MiB")

    print("\nLoading checkpoint (map_location=cuda)...")
    t0 = time.time()
    before = get_cuda_mem()
    ckpt = torch.load(ckpt_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    after_load = get_cuda_mem()
    print(f"After torch.load: {after_load:.1f} MiB (delta {after_load - before:.1f} MiB) in {time.time()-t0:.2f}s")

    print("\nCheckpoint keys (top level):")
    for k in list(ckpt.keys())[:15]:
        print(f"  - {k}")
    print(f"  ... total {len(ckpt)} top-level keys")

    # Look for known heavy objects
    if 'model_state' in ckpt or 'model' in ckpt:
        state = ckpt.get('model_state', ckpt.get('model', {}))
        print(f"\nModel state dict has {len(state)} tensors")

    print("\n=== CONCLUSION FROM THIS RUN ===")
    print("If this lightweight load stays under ~2 GiB but the full trainer dies in 8-10s,")
    print("the killer is NOT the model weights.")
    print("The killer is runtime state created AFTER load:")
    print("  - BrainMimeticTripleMemory (Working + Attractor + Provenance buffers)")
    print("  - AttractorSolverModule full H/L cycle state + proposal buffers")
    print("  - Rehearsal buffer + K-candidate trajectories (even with small K)")
    print("  - Internal fast recurrent states")
    print("\nThis matches the pattern: the 50-step 'successful' run from 590 never")
    print("had time to hit the cgroup limit. Any attempt at longer horizon does.")

if __name__ == "__main__":
    main()