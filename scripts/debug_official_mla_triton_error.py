#!/usr/bin/env python3
"""
Deep debugging script for:
"ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)"

Focus: Official FLA MultiheadLatentAttention inside OneBodyParallelHybridBlock.

This script adds heavy device/dtype tracing to find where a CPU tensor is leaking
into the Triton kernel path.
"""

import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import traceback

from src.qtrm_mm.config import QTRMConfig
from src.qtrm_mm.blocks import OneBodyParallelHybridBlock


def print_tensor_info(name: str, tensor: torch.Tensor):
    if tensor is None:
        print(f"  {name}: None")
        return
    print(f"  {name}: shape={tuple(tensor.shape)}, device={tensor.device}, dtype={tensor.dtype}, "
          f"requires_grad={tensor.requires_grad}")


def debug_forward_with_tracing(block, x, attention_mask=None, stochastic_breadth_noise=None):
    print("\n=== ENTERING HybridBlock.forward with tracing ===")
    print_tensor_info("input x", x)

    # Monkey-patch the attention heads' forward to trace
    original_att_forwards = []
    for i, attn in enumerate(block.attention_heads):
        orig_forward = attn.forward

        def make_traced_forward(idx, orig_fn):
            def traced_forward(hidden_states, attention_mask=None):
                print(f"\n[ATTN HEAD {idx}] Forward called")
                print_tensor_info(f"  hidden_states to MLA", hidden_states)
                try:
                    return orig_fn(hidden_states, attention_mask=attention_mask)
                except Exception as e:
                    print(f"  !!! ERROR in MLA head {idx}: {e}")
                    traceback.print_exc()
                    raise
            return traced_forward

        attn.forward = make_traced_forward(i, orig_forward)
        original_att_forwards.append(orig_forward)

    try:
        print("\n--- Calling block(x) ---")
        out = block(x, attention_mask=attention_mask, stochastic_breadth_noise=stochastic_breadth_noise)
        print("\n=== Forward succeeded ===")
        print_tensor_info("output", out)
        return out
    finally:
        # Restore original forwards
        for attn, orig in zip(block.attention_heads, original_att_forwards):
            attn.forward = orig


def main():
    print("=" * 70)
    print("DEEP DEBUG: Official MLA Triton CPU Tensor Error")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    cfg = QTRMConfig(
        d_model=128,
        n_heads=8,
        n_kv_heads=4,
        d_ff=256,
        max_seq_len=512,
        delta_backend="official_gated_delta2",   # Try to force official GDN2 too
    )

    print("\nCreating OneBodyParallelHybridBlock with attention_type='mla'...")
    block = OneBodyParallelHybridBlock(
        cfg=cfg,
        recurrence_head_count=2,
        attention_head_count=2,
        attention_type="mla",
    )

    # Print what was actually loaded
    print(f"\nRecurrence head 0 type: {type(block.recurrence_heads[0]).__name__}")
    print(f"Attention head 0 type : {type(block.attention_heads[0]).__name__}")

    # === ROOT CAUSE DIAGNOSIS ===
    print("\n=== DEVICE STATE AFTER CREATION (before any .to()) ===")
    print(f"  block.norm1.weight.device = {block.norm1.weight.device}")
    print(f"  block.norm2.weight.device = {block.norm2.weight.device}")

    # Create input on CUDA explicitly
    x = torch.randn(2, 8, 128, device=device, dtype=torch.float32)
    print_tensor_info("\nInput tensor (explicitly on CUDA)", x)

    print("\n>>> This is the ROOT CAUSE: The entire block (including all loaded official modules)")
    print(">>> was created on CPU. Its parameters were never moved to CUDA.")
    print(">>> Passing a CUDA tensor directly causes device mismatch on the very first norm.")

    # Demonstrate the fix
    print("\n>>> Applying the fix: block = block.to(device)")
    block = block.to(device)

    print(f"After .to(device):")
    print(f"  block.norm1.weight.device = {block.norm1.weight.device}")

    # Try the traced forward again
    try:
        debug_forward_with_tracing(block, x)
    except Exception as e:
        print(f"\n\n=== ERROR AFTER FIX (should be different) ===")
        print(f"{type(e).__name__}: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
