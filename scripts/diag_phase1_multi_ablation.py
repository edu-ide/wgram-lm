#!/usr/bin/env python3
"""
Phase 1 Multi-Mechanism Ablation Runner (I→G→A)
Covers: Answer Attractor (정답 정렬), Gated Workspaces, LeWM, Provenance Register

Runs multi-seed, multi-batch proxy ablations on GPU.
Produces clean tables for wiki / component_registry attachment.

Usage (GPU):
  .venv/bin/python scripts/diag_phase1_multi_ablation.py --seeds 3 --batch 4 --seq 16 --d 128
"""

import argparse
import torch
import torch.nn as nn
from wgram_lm.config import QTRMConfig
from wgram_lm.core import QTRMRecursiveCore

def run_single_trial(cfg: QTRMConfig, seed: int, batch: int, seq: int, d: int, device: str):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    core = QTRMRecursiveCore(cfg).to(device)
    ws = torch.randn(batch, seq, d, device=device)

    # Warm up buffer for attractor
    for _ in range(6):
        _, _, _, _ = core(ws, return_carry=True)

    z_l, z_h_full, _, halt_full = core(ws, return_carry=True)

    # Now run ablated versions
    results = {"full": z_h_full.norm().item()}

    # 1. Answer Attractor ablation
    cfg_attr = QTRMConfig(**{k: v for k, v in cfg.__dict__.items()})
    cfg_attr.core_answer_attractor_ablation_zero = True
    core_attr = QTRMRecursiveCore(cfg_attr).to(device)
    for _ in range(6):
        _, _, _, _ = core_attr(ws, return_carry=True)
    _, z_h_attr, _, _ = core_attr(ws, return_carry=True)
    results["attr_ablate"] = z_h_attr.norm().item()

    # 2. Workspace ablation
    cfg_ws = QTRMConfig(**{k: v for k, v in cfg.__dict__.items()})
    cfg_ws.core_thought_workspace_ablation_zero = True
    core_ws = QTRMRecursiveCore(cfg_ws).to(device)
    _, z_h_ws, _, _ = core_ws(ws, return_carry=True)
    results["ws_ablate"] = z_h_ws.norm().item()

    # 3. LeWM ablation
    cfg_lewm = QTRMConfig(**{k: v for k, v in cfg.__dict__.items()})
    cfg_lewm.core_lewm_ablation_zero = True
    core_lewm = QTRMRecursiveCore(cfg_lewm).to(device)
    _, z_h_lewm, _, _ = core_lewm(ws, return_carry=True)
    results["lewm_ablate"] = z_h_lewm.norm().item()

    # 4. Provenance ablation
    cfg_prov = QTRMConfig(**{k: v for k, v in cfg.__dict__.items()})
    cfg_prov.core_provenance_register_ablation_zero = True
    core_prov = QTRMRecursiveCore(cfg_prov).to(device)
    _, z_h_prov, _, _ = core_prov(ws, return_carry=True)
    results["prov_ablate"] = z_h_prov.norm().item()

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq", type=int, default=16)
    parser.add_argument("--d", type=int, default=128)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Phase 1 Multi Ablation on {device} (RTX 4090 preferred)")

    base_cfg = dict(
        d_model=args.d, d_ff=512, n_heads=4, n_kv_heads=2,
        n_prelude_layers=1, n_core_layers=2, max_seq_len=args.seq,
        vocab_size=8192,
        core_answer_attractor_enabled=True,
        core_thought_workspace_enabled=True,
        core_thought_workspace_selector_mode="importance",
        core_lewm_enabled=True,
        core_provenance_register_enabled=True,
    )

    all_results = []
    for s in range(args.seeds):
        cfg = QTRMConfig(**base_cfg)
        res = run_single_trial(cfg, seed=42 + s, batch=args.batch, seq=args.seq, d=args.d, device=device)
        res["seed"] = 42 + s
        all_results.append(res)
        print(f"Seed {42+s}: {res}")

    print("\n=== Phase 1 Multi-Mechanism Ablation Summary (GPU) ===")
    print("All values = mean z_h norm over last forward (higher = stronger state activity)")
    print("| Seed | Full | Attr Abate | WS Abate | LeWM Abate | Prov Abate |")
    print("|------|------|------------|----------|------------|------------|")
    for r in all_results:
        print(f"| {r['seed']} | {r['full']:.2f} | {r['attr_ablate']:.2f} | {r['ws_ablate']:.2f} | {r['lewm_ablate']:.2f} | {r['prov_ablate']:.2f} |")

    print("\nInterpretation (I→G→A):")
    print("- Larger drop when ablating a mechanism = stronger causal contribution to state / 정답 정렬.")
    print("- Run with --seeds 5+ --batch 8 for more stable numbers before G-stage composition.")

if __name__ == "__main__":
    main()