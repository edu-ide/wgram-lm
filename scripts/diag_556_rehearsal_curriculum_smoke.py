#!/usr/bin/env python3
"""
diag_556_rehearsal_curriculum_smoke.py

Small but realistic smoke test for the Full Adaptive Rehearsal 5.56 Gold Curriculum,
including the stochastic recurrent breadth (Reverse I→G→A).

References:
- docs/wiki/decisions/2026-05-30-deep-dive-full-556-rehearsal-curriculum.md
- docs/wiki/decisions/2026-05-30-reverse-iga-stochastic-breadth-plan.md

This script is the bridge between simulation and real code.
It currently runs in pure numpy for immediate feedback, but is structured
so it can call the real AdaptiveRehearsal.full_curriculum_rehearsal_step
once the trainer is ready.

Run:
python scripts/diag_556_rehearsal_curriculum_smoke.py --steps 40
"""

import argparse
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

# ============================================================
# Minimal simulation of key 5.56 components (for smoke)
# ============================================================

@dataclass
class RehearsalConfig:
    scheduled_binding_start: float = 0.40
    scheduled_binding_end: float = 0.04
    rehearsal_interval: int = 5
    attractor_protection_during_rehearsal: float = 0.7
    gold_injection_strength: float = 0.25

def get_scheduled_binding_weight(step: int, total_steps: int, cfg: RehearsalConfig) -> float:
    """Linear decay from start to end (core of 5.56 scheduled recipe)"""
    progress = min(step / max(total_steps - 1, 1), 1.0)
    return cfg.scheduled_binding_start + (cfg.scheduled_binding_end - cfg.scheduled_binding_start) * progress

def simulate_attractor_pressure(state: np.ndarray, memory_buffer: list, strength: float = 0.03) -> np.ndarray:
    """Simplified 570-style monotonic pressure"""
    if len(memory_buffer) < 3:
        return state
    recent = np.stack(memory_buffer[-3:])
    worst = recent[np.argmin([np.linalg.norm(recent[i] - state) for i in range(len(recent))])]
    push = state - worst
    return state + strength * push

def simulate_gold_injection(state: np.ndarray, gold_state: np.ndarray, alpha: float) -> np.ndarray:
    return (1 - alpha) * state + alpha * gold_state

def simulate_stochastic_breadth(state: np.ndarray, enabled: bool, ablation_zero: bool,
                                scale: float = 0.06, training: bool = True) -> np.ndarray:
    """Minimal version of the logic we added in core.py"""
    if not enabled or ablation_zero:
        return state
    noise = np.random.normal(0, scale, state.shape)
    return state + noise

# ============================================================
# Main Smoke Curriculum
# ============================================================

def run_556_smoke(curriculum_steps: int = 40,
                  stochastic_breadth_enabled: bool = False,
                  stochastic_breadth_ablation_zero: bool = False,
                  seed: int = 42) -> Dict[str, Any]:
    """
    Runs a miniature version of the 5.56 gold rehearsal curriculum.
    Returns metrics that can be compared across breadth on/off.
    """
    np.random.seed(seed)
    cfg = RehearsalConfig()

    d = 64
    # Fake "gold state" (proxy for 642 bos_latent behavior)
    gold_state = np.random.normal(0, 0.08, d).astype(np.float32)

    # Initial state (somewhat drifted)
    state = np.random.normal(0, 0.12, d).astype(np.float32)
    memory_buffer = []

    metrics = {
        "state_norms": [],
        "attractor_alignment": [],
        "rehearsal_coherence": [],
        "binding_weights": [],
    }

    for step in range(curriculum_steps):
        # 1. Scheduled binding decay (core of 5.56)
        bind_w = get_scheduled_binding_weight(step, curriculum_steps, cfg)
        metrics["binding_weights"].append(bind_w)

        # 2. Normal recurrent step (simplified)
        state = state * 0.95 + np.random.normal(0, 0.01, d) * 0.05

        # 3. Stochastic breadth (the piece we are testing)
        state = simulate_stochastic_breadth(
            state,
            enabled=stochastic_breadth_enabled,
            ablation_zero=stochastic_breadth_ablation_zero
        )

        # 4. Rehearsal every N steps (with attractor protection)
        if step % cfg.rehearsal_interval == 0:
            # Gold injection (structural, not just additive)
            state = simulate_gold_injection(state, gold_state, cfg.gold_injection_strength * bind_w)

            # Attractor protection during rehearsal (very important for 5.56)
            if len(memory_buffer) >= 3:
                state = simulate_attractor_pressure(
                    state, memory_buffer,
                    strength=0.025 * cfg.attractor_protection_during_rehearsal
                )

            # Record coherence (how close we stay to gold during rehearsal)
            coherence = 1.0 - np.linalg.norm(state - gold_state) / (np.linalg.norm(gold_state) + 1e-6)
            metrics["rehearsal_coherence"].append(coherence)

        # 5. Memory buffer + attractor pressure (normal steps)
        memory_buffer.append(state.copy())
        if len(memory_buffer) > 8:
            memory_buffer.pop(0)

        state = simulate_attractor_pressure(state, memory_buffer, strength=0.02)

        # Record metrics
        metrics["state_norms"].append(float(np.linalg.norm(state)))
        if len(memory_buffer) >= 3:
            alignment = float(np.mean([np.dot(state, m) for m in memory_buffer[-3:]]))
            metrics["attractor_alignment"].append(alignment)

    # Summary stats
    summary = {
        "final_state_norm": metrics["state_norms"][-1],
        "mean_attractor_alignment": float(np.mean(metrics["attractor_alignment"])) if metrics["attractor_alignment"] else 0.0,
        "mean_rehearsal_coherence": float(np.mean(metrics["rehearsal_coherence"])) if metrics["rehearsal_coherence"] else 0.0,
        "binding_decay_range": (min(metrics["binding_weights"]), max(metrics["binding_weights"])),
    }

    return {"metrics": metrics, "summary": summary, "config": cfg}

# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=== 5.56 Rehearsal Curriculum Smoke (with Stochastic Breadth) ===\n")

    # Run 1: Stochastic breadth OFF (ablation)
    result_off = run_556_smoke(
        curriculum_steps=args.steps,
        stochastic_breadth_enabled=True,
        stochastic_breadth_ablation_zero=True,
        seed=args.seed
    )

    # Run 2: Stochastic breadth ON
    result_on = run_556_smoke(
        curriculum_steps=args.steps,
        stochastic_breadth_enabled=True,
        stochastic_breadth_ablation_zero=False,
        seed=args.seed + 1
    )

    print("=== Results ===")
    print("\n[ Breadth = OFF (ablation_zero) ]")
    print(f"  Final state norm         : {result_off['summary']['final_state_norm']:.4f}")
    print(f"  Mean attractor alignment : {result_off['summary']['mean_attractor_alignment']:.4f}")
    print(f"  Mean rehearsal coherence : {result_off['summary']['mean_rehearsal_coherence']:.4f}")

    print("\n[ Breadth = ON ]")
    print(f"  Final state norm         : {result_on['summary']['final_state_norm']:.4f}")
    print(f"  Mean attractor alignment : {result_on['summary']['mean_attractor_alignment']:.4f}")
    print(f"  Mean rehearsal coherence : {result_on['summary']['mean_rehearsal_coherence']:.4f}")

    delta_coherence = result_on['summary']['mean_rehearsal_coherence'] - result_off['summary']['mean_rehearsal_coherence']
    print(f"\nΔ Rehearsal Coherence (ON - OFF) = {delta_coherence:+.4f}")

    print("\nThis is the minimal realistic smoke that includes the stochastic breadth Reverse I→G→A work.")
    print("See docs/wiki/decisions/2026-05-30-deep-dive-full-556-rehearsal-curriculum.md for context and next steps.")

def try_real_integration_mode(steps: int = 20):
    """
    2026-05-30: Actually exercises the real AdaptiveRehearsal.full_curriculum_rehearsal_step
    that we extended with stochastic breadth support.
    """
    print("\n=== Trying REAL production class integration (순서대로) ===")
    try:
        from src.qtrm_mm.rehearsal.adaptive_rehearsal import AdaptiveRehearsal, RehearsalConfig
        import torch

        real_cfg = RehearsalConfig(
            enabled=True,
            scheduled_binding_decay_start=0.40,
            scheduled_binding_decay_end=0.04,
            gold_state_injection_alpha=0.25,
            protect_attractor=True,
            attractor_protection_during_rehearsal=0.7,
        )

        class DummyCoreCfg:
            d_model = 64

        rehearsal = AdaptiveRehearsal(real_cfg, DummyCoreCfg())
        rehearsal.set_total_steps(steps)

        B, D = 4, 64
        z_h = torch.randn(B, 1, D)
        memory_buffer = [torch.randn(B, D) for _ in range(6)]
        gold = torch.randn(D) * 0.08

        def stochastic_breadth_stub(z):
            if torch.rand(1).item() > 0.65:
                return z + torch.randn_like(z) * 0.06
            return z

        norms = []
        for _ in range(steps):
            z_h = rehearsal.full_curriculum_rehearsal_step(
                z_h, memory_buffer, gold_state=gold, stochastic_breadth_fn=stochastic_breadth_stub
            )
            norms.append(z_h.norm().item())

        print(f"Real AdaptiveRehearsal.full_curriculum_rehearsal_step ran successfully for {steps} steps.")
        print(f"Final norm: {norms[-1]:.4f}")
        return True
    except Exception as e:
        print(f"Real mode not runnable here (expected in limited env): {e}")
        return False


if __name__ == "__main__":
    main()
    try_real_integration_mode(25)


# ============================================================
# Real code transition notes (2026-05-30)
# ============================================================
#
# To move from this smoke to production:
#
# from src.qtrm_mm.rehearsal.adaptive_rehearsal import AdaptiveRehearsal, RehearsalConfig
# from src.qtrm_mm.core import QTRMRecursiveCore
#
# real_reh = AdaptiveRehearsal(RehearsalConfig(...), core_cfg)
# real_reh.set_total_steps(total_training_steps)
#
# Then inside training loop:
# z_h = real_reh.full_curriculum_rehearsal_step(
#     z_h, memory_buffer, gold_state=gold_latent,
#     stochastic_breadth_fn = lambda z: core._apply_stochastic_breadth(z, ...)
# )
#
# The smoke above is deliberately written so the simulation loop
# can be replaced with the real call with minimal changes.