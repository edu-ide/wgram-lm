"""
example_556_full_curriculum_wiring.py

Minimal, copy-paste ready example showing how to wire the real
AdaptiveRehearsal (with full 5.56 curriculum support) + stochastic breadth
into a training loop.

This is the direct follow-up to the smoke script and the production class
extensions done on 2026-05-30.

Usage in a real trainer:
    from this file import build_556_rehearsal, make_stochastic_breadth_fn
    ...
    rehearsal = build_556_rehearsal(core_cfg, total_steps=...)
    stochastic_fn = make_stochastic_breadth_fn(core)
"""

from typing import Optional, Callable
import torch

# Import the production classes we extended
try:
    from src.qtrm_mm.rehearsal.adaptive_rehearsal import AdaptiveRehearsal, RehearsalConfig
except ImportError:
    AdaptiveRehearsal = None
    RehearsalConfig = None


def build_556_rehearsal(core_cfg: object, total_steps: int) -> Optional["AdaptiveRehearsal"]:
    """
    Factory for the full 5.56 Adaptive Rehearsal module.
    """
    if AdaptiveRehearsal is None:
        raise RuntimeError("AdaptiveRehearsal not available")

    cfg = RehearsalConfig(
        enabled=True,
        scheduled_binding_decay_start=0.40,
        scheduled_binding_decay_end=0.04,
        gold_state_injection_alpha=0.25,
        protect_attractor=True,
        attractor_protection_during_rehearsal=0.7,
    )

    rehearsal = AdaptiveRehearsal(cfg, core_cfg)
    rehearsal.set_total_steps(total_steps)
    return rehearsal


def make_stochastic_breadth_fn(core: "QTRMRecursiveCore") -> Callable:
    """
    Returns a function compatible with full_curriculum_rehearsal_step's stochastic_breadth_fn hook.
    This is the hook for the Reverse I→G→A stochastic breadth work.
    """
    def stochastic_breadth(z_h: torch.Tensor) -> torch.Tensor:
        if not getattr(core, "_stochastic_breadth_enabled", False):
            return z_h
        if getattr(core, "_stochastic_breadth_ablation_zero", False):
            return z_h
        # Delegate to the implementation we added in QTRMRecursiveCore
        if hasattr(core, "_apply_stochastic_breadth"):
            # Simplified call for trainer context
            pooled = z_h.mean(dim=1)
            ctx = pooled  # or memory signal
            return core._apply_stochastic_breadth(z_h, pooled, ctx)
        return z_h
    return stochastic_breadth


# Example usage inside a training step
def example_training_step(
    core: "QTRMRecursiveCore",
    rehearsal: "AdaptiveRehearsal",
    z_h: torch.Tensor,
    memory_buffer: list,
    gold_state: Optional[torch.Tensor],
):
    stochastic_fn = make_stochastic_breadth_fn(core)

    z_h = rehearsal.full_curriculum_rehearsal_step(
        z_h=z_h,
        memory_buffer=memory_buffer,
        gold_state=gold_state,
        stochastic_breadth_fn=stochastic_fn,
    )

    # ... continue with normal core forward, loss, etc.
    return z_h


if __name__ == "__main__":
    print("This is an example wiring file. Import build_556_rehearsal and make_stochastic_breadth_fn in your trainer.")
    print("See docs/wiki/decisions/2026-05-30-deep-dive-full-556-rehearsal-curriculum.md for context.")