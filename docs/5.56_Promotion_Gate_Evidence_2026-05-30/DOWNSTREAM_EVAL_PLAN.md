# Downstream Evaluation Plan for 5.56 Curriculum Checkpoints

**Date**: 2026-05-30
**Purpose**: Move from training dynamics proxies to the actual historical target metrics (state_ablation_median ~5.5x range on hard-family cases).

## The Original Target (What 5.53~5.56 Actually Meant)

The famous numbers came from **post-training evaluation**, not training loss:
- After running the full 5.56 Adaptive Rehearsal curriculum on 642 gold,
- Measure **state_ablation_median** on held-out hard-family reasoning cases.
- The model that had gone through the curriculum showed significantly better robustness when its recurrent state was ablated.

This is the true signal we are trying to recover.

## Current Limitation

The current `train_556_full_curriculum_minimal.py` is a **small-scale curriculum trainer** using synthetic random workspaces. It is excellent for studying training dynamics (decay, stochastic breadth, protection), but does not produce checkpoints that can be directly evaluated on the original hard-family tasks.

## Recommended Path to Close G-Stage

### Phase 1: Proxy Downstream (Doable Now)

Create a simple "state stability under ablation" probe that works on the current small model:

1. Load a trained checkpoint from the 5.56 curriculum.
2. Run a batch of synthetic "reasoning" sequences.
3. At various points, ablate or noise the recurrent state (z_h) in different ways.
4. Measure how much the next prediction / trajectory quality degrades (analogous to state_ablation_median).

This can give us a **relative** signal: "Did the 5.56 curriculum + stochastic breadth produce more robust states than ablated versions?"

### Phase 2: Real Downstream (The Actual Goal)

To truly know if we recovered the 5.5x signal, we eventually need to:
- Port the best 5.56 curriculum checkpoints into a larger model that has the original hard-family evaluation harness.
- Or build a proper hard-family dataset + evaluator on top of the current QTRMRecursiveCore.

## Starter Script (Phase 1 Proxy)

Below is a minimal starter you can extend.

**File to create**: `scripts/probe_state_ablation_robustness.py`

```python
"""
Minimal state ablation robustness probe for 5.56 curriculum checkpoints.
This is a proxy for the historical state_ablation_median idea.
"""

import torch
from wgram_lm.config import QTRMConfig
from wgram_lm.core import QTRMRecursiveCore

def probe_checkpoint(ckpt_path, steps=50, ablation_strength=0.5, num_trials=8):
    """
    Load a curriculum-trained checkpoint and measure how much
    state ablation hurts future trajectory quality.
    """
    # TODO: implement loading logic compatible with the trainer's save format
    # Then run forward passes with and without state perturbation
    # Compute relative degradation (the proxy for "ablation median")
    pass

if __name__ == "__main__":
    # Example usage after a matrix finishes
    # probe_checkpoint("local_556_real642_full_gstage_matrix_.../01_full_556_real_gold_stoch_on/best.pt")
    print("Implement the actual probe based on your evaluation needs.")
```

## Action Items for True G-Stage Closure

1. Finish the current full matrix (now running with fixed launcher).
2. Use the best checkpoints from the matrix to run the proxy probe above.
3. Compare relative robustness:
   - Full 5.56 + stoch ON vs stoch zero
   - Full vs protection-off, etc.
4. If any variant shows clearly superior state robustness, that becomes the candidate for Phase 2 (real hard-family eval).

**Bottom line**: The training dynamics side of G-stage is now very strong.
The remaining gap to the original 5.5x is almost entirely in the **evaluation** layer, not the curriculum itself.
