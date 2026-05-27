# Recommended Next Experiment after 180-step Real Gold Run

**Date**: 2026-05-30  
**Current Status**: 180-step real 642 gold + stochastic ON completed with strong, stable signals.

## Recommended Next Step (Highest Priority)

**Full Controlled Ablation Matrix on Real 642 Gold**

This is the direct next step required by the original plan and the research-driven-architecture-debugging skill for Promotion Gate evidence.

### Proposed Matrix (Minimal but Powerful)

Run the following 5 variants, each for **120–150 steps** on the same real 642 gold checkpoint:

1. **Full 5.56** (baseline — decay + protection + stochastic ON + real gold)
2. **Stochastic OFF** (`--stochastic_ablation_zero true`)
3. **Protection OFF** (temporarily set `attractor_protection_during_rehearsal=0.0` or equivalent)
4. **Decay Fixed High** (modify rehearsal config to keep binding weight high instead of decaying)
5. **Synthetic Gold Only** (no --gold_path, pure proxy)

### Why This Matrix?

- Directly measures the causal contribution of each historical 5.56 ingredient under real gold structural bias.
- The most interesting current signal (higher stochastic diversity with real gold path) can be isolated.
- Gives us the data needed for a proper Promotion Gate decision.

### Exact Commands (copy-paste ready)

```bash
# Recommended: Use the matrix runner (best)
PYTHONPATH=. python scripts/run_556_ablation_matrix.py \
    --steps 130 \
    --gold_path local_eval/642_adaptive_fine_tuned_200step/adaptive_phase2_checkpoint.pt \
    --base_save_dir local_556_real642_full_matrix_$(date +%Y%m%d)

# Alternative: Run individual variants manually if you want more control
```

After the matrix finishes, run the analyzer across all variants:

```bash
python scripts/analyze_556_curriculum_metrics.py \
    local_556_real642_full_matrix_*/**/metrics.json \
    --output real642_full_matrix_analysis.md
```

### Alternative (if matrix feels too heavy right now)

One longer single run:
- 300–400 steps, real gold + stochastic ON
- Goal: test whether the strong diversity signal and clean decay continue even further, or if any late-stage degradation appears.

### Decision

Reply with one of:
- "Full matrix 해" (recommended)
- "300 step single run 해"
- "다른 거 먼저 해" (tell me what)

All supporting code, hardening, and analysis tools are already in place.
