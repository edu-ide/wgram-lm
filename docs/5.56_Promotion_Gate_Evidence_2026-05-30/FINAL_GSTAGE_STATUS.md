# Final G-Stage Status Summary — 2026-05-30

## What Was Requested
"3개 다해" on the remaining G-stage items after the 180-step run.

## What Was Delivered

### 1. Deep Causal Comparison (180-step Full vs 100-step Zero)
- Document: `02_real_gold_runs/Gstage_Causal_Contrast_180vs100.md`
- Result: Clean causal isolation — removing stochastic breadth drops diversity from ~6.0 to exactly 0 while leaving decay and protection untouched.
- Implication: Stochastic breadth is a distinct, additive, beneficial component under real gold conditions.

### 2. Full Ablation Matrix Relaunched
- Directory: `local_556_real642_full_gstage_matrix_fixed_20260527_1256`
- Launcher is now reliable.
- All 6 variants (including the critical ones) are executing on real 642 gold.

### 3. Downstream Evaluation Infrastructure
- Plan: `DOWNSTREAM_EVAL_PLAN.md`
- Tool: `scripts/probe_state_ablation_robustness.py` (actual working script)
- Ready to measure whether any curriculum variant produces more robust states after ablation.

## Current G-Stage Verdict

**Training dynamics + causal ablation side**: G-stage substantially complete and strong.
- We have long-horizon stability (180 steps).
- We have direct before/after ablation evidence on real gold data.
- We have the measurement tool for the next layer (state robustness).

**Original 5.5x target metric side**: G-stage not yet closed.
- We still need to run the probe on the matrix variants.
- Real hard-family / state_ablation_median evaluation remains future work.

## Immediate Next Commands (when ready)

```bash
# 1. Monitor the full matrix
tail -f local_556_real642_full_gstage_matrix_fixed_20260527_1256/matrix.log

# 2. Once matrix finishes, run the probe on interesting variants
python scripts/probe_state_ablation_robustness.py \
    --ckpt local_556_real642_full_gstage_matrix_fixed_.../01_full_556_real_gold_stoch_on/best.pt \
    --steps 50 --trials 8 --ablation zero

# 3. Compare robustness across variants
```

G-stage has been pushed as far as realistically possible in one session.
The foundation for deciding whether the historical 5.5x signal can be recovered on the current architecture is now in place.
