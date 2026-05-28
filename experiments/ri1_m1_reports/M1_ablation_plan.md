# Minimal M1 Ablation Diagnostic Plan (for I-stage closure)

**Goal**: Prove that the variable depth training (M1) is causally responsible for the improved test-time depth scaling.

## Proposed Experiment

**Base**: Use the final checkpoint from a longer M1 run (e.g. the 25-step or upcoming 50-step clean run).

**Two matched short continuations** (same base, same number of steps, same everything except M1):

1. **M1 Enabled** (control)
   - `--enable_ri1_variable_depth`
   - `--ri1_depth_mean 5 --ri1_depth_max 12`
   - Run for 10~15 steps

2. **M1 Disabled** (ablation)
   - `--ri1_depth_ablation_fixed` (forces fixed depth, no variable sampling)
   - Same other flags
   - Run for the exact same number of steps

**Measurement** (after both finish):
- Run strict-B depth sweep (d=1, 4, 8) on both final checkpoints.
- Compare the scaling curves:
  - How much does d=8 improve over d=1 in each case?
  - If M1-enabled shows significantly better scaling (especially larger lift at high depth), this is strong causal evidence.

**Expected outcome for I-closure**:
- M1-enabled run should show clearly better monotonic scaling than M1-disabled run.
- This directly shows "variable depth training during continuation is what improved the depth scaling capability".

## Exact Commands (example, adjust paths)

```bash
# From the 25-step or 50-step final ckpt
BASE_CKPT=checkpoints/hybrid_ri4_ri1_m1_long_.../hybrid_ri4_cont_step25.pt

# M1 Enabled continuation (15 steps)
PYTHONPATH=. uv run python scripts/train_hybrid_ri4_real_continuation_minimal.py \
  --steps 15 --resume_from $BASE_CKPT \
  --out_dir experiments/ri1_m1_ablation/m1_on \
  --all-three-tracks --enable_ri1_variable_depth \
  --ri1_depth_mean 5 --ri1_depth_max 12 \
  --save_every 5

# M1 Disabled (ablation)
PYTHONPATH=. uv run python scripts/train_hybrid_ri4_real_continuation_minimal.py \
  --steps 15 --resume_from $BASE_CKPT \
  --out_dir experiments/ri1_m1_ablation/m1_off \
  --all-three-tracks --ri1_depth_ablation_fixed \
  --save_every 5

# Then measure both
for d in 1 4 8; do
  echo "=== M1 ON, depth=$d ==="
  PYTHONPATH=. uv run python experiments/matched_port_evaluation_a9617cd8/compute_72_accuracy_from_base.py \
    --checkpoint experiments/ri1_m1_ablation/m1_on/hybrid_ri4_cont_step15.pt \
    --all-three --effective-depth $d

  echo "=== M1 OFF, depth=$d ==="
  PYTHONPATH=. uv run python experiments/matched_port_evaluation_a9617cd8/compute_72_accuracy_from_base.py \
    --checkpoint experiments/ri1_m1_ablation/m1_off/hybrid_ri4_cont_step15.pt \
    --all-three --effective-depth $d
done
```

Once this diagnostic is executed and shows clear causal difference, we can mark "M1 causal ownership proven" and move much closer to declaring I-stage closed.
