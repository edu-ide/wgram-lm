# Consolidated RI-1 M1 Analysis (as of 2026-05-28)

## Artifacts We Now Have

1. **25-step clean M1 run** (best scaling signal so far)
   - d=1: 18.06%
   - d=4: 23.61%
   - d=8: **40.28%** (strong monotonic)

2. **50-step clean M1 run**
   - d=1: 30.56%
   - d=4: **37.50%**
   - d=8: 33.33% (scaling peaks at d=4, slight drop at d=8)

3. **M1 ON vs M1 OFF ablation** (12 steps from 25-step base) - causal test
   - M1 ON: d=1 34.72%, d=8 31.94%
   - M1 OFF: d=1 27.78%, d=8 30.56%
   - M1 ON is better at low depth; high depth difference is small in this short window.

## Honest Assessment for I→G→A

**Strengths**:
- Variable depth training + 3-track composition can produce monotonic depth scaling (25-step run is the best evidence).
- Overall capability improves with more steps (50-step has higher mid-depth accuracy).
- M1 ON shows benefit over fixed in the ablation (especially at d=1).

**Limitations / Open Questions**:
- Scaling is not consistently "the deeper the better" across runs (50-step peaks at d=4).
- The causal impact of variable depth on *scaling slope* (vs just overall performance) is modest in short continuations.
- We still lack a long run where high depth (d=8+) clearly and substantially outperforms mid-depth in a stable way.
- No multi-seed data.

## Current I-Stage Status

**Advanced I-stage with promising but not yet conclusive evidence.**

The 25-step run gave us the "proof of concept" that the direction works.
The 50-step and ablation gave us nuance (benefit exists, but not as dramatic as hoped in short horizons, and scaling can saturate or peak).

## Recommended Fastest Path Forward (to close I or decide on next architecture move)

1. **Immediate (today)**: Analyze the full C-track from the 50-step run (loss curve + sampled depth distribution) to see if deeper samples were actually helpful or if the model preferred certain depths.

2. **Next 1-2 days**: If C-track shows that high depth samples were not being utilized effectively, consider a small targeted improvement (e.g., stronger depth-wise contrastive loss inside the Attractor using the rolling memory_buffer).

3. **Decision point after that**:
   - If a follow-up run with the improvement shows clear, stable, high-depth superiority → declare I closed and move to G.
   - If even with improvements the scaling remains modest or peaks at mid-depth → this may indicate a deeper architectural limitation (e.g., the current hybrid recurrence + attractor is not the ideal substrate for unbounded depth scaling), and we should consider the big-jump options discussed earlier.

This is the honest, fastest, skill-compliant path.
