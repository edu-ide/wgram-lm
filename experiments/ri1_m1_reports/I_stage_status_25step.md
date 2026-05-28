# RI-1 M1 I-Stage Status Report (after 25-step clean run)

**Date**: 2026-05-28
**Run**: checkpoints/hybrid_ri4_ri1_m1_long_20260528_1303 (25 steps, clean from scratch, d=128, enhanced M1 sampler with progress bias, --all-three-tracks)

## Latest Numbers (strict B on pure_recursive_reasoning_heldout_72)

| Depth | Accuracy  | vs 8-step M1 | vs Pre-M1 baseline |
|-------|-----------|--------------|--------------------|
| 1     | 18.06%    | lower        | lower              |
| 4     | 23.61%    | lower        | lower              |
| 8     | **40.28%**| +5.56pp      | +19.45pp (was 20.83%) |

**Key observation**: First time we have strong monotonic scaling with a clear jump at d=8 in a longer run. d=8 is now the best by a significant margin.

## I-Stage Assessment (per research-driven-architecture-debugging skill)

**Current phase**: Advanced I-stage (Improvement), **not yet closed**.

### What improved (positive)
- Clear monotonic depth scaling emerged (unlike pre-M1 degradation).
- d=8 accuracy reached the highest value seen so far under our current matched protocol.
- M1 variable depth sampling (with deeper bias in later steps) + 3-track composition produced the desired directional signal.
- Loss descent + active variable depth sampling throughout training confirmed (C-track good).

### What is still missing for I-stage closure
- Narrow contract not strong enough: 25 steps on synthetic base is still very limited horizon.
- No direct causal ablation: we do not yet have "M1 variable depth training on vs off" on the *same base and recipe* showing that turning off the variable depth sampling destroys the scaling gain.
- Low-depth performance dropped compared to shorter previous runs (possible capacity or initialization effect — needs diagnosis).
- Single run, no seed stability data.
- Composition with full 3-tracks is active but not yet stress-tested for destructive interference on the depth scaling axis.

### I-Stage Closure Criteria (proposed minimal bar before G)
1. At least one longer matched continuation (50+ steps) showing sustained or stronger monotonic scaling.
2. Explicit small M1 ablation diagnostic (same base, M1 sampling enabled vs disabled) where scaling gain materially shrinks when M1 is off.
3. At least one additional seed or clear note on seed sensitivity.
4. Written "narrow contract closed" declaration in wiki + bias map.

## Recommendation

**Do not move to G (Generalization) or A (Architecture-ization) yet.**

We are in the strongest I-stage position we have ever been for RI-1, but the skill requires closing the narrow contract with clearer causal ownership before generalizing.

Next immediate actions (I-stage strengthening):
- Run 50-step version of the current recipe (or longer).
- Add a minimal M1 on/off ablation experiment (even short) on the same starting point.
- If the next longer run strengthens the signal and we can get an ablation, then declare I closed and move to G.

**Current honest label**: "RI-1 M1 — Advanced I-stage with best scaling signal to date. I not closed."


## M1 ON vs M1 OFF Ablation Results (12-step continuation from 25-step base)

**Setup**: Same base (25-step M1 checkpoint), 12 additional steps, only difference = variable depth sampling enabled (ON) vs fixed (OFF via ablation flag). 3-tracks active in both.

**Results (strict B on pure_72)**:

**M1 ON (variable depth active)**:
- Depth 1: 25/72 (34.72%)
- Depth 8: 23/72 (31.94%)

**M1 OFF (variable depth disabled)**:
- Depth 1: 20/72 (27.78%)
- Depth 4: 24/72 (33.33%)
- Depth 8: 22/72 (30.56%)

**Observation**:
- M1 ON shows higher accuracy at low depth (d=1) compared to OFF.
- At high depth (d=8), the difference is small.
- In this short 12-step window, the variable depth training did not produce dramatically stronger scaling than fixed depth.
- This suggests the benefit of variable depth may require longer horizons to fully manifest in the scaling behavior, or the current pressure implementation's effect is more on overall capability than on the depth-scaling slope in short continuations.

**Implication for I-stage**:
- Some positive signal for M1 (better low-depth performance when variable depth was used in training).
- However, the "strong causal proof that variable depth training is what creates superior depth scaling" is not yet clear from this short ablation.
- The 50-step run (still running) will be more decisive for whether longer training with variable depth produces clearer scaling advantage.

This ablation advanced our understanding but did not yet provide the slam-dunk causal evidence needed to fully close I-stage.

## Update after Attractor strengthening (2026-05-28)

To address the observation that scaling did not continue reliably to high depth in the 50-step run, a minimal strengthening was added to the monotonic pressure when M1 variable depth is active:

- When a high-depth sample (think_steps >=6) is drawn, we add a small bonus for achieving good final alignment.
- This directly pressures the Answer Align Attractor to produce better states on deeper rollouts, which is the core of RI-1.

This is a targeted I-stage improvement to the Attractor component (the depth-wise pressure mechanism) to make variable depth training lead to more robust test-time scaling.

Future runs (post this patch) should show stronger and more consistent gains at higher depths.

## Post-Attractor Strengthening Experiment (launched 2026-05-28)

To test whether strengthening the Answer Align Attractor (cross-depth bonus on high-depth samples when M1 is active) produces more robust and monotonic scaling at higher depths, the following run was launched:

- 50 steps, clean (no resume), d=128
- M1 variable depth active (mean=6, max=14, progress-biased)
- All three tracks active
- Answer pressure weight 0.8 + monotonic weight 0.4 (same as previous strong runs)
- **Key difference**: The minimal Attractor cross-depth strengthening patch (high depth samples get bonus for good final alignment)

This is the direct test for "Answer Align Attractor 개선" as the leading candidate to close the remaining I-stage gap for RI-1.

Expected: If this run shows clearer, more sustained gains at d=8+ (compared to the pre-patch 50-step), it strongly supports that the Attractor was the limiting factor and the improvement is effective.

Out dir: checkpoints/hybrid_ri4_ri1_m1_50step_post_attractor_...

Will measure with full depth sweep (1/4/8) upon completion.
