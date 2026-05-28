# RI-1 M1 I-Stage Closure Criteria (Minimal Bar before G)

**Purpose**: Define the smallest set of evidence required to honestly declare the narrow I-stage contract for "Variable Depth Training inducing test-time depth scaling" as closed, before any talk of moving to Generalization (G) or Architecture-ization (A).

## Mandatory Evidence (all must exist on matched artifacts)

1. **Stronger Scaling Signal on Longer Horizon**
   - At least one 50+ step continuation (or longer) under the current recipe showing sustained or improved monotonic accuracy gain on pure_72 strict B when depth is increased (especially clear lift at d=8 vs d=1/4).
   - The gain at higher depth should be material compared to the 25-step result (40.28% at d=8).

2. **Causal Ownership via M1 Ablation**
   - Explicit small diagnostic: same base checkpoint + same total steps, only difference = M1 variable depth sampling enabled vs disabled (or strongly reduced).
   - The run with M1 sampling must show meaningfully better depth scaling (higher d=8 accuracy and/or larger d=1 to d=8 delta) than the M1-off counterpart.
   - This is the key "the variable depth training is what caused the scaling" evidence.

3. **Composition Without Destructive Interference**
   - The scaling gain must survive (or improve) when all three historical tracks are active together (already default in our runs).
   - At minimum, no major regression when Workspaces + Attractor + Provenance are all on compared to subsets.

4. **Basic Reproducibility / Stability Note**
   - Either a second seed showing similar directional scaling, or an explicit written note on observed seed sensitivity with a plan to address it in G-stage.

5. **Written Declaration**
   - A short "I-stage closed" note in the bias map + roadmap + decision record, stating:
     - What the narrow contract was.
     - Which evidence closed it.
     - What remains for G (multi-seed, broader families, full ablation matrix on scaling, etc.).

## Anti-Patterns (do not declare I closed if these are true)
- Only short synthetic runs (≤25 steps) exist.
- No direct M1 on/off comparison on the scaling metric.
- Scaling looks good only when other tracks are off.
- Claiming "the mechanism works" without showing it is the thing that improved the scaling (vs just longer training or other factors).

Only after the above five items are met (even minimally) should we seriously discuss moving to G-stage.

## Active Experiments for I-Closure (as of 2026-05-28)

1. **50-step clean M1 run** (launched)
   - Background task running.
   - Target: stronger and more sustained monotonic scaling than 25-step.

2. **Minimal M1 on/off ablation diagnostic** (plan ready)
   - See `M1_ablation_plan.md`
   - Will be executed as soon as we have a solid base checkpoint (ideally after the 50-step run, or using the 25-step if we want faster signal).

Once both are complete and show positive results, we will have the two biggest missing pieces for declaring I-stage closed.
