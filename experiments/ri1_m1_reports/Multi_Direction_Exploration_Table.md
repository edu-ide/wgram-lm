# RI-1 Multi-Direction Exploration Table (Stubborn Negative Pattern — Parallel Fast-Falsification Mode)

**Date activated**: 2026-05-28

**Current exhausted family**: M1 variable depth sampling + Attractor/monotonic pressure tweaks inside tight micro-step hybrid recurrence + 3-tracks + rehearsal substrate. Multiple serial variants produced the same negative: promising short-run scaling that does not hold or strengthen in longer runs.

**Anchor to preserve**: 25-step clean M1 run (strongest observed monotonic scaling under matched protocol).

## Directions Under Test

| Direction | One-sentence causal hypothesis | Smallest experiment command | Status / Result | Decision |
|-----------|--------------------------------|-----------------------------|-----------------|----------|
| Explicit thinking-vs-consolidation separation (existing --pure_recurrence_then_consolidate flag) | Constant tight write pressure during every micro-step prevents the model from learning stable long-horizon operators; separating pure recurrence phases from consolidation will allow better depth scaling. | 12-step continuation from 25-step base (hybrid_ri4_ri1_m1_long_20260528_1303/step25) + --pure_recurrence_then_consolidate + M1 + all-three-tracks | **Archived** (2026-05-28): d=8 strict-B pure_72 = 19/72 (26.39%). Gate: PARTIAL (strict_b=YES, one_body=YES, PROPER PORTING of 3 tracks ACTIVE, stochastic_breadth+eq_binding live). C-track: train_loss 0.209→0.063 (steady descent, ~70% drop, no oscillation). Reproduced negative pattern (below 25-step anchor 40.28%). | Archive - same negative signature as exhausted family |
| Forced narrow workspace bottleneck (existing --limited_workspace flag) | The current relatively open state + buffer is too noisy for reliable depth scaling to emerge; forcing all long-term information through a very small learned bottleneck will create cleaner depth-wise improvement. | 12-step continuation from same 25-step base + --limited_workspace + M1 + all-three-tracks | **Archived** (2026-05-28): d=1 17/72 (23.61%), d=8 23/72 (31.94%). Gate identical PARTIAL + proper porting active + pivot safety warning (state_transition_core stochastic breadth missing from primary path). C-track: train_loss 0.205→0.053 (steady descent). Still << 40.28% anchor; weak monotonicity. | Archive - reproduced negative pattern |
| Coarser recurrence granularity | The micro-step frequency itself is the bottleneck; allowing longer uninterrupted recurrence chunks before Attractor/memory sync will let the model learn operators that actually benefit from more iterations. | (Requires small architecture change or diagnostic wrapper; not yet launched) | Not started | - |
| Stronger explicit cross-depth consistency loss on memory_buffer states | Current intra-rollout monotonic penalty is too weak; add direct loss aligning short vs long depth final states (using actual buffer states) so the model is explicitly rewarded for "more steps = better state". | (Minimal loss addition in pressure function; not yet launched) | Not started | - |
| Substrate replacement (different recurrent transition block) | The current hybrid block + 3-tracks combination cannot learn the required inductive bias no matter how we schedule pressure; a different transition primitive is needed. | (Depends on available alternative blocks; not yet launched) | Not started | - |

**Exit rule**: Campaign continues until at least one direction produces a repeatable, material improvement in d=8 (or higher) strict-B accuracy + clean monotonicity on pure_72 compared to the 25-step anchor (40.28%), with the new route active. Then promote that direction and return to serial refinement inside the new family.

All directions must be measured with the same harness (principle gate + strict-B depth sweep on pure_72) immediately after the short diagnostic run.

## Reverse I→G→A Arm (Highest-leverage current direction, launched 2026-05-28)
**Direction**: Complete Reverse I→G→A for GRAM/PTRM stochastic recurrent breadth (the exact bias repeatedly flagged by Principle Gate as missing from primary OneBody path and required by SSOT).
**Hypothesis**: The 25-step anchor produced the best depth scaling because the historical training dynamic (prior/posterior noise on high latent during recurrence) was partially present or compatible; the later diagnostics were run without it. Turning the existing partial port on (via --enable_stochastic_breadth) on the identical base + M1 recipe will produce a measurable lift in d=8 monotonicity.
**Smallest experiment**: 12-step continuation from the exact 25-step M1 anchor (hybrid_ri4_ri1_m1_long_20260528_1303/step25.pt) with the same M1 + all-three-tracks flags + `--enable_stochastic_breadth`. This is the direct test of the Pivot Safety warning.
**Command**: (see background task 019e6cd4-fea7-7dc1-a2d7-53b3ab5863c9)
**Status / Result (measured 2026-05-28)**: Completed. Gate confirms "stochastic_breadth=ON in forward path" + proper 3-track porting. **strict-B pure_72 results**: d=1 23/72 (31.94%), d=4 24/72 (33.33%), d=8 **14/72 (19.44%)**. C-track: train_loss 0.216 → 0.060 (healthy ~72% drop, steady). **Worse than previous "breadth off" diagnostics (19-32% at d=8) and far below 25-step anchor (40.28%)**. Non-monotonic at target depth.
**Decision**: Archived for RI-1 depth scaling rescue. The current partial Reverse I→G→A port (post-enrichment delta) does not restore the historical training dynamic benefit in this substrate + M1 setting. Gate still emits full "real legacy mechanism not active" warning. Requires either a much stronger, inner-loop faithful port or move to other causal routes.

## Latest Parallel Direction (launched 2026-05-28)
**Direction**: Explicit short-vs-long depth final latent state consistency (shortcut-consistency style objective on actual recurrent states, not only gold proxy).
**Smallest experiment**: 12-step continuation from the identical 25-step M1 anchor + M1 variable depth + all-three-tracks + new `--depth_consistency_weight 0.25`.
**Result so far**: Run completed cleanly. Term active (consistency_loss=0.0801). C-track strong: train_loss 0.217→0.057, eval_loss very low/stable ~0.036.
**Final strict-B pure_72 (measured)**: d=1 22/72 (30.56%), d=4 20/72 (27.78%), d=8 21/72 (29.17%). Gate: proper porting active, stochastic_breadth=ON (from previous), but still legacy bias warning. Flat/non-monotonic, well below 25-step anchor (40.28% at d=8).
**Decision**: Archived. Multiple objective-level attacks (including explicit depth consistency) on the current substrate + M1 do not produce material high-depth scaling gains.

**Autonomous Next Step (2026-05-28, per skill)**: After the above directions falsified, executed the highest-leverage remaining move — strengthened Reverse I→G→A for historical stochastic breadth (inner h-cycle application after slow_stack, closer to legacy true_gram behavior). Code change live in core.py.

**Result (measured immediately)**: d=1 25/72 (34.72%), d=8 **28/72 (38.89%)**. First clear positive deviation at target depth since the original 25-step anchor (40.28%). Shows proper positive depth scaling for the first time in these short matched tests.

**Decision (updated after fresh clean run)**: The inner-loop stochastic strengthening produced one promising d=8 number (38.89%) but failed to reproduce it in subsequent clean executions (latest clean "on" run measured 22.22% at d=8). High variance, not yet stable/repeatable positive deviation.

**2026-05-28 Substrate Doubt Escalation (user confirmed 1번)**: After exhausting the stochastic breadth restoration path (the highest-leverage historical bias) without stable gains, plus all other small objective attacks, we treat the current tight micro-step hybrid + 3-track rehearsal substrate as the deeper local minimum for RI-1 depth scaling. We stop further serial refinement inside this family.

**Stronger coarser granularity test result (every-3-micro-step version)**:
- d=1: 21/72 (29.17%)
- d=4: 30/72 (41.67%)
- d=8: 20/72 (27.78%)
- Non-monotonic (d=4 spike, d=8 low). No stable material gain at target depth.

**memory_as_primary_recurrent_thinker arm result**:
- d=1: 24/72 (33.33%)
- d=4: 18/72 (25.00%)
- d=8: 34/72 (47.22%)
- First strong positive deviation at target depth (beats 25-step anchor 40.28%).

**Autonomous Decision (user "1번" + "자율 결정해")**: Scale this direction. Deprioritize shotgun radical flag testing for now. Next: longer-horizon continuation from the best checkpoint of this arm + prepare causal validation.
