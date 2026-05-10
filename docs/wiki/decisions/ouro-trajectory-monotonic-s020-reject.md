# Ouro Trajectory-Monotonic S020 Reject

Status: rejected probe, 2026-05-06.

## Failure Ledger

Failure:

```text
Accepted tail-negative checkpoint still solves only 4/8 held-out mixed
composition cases and fails the remaining four by emitting the pre-subtract
sum. Answer-side binder probes regressed to 2/8.
```

Root architecture hypothesis:

```text
The bottleneck might be insufficient process credit across recursive loop
depths, not an answer-side readout problem.
```

Information path needed:

```text
prompt -> donor hidden states -> QTRM recursive core / answer-state loop
-> LM logits
```

Current path preserved:

```text
yes. No MemoryOS, retrieval, hidden answer channel, or runtime solver is used.
```

Architecture candidates considered:

```text
1. Adjacent-depth trajectory monotonic margin.
2. Later-depth shortcut KL into earlier depths.
3. Latent-step reward / RL-style process credit.
```

Recommended candidate:

```text
1. It is the smallest falsifier and directly tests whether depth-to-depth
answer log-prob regression is the current bottleneck.
```

Acceptance gate:

```text
Promote only if full core8 beats the accepted 4/8 baseline, or preserves 4/8
while an ablated bridge/core path drops below full. Action-code must remain
32/32.
```

Kill criterion:

```text
Reject if full core8 ties baseline and bridge-off/core-off matches the full
advantage.
```

## Implementation

Added training args:

```text
--depth-trajectory-monotonic-weight
--depth-trajectory-monotonic-margin
```

Mechanism:

```text
For adjacent recursive depths, compute target sequence mean log-prob.
Penalize: next_depth_logp < previous_depth_logp + margin
```

This differs from the older `progress_margin` because it assigns process credit
to every adjacent transition, rather than only comparing all previous depths to
the final depth.

Artifacts:

```text
config:
  configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_trajectory_monotonic_s020.yaml

runner:
  scripts/242_run_qtrm_ouro_trajectory_monotonic_s020.sh

checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_trajectory_monotonic_s020_from_tail_s020/last.pt
  deleted after rejection to recover local disk; eval JSON artifacts retained.

eval:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_trajectory_monotonic_s020_from_tail_s020/lm_causal_forced_choice_smoke8_with_baselines.jsonl
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_trajectory_monotonic_s020_from_tail_s020/action_code_eval32.json
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_trajectory_monotonic_s020_from_tail_s020/tail_error_summary_smoke8.json
```

## Result

Held-out smoke8 causal forced-choice:

```text
donor_only:   0/8
core_off:     0/8
core_steps 1: 2/8
core_steps 2: 2/8
core_steps 4: 2/8
full core8:   4/8
bridge_off:   4/8
```

Action-code gate:

```text
exact:        32/32
step_acc:     1.0000
finality_acc: 1.0000
halted_exact: 32/32
```

Tail breakdown:

```text
full core8:
  correct_final:     4
  pre_subtract_sum:  4

bridge_off:
  correct_final:     4
  pre_subtract_sum:  4
```

Training telemetry:

```text
depth_trajectory_monotonic: 0.0000
depth_trajectory_step_delta: positive on logged core_steps=4 sample
```

This means the accepted checkpoint already satisfied the adjacent target
log-prob monotonic condition on the sampled training rows. The new objective
therefore did not attack the active held-out failure.

## Decision

Reject as canonical.

The probe preserves the 4/8 accepted score and keeps the transition/action
controller perfect, but it removes the previous causal bridge gap: full and
bridge-off both score 4/8. A loss that is already zero during training is not a
useful bottleneck intervention.

## Next Hypothesis

The next experiment should not add another answer-side binder or simple
monotonic target-logp margin. The failure is more specific:

```text
the model has the correct transition trace, but the LM answer path cannot
reliably perform or retain the final subtract transformation.
```

Next candidate:

```text
state-delta contrast / counterfactual subtract-tail objective
```

The objective should compare final answer against minimally different
counterfactual tails, e.g. `sum`, `sum-offset+1`, `sum-offset-1`, and should be
tested with bridge/core ablations so it cannot pass by memorizing one surface.
