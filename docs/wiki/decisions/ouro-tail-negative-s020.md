# Ouro Tail-Negative S020

Status: accepted smoke probe, 2026-05-06.

## Question

The accepted causal-prefix tail checkpoint still failed only by selecting the
pre-subtract sum:

```text
full core8:
  correct_final:     4
  pre_subtract_sum:  4
```

This probe adds a narrow process-preference loss:

```text
final answer > first preterminal state
```

For mixed list arithmetic this is:

```text
sum(doubled_values) - offset > sum(doubled_values)
```

The negative is derived from in-row `depth_targets` and
`transition_finality_targets`. It is not an external runtime solver and it does
not change the universal LLM inference path.

## Implementation

Added training options:

```text
--tail-negative-margin-weight
--tail-negative-margin
--tail-negative-family-filter
```

Added helpers:

```text
tail_negative_rejected_texts(...)
tail_negative_sequence_margin_loss(...)
```

The objective requires causal-prefix supervision, so it applies to answer-token
prefix examples rather than full-answer teacher-forced leakage.

## Artifacts

```text
checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_tailneg_s020_from_tail_s020/last.pt

eval:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_tailneg_s020_from_tail_s020/lm_causal_forced_choice_smoke8_with_baselines.jsonl
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_tailneg_s020_from_tail_s020/tail_error_summary_smoke8.json
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_tailneg_s020_from_tail_s020/action_code_eval32.json
```

## Result

Baseline-inclusive held-out smoke8:

```text
donor_only:   0/8
core_off:     0/8
bridge_off:   2/8
full core8:   4/8
action-code: 32/32
finality:     1.0000
halted_exact: 32/32
```

Tail breakdown:

```text
full core8:
  correct_final:     4
  pre_subtract_sum:  4

bridge_off:
  correct_final:     2
  pre_subtract_sum:  6
```

Previous accepted checkpoint:

```text
full core8:   4/8
bridge_off:   3/8
```

## Decision

Accept as a causal-gap improvement, not as a final-tail solution.

The full model keeps the 4/8 score while bridge-off drops from 3/8 to 2/8 and
the action controller remains 32/32. However, the full model still has four
pre-subtract-sum failures, so the raw answer quality did not improve.

## MixedX4 Follow-Up Reject

Oversampled `mixed_list_arithmetic=4` and trained for 40 steps with stronger
tail-negative pressure.

Artifacts:

```text
checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_tailneg_mixedx4_s040_from_tail_s020/last.pt

eval:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_tailneg_mixedx4_s040_from_tail_s020/lm_causal_forced_choice_smoke8_with_baselines.jsonl
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_tailneg_mixedx4_s040_from_tail_s020/tail_error_summary_smoke8.json
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_tailneg_mixedx4_s040_from_tail_s020/action_code_eval32.json
```

Result:

```text
donor_only:   0/8
core_off:     0/8
bridge_off:   4/8
full core8:   3/8
action-code: 32/32
```

Tail breakdown:

```text
full core8:
  correct_final:     3
  pre_subtract_sum:  2
  doubled_list:      3

bridge_off:
  correct_final:     4
  pre_subtract_sum:  3
  doubled_list:      1
```

Reject. Stronger/oversampled tail-negative pressure reduces pre-subtract errors
but reintroduces doubled-list errors and makes bridge-off outperform the full
model.

## Next

The next candidate should not merely increase negative pressure. The failure is
now more specifically:

```text
The model can learn which operation comes next, but the answer-state loop does
not reliably bind the final operation result to the emitted token sequence.
```

Testable next candidates:

```text
1. final-operation marker/value binder inside the answer-state loop;
2. low-weight tail-negative only on the final digit positions;
3. transition-state-to-answer delta restricted to the final operation depth.
```
