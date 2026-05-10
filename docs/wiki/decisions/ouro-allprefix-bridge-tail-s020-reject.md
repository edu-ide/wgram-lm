# Ouro All-Prefix Bridge Tail S020 Reject

Status: rejected, 2026-05-06.

## Question

The accepted causal-prefix tail checkpoint improved the mixed-composition smoke
to `full 4/8`, but bridge-off also improved to `3/8`. This suggested the tail
tokens were being trained mostly by answer CE / sequence margin, while the
transition-joint answer bridge remained only weakly causal.

This probe changed the training contract so bridge contrast can apply to every
causal-prefix answer-token example, not only the first token.

## Implementation

Added:

```text
--transition-joint-answer-bridge-contrast-all-prefix-tokens
```

Default behavior stays unchanged: bridge contrast applies only to
`example_index == 0`. With the new flag, it applies to all causal-prefix answer
token examples.

## Artifacts

```text
checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_allprefix_bridge_tail_s020_from_tail_s020/last.pt

eval:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_allprefix_bridge_tail_s020_from_tail_s020/lm_causal_forced_choice_smoke8_with_baselines.jsonl
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_allprefix_bridge_tail_s020_from_tail_s020/action_code_eval32.json
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_allprefix_bridge_tail_s020_from_tail_s020/tail_error_summary_smoke8.json
```

## Result

```text
donor_only:   0/8
core_off:     0/8
bridge_off:   2/8
full core8:   2/8
action-code: 32/32
finality:     1.0000
halted_exact: 32/32
```

Tail-error breakdown:

```text
full core8:
  correct_final:     2
  pre_subtract_sum:  4
  doubled_list:      2

bridge_off:
  correct_final:     2
  pre_subtract_sum:  4
  doubled_list:      2
```

The transition/action controller is preserved, but the answer path regresses
from the previous accepted checkpoint:

```text
previous accepted:
  bridge_off: 3/8
  full:       4/8

all-prefix bridge contrast:
  bridge_off: 2/8
  full:       2/8
```

## Decision

Reject.

Naively applying bridge contrast to every causal-prefix answer token is too
strong or mis-targeted. It does not increase bridge causality and it removes the
previous full-vs-bridge-off gap. It also reintroduces earlier-stage doubled-list
errors, so the objective is not a clean final-tail fix.

## Next

Do not keep tuning this objective blindly. Build a tail-negative gate that
separately reports:

```text
correct final answer
pre-subtract sum selected
doubled-list selected
forced-choice tie
other miss
```

Then test narrower candidates, for example:

```text
1. low-weight tail-only bridge contrast;
2. answer-token final-operation marker supervision;
3. held-out tail-negative margin only when the rejected candidate is the
   pre-subtract sum.
```
