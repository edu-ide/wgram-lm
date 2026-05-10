# Ouro Subtract-Tail Counterfactual S020 Reject

Status: rejected probe, 2026-05-06.

## Question

The accepted tail-negative checkpoint still fails four held-out mixed
composition cases by choosing the pre-subtract sum:

```text
sum(doubled_values) instead of sum(doubled_values) - offset
```

The trajectory-monotonic probe showed that adjacent target log-prob improvement
was not the active bottleneck. This probe asks whether a tighter counterfactual
negative set around the subtract tail can make the final answer path keep the
last operation.

## Implementation

Added training options:

```text
--subtract-tail-counterfactual-margin-weight
--subtract-tail-counterfactual-margin
--subtract-tail-counterfactual-family-filter
```

Added helpers:

```text
subtract_tail_counterfactual_rejected_texts(...)
subtract_tail_counterfactual_sequence_margin_loss(...)
```

For mixed list arithmetic, the rejected set is:

```text
preterminal sum
final_answer - 1
final_answer + 1
```

The loss is train-only and stays on the universal LLM path:

```text
prompt -> donor hidden states -> recursive core / answer-state loop
-> LM logits
```

No runtime solver, MemoryOS, retrieval, or hidden answer channel is added.

## Artifacts

```text
runner:
  scripts/243_run_qtrm_ouro_subtract_tail_counterfactual_s020.sh

checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_subtract_tail_counterfactual_s020_from_tail_s020/last.pt
  deleted after rejection to recover local disk; eval JSON artifacts retained.

eval:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_subtract_tail_counterfactual_s020_from_tail_s020/lm_causal_forced_choice_smoke8_with_baselines.jsonl
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_subtract_tail_counterfactual_s020_from_tail_s020/action_code_eval32.json
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_subtract_tail_counterfactual_s020_from_tail_s020/tail_error_summary_smoke8.json
```

## Result

Held-out smoke8 causal forced-choice:

```text
donor_only:   0/8
core_off:     0/8
core_steps 1: 2/8
core_steps 2: 2/8
core_steps 4: 4/8
full core8:   3/8
bridge_off:   2/8
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
core_steps 4:
  correct_final:     4
  pre_subtract_sum:  4

full core8:
  correct_final:     3
  pre_subtract_sum:  5

bridge_off:
  correct_final:     2
  pre_subtract_sum:  6
```

## Decision

Reject as canonical.

The bridge ablation becomes causal again, but the full model regresses from the
accepted 4/8 baseline to 3/8. The result is therefore not a raw-intelligence
improvement.

## New Bottleneck

This probe reveals a more precise failure:

```text
depth 4 can match the accepted 4/8 score, but running to depth 8 worsens the
answer path to 3/8.
```

That points to recursive-depth overshoot rather than lack of hard negatives.
The next candidate should test a learned or supervised halt/readout policy that
uses the transition finality signal to stop answer-state updates when the
latent trace is already terminal.

Acceptance gate for the next candidate:

```text
dynamic halt/readout must beat or match core_steps4 while core_off and
halt-off/bridge-off drop. If dynamic halt only wraps a fixed depth-4 shortcut
without causal dependence, reject.
```
