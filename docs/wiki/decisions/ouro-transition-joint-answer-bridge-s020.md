# Ouro Transition Joint Answer Bridge S020

Status: rejected probe, 2026-05-06.

## Question

The S80 Ouro recurrent checkpoint has a correct transition-state joint trace
on the 32-case action-code smoke, but the LM answer path only reaches 2/8.
The finality selector showed that selecting a depth after the answer loop does
not help. This probe tests a stronger causal path:

```text
transition_state_joint_logits[depth]
-> soft joint-code distribution
-> trainable projection to d_model
-> gated delta inside answer-state recurrence
-> LM logits
```

## Implementation

Added:

```text
transition_state_joint_answer_bridge_enabled
transition_state_joint_answer_gate_init_bias
transition_state_joint_answer_gate_min
disable_transition_state_joint_answer_bridge
```

The bridge is not an external answer channel. It is derived from the learned
transition joint logits and modifies the answer hidden state before the shared
LM head.

## Artifacts

```text
config:
  configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_joint_answer_bridge_s020.yaml

checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_joint_answer_bridge_len579_s020_from_s080/last.pt

eval:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_joint_answer_bridge_len579_s020_from_s080/lm_causal_forced_choice_smoke8.jsonl
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_joint_answer_bridge_len579_s020_from_s080/action_code_eval32.json
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_joint_answer_bridge_len579_s020_from_s080/gate_selection.json
```

## Result

Training signal:

```text
final_path_ce: 3.9788 -> 2.6370 on logged batches
transition_state_joint_acc: 0.7500 -> 0.8750 on logged batches
```

Held-out gate:

```text
LM causal forced-choice smoke8:
  full:       2/8
  bridge off: 2/8

action-code:
  exact:       32/32
  step_acc:     1.0000
  finality_acc: 1.0000

gate selector:
  s080_recurrent_baseline: accepted smoke baseline
  s020_joint_answer_bridge: rejected, ablation_drop_below_min
```

## Decision

Reject as a canonical architecture upgrade.

The bridge preserves the correct transition controller but does not create a
measurable answer-path dependency. Lower train CE is again not accepted because
the held-out full path and bridge-off path are identical on the smoke gate.

## Interpretation

The bottleneck is not only "the answer path cannot see the transition trace."
It can now see it, but the objective does not force the answer recurrence to
use it for value computation. The next candidate must create causal credit for
the recurrent answer computation itself, not just add another side input.

Candidate directions:

```text
1. Bridge-contrast objective:
   train full > bridge_off on choice candidates, so the bridge must carry
   useful information instead of becoming ignorable.

2. Depth-wise answer-process supervision:
   supervise intermediate answer logits from the same answer recurrence against
   solver trace states, while preserving the final held-out gate.

3. Smaller controlled task first:
   prove bridge causality on a tiny held-out arithmetic subset before touching
   the full mixed-composition split again.
```
