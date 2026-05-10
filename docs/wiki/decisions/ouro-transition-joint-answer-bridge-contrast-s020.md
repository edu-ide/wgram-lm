# Ouro Transition Joint Answer Bridge Contrast S020

Status: accepted smoke probe, 2026-05-06.

## Question

The plain transition-joint answer bridge made the answer-state loop see the
transition trace, but held-out full and bridge-off both scored 2/8. This showed
that visibility alone is not causality.

This probe adds explicit bridge contrast:

```text
loss += relu(margin - (logp_gold(full) - logp_gold(bridge_off)))
```

The target is not higher train CE performance. The target is a held-out
ablation drop: full must beat bridge-off while preserving the learned
transition controller.

## Implementation

Added training infrastructure:

```text
--transition-joint-answer-bridge-contrast-weight
--transition-joint-answer-bridge-contrast-margin

transition_joint_answer_bridge_contrastive_loss(...)
```

During training, the script runs a second forward pass with:

```text
disable_transition_state_joint_answer_bridge=True
```

and contrasts the gold answer log-probability against the full path.

## Artifacts

```text
config:
  configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_joint_answer_bridge_s020.yaml

checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_joint_answer_bridge_contrast_len579_s020_from_s080/last.pt

eval:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_joint_answer_bridge_contrast_len579_s020_from_s080/lm_causal_forced_choice_smoke8.jsonl
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_joint_answer_bridge_contrast_len579_s020_from_s080/action_code_eval32.json
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_joint_answer_bridge_contrast_len579_s020_from_s080/gate_selection.json
```

## Result

Training signal:

```text
final_path_ce: 4.0516 -> 2.6656 on logged batches
bridge contrast delta: -0.0525 -> -0.0059 on logged batches
transition_state_joint_acc: 0.7500 -> 0.8750 on logged batches
```

Held-out gate:

```text
LM causal forced-choice smoke8:
  full:       2/8
  bridge off: 0/8

action-code:
  exact:       32/32
  step_acc:     1.0000
  finality_acc: 1.0000

gate selector:
  s020_joint_answer_bridge_contrast: accepted
  s080_recurrent_baseline: accepted
```

## Decision

Accept as a smoke probe, not as a quality improvement.

This does not beat the S80 baseline on full accuracy, so it is not a better
checkpoint yet. It is still important because it converts the transition-joint
answer bridge from an ignorable side input into a held-out causal dependency:
bridge-off drops from 2/8 to 0/8 while action-code remains 32/32.

## Next

Scale this only under validation gates:

```text
1. save checkpoints every 20 or 40 steps;
2. require full > bridge_off and action-code 32/32;
3. reject any continuation where full regresses to 0/8 or bridge-off matches full;
4. only after the smoke is stable, expand beyond 8 held-out cases.
```

## S080 Continuation From S020

Continued the accepted S020 bridge-contrast checkpoint for 60 more steps with
`--save-every 20`.

Artifacts:

```text
checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_joint_answer_bridge_contrast_len579_s080_from_s020/step_000060.pt

eval:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_joint_answer_bridge_contrast_len579_s080_from_s020/lm_causal_forced_choice_smoke8_step20.jsonl
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_joint_answer_bridge_contrast_len579_s080_from_s020/lm_causal_forced_choice_smoke8_step60.jsonl
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_joint_answer_bridge_contrast_len579_s080_from_s020/action_code_eval32_step60.json
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_joint_answer_bridge_contrast_len579_s080_from_s020/gate_selection_step60.json
```

Result:

```text
step20:
  full:       2/8
  bridge off: 0/8

step60:
  full:       2/8
  bridge off: 0/8
  action-code: 32/32
```

Decision:

```text
Accept as causal-preservation only.
Do not claim quality improvement.
```

The bridge dependency survives to the S080 continuation, but full answer
accuracy remains at 2/8. The next experiment should not simply keep extending
this same objective. It should add a validation-gated quality pressure, such as
choice-candidate contrast in addition to bridge contrast, and reject any
checkpoint that does not exceed the 2/8 smoke ceiling.
