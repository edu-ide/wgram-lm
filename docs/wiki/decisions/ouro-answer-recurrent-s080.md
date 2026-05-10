# Ouro Answer Recurrent S080

Date: 2026-05-06

Status: S80 smoke accepted; S240 and choice-margin continuations regressed.

## Why This Replaces The Bridge

The role-value answer bridge failed:

```text
action-code exact: 32/32
LM causal forced-choice:
  donor_only:              0/8
  core_off:                0/8
  core_steps_8:            0/8
  core_steps_8 bridge_off: 0/8
```

The problem is not just missing value-state tokens. The answer hidden state
itself needs recurrent computation in the normal LM path.

## Ouro/LoopLM Mapping

Ouro/LoopLM suggests:

```text
decoder hidden state
-> shared recurrent transformer block
-> repeated latent computation
-> LM logits
```

QTRM candidate:

```text
prompt/donor hidden context
-> latent workspace
-> mandatory recursive core
-> answer_state_loop cross-attention
-> shared causal recurrent answer block
-> LM logits
```

This is still smaller than full Ouro pretraining. It is a falsifier that tests
whether updating the answer hidden state itself is more effective than feeding
side-state tokens into the answer loop.

## Config

```text
configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_recurrent_s080.yaml
```

Initial checkpoint:

```text
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_core_role_value_state_joint_len579_s240_from_joint_s240/last.pt
```

## Implementation

Added:

```text
answer_state_loop_recurrent_block_enabled
answer_state_loop_recurrent_layers
answer_state_loop_recurrent_gate_init_bias
answer_state_loop_recurrent_gate_min
disable_answer_state_loop_recurrent
```

The recurrent block is parameter-shared across answer-state-loop depth steps.
It runs over the answer hidden sequence and feeds the same LM head.

## Acceptance

Promote only if:

```text
action-code exact remains 32/32
core_steps_8 LM forced-choice > donor_only
core_steps_8 LM forced-choice > core_off
core_steps_8 answer_state_recurrent_off drops versus full
```

Reject if final CE improves in training but held-out LM forced-choice remains
0/8 or recurrent-off matches full.

## Result

Run S80:

```text
checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_recurrent_len579_s080_from_len579_s240/last.pt

training signal:
  final_path_ce: 10.9580 -> 2.6068 on logged batches
  final_path_acc: up to 0.5000 on the final logged batch
  transition_state_joint_acc: 1.0000 on logged batches

action-code heldout:
  rows:          32
  exact:         32/32
  step_acc:       1.0000
  finality_acc:   1.0000
  halted_exact:  32/32

LM causal forced-choice smoke8:
  donor_only:                         0/8
  core_off:                           0/8
  core_steps_8:                       2/8
  core_steps_8 answer_recurrent_off:  0/8

decision:
  Accept as the first causal answer-path gain. Do not claim general reasoning
  yet. Scale to S240/S480 and larger heldout before promoting to canonical.
```

## Interpretation

This is the first probe in this branch where all three required signals appear
together:

```text
the action-code policy is preserved
the normal LM answer path beats donor/core-off
the gain disappears when the recurrent answer block is disabled
```

The result supports the Ouro/LoopLM hypothesis more than bridge-style side
tokens. The model still misses 6/8 smoke cases, so the next step is not SubQ/MSA
long-context scaling. The next step is to scale this recurrent answer path and
retest the same ablation.

## S240 Continuation

Continuation command used S80 as the init checkpoint and trained 160 more steps
with the same loss.

```text
checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_recurrent_len579_s240_from_s080/last.pt

training signal:
  final_path_ce: 4.3030 -> 1.0415 on logged batches
  final_path_acc: up to 1.0000, final logged batch 0.7500
  transition_state_joint_acc: 1.0000

action-code heldout:
  rows:          32
  exact:         32/32
  step_acc:       1.0000
  finality_acc:   1.0000
  halted_exact:  32/32

LM causal forced-choice smoke8:
  donor_only:                         0/8
  core_off:                           0/8
  core_steps_8:                       0/8
  core_steps_8 answer_recurrent_off:  0/8
```

S240 is rejected as an answer-path scale-up. It improves in-sample CE but loses
the S80 held-out causal answer gain. The failure mode changes from intermediate
list strings to near-final distractors such as `300024`, `300036`, `400040`,
and `400056`, which means the recurrent path is learning task surface structure
without stable final-value selection.

## Choice-Margin Continuation

The next test added sequence choice-margin pressure from the existing
`choices` field, with `--save-every 40`, and continued from the accepted S80
checkpoint.

```text
checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_recurrent_choice_margin_len579_s080_from_s080/step_000080.pt

training signal:
  final_path_ce: down to 0.9753 on the final logged batch
  final_path_acc: 1.0000 on the final logged batch
  transition_state_joint_acc: 1.0000
  choice_sequence_margin_final_path: 0.1333

action-code heldout at step80:
  rows:          32
  exact:         32/32
  step_acc:       1.0000
  finality_acc:   1.0000
  halted_exact:  32/32

LM causal forced-choice smoke8:
  step40 core_steps_8:                      0/8
  step40 core_steps_8 answer_recurrent_off: 0/8
  step80 core_steps_8:                      0/8
```

Decision: reject this continuation as a canonical objective. The sequence
choice-margin implementation is useful infrastructure, but the experiment
confirms the same failure class as S240: action-code/transition state stays
perfect while the held-out LM answer path loses the S80 causal gain.

## Next

```text
1. Keep S80 as the best observed recurrent-answer checkpoint.
2. Do not continue the same CE-only objective blindly.
3. Use validation-gated checkpoint selection; do not accept lower in-sample CE
   without held-out LM causal forced-choice improvement.
4. Replace local objective tuning with a root answer-path candidate: recurrent
   value-state credit, selective context routing, or a SubQ/MSA-like sparse
   router that still feeds the universal LM causal path.
5. If the answer path still cannot select final values stably, add the
   SubQ/MSA-like selective context router as the next causal information-routing
   candidate.
```

Implementation note:

```text
scripts/196_train_pure_recursive_depth_supervised.py now accepts --save-every.
Use --save-every 40 for this family so step_000040.pt, step_000080.pt, and
later snapshots can be evaluated before accepting a longer run.

The same script can derive rejected choice-margin targets from `choices` when a
row has no explicit `rejected` field. This is experimental infrastructure, not
an accepted training recipe for this architecture family.
```
