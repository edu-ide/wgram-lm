# Role-Value Answer Bridge S120

Date: 2026-05-05

Status: rejected after S80 falsification.

## Failure Ledger

Failure:

```text
Final-answer-only CE preserves the action-code policy but does not improve
held-out LM causal forced-choice. The model still prefers intermediate state
strings over final scalar answers.
```

Evidence:

```text
Final Answer Bridge S120:
  action-code exact: 32/32
  LM causal forced-choice smoke8:
    donor_only:   0/8
    core_off:     0/8
    core_steps_1: 0/8
    core_steps_8: 0/8
```

Known limitation class:

```text
Non-causal probe success. The core can emit useful action/value telemetry, but
the LM answer path is not forced to consume the computed value state.
```

Could the big structure be wrong?:

```text
Yes, if role-value states remain side heads. They must become internal tokens
read by the same answer-state loop that produces LM logits, or the architecture
is only a probe scaffold.
```

Information path needed:

```text
prompt tokens
-> frozen donor hidden context
-> latent workspace
-> mandatory recursive core
-> role-value state logits
-> soft value-state tokens
-> answer-state loop
-> LM logits
```

Current candidate:

```text
core_role_value_state_logits
-> softmax over value vocabulary per role
-> learned value embedding + learned role embedding
-> gated role-value answer tokens
-> answer_state_loop cross-attention
-> LM logits
```

The bridge is not an executor and does not compute the final answer outside the
model. It is a learned, ablatable internal bottleneck that must improve the
universal LLM causal path.

## LoopLM/Ouro Check

`Scaling Latent Reasoning via Looped Language Models` is more relevant to this
probe than Samsung TRM because Ouro is a general decoder-only LLM with recurrent
latent computation trained through the LM path.

QTRM implication:

```text
Do not copy TRM's fixed-output puzzle answer update literally.
Move toward LoopLM-style recurrent hidden-state updates that feed LM logits.
```

This bridge is therefore only a small falsifier. It is promoted only if the
role-value state becomes causal for the normal answer logits. If it fails, the
next candidate should be a deeper Ouro-style answer-state recurrent block with
depth regularization, not another side head.

## Config

```text
configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_role_value_answer_bridge_s120.yaml
```

Initial checkpoint:

```text
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_core_role_value_state_joint_len579_s240_from_joint_s240/last.pt
```

## Training Objective

First falsifier:

```text
causal-prefix final answer CE
+ algorithmic role-value state CE
+ transition_state_joint CE
```

Depth CE is disabled in the first smoke to keep the falsifier short. The code
path still supports role-value bridge tokens in both final and depth
answer-state-loop logits.

## Acceptance

Promote only if all hold:

```text
action-code exact remains 32/32
LM causal forced-choice core_steps_8 > donor_only
LM causal forced-choice core_steps_8 > core_off
LM causal forced-choice drops when role_value_answer_bridge_off
```

Reject if the bridge trains in-sample but held-out LM forced-choice remains
0/8 or does not drop under `role_value_answer_bridge_off`.

## Implementation

Added fields:

```text
core_role_value_state_answer_bridge_enabled
core_role_value_state_answer_bridge_gate_init_bias
core_role_value_state_answer_bridge_gate_min
```

Added ablation:

```text
qtrm_core_steps_8_role_value_answer_bridge_off_no_evidence
```

Tested contract:

```text
enabled bridge changes answer_state_loop_logits
enabled bridge changes core_depth_last_logits
enabled bridge changes core_depth_text_logits
disabled bridge returns an empty gate tensor
```

## Result

Checkpoint:

```text
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_role_value_answer_bridge_len579_s080_from_len579_s240/last.pt
```

Training signal:

```text
final_path_ce: 10.95 -> 4.39 on logged batches
algorithmic_role_value_state_step_exact: up to 0.75 on the final logged batch
transition_state_joint_acc: 1.0 on the final logged batch
```

Held-out action-code gate:

```text
rows:          32
exact:         32/32
step_acc:       1.0000
finality_acc:   1.0000
halted_exact:  32/32
```

Held-out LM causal forced-choice smoke8:

```text
donor_only:                   0/8
core_off:                     0/8
core_steps_8:                 0/8
core_steps_8 bridge_off:      0/8
```

Observed completions:

```text
donor_only:
  intermediate list strings such as 100004,100008,100012

core_off:
  __FORCED_CHOICE_TIE__

core_steps_8:
  same intermediate list strings plus scalar-like distractors such as 400040

core_steps_8 bridge_off:
  same pattern as core_steps_8
```

## Decision

Reject.

The bridge is trainable and preserves the action-code policy, but it does not
improve held-out LM forced-choice and it does not produce a bridge-off causal
drop. The core still solves the action/finality trace as telemetry, while the
normal answer logits do not consume the value state enough to select the final
scalar answer.

## Next

Do not scale this bridge to S120/S480 as the next move. The next candidate
should follow the Ouro/LoopLM direction more directly:

```text
prompt hidden state
-> mandatory recurrent answer-state block
-> shared-depth hidden-state updates
-> entropy/depth regularized halt or allocation head
-> LM logits
```

Promotion requires depth-8 to beat donor/core-off and to degrade under
recurrent-answer-state-off on held-out LM causal forced-choice.
