# Core Role-Value State Prompt-Extract S240

Date: 2026-05-05

## Question

Does adding a role-conditioned prompt extraction cross-attention before the
mandatory recurrent core improve exact value binding?

## Candidate

Add gated role-token extraction:

```text
role embeddings
  + gated cross-attention over prompt token states
  -> role-value tokens
  -> mandatory recurrent core
```

The gate is initialized low so the new path starts close to the previous
accepted scaffold.

Config:

```text
configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_core_role_value_state_prompt_extract_joint_s120.yaml
```

Checkpoint:

```text
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_core_role_value_state_prompt_extract_len579_s240_from_len579_s240/last.pt
```

## Result

Held-out core role-value:

```text
rows:                 32
trace exact:          0/32
step exact:           16/256 = 0.0625
value accuracy:       130/624 = 0.2083
```

Action-code preservation:

```text
trace exact:          32/32
halted exact:         32/32
step accuracy:        1.0000
finality accuracy:    1.0000
```

Comparison:

```text
best len579 scaffold: 0.2949 value accuracy, 16/256 step exact
prompt-extract path:  0.2083 value accuracy, 16/256 step exact
```

## Decision

Reject this prompt-extract candidate as implemented.

The candidate preserves action-code behavior, but it regresses value accuracy.
It should not replace the current best checkpoint.

## Interpretation

The failure does not prove that prompt-to-role extraction is unnecessary. It
shows that a newly initialized cross-attention path inserted before the core is
not enough under this short training schedule. It may perturb role-token
geometry more than it helps binding.

Next candidates should prefer lower-risk binding pressure:

```text
1. Train longer from the current best len579 scaffold without new modules.
2. Add explicit step-to-step role-transition supervision.
3. If adding prompt extraction again, pretrain the extractor against role
   targets before letting it affect the recurrent core.
```
