# Typed Register Prompt Binder S120

Date: 2026-05-05

Status: rejected experiment.

## Question

Did typed-register v1/v2 fail because the register path could not directly
read exact prompt values?

## Root Architecture Hypothesis

The v1/v2 typed-register path used a weak context summary:

```text
context tokens -> mean pooled context -> operation selector/register update
```

If exact value binding was the blocker, then opening an existing prompt-token
cross-attention binder before the recurrent core should improve held-out
role-value accuracy.

## Implementation

Added a narrow trainable policy:

```text
trainable_param_policy: core_typed_register_executor_and_prompt_extract
```

Trainable prefixes:

```text
core_typed_register_*
core_role_value_state_prompt_*
```

Config:

```text
configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_typed_register_prompt_binder_s120.yaml
```

Training objective:

```text
core_typed_register_value CE
+ core_typed_register_operation CE
```

No external solver is used at inference. Prompt extraction is inside the
canonical token path.

## Result

Checkpoint:

```text
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_typed_register_prompt_binder_len579_s120_from_len579_s240/last.pt
```

Held-out full:

```text
value accuracy: 104/624 = 0.1666666667
step exact:       0/256 = 0.0
trace exact:      0/32
```

Typed-register-off:

```text
value accuracy: 168/624 = 0.2692307692
step exact:       0/256 = 0.0
trace exact:      0/32
```

Action-code preservation:

```text
exact:        32/32
step acc:      1.0
finality acc:  1.0
halted exact: 32/32
```

Baseline before prompt-binder:

```text
value accuracy: 184/624 = 0.2948717949
step exact:      16/256 = 0.0625
```

## Decision

Reject as canonical.

Prompt-token access alone does not fix value reasoning, and training the prompt
binder also degrades the underlying role-value baseline:

```text
baseline:                 184/624
prompt-binder full:       104/624
prompt-binder register-off 168/624
```

This falsifies the simple "mean context was the only blocker" hypothesis.

## Next Hypothesis

The blocker is not just prompt access. The model still lacks a supervised,
causal recurrent transition objective for exact value updates.

Next candidate should avoid another independent readout head and instead add:

```text
depth t register state -> predicted depth t+1 register state
latent lookahead/process reward over depths 1/2/4/8
teacher-forced transition consistency before free rollout
held-out full/off/depth-sweep gate
```

Acceptance must require:

```text
full > 184/624
typed-register-off < full
trace exact > 0/32
action-code exact = 32/32
```
