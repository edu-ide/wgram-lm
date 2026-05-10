# Typed Register Executor Only S120

Date: 2026-05-05

## Question

Can a TransNAR/CLRS-style typed register executor improve QTRM role-value state
reasoning while preserving the universal LLM causal path?

## Universal LLM Path Constraint

This candidate preserves the canonical path:

```text
prompt/chat-template tokens
-> tokenizer
-> frozen donor hidden states
-> QTRM workspace + mandatory recurrent core
-> learned operation selector + typed register state
-> role-value readout
-> LM answer path
```

No external solver computes the final answer. The verifier/targets only train
or evaluate the internal state.

## Baseline

Canonical value-state baseline:

```text
checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_core_role_value_state_joint_len579_s240_from_joint_s240/last.pt

value accuracy:
  184/624 = 0.2948717949

step exact:
  16/256 = 0.0625

trace exact:
  0/32

action-code exact:
  32/32
```

## Implementation

Added model scaffold:

```text
core_typed_register_executor_enabled
core_typed_register_num_operations
core_typed_register_hidden_dim
core_typed_register_gate_init_bias
core_typed_register_gate_min

outputs:
  core_typed_register_operation_logits
  core_typed_register_value_logits
  core_typed_register_gate_mean
```

Mechanism:

```text
core trajectory + role tokens
-> operation logits
-> soft operation embedding
-> persistent typed register state
-> gated recurrent register update
-> role-value logits
```

This differs from the rejected `core_value_delta_code` path because it has a
persistent register state across recurrent steps, not independent per-step
value logits.

Config:

```text
configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_typed_register_executor_only_s120.yaml
```

Training policy:

```text
trainable_param_policy: core_typed_register_executor_only
trainable params: 605,835
```

Training hook:

```text
scripts/196_train_pure_recursive_depth_supervised.py
--core-typed-register-ce-weight
```

Eval ablation:

```text
scripts/238_eval_qtrm_algorithmic_value_state.py
--disable-core-typed-register-executor
```

## Command

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src .venv/bin/python scripts/196_train_pure_recursive_depth_supervised.py \
  --config configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_typed_register_executor_only_s120.yaml \
  --data-jsonl data/filtered/pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_len579_train40000_v0to5_mixed_only.jsonl \
  --init-checkpoint local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_core_role_value_state_joint_len579_s240_from_joint_s240/last.pt \
  --steps 120 --depth-steps 1,2,4,8 \
  --out-dir local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_typed_register_executor_only_len579_s120_from_len579_s240 \
  --final-logit-ce-weight 0.0 --depth-final-ce-weight 0.0 --progress-margin-weight 0.0 \
  --core-typed-register-ce-weight 1.0
```

## Results

Untrained typed-register path:

```text
value accuracy: 0/624 = 0.0
step exact:     0/256 = 0.0
trace exact:    0/32
```

Trained full path:

```text
value accuracy: 106/624 = 0.1698717949
step exact:       0/256 = 0.0
trace exact:      0/32
```

Typed-register-off ablation:

```text
value accuracy: 184/624 = 0.2948717949
step exact:      16/256 = 0.0625
trace exact:      0/32
```

Action-code preservation:

```text
action-code exact: 32/32
step accuracy:      1.0
finality accuracy:  1.0
```

## Interpretation

The path is causal but harmful:

```text
typed-register full: 106/624
typed-register off:  184/624
```

This proves the new executor is on the measured value path, but the current
training signal is not enough to learn useful held-out register updates.
Importantly, the action-code controller remains intact, so the failure is
localized to value register learning rather than global model damage.

## Decision

Reject `typed_register_executor_only_s120` as canonical.

Keep the scaffold because it preserves the universal LLM path and provides an
ablatable internal register mechanism. Do not promote it until it beats the
184/624 baseline.

## Next Candidate

The next candidate should train the operation selector and register update
more explicitly:

```text
operation CE from solver/transition labels
+ register-update consistency loss
+ role-value CE
+ typed-register-off ablation
```

The key change is to stop asking a randomly initialized register executor to
discover operation semantics from value CE alone. CLRS/TransNAR-style process
supervision should provide operation/state hints while preserving the same
prompt-token-to-logits LLM path.

Acceptance gate:

```text
full value accuracy > 184/624
step exact          > 16/256
trace exact         > 0/32
action-code exact   = 32/32
typed-register-off  < full
operation-off/shuffle < full
```
