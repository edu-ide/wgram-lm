# Core Value-Delta Code Only S120

Date: 2026-05-05

## Question

Can a Discrete NAR-style `value_delta_code` bottleneck make the mandatory QTRM
recursive core learn exact role-value state updates on the held-out mixed
composition gate?

## Baseline

Canonical baseline:

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

## Prior-Backed Hypothesis

The local carry/delta MLP probes failed because exact value updates need a
discrete algorithmic execution state, not another continuous hidden-state
adapter.

Prior work behind this candidate:

```text
CLRS / neural algorithmic reasoning:
  step/state hint supervision and executable algorithm traces

Discrete Neural Algorithmic Reasoning:
  discrete code path for algorithmic state

TransNAR:
  Transformer features connected to an algorithmic reasoner

LoopFormer / RLTT / LoopRPT:
  recurrent latent trajectory should be trained and probed by depth
```

## Implementation

Files:

```text
src/wgram_lm/config.py
src/wgram_lm/wgram_model.py
src/wgram_lm/training/train.py
scripts/196_train_pure_recursive_depth_supervised.py
scripts/238_eval_qtrm_algorithmic_value_state.py
configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_core_value_delta_code_only_s120.yaml
```

Added scaffold:

```text
core_value_delta_code_logits
core_value_delta_code_gate_mean
trainable_param_policy: core_value_delta_code_only
--core-value-delta-code-ce-weight
--disable-core-value-delta-code
--use-core-value-delta-code
```

Trainable parameters:

```text
132,737 parameters
7 tensors
```

The scaffold trains only the new value-delta code path from the accepted
role-value baseline. The existing core and role-value state heads are frozen.

## Commands

Training:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src .venv/bin/python scripts/196_train_pure_recursive_depth_supervised.py \
  --config configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_core_value_delta_code_only_s120.yaml \
  --data-jsonl data/filtered/pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_len579_train40000_v0to5_mixed_only.jsonl \
  --init-checkpoint local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_core_role_value_state_joint_len579_s240_from_joint_s240/last.pt \
  --steps 120 --depth-steps 1,2,4,8 \
  --out-dir local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_core_value_delta_code_only_len579_s120_from_len579_s240 \
  --final-logit-ce-weight 0.0 --depth-final-ce-weight 0.0 --progress-margin-weight 0.0 \
  --algorithmic-role-value-state-ce-weight 1.0 \
  --core-value-delta-code-ce-weight 1.0
```

Direct code-logit readout:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src .venv/bin/python scripts/238_eval_qtrm_algorithmic_value_state.py \
  --config configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_core_value_delta_code_only_s120.yaml \
  --checkpoint local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_core_value_delta_code_only_len579_s120_from_len579_s240/last.pt \
  --data-jsonl data/eval/pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_len579_eval50000_v6to7_len7_9.jsonl \
  --out-json local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_core_value_delta_code_only_len579_s120_from_len579_s240/eval_value_code_logits_direct.json \
  --core-steps 8 --use-role-value-state --use-core-value-delta-code
```

## Results

Untrained control:

```text
value accuracy: 184/624 = 0.2948717949
step exact:      16/256 = 0.0625
trace exact:      0/32
```

Full trained path:

```text
value accuracy: 184/624 = 0.2948717949
step exact:      16/256 = 0.0625
trace exact:      0/32
```

Code-off ablation:

```text
value accuracy: 184/624 = 0.2948717949
step exact:      16/256 = 0.0625
trace exact:      0/32
```

Direct code-logit readout:

```text
value accuracy:  63/624 = 0.1009615385
step exact:       0/256 = 0.0
trace exact:      0/32
```

Action-code preservation:

```text
action-code exact: 32/32
step accuracy:      1.0
finality accuracy:  1.0
```

Depth sweep:

```text
depth 1:  64/120 = 0.5333333333, step exact 16/32
depth 2:  96/240 = 0.4000000000, step exact 16/64
depth 4: 120/368 = 0.3260869565, step exact 16/128
depth 8: 184/624 = 0.2948717949, step exact 16/256
```

## Interpretation

The new code path does not improve the held-out value-state gate. Disabling it
does not change the full result, and reading its logits directly is worse than
the baseline. This means the failure is not merely a final readout routing
problem. The learned finite code itself is not carrying the required exact
value-state transition signal.

The depth sweep also rejects raw recursive improvement: deeper recurrence does
not improve held-out value/step accuracy.

## Decision

Reject as canonical.

Keep the implementation only as a probe scaffold. Do not promote
`core_value_delta_code_only` to the main QTRM architecture.

## Next Architecture Candidate

The next candidate should be stricter and more algorithmic:

```text
canonical prompt stream
-> role binder
-> mandatory recurrent core
-> explicit typed registers
-> learned operation selector
-> deterministic or verifier-checked register update
-> role-value readout
```

This is closer to TransNAR/CLRS than to another hidden-state adapter. The key
change is that value updates must be executed as state transitions, not only
predicted as independent per-step logits.

Acceptance gate:

```text
held-out value accuracy > 184/624
step exact             > 16/256
trace exact            > 0/32
action-code exact      = 32/32
operation/register ablation drops below full
depth 8 beats depth 1 on held-out value or step metrics
```
