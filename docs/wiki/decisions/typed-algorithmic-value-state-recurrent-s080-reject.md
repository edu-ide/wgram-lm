# Typed Algorithmic Value-State Recurrent S080 Reject

Date: 2026-05-08

## Question

Can pushing typed algorithmic value-state into a recurrent update improve
held-out numeric/register binding over the previous typed readout head?

The prior non-recurrent typed head reached:

```text
content-field accuracy: 352/1024 = 0.34375
trace exact:              0/32
```

## Setup

Config:

```text
configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_typed_algorithmic_value_state_recurrent_s080.yaml
```

Init:

```text
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_typed_algorithmic_value_state_len1113_s080_from_joint_s080/last.pt
```

Output:

```text
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_typed_algorithmic_value_state_recurrent_len1113_s080_from_typed_s080/last.pt
```

Training:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src .venv/bin/python \
  scripts/196_train_pure_recursive_depth_supervised.py \
  --config configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_typed_algorithmic_value_state_recurrent_s080.yaml \
  --data-jsonl data/filtered/pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_len1113_probe_train40000_v0to5_mixed_only.jsonl \
  --init-checkpoint local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_typed_algorithmic_value_state_len1113_s080_from_joint_s080/last.pt \
  --steps 80 --depth-steps 1,2,4,8 \
  --out-dir local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_typed_algorithmic_value_state_recurrent_len1113_s080_from_typed_s080 \
  --transition-state-joint-ce-weight 1.0 \
  --typed-algorithmic-value-state-ce-weight 1.0 \
  --final-logit-ce-weight 0.0 \
  --depth-final-ce-weight 0.0 \
  --all-depth-ce-weight 0.0 \
  --progress-margin-weight 0.0
```

## Held-Out Eval

Eval split:

```text
data/eval/pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_len1113_probe_eval60000_v6to7_len11_13_mixed_only.jsonl
```

Full recurrent typed-state:

```text
rows:                   32
trace exact:             0/32
step exact:              0/256
field accuracy:        336/1120 = 0.30000
content-field accuracy:304/1024 = 0.296875
```

Recurrent-off ablation:

```text
rows:                   32
trace exact:             0/32
step exact:              0/256
field accuracy:        384/1120 = 0.342857
content-field accuracy:352/1024 = 0.34375
```

## Decision

Reject as canonical.

The recurrent update is on the measured path, but it hurts held-out typed value
state:

```text
full recurrent: 0.296875 content accuracy
recurrent-off:  0.343750 content accuracy
```

This matches the state-conditioned renderer result:

```text
adding more state heads/adapters does not fix exact value preservation.
```

## Next Constraint

Do not continue with larger readout heads or larger soft-prefix bridges until
the value codec itself is simplified and made exact enough.

Next candidate should use a minimal affine/register state machine:

```text
prompt tokens
-> recurrent core
-> explicit typed registers:
     list_base
     even_count
     doubled_sum_coeff
     residual
     finality
-> answer path
```

Acceptance must require:

```text
state trace exact > 0 on held-out range
full answer exact > state_off/core_off
no external solver computes the answer
```

