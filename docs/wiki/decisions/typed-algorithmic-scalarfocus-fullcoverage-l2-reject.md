# Typed Algorithmic Scalarfocus Fullcoverage L2 Reject

Date: 2026-05-08

## Question

Can short scalar-focused full-coverage training turn the typed algorithmic
value-state codec into a causal L2 local gate for mixed list-to-arithmetic
reasoning?

Target level:

```text
L2 local gate
```

Required signal:

```text
full typed state > typed-head-off
full typed state > recurrent-off
full typed state > typed-register-executor-off
scalar_residual / final_residual must be non-zero on held-out cases
```

The important distinction is whether the gain comes from recurrent latent
reasoning or from a static auxiliary readout.

## Run

Checkpoint:

```text
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_typed_algorithmic_scalarfocus_len1113_s576_fullcoverage/last.pt
```

Training command:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src \
  .venv/bin/python scripts/196_train_pure_recursive_depth_supervised.py \
  --config configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_typed_algorithmic_value_state_s080.yaml \
  --data-jsonl data/filtered/pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_len1113_probe_train40000_v0to5_mixed_only.jsonl \
  --init-checkpoint local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_typed_algorithmic_value_state_len1113_s080_from_joint_s080/last.pt \
  --steps 576 \
  --depth-steps 1,2,4,8 \
  --transition-state-joint-ce-weight 1.0 \
  --typed-algorithmic-value-state-ce-weight 1.0 \
  --typed-algorithmic-kind-ce-multiplier 0.25 \
  --typed-algorithmic-list-ce-multiplier 0.25 \
  --typed-algorithmic-scalar-ce-multiplier 6.0 \
  --final-logit-ce-weight 0.0 \
  --depth-final-ce-weight 0.0 \
  --all-depth-ce-weight 0.0 \
  --progress-margin-weight 0.0
```

This fixed the previous schedule issue where a short run saw only a small part
of the scalar-focused training surface.

## Held-Out Results

Held-out full:

```text
rows:                   32
trace_exact:             0/32
step_exact:              0/256
field_accuracy:        464/1120 = 0.4143
content_field_accuracy:432/1024 = 0.4219
```

Typed head disabled:

```text
field_accuracy:          96/1120 = 0.0857
content_field_accuracy:   0/1024 = 0.0000
```

Recurrent-off:

```text
field_accuracy:        464/1120 = 0.4143
content_field_accuracy:432/1024 = 0.4219
```

Typed-register-executor-off:

```text
field_accuracy:        464/1120 = 0.4143
content_field_accuracy:432/1024 = 0.4219
```

Field breakdown for full:

```text
kind:                 256/256 = 1.0000
raw_list_offsets:      80/224 = 0.3571
doubled_list_offsets:  80/224 = 0.3571
scalar_coeff:          48/192 = 0.2500
scalar_residual:        0/192 = 0.0000
final_residual:         0/32  = 0.0000
```

The scalar predictions are still mostly constant-output shortcuts:

```text
scalar_coeff:    target 13 -> predicted 11 for 96 examples
scalar_residual: many targets -> predicted 25
final_residual:  targets -> predicted 25 or 34
```

## Decision

Reject as L2 recursive-state progress.

The typed head clearly carries some easy field information, because disabling
the entire typed head drops content accuracy to zero. However, disabling the
recurrent typed-state path or the typed-register executor does not change the
score at all:

```text
full == recurrent_off == executor_off
```

Therefore the gain is not causal to recursive latent reasoning. It is a static
auxiliary readout that learns `kind` and some list-offset patterns, but it does
not learn exact scalar residuals or final answers.

## Root Diagnosis

The blocker is no longer train coverage. The full-coverage run saw the intended
surface and still failed on the exact scalar fields.

Current bottleneck:

```text
value-state representation is not forced through a recurrent causal state that
must update and preserve exact scalar values before final answer selection.
```

The architecture claim remains probe-only until an ablation shows:

```text
core/recurrent-on improves scalar_residual or final_residual
core/recurrent-off drops
final answer path also drops when the same causal state is removed
```

## Next

Do not spend more steps on this static typed-state CE path.

Shortest path:

```text
1. Use the accepted transition/action controller as the stable scaffold.
2. Rebuild a reproducible Ouro/LoopLM-style answer recurrent checkpoint.
3. Train with validation-gated checkpoint selection, not final training loss.
4. Promote only if held-out causal forced-choice beats donor/core-off and
   answer-recurrent-off drops.
5. Only after that, reconnect a value-state codec to the answer recurrent path.
```

This keeps the canonical path as:

```text
prompt tokens -> donor/QTRM hidden states -> mandatory recursive core
-> recurrent answer hidden state -> LM logits
```

and avoids turning typed CE into a hidden calculator or non-causal answer
channel.
