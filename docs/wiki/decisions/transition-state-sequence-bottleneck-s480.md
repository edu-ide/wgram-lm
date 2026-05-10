# Transition State Sequence Bottleneck S480

Date: 2026-05-05

## Claim

After the accepted mixed-composition gate, test whether the model is learning
value-bearing latent state content, not only the supervised latent action code.

Accepted baseline:

```text
checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_balanced_interleave_s720_from_s240/last.pt

action-code gate:
  full exact:         32/32
  transition-off:     0/32
  code shuffle:       0/32
  code dropout:       0/32
```

## New Probe

Added a full state-sequence evaluator:

```text
script:
  scripts/236_eval_qtrm_core_state_sequence.py

test:
  tests/test_qtrm_core_state_sequence_eval.py
```

It compares every labelled token in `depth_targets[depth]`, not only the first
token.

## Results

Baseline accepted checkpoint, `core_depth_text_logits`:

```text
state trace exact: 0/32
state step exact:  0.0000
state token acc:   0.0040

transition-state-off:
  same token acc:  0.0040
```

This means the accepted action-code checkpoint does not expose useful state
content through the existing depth text readout.

Jointly training core/answer-state-loop with staged sequence CE:

```text
checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_state_sequence_s480_from_mixed_s720/last.pt

state token acc:   0.3713
state trace exact: 0/32

action-code exact: 0/32
```

Reject: state-token accuracy rises, but the accepted recursive action policy is
destroyed.

Readout-only training:

```text
new policy:
  answer_state_loop_only

checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_state_readout_only_s480_from_mixed_s720/last.pt

state token acc:   0.2273
state trace exact: 0/32
action-code exact: 32/32
```

Reject as a state-content solution. It preserves the core but does not recover
exact state traces.

Direct transition-state sequence head:

```text
new config fields:
  model.transition_state_sequence_enabled
  model.transition_state_sequence_max_tokens

new policy:
  transition_state_sequence_only

checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_transition_state_sequence_s480_from_mixed_s720/last.pt

state token acc:   0.1418
state trace exact: 0/32
action-code exact: 32/32
```

Reject. It preserves the action-code policy but the fixed latent states still do
not decode value-bearing traces reliably.

## Decision

The current accepted QTRM primitive core is a causal latent-action controller,
not yet a value-bearing neural state-transition model.

Accepted:

```text
The core/action-code path can choose the correct operation sequence and halt.
```

Rejected:

```text
The current latent state contains enough stable numeric/list content to decode
the full internal trace under held-out values and lengths.
```

## Next Bottleneck

Stop adding readout heads. The missing capability is not another decoder; it is
value preservation inside the recurrent latent update.

Next architecture candidate:

```text
Value-Bearing Latent State Gate

1. Add compact differentiable value slots to the recursive core state.
2. Supervise slots with numeric/list state targets using bounded vocab or scalar
   digit heads, not a 248K full-vocab projection.
3. Require:
   action-code full exact remains 32/32
   value-state exact rises above readout-only baseline
   value-state ablation destroys final mixed-composition success
```

Kill criterion:

```text
If value slots improve only train rows or do not affect final answers under
ablation, reject and redesign the core transition itself.
```
