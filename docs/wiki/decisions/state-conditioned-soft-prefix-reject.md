# State-Conditioned Soft-Prefix Reject

Date: 2026-05-08

## Question

Can explicit QTRM value-state logits make the donor soft-prefix renderer produce
the correct final answer?

This tested the stricter causal path:

```text
prompt tokens
-> frozen donor hidden states
-> mandatory QTRM recurrent core
-> core_role_value_state_logits
-> learned donor soft-prefix
-> donor LM answer tokens
```

The decisive ablation is:

```text
full > donor_only
full > core_off
full > state_off
```

If `state_off` is as good as, or better than, `full`, the value-state is not
causally helping answer rendering.

## Implementation

Extended `scripts/304_train_core_soft_prefix_donor.py` with optional explicit
state features:

```text
--state-logits-key core_role_value_state_logits
--state-feature-mode softmax | logits | argmax_onehot
```

The adapter now supports:

```text
core_hidden + state_features -> virtual donor embedding prefix
```

and evaluates four modes:

```text
donor_only_no_evidence
soft_core_off_no_evidence
soft_state_off_no_evidence
soft_full_no_evidence
```

## Experiment A: Arithmetic Smoke

Checkpoint:

```text
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_core_role_value_state_joint_len579_s240_from_joint_s240/last.pt
```

Data:

```text
train: data/filtered/pure_recursive_reasoning_arith_chain_train64_start18.jsonl
eval:  data/eval/pure_recursive_reasoning_arith_chain_heldout4.jsonl
```

Result:

```text
state_dim: 1280
teacher-forced full:      0.8750
teacher-forced core_off:  0.5625
teacher-forced state_off: 0.9375
generation full exact:    0/4
```

Decision: reject. The explicit state features hurt teacher-forced rendering.

## Experiment B: Native Mixed-Composition Distribution

Data:

```text
train: data/filtered/pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_len579_train40000_v0to5_mixed_only.jsonl
eval:  data/eval/pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_len579_eval50000_v6to7_len7_9.jsonl
```

Softmax features:

```text
teacher-forced full:      0.6094
teacher-forced core_off:  0.5859
teacher-forced state_off: 0.5938
generation full exact:    0/16
```

Normalized logit features:

```text
teacher-forced full:      0.5703
teacher-forced core_off:  0.5703
teacher-forced state_off: 0.5859
generation full exact:    0/16
```

Generated answers stayed in the wrong numeric pattern, for example:

```text
expected 300015, generated 240013
expected 400037, generated 240013
```

## Interpretation

The bottleneck is not just the donor soft-prefix adapter.

Current `core_role_value_state_logits` uses compact role/value vocabularies and
does not preserve enough exact numeric information to render large final
answers. The donor LM can learn a local numeric style, but it does not recover
the underlying arithmetic state.

A follow-up using the more structured typed algorithmic fields also rejected:

```text
typed_algorithmic_* -> soft-prefix -> donor LM

teacher-forced full:      0.546875
teacher-forced core_off:  0.546875
teacher-forced state_off: 0.554688
generation full exact:    0/16
```

The typed recurrent value-state update was also rejected separately:

```text
full recurrent content accuracy: 0.296875
recurrent-off content accuracy:  0.343750
```

Therefore:

```text
Do not promote state-conditioned soft-prefix as canonical.
Do not spend more time on larger soft-prefix adapters before fixing the state codec.
```

## Next Architecture Constraint

The next candidate must make value state itself explicit enough to support
answer rendering:

```text
prompt numeric facts
-> mandatory recurrent core
-> typed value registers with compositional numeric representation
-> final answer renderer
```

Acceptance must require:

```text
full answer exact > state_off
full answer exact > core_off
state trace exact > 0 on held-out range
no external solver computes the answer
```
