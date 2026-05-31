# Typed Algorithmic Value-State Len1113 S080

Date: 2026-05-07

## Question

Can a role-separated typed value-state head learn mixed list-arithmetic
intermediate values on held-out list lengths 11/13 while preserving the
accepted dynamic-halt action-code controller?

## Failure Ledger

```text
Failure:
  Generic/factorized value slots learned phase/kind but not numeric content.

Evidence:
  algorithmic-value-state-s480 kept trace exact at 0/32 and content-slot
  accuracy near zero; typed-register executor variants also failed to beat the
  prior role-value baseline.

Known limitation class:
  Neural value binding and transition, not action routing.

Root architecture hypothesis:
  One shared value vocabulary mixes list offsets, doubled-list offsets, scalar
  coefficients, scalar residuals, final residuals, and padding. This violates
  the role/filler state contract.

Could the big structure be wrong?:
  Yes. If typed fields still cannot reach held-out trace exact, readout heads
  are likely insufficient and the recurrent core itself must carry/update the
  value state.

Information path needed:
  prompt tokens -> frozen donor hidden states -> QTRM workspace/core trajectory
  -> typed internal value fields -> ablatable metric path.

Current information path:
  Same universal LLM path; no external solver computes the evaluated fields.

Prior work to check:
  Role-filler/TPR, Slot Attention, CLRS process supervision, TransNAR-style
  algorithmic registers.

Recommended candidate:
  Add field-specific heads for raw list offsets, doubled list offsets, scalar
  coefficient, scalar residual, and final residual.

Smallest next experiment:
  S80 from the accepted len11/13 action checkpoint on mixed-only rows.

Acceptance gate:
  held-out content-field accuracy above head-off and previous generic-slot
  result; action-code exact remains 32/32.

Kill criterion:
  If trace exact remains 0/32 after short typed-field runs, do not promote the
  head as canonical; move value transition into the recurrent state update.
```

## Implementation

Added typed targets/scorers:

```text
src/wgram_lm/algorithmic_value_state.py
```

Added model outputs:

```text
typed_algorithmic_kind_logits
typed_algorithmic_raw_list_offset_logits
typed_algorithmic_doubled_list_offset_logits
typed_algorithmic_scalar_coeff_logits
typed_algorithmic_scalar_residual_logits
typed_algorithmic_final_residual_logits
```

Added train/eval hooks:

```text
scripts/196_train_pure_recursive_depth_supervised.py
  --typed-algorithmic-value-state-ce-weight

scripts/238_eval_qtrm_algorithmic_value_state.py
  --use-typed-algorithmic-value-state
  --disable-typed-algorithmic-value-state
```

Config:

```text
configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_typed_algorithmic_value_state_s080.yaml
```

Mixed-only splits:

```text
data/filtered/pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_len1113_probe_train40000_v0to5_mixed_only.jsonl
data/eval/pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_len1113_probe_eval60000_v6to7_len11_13_mixed_only.jsonl
```

Checkpoint:

```text
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_typed_algorithmic_value_state_len1113_s080_from_joint_s080/last.pt
```

## Training Signal

The first mixed-only log shows the target path is active:

```text
typed samples:         35
typed content samples: 17
typed CE:              28.3740
content accuracy:       0.0000
```

By step 40:

```text
typed CE:              6.2605
field accuracy:        0.5143
content accuracy:      0.5882
step exact:            0.1250
transition joint acc:  1.0000
```

## Held-Out Results

Held-out len11/13 mixed-only typed fields:

```text
rows:                   32
trace exact:             0/32
step exact:              0/256
field accuracy:        384/1120 = 0.3428571429
content-field accuracy:352/1024 = 0.34375
```

Typed-head-off ablation:

```text
rows:                   32
trace exact:             0/32
field accuracy:         96/1120 = 0.0857142857
content-field accuracy:  0/1024 = 0.0
```

Action-code preservation:

```text
len11/13 full:
  exact:        32/32
  step acc:      1.0000
  finality acc:  1.0000
  halted exact: 32/32

canonical len7/9:
  exact:        32/32
  step acc:      1.0000
  finality acc:  1.0000
  halted exact: 32/32
```

## Decision

Accept only as a causal Stage-2 probe improvement, not as canonical neural
transition reasoning.

The typed-field head beats the head-off ablation and the previous generic-slot
content result while preserving the accepted action-code controller. However,
trace exact remains 0/32, so the model still does not have an exact learned
value transition. The next architecture must push typed value state into the
recurrent update itself or add process-supervised state-delta training; another
readout-only head is unlikely to close the gap.

## Scope Correction: Not A Universal LLM Objective

Typed CE is not a general LLM training objective. It is a process-supervision
probe for numeric/register binding.

Canonical QTRM claims must stay on this path:

```text
prompt/chat template
-> tokenizer
-> donor hidden states or token embeddings
-> QTRM recurrent core / workspace
-> LM logits
-> autoregressive text answer
```

The typed fields are allowed only if they are derived from the same token
stream and feed, supervise, or diagnose the causal LM answer path. They must
not become a hidden executor or a second answer channel.

Therefore:

```text
accepted:
  typed content-field head beats head-off as an internal probe

not accepted:
  typed CE as the final reasoning objective
  typed registers as runtime answer computation
  typed recurrent cells as canonical unless LM logits/text improve and
  typed-field-off/core-off ablations prove causal loss
```

The canonical branch should now use the Ouro/LoopLM-style answer path and raw
intelligence gates, where deeper recurrent computation improves LM
forced-choice or generation over donor/core-off.

## Next

```text
1. Keep typed value-state work as probe-only.

2. Do not promote typed recurrent value cells unless they improve the normal
   LM answer path, not just field metrics.

3. Canonical next gate:
   donor/core-off/depth1/2/4/8
   + recurrent/halt/module-off ablations
   + LM causal forced-choice and greedy generation metrics.

4. If typed state is used again, it must be an auxiliary process target whose
   removal hurts LM answer quality on held-out cases.
```
