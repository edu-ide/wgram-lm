# Pointer/Copy Renderer Prior

Date: 2026-05-09

## Why This Matters

The QTRM absolute ordered-state gate rejected after the corrected class-coverage
split and prompt-open 300-step run. The model can create a small causal value
signal, but it does not generalize exact ordered traces through a flat absolute
value-class head.

For list-transform tasks, a better prior-backed route is:

```text
prompt tokens
-> source-position pointer state
-> recurrent filter/map state
-> copy/edit renderer over source tokens and normal LM logits
```

This keeps the state general over variable input values. The model should point
to source positions and learn transformations, instead of learning one output
class per numeric value.

## Prior Sources

- Pointer Networks, Vinyals et al. 2015:
  https://arxiv.org/abs/1506.03134

  Core idea: output elements can be positions in the input sequence. This is a
  better fit for variable-size value dictionaries than fixed absolute classes.

- CopyNet, Gu et al. 2016:
  https://arxiv.org/abs/1603.06393

  Core idea: sequence generation can combine vocabulary generation with explicit
  copying from the source sequence.

- Pointer-Generator Networks, See et al. 2017:
  https://arxiv.org/abs/1704.04368

  Core idea: hybrid generate/copy distributions improve factual reproduction
  while preserving normal generation.

## Mapping To QTRM

Accepted local evidence already exists for source-position pointer state:

```text
/mnt/nvme1n1p2/qtrm-runs/primitive_field_heads_source_pointer_s120_seed16/accepted_l2_source_pointer_step_000040.pt
```

The rejected absolute-value route shows why pointer state should be canonical
for this bottleneck:

```text
absolute classes:
  pro: answer-renderable in principle
  con: poor held-out trace exact, brittle value vocabulary

source-position pointers:
  pro: generalizes over values and order
  con: needs a learned copy/edit renderer to reach LM logits
```

## Next Gate

Target level:

```text
L2 local / L3 candidate
```

Major bottleneck:

```text
source-position recurrent state must causally improve normal answer generation
through a copy/edit renderer.
```

Required comparisons:

```text
full > donor-only
full > core-off
full > source-pointer-off
full > copy-renderer-off
```

Reject if:

```text
source-pointer-off or copy-renderer-off ties full
or generation remains 0 exact on held-out list-transform cases.
```

Promotion requires:

```text
normal autoregressive text generation improves without hidden solver/rule-copy
and ablation drops prove the source-position state is causal.
```
