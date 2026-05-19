# HRM-Text Source

Date: 2026-05-19

Purpose: record the first strong public prior that ports HRM-style recurrent
latent reasoning into an actual text-generation language model.

## Sources

- Model: <https://huggingface.co/sapientinc/HRM-Text-1B>
- Repo: <https://github.com/sapientinc/HRM-Text>
- Local clone: `references/official/hrm-text`
- Local commit: `f99410a`
- HRM paper: <https://arxiv.org/abs/2506.21734>
- TRM paper: <https://arxiv.org/abs/2510.04871>
- TRM repo: `references/official/tiny-recursive-models`

## What HRM-Text Is

HRM-Text is a text-generation LM built around the Hierarchical Reasoning Model
family. The public model card describes it as a 1B-parameter checkpoint trained
from scratch on public text data with a PrefixLM objective.

Key model-card facts:

```text
parameters: about 1B
hidden size: 1536
H/L stack layers: 16 each
attention: MHA with gated attention output
cycles: H_cycles x L_cycles = 2 x 3
max sequence length: 4096
training tokens: 40B
objective: PrefixLM
status: pre-alignment / not chat tuned
```

The recurrent core, in simplified form:

```text
z_H = embed(input_ids) * embedding_scale
z_L = learned z_L_init

for H cycle:
  for L cycle:
    z_L = L_module(z_L + z_H)
  z_H = H_module(z_H + z_L)

return z_H -> LM head
```

## Implementation Notes From Official Repo

Relevant files:

```text
models/baselines/hrm_nocarry_bp_warmup.py
models/baselines/trm_nocarry.py
models/baselines/ut_nocarry.py
models/layers.py
models/flash_attention_prefixlm_v2.py
dataset_new.py
pretrain.py
conversion/convert_to_hf.py
```

Important details:

- `hrm` has separate `H_level` and `L_level` recurrent blocks.
- `trm` keeps the `z_H`, `z_L` dual state but uses the same `L_level` block for
  both updates.
- HRM-Text uses no-carry text pretraining baselines rather than the puzzle
  repo's ACT carry wrapper as the primary text-LM path.
- PrefixLM is canonical for HRM-Text: prefix tokens attend bidirectionally and
  response tokens attend causally.
- The training stack is not a small 4090 recipe: the reference L/XL runs assume
  Hopper-class multi-GPU training with FlashAttention 3 and FSDP2.

## HRM vs TRM Correction

TRM is not "single-state" in the official code. It keeps two latent states:

```text
z_H
z_L
```

The real simplification is module sharing:

```text
HRM:
  z_L = L_level(z_L, z_H)
  z_H = H_level(z_H, z_L)

TRM:
  z_L = L_level(z_L, z_H)
  z_H = L_level(z_H, z_L)
```

Therefore:

```text
dual state != dual module
```

For QTRM, this distinction matters. If we keep `z_H/z_L` but use one shared
recurrent block, we are closer to TRM. If we use separate high/low recurrent
blocks, we are closer to HRM.

## QTRM Implications

HRM-Text changes the prior for QTRM-native:

```text
Before:
  TRM/HRM were mainly puzzle-reasoning priors.

After HRM-Text:
  HRM/TRM-style recurrent latent reasoning has a public text-LM scaling prior.
```

Concrete design consequences:

1. Keep the canonical path as:

```text
tokens -> embeddings -> recurrent z_H/z_L core -> z_H readout -> LM logits
```

2. Treat `z_H/z_L` as canonical, but do not assume separate H/L modules are
   automatically better.

3. Add two QTRM-native ablation families:

```text
TRM-style:
  shared recurrent block for z_H and z_L updates

HRM-style:
  separate H_module and L_module
```

4. Compare these under the same held-out LM-generation gate:

```text
full_generation_exact
think0_generation_exact
full_minus_think0
full_minus_worst_ablation
min_family_generation_exact
state_reset / z_l_zero / z_h_zero / op_zero
```

5. Do not claim HRM-Text-level capability unless QTRM has actual text-LM
   pretraining or healing. HRM-Text used large-scale text pretraining; a small
   synthetic gate only proves recurrent-core causality.

## Recommended Next Experiment

Run a controlled QTRM-native comparison with no architecture shopping:

```text
same tokenizer
same d_model
same data
same H_cycles x L_cycles
same eval seed
same acceptance gate

candidate A:
  TRM-style shared z_H/z_L recurrent block

candidate B:
  HRM-style separate H/L recurrent blocks

candidate C:
  current best QTRM nested core baseline
```

Accept only if a candidate improves the same held-out greedy LM-generation
metric and loses under destructive state/core ablations.
