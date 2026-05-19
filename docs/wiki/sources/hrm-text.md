# HRM-Text Source

Date: 2026-05-19

Purpose: record the first strong public prior that ports HRM-style recurrent
latent reasoning into an actual text-generation language model.

## Sources

- Model: <https://huggingface.co/sapientinc/HRM-Text-1B>
- Repo: <https://github.com/sapientinc/HRM-Text>
- Local clone: `references/official/hrm-text`
- Local commit: `f99410a`
- Data pipeline repo: <https://github.com/sapientinc/data_io>
- Data pipeline local clone: `references/official/data_io`
- Data pipeline local commit: `4f9bc38`
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

Reference resource targets from the official README:

```text
L  / 0.6B parameters:  8 H100s,  single node, about 50 hours
XL / 1.0B parameters: 16 H100s, two nodes,  about 46 hours
```

QTRM implication:

```text
HRM-Text is strong evidence for recurrent text-LM pretraining, but its
reference runs are not a fast single-DGX-Spark recipe. For this project,
HRM-Text should guide the data/objective/training stack, while the main
near-term model route uses Qwen3.5 pretrained initialization plus a strict
TRM recurrent core.
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

## Data IO Pipeline Notes

HRM-Text's companion `sapientinc/data_io` repo matters for QTRM because it
defines the data discipline, not only a preprocessing convenience.

Key source fact:

```text
HRM-Text Data IO produces instruction-style question-answer pairs and sampled
tokenized datasets, instead of directly streaming arbitrary web documents.
```

Canonical cleaned row:

```json
{
  "condition": "cot,noisy",
  "instruction": "Question or prompt text",
  "response": "Answer or completion text"
}
```

Tokenized output preserves boundaries:

```text
tokens.npy
inst_start.npy
inst_len.npy
resp_start.npy
resp_len.npy
metadata.json
```

Sampler:

```text
sample_tokenized.py
prefix_config.yaml
```

Useful sampling principles:

```text
1. sample by dataset/task prefix, not uniform raw token stream;
2. cap over-large sources with max_per_file;
3. upsample small high-quality datasets with repeat;
4. track coverage by task/category before training;
5. treat any token distribution change as a benchmarked breaking change.
```

Important resource note:

```text
The full cleaning pipeline states a roughly 512 GiB RAM requirement. For QTRM,
do not copy that full cleaning path into the fast loop. Prefer the released
cleaned data or a small Data-IO-compatible local subset first.
```

QTRM import rule:

```text
Use Data IO's instruction/response boundaries and stratified sampling style
for QTRM language healing. Do not switch tokenizers away from Qwen unless the
goal is a from-scratch native model; Qwen-preinit experiments should keep the
Qwen tokenizer and only import Data IO's row schema and sampling discipline.
```

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

## QTRM Experiment: Separate H/L Candidate

Date: 2026-05-19

Implemented candidate:

```text
think_structure=trm_dual_z_hrm_separate

z_L update:
  shared QTRM/TRM-style low recurrent stage

z_H update:
  separate HRM-style high recurrent stage
```

Code:

```text
scripts/335_train_qtrm_native_etd_probe.py
scripts/342_qtrm_native_l5d_backbone_compare.py
```

DGX run:

```text
worktree:
  /mnt/data4tb/qtrm_hrm_exp_20260519

python:
  /mnt/data4tb/venv_sglang_pr23000/bin/python

report:
  local_eval/hrm_text_prior_core_compare_v2_short_dgx_torch_20260519/
    backbone_compare_summary.json
```

Result:

```text
candidate                                      full_exact  depth_gain  worst_ablation_drop
trm_dual_z_official_trm_think                  0.03125     0.00521     -0.00521
trm_dual_z_hrm_separate_official_trm_think     0.06250     0.02083      0.03125
```

Interpretation:

```text
HRM-style separate H/L modules produced a better short-gate signal than the
shared TRM-style update under the same seed and data.

This is not enough to claim a solved language model or a final architecture.
The absolute exact score is still low, so this is a promotion candidate for a
larger controlled gate, not a capability claim.
```

## Strict TRM Condition Follow-up

Date: 2026-05-19

Concern:

```text
HRM-style separate H/L should not be assumed better than TRM-style sharing until
the TRM candidate receives a fairer recurrence schedule.
```

Added comparison candidates:

```text
trm_dual_z_official_trm_l3_halt_think
trm_dual_z_hrm_separate_l3_halt_official_trm_think
```

Strict schedule:

```text
L cycles: 3
halt head: dedicated
halt loss: active-length cumulative prefixes
halt-depth final-answer loss: 0.25
adaptive halt eval: enabled
```

Result:

```text
candidate                                           full_exact  depth_gain  ablation_drop
trm_dual_z_official_trm_think                       0.03125     0.00521    -0.00521
trm_dual_z_official_trm_l3_halt_think               0.04167     0.03646     0.00000
trm_dual_z_hrm_separate_official_trm_think          0.06250     0.02083     0.03125
trm_dual_z_hrm_separate_l3_halt_official_trm_think  0.04688     0.00000     0.01042
```

Conclusion:

```text
The stricter TRM recurrence recipe helped the shared TRM candidate but did not
beat the simpler HRM-style separate H/L candidate in the current short gate.

Therefore, for QTRM-native near-term work, HRM-style role separation remains
the better empirical prior. TRM-style sharing remains a research candidate, but
must show positive destructive-ablation drop before promotion.
```

## Multi-Seed Follow-up

Date: 2026-05-19

Ran three short seeds comparing the best strict-TRM recipe against the simple
HRM-style separate H/L recipe.

```text
seed/eval_seed  strict_TRM_exact  strict_TRM_causal  HRM_exact  HRM_causal  winner
337/9337        0.04167           false              0.06250    true        HRM
338/9338        0.02083           false              0.05729    true        HRM
339/9339        0.05208           true               0.04167    false       TRM
```

Aggregate:

```text
strict_TRM mean exact: 0.03819
HRM mean exact:        0.05382

strict_TRM causal_ok: 1/3
HRM causal_ok:        2/3
```

Conclusion:

```text
For this QTRM-native short gate, HRM-style separate H/L is the better empirical
default. The result is not a general proof against TRM; it is a local
architecture decision for the current training regime.
```

## Length-Scaling Revision

Date: 2026-05-19

Follow-up len6/len7 gates changed the practical baseline decision.

```text
len6:
  strict TRM exact:   0.05990
  HRM separate exact: 0.05990

  strict TRM has better depth gain and min-family.
  HRM has slightly better ablation drop.

len7:
  strict TRM exact:   0.05208
  HRM separate exact: 0.04167

  strict TRM also has better depth gain, ablation drop, and min-family.
```

Revised conclusion:

```text
HRM-style separate H/L is a useful optimization prior at len4, but strict TRM
shared recurrence scales better in the current len6/len7 checks.

Use HRM-style separate H/L as a diagnostic candidate, not the canonical
reasoning core. Use strict TRM L=3 + halt/depth objectives as the current
scaling baseline until a broader multi-seed len6/len7 run disproves it.
```
