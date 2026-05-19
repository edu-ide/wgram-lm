# Qwen3.5 Preinit Core Interpolation Stabilization

Date: 2026-05-19

## Question

Can the 128-case accepted Qwen3.5-pretrained mandatory-core QTRM checkpoint
composition be stabilized into a 256-case accepted checkpoint without changing
the architecture?

## Setup

Canonical path:

```text
prompt tokens
-> Qwen3.5 tokenizer / embeddings
-> Qwen3.5 original backbone
-> mandatory shared z_H/z_L TRM-style recurrent core
-> Qwen3.5 LM head
-> LM logits
```

Checkpoint A:

```text
local_eval/qwen35_preinit_strict_trm_partial_l3_gate_s80_20260519/last_core.pt
```

Checkpoint B:

```text
local_eval/qwen35_preinit_strict_trm_partial_l3_checksum_repair_s80_20260519/last_core.pt
```

Best previous scalar interpolation:

```text
alpha=0.25
128-case: accepted, gain 0.0390625, language_top1 1.0
256-case: rejected, gain 0.01953125, language_top1 1.0
```

The 256-case gate requires `gain >= 0.02`, so alpha=0.25 misses by one
additional core-over-base correct case.

## What Was Tried

Direct stabilization continuation from alpha=0.25:

```text
local_eval/qwen35_preinit_alpha025_stabilize_s60_20260519
decision: rejected
gain: 0.015625
language_top1_agreement: 1.0
```

Conclusion: low-LR continuation with KL and core-advantage pressure worsened
the accepted basin instead of stabilizing it.

Selective qwen/core interpolation was added:

```text
scripts/411_interpolate_trainable_checkpoints.py
  --qwen-alpha
  --core-alpha
  --qwen-attn-alpha
  --qwen-mlp-alpha
  --qwen-norm-alpha
  --core-state-alpha
  --core-adapter-alpha
```

Representative 256-case results:

```text
alpha0.25 / q0.25_c0.30:
  accepted: false
  gain: 0.01953125
  min_family_gain: 0.0
  min_family_core_accuracy: 0.0930232558
  language_top1_agreement: 1.0

q0.25_c0.32:
  accepted: false
  gain: 0.01953125
  language_top1_agreement: 1.0

q0.25_c0.35:
  accepted: false
  gain: 0.01953125
  language_top1_agreement: 1.0

q0.10_c0.50:
  accepted: false
  gain: 0.0078125
  min_family_gain: 0.0
  min_family_core_accuracy: 0.1162790698
  language_top1_agreement: 1.0

q0.15_c0.50:
  accepted: false
  gain: 0.0
  min_family_gain: -0.0232558140
  language_top1_agreement: 0.875

q0.20_c0.50:
  accepted: false
  gain: 0.00390625
  min_family_gain: -0.0348837209
  language_top1_agreement: 1.0
```

Group interpolation gave two 128-case passes but did not survive 256 cases:

```text
qa25_qm25_qn25_cs50_ca25:
  128-case: accepted, gain 0.0390625
  256-case: rejected, gain 0.01953125

qa25_qm25_qn10_cs50_ca30:
  128-case: accepted, gain 0.0234375
  256-case: rejected, gain 0.0078125
```

## Interpretation

The result is useful but not yet a robust promotion.

What is real:

```text
1. Qwen3.5-pretrained mandatory-core QTRM can preserve language logits.
2. A QTRM core gain over core-off exists in a nearby weight basin.
3. Family-floor and aggregate-gain objectives can coexist at 128-case scale.
```

What is not proven:

```text
1. 256-case stable acceptance.
2. public benchmark gain.
3. Qwen3.6-27B comparison.
4. a trained stable checkpoint after continuation.
```

The decisive bottleneck is no longer loading, language collapse, or scalar
interpolation. The bottleneck is recurrent-objective stability: training must
make the core's causal gain robust across families instead of merely moving
between two fragile rejected checkpoints.

## Decision

Do not spend more time on scalar or coarse group interpolation unless it is used
as a diagnostic. The next credible step is an objective/selection change:

```text
1. keep the canonical Qwen3.5-pretrained mandatory-core path;
2. train with fixed 256-case family-balanced selection during training;
3. select checkpoints by the actual promotion gate, not only loss;
4. add core-off and family-floor metrics to every periodic checkpoint;
5. promote only if 256-case gain clears the threshold by margin.
```

This preserves the QTRM-native claim because the model remains a single graph
and the improvement must pass through token -> Qwen backbone -> mandatory core
-> LM logits.

## Follow-Up Selection Run

Added runner:

```text
scripts/412_run_qwen35_preinit_family_balanced_selection.sh
```

DGX output:

```text
local_eval/qwen35_preinit_family_balanced_select_s120_20260519
```

Configuration:

```text
init_checkpoint: alpha_0.25.pt
steps: 120
eval_cases: 256
eval_every_steps: 10
lr: 1.0e-5
qwen_lr: 0.0
kl_weight: 0.10
language_kl_weight: 0.10
core_advantage_weight: 0.20
core_advantage_mode: label_choice_margin
family_loss_weights: chain5=1.2,checksum4=1.2,select_pair=1.2
```

Result:

```text
decision: rejected
best_periodic_eval_step: 10
best_periodic_gain: 0.01953125
final_gain: 0.01953125
language_top1_agreement: 1.0

family gains:
  chain5:      +0.0235294118
  checksum4:   0.0
  select_pair: +0.0352941176
```

Interpretation:

```text
The fixed 256-case selection loop preserves the alpha=0.25 near-miss but does
not improve it. The bottleneck is now sharper: the current continuation
objective cannot create checksum4 gain without degrading other families. The
next experiment should add a checksum-specific contrastive/counterfactual
objective or a recurrent trajectory loss, not more low-LR CE continuation.
```

## Checksum Counterfactual Objective 2026-05-19

Added training-only checksum4 counterfactual augmentation:

```text
scripts/362_train_qwen_backbone_qtrm_core_gate.py
  --checksum-counterfactual-weight
  --checksum-counterfactual-variants
```

Mechanism:

```text
For checksum4 training cases, parse a,b,c,d from the prompt.
Create counterfactual prompts by changing one digit.
Train the same canonical LM-logit path to answer the changed prompt.
No runtime sidecar or external solver is used at inference.
```

DGX run:

```text
local_eval/qwen35_preinit_checksum_cf_w05_v2_s80_20260519
```

Configuration:

```text
init_checkpoint: alpha_0.25.pt
steps: 80
eval_cases: 256
lr: 2.0e-5
qwen_lr: 0.0
checksum_counterfactual_weight: 0.5
checksum_counterfactual_variants: 2
family_loss_weights: chain5=1.1,checksum4=1.8,select_pair=1.1
```

256-case result:

```text
decision: accepted
gain: 0.0234375
language_top1_agreement: 1.0

family gains:
  chain5:      +0.0352941176
  checksum4:   0.0
  select_pair: +0.0352941176
```

512-case expansion:

```text
local_eval/qwen35_preinit_checksum_cf_w05_v2_eval512_20260519

decision: rejected
gain: 0.015625
language_top1_agreement: 1.0

family gains:
  chain5:      +0.0292397661
  checksum4:   0.0
  select_pair: +0.0176470588
```

Interpretation:

```text
This is a real step: the Qwen3.5-pretrained mandatory-core path crossed the
256-case promotion threshold while preserving language logits. It is not yet a
robust breakthrough because the 512-case expansion rejects and checksum4 still
has zero core-over-base gain.

The counterfactual objective helped the aggregate gate, mostly by improving
chain5 retention/gain. It did not solve the intended checksum4 bottleneck.
```

Next required objective:

```text
base-error targeted checksum4 objective:
  compute base/core label-choice logits on checksum4 training cases
  apply extra margin only where base is wrong or weak
  select by 512-case gain and require checksum4 positive gain

Do not call the 256-case checkpoint robust until this passes.
```

## Base-Error Targeted Checksum Objective 2026-05-19

Added:

```text
--checksum-base-error-advantage-weight
--checksum-base-error-margin
--checksum-base-error-base-margin-threshold
```

Mechanism:

```text
On checksum4 training cases, compute base/core label-choice margins.
If the base/core-off path is wrong or weak, add extra loss that pushes the
core target digit above the strongest wrong digit.
```

DGX run:

```text
local_eval/qwen35_preinit_checksum_baseerr_w06_s80_20260519
```

Result:

```text
256-case decision: accepted
gain: 0.0234375
language_top1_agreement: 1.0

family gains:
  chain5:      +0.0352941176
  checksum4:   0.0
  select_pair: +0.0352941176
```

Interpretation:

```text
The stricter targeted loss still did not move checksum4. The 256-case
acceptance is therefore not evidence that the intended checksum bottleneck is
solved. The accepted gain is again carried by chain5/select_pair.

Conclusion: checksum4 is not responding to final-token CE/margin pressure at
this scale. The next diagnostic must inspect whether the recurrent core changes
checksum4 predictions/logit ranks at all, and whether z_H/z_L carries operand
information. If the latent state is not binding a,b,c,d, the next fix is a
state/trajectory supervision or operand-binding objective, not a stronger
answer-margin loss.
```
