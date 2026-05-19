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
