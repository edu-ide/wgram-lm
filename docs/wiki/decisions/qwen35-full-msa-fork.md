# Qwen3.5-2B Full-MSA Fork

Date: 2026-04-30.

Status: conversion scaffold and tiny Qwen3.5-native MSA text-forward prototype
implemented; trained checkpoint not produced yet.

## Decision

Proceed with the aggressive path requested by the user:

```text
Qwen3.5-2B hybrid donor
-> replace all text token mixers with Qwen3.5-native Memory Sparse Attention
-> heal the donor with continual pretraining
-> use the healed full-MSA donor under QTRM
```

This is different from using MSA-4B as a donor. It attempts to turn the current
Qwen3.5-2B donor itself into an MSA-style donor.

## Source Architecture

Local source config:
`references/model_configs/qwen35_2b_base/config.json`

Qwen3.5-2B text layers:

- 24 text layers total;
- 18 `linear_attention` layers;
- 6 `full_attention` layers;
- pattern: three linear-attention layers followed by one full-attention layer.

The MSA reference implementation is local at:
`references/official/msa@30405b2a134c`

MSA source page:
[Memory Sparse Attention](../sources/memory-sparse-attention.md).

## Why This Is A Real Fork

Full MSA requires more than a retrieval module:

```text
doc_ids
-> document-wise RoPE
-> chunk-pooled K/V/router K
-> router Q/K projection
-> top-k document selection
-> sparse attention over selected memory K/V + query K/V
-> auxiliary routing loss
-> Memory Parallel / Memory Interleave runtime
```

Qwen3.5-2B currently uses `Qwen3_5GatedDeltaNet` for most layers. Those
conv/recurrent delta weights do not map cleanly into sparse-attention q/k/v/o
weights. Therefore this fork needs donor healing or continual pretraining
before it can be used as a reliable donor.

## Implemented Scaffold

Files:

- `src/wgram_lm/msa_qwen35.py`
- `src/wgram_lm/qwen35_full_msa.py`
- `src/wgram_lm/qwen35_full_msa_healing.py`
- `scripts/129_prepare_qwen35_full_msa_fork.py`
- `scripts/130_train_qwen35_full_msa_healing.py`
- `tests/test_qwen35_full_msa_checkpoint.py`
- `tests/test_qwen35_full_msa_fork.py`
- `tests/test_qwen35_full_msa_model.py`
- `tests/test_qwen35_full_msa_weight_copy.py`
- `tests/test_qwen35_full_msa_healing.py`

Generated artifact command:

```bash
PYTHONPATH=src python3 scripts/129_prepare_qwen35_full_msa_fork.py
```

Default output:
`runs/qwen35_2b_full_msa_fork_plan/`

Artifacts:

- `config.json`: target full-MSA fork config;
- `conversion_manifest.json`: layer conversion and weight reuse policy;
- `README.md`: human-readable conversion summary.

## Weight Reuse Policy

Reusable without shape change:

- token embeddings;
- tied LM head;
- decoder MLP weights;
- input/post-attention RMSNorm weights;
- final RMSNorm;
- vision tower weights if the multimodal path is preserved.

Reusable as MSA seed:

- original full-attention layers: `[3, 7, 11, 15, 19, 23]`;
- q/k/v/o projections and q/k norms, provided the implementation preserves the
  Qwen3.5 gated query projection contract.

Must reinitialize or heal:

- original linear-attention layers:
  `[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22]`;
- new MSA router q/k projections;
- MSA cache/routing runtime state.

## Qwen3.5-Native MSA Prototype

The first custom text-forward prototype now exists:

```text
Qwen35FullMsaAttention
Qwen35FullMsaDecoderLayer
Qwen35FullMsaTextModel
```

It preserves these Qwen3.5-specific details:

- gated q projection;
- q/k RMSNorm;
- partial rotary / mRoPE behavior;
- MSA-style doc-id routing with chunk-pooled document scoring.

It does not yet implement:

- full Hugging Face `PreTrainedModel` registration;
- multimodal `ForConditionalGeneration`;
- MSA Memory Parallel CPU/GPU cache service;
- generation-time stage1/stage2 KV prefill;
- copied Qwen3.5 donor weights.

## Next Implementation Gate

Acceptance gate:

1. load the fork config;
2. instantiate a tiny/random model; **done in test**;
3. run a doc_ids forward pass; **done in test**;
4. implement custom checkpoint save/load roundtrip; **done in test**;
5. implement full HF model registration and weight-copy script;
6. copy reusable weights from Qwen3.5-2B;
7. run a tiny donor-healing batch; **done in tiny smoke**;
8. compare donor perplexity/logit KL before plugging into QTRM.

## Safe Healing Smoke

The first safety-oriented healing runner is intentionally tiny:

```bash
PYTHONPATH=src uv run --with torch --with 'transformers>=4.57.0' \
  python scripts/130_train_qwen35_full_msa_healing.py \
  --mode tiny-smoke --steps 2
```

It uses a tiny teacher Qwen3.5-style model and a tiny full-MSA student. The
stage-1 freeze policy trains only MSA attention/router parameters while copied
embeddings, MLPs, norms, and LM head stay frozen.

Loss:

```text
LM next-token CE + donor_KL(original Qwen3.5 teacher || full-MSA fork)
```

This smoke does not claim quality. It proves the safe-healing loop updates only
the intended parameters and writes `healing_report.json`.

Smoke result:

- command:
  `PYTHONPATH=src uv run --with torch --with 'transformers>=4.57.0' python scripts/130_train_qwen35_full_msa_healing.py --mode tiny-smoke --steps 2 --out-dir runs/qwen35_full_msa_healing_tiny_smoke`
- report: `runs/qwen35_full_msa_healing_tiny_smoke/healing_report.json`
- trainable params: `11296`;
- frozen params: `20640`;
- updated trainable L1: `15.0766`;
- final loss: `4.9102`;
- final LM loss: `4.8438`;
- final donor KL: `0.0664`.

## Claim Boundary

This scaffold does not yet prove a working full-MSA Qwen3.5 donor. It proves
the conversion boundary and makes the destructive part explicit: 18 of 24
token-mixer layers require MSA replacement and healing.
