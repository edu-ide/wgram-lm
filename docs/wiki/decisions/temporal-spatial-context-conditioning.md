# Temporal-Spatial Context Conditioning

Status: implemented as a minor architecture extension; not yet promoted as a
capability result.

## Why

QTRM needs time and space awareness for metacognition, memory validity, visual
layout, embodied/world-model reasoning, and long-running agent work. The
architecture should not hide this information in runtime side channels. It
should put SSOT-derived temporal/spatial facts on the model's forward path so
the workspace and recursive core can use them causally.

Prior work motivating this axis:

```text
temporal grounding/alignment:
  Set the Clock: Temporal Alignment of Pretrained Language Models
  Time-aware/temporal RAG systems such as TimeR4 and TempRALM

spatial / spatial-temporal memory:
  3D-Mem
  3DLLM-Mem
  OmniSpatial and Spatial-DISE as spatial reasoning gates
```

## Architecture

The root architecture remains:

```text
SSOT input
  -> Qwen donor / token embedding
  -> QTRM workspace
  -> recursive core
  -> coda / answer readout
  -> logits
```

The extension adds temporal-spatial context tokens before the prelude:

```text
prompt tokens
+ optional donor hidden tokens
+ optional visual tokens
+ optional workspace/memory tokens
+ temporal_spatial_context tokens
       |
       v
prelude -> LatentWorkspace -> RecursiveCore -> coda/readout
```

Implementation detail:

```text
temporal_spatial_context: [batch, tokens, temporal_spatial_context_dim]
  or [batch, temporal_spatial_context_dim] for one token

projection:
  Linear(temporal_spatial_context_dim -> d_model)
  + learned positional embedding
  + RMSNorm

ablation:
  disable_temporal_spatial_context=True
```

The context tensor must be derived from visible canonical input, memory
metadata, visual detections, UI geometry, or tool observations. If it is filled
by hidden labels unavailable to a real model, the experiment is invalid.

## Config

```text
model.temporal_spatial_context_enabled: false
model.temporal_spatial_context_dim: 8
model.temporal_spatial_context_max_tokens: 4
```

New trainable policy:

```text
core_and_temporal_spatial_context
```

This trains:

```text
core.*
temporal_spatial_context_proj.*
temporal_spatial_context_norm.*
temporal_spatial_context_pos
```

and freezes the base embedding/prelude/coda path.

## Acceptance Gate

This change is only accepted as real temporal/spatial intelligence if a held-out
gate shows:

```text
full temporal-spatial context > temporal_spatial_context_off
full temporal-spatial context > prompt-only baseline
core_off or workspace_off loses the measured gain
```

Candidate eval families:

```text
temporal:
  stale vs fresh fact
  valid_until / observed_at conflict
  session elapsed / deadline routing

spatial:
  left/right/inside/behind relation
  2D UI/layout reference
  3D object permanence or room memory

composition:
  object location changes over time
  newest observation overrides stale memory
```

## Current Gate Wiring

The held-out gate now has a concrete data/eval path:

```text
data builder:
  scripts/207_build_temporal_spatial_context_cases.py

train/eval runner:
  scripts/208_run_temporal_spatial_context_gate.sh

probe config:
  configs/qwen35_2b_4090_temporal_spatial_context_probe.yaml

train data:
  data/train/temporal_spatial_context_train_120.jsonl

held-out eval data:
  data/eval/temporal_spatial_context_heldout_24.jsonl

gate output:
  docs/wiki/decisions/temporal-spatial-context-gate.md
  docs/wiki/decisions/temporal-spatial-context-gate-summary.json
```

The eval compares:

```text
context on:
  qtrm_core_steps_8_no_evidence

context off:
  qtrm_core_steps_8_temporal_spatial_off_no_evidence
```

The gate is accepted only if context-on beats context-off on the same held-out
cases, all full-mode records have temporal/spatial context tokens, all ablation
records have the context path disabled, and no MemoryOS/retrieval shortcut is
present.

Important limitation: these cases keep the source facts visible in the prompt
and put only SSOT-derived structured features on the context-token path. This
preserves the single source of truth. It also means a strong prompt-only model
can still solve some cases, so the gate measures whether the structured
context path adds causal value, not whether hidden labels were smuggled into
the model.

## Files

```text
src/wgram_lm/config.py
src/wgram_lm/wgram_model.py
src/wgram_lm/training/train.py
src/wgram_lm/eval/raw_intelligence_gate.py
scripts/196_train_pure_recursive_depth_supervised.py
scripts/192_eval_raw_intelligence.py
scripts/207_build_temporal_spatial_context_cases.py
scripts/208_run_temporal_spatial_context_gate.sh
configs/qwen35_2b_4090_temporal_spatial_context_probe.yaml
tests/test_model_config.py
tests/test_training_checkpoint_init.py
tests/test_raw_intelligence_eval_script.py
tests/test_temporal_spatial_context_cases.py
tests/test_temporal_spatial_context_runner.py
```
