# 2026-05-31 Active BLT Runtime Architecture SSOT

## Decision

The active 82M BLT language runs use this executable path:

```text
scripts/557_train_blt_d_prefixlm_dataio.py
-> wgram_lm.models.blt_prefixlm.BLTDByteLatentPrefixLM
-> scripts/534_train_native_prefixlm_dataio.py::build_model
-> scripts/335_train_qtrm_native_etd_probe.py::NativeQTRMETDLM
-> hnet_causal_speaker / same LM head
```

The current canonical run settings are:

```text
patch_boundary_mode = hnet_dechunk
decoder_latent_mode = one_body
backbone = trm_qwen35_3to1
think_structure = trm_dual_z
delta_backend = official_gated_delta2
train_think_steps = 4
imta_trajectories = 3 for new RI/IMTA runs
```

## Runtime Diagram

```text
DataIO PrefixLM row
  instruction bytes + supervised response bytes
        |
        v
BLTDByteLatentPrefixLM
        |
        v
byte_embed(input_ids)
        |
        v
semantic_boundary_scorer
  - UTF-8 continuation bytes cannot open a boundary
  - dynamic_min_patch_size prevents tiny noisy chunks
  - patch_size is the hard boundary cap
        |
        v
causal chunk summaries at selected boundaries
  - summary covers previous_boundary+1 through current boundary only
  - non-boundary bytes influence the recurrent latent input
  - no next chunk is read early
        |
        v
NativeQTRMETDLM global core
  backbone = trm_qwen35_3to1
  block mixer = gated_delta, gated_delta, gated_delta, attention
  delta_backend = official_gated_delta2
  think_structure = trm_dual_z
  train_think_steps = 4
        |
        v
IMTA / GRAM-PTRM same-body breadth
  K learned-offset latent trajectories when imta_trajectories > 1
  per-trajectory latent adapters create answer-facing internal breadth
  training-only stochastic noise may be added to non-anchor trajectories
  speaker-space selector/aggregator chooses latent states, not answer strings
        |
        v
EMA/dechunk selected latent states back to byte positions
        |
        v
hnet one-body bridge
  bridged_latent = dechunked + hnet_latent_bridge(dechunked)
  token_state = sigmoid(byte_gate) * byte_embed
              + sigmoid(latent_gate) * bridged_latent
  default gates:
    byte_gate_init = -2.0
    latent_gate_init = 2.0
        |
        v
hnet_causal_speaker
  BLTDLocalDecoder over the full byte sequence
  causal self-attention + FFN local language modeling
  tied output head with legacy hnet_byte_speaker weight where possible
        |
        v
autoregressive byte logits
        |
        v
free generation / generation gates
```

## Active Components

| Component | Active path | Role |
|---|---|---|
| Entry script | `scripts/557_train_blt_d_prefixlm_dataio.py` | Owns run args, resume, logging, eval, checkpoints |
| BLT wrapper | `wgram_lm.models.blt_prefixlm.BLTDByteLatentPrefixLM` | Byte input, dynamic patching, dechunking, byte speaker |
| Boundary mode | `hnet_dechunk` | Learned semantic boundaries with UTF-8-safe constraints |
| Boundary state | causal chunk summary | Learned summary of already-present bytes, not only the boundary byte |
| Global core | `NativeQTRMETDLM` | Recurrent latent thinking over selected patch states |
| IMTA breadth | `imta_trajectories > 1` | Same-body GRAM/PTRM-style K latent trajectories with per-trajectory adapters before the speaker |
| Backbone | `trm_qwen35_3to1` | Three GatedDeltaNet2-style mixers per attention mixer |
| Delta runtime | `official_gated_delta2` | Fail-fast official runtime; fallback is not canonical |
| Answer path | `hnet_causal_speaker` | Causal byte local decoder fed by gated latent-dominant states |
| Generation | autoregressive free generation | Only decoded free-generation samples count as answer-quality evaluation |

## Required Gap Flags

This file names the executable path, so it must distinguish "active now" from
"required for the next RI claim."

| Requirement | Current checkpoint status | Next required integration |
|---|---|---|
| Causal chunk summary | Implemented in code after the original 82M checkpoint | Train a fresh checkpoint; do not compare old checkpoints as if they had this path |
| IMTA / GRAM / PTRM stochastic breadth | Implemented with `--imta-trajectories > 1`, per-trajectory adapters, speaker-space selector, and optional diversity loss; older checkpoints with K=1 are not RI-complete | K>1 must beat K=1 in decoded free generation under matched budget |
| Own-latent prediction from arXiv:2605.27734 | Executable when `--own-latent-prediction-weight > 0`; older checkpoints may only have telemetry or missing weights | Same-body latent-prediction auxiliary over causal chunk/core/dechunked states that are already answer-causal |
| LeWM external world-model branch | Not active and not canonical | May return only as a same-body semantic/answer-causal latent predictor |

## Inactive / Non-Canonical Components

| Name | Status |
|---|---|
| `src/wgram_lm/blocks.py::OneBodyParallelHybridBlock` | Not wired into this BLT runtime |
| LeWM / LeJEPA world model | Not instantiated in the active BLT answer path |
| Legacy external GRAM/PTRM search | Not part of baseline free generation; the required future version must be same-body IMTA, not detached answer reranking |
| `hnet_byte_speaker` as answer path | Deprecated for canonical hnet runs; kept for checkpoint compatibility/head weight reuse |

## Naming Clarification

`decoder_latent_mode=one_body` in `BLTDByteLatentPrefixLM` means the non-hnet
`clean_decoder` branch does not receive the direct byte-embedding shortcut. It
must speak from the latent/global path plus positional conditioning.

The current canonical `hnet_dechunk` runs do not use that `clean_decoder` branch
for logits. They use `hnet_causal_speaker`: a full causal byte-level local
speaker over token states formed from a small gated byte residual plus dechunked
recurrent latent hidden. This replaces the older `LayerNorm -> Linear`
`hnet_byte_speaker` answer path, which was too weak for language modeling and
made `think_steps` too easy to bypass.

It does **not** mean `src/wgram_lm/blocks.py::OneBodyParallelHybridBlock` is the
active block. That class is a separate experimental path. The file itself says
it is not wired into the default `QTRMBlockStack` path.

## LeWM / World Model Status

LeWorldModel code remains in:

```text
src/wgram_lm/world_model.py
src/wgram_lm/wgram_model.py
src/wgram_lm/training/train.py
```

It is not in the active BLT PrefixLM runtime above. Do not describe the current
82M BLT run as "predicting with a LeJEPA/LeWM world model before thinking".

The existing LeWM SSOT still applies:

```text
core_world_model_enabled = false
loss_core_world_model_weight = 0.0
```

LeWM can return to the answer path only after a separate semantic-transition or
answer-causal gate passes.

## Own-Latent Prediction Status

arXiv:2605.27734 is adopted as a methodology source:

```text
token/byte CE teaches the model to speak
own-latent prediction teaches hidden compositional structure
```

It does not contradict the LeWM demotion. The rejected LeWM probes showed that
a latent transition loss can become non-semantic when the target is not
answer-causal. The new rule is:

```text
Do add a same-body own-latent predictor as the next methodology candidate.
Do not claim it is active in the current 82M checkpoint.
Do not let it bypass hnet_causal_speaker or the same LM head.
```

Canonical reference:

- [2026-05-31 Own-Latent Prediction Methodology SSOT](2026-05-31-own-latent-prediction-methodology-ssot.md)

## Answer Attractor Status

`answer_attractor` in the active BLT runs is a training auxiliary over multiple
`think_steps`. It adds CE and monotonic-depth pressure. It is not a separate
runtime answer module, and it does not by itself prove test-time reasoning depth
scaling.

## GRAM/PTRM Status

Legacy external GRAM/PTRM and the `OneBodyParallelHybridBlock`
stochastic-breadth work exist elsewhere in the repo. The active BLT runtime now
has its own same-body IMTA switch:

```text
imta_trajectories = 1:
  disabled; useful as the K=1 ablation

imta_trajectories > 1:
  active same-body GRAM/PTRM-style latent breadth before hnet_causal_speaker
  each trajectory has a small latent adapter
  selector observes pooled trajectory state plus shallow speaker-space state
  optional imta_diversity_weight discourages collapsed identical trajectories
```

For any RI/IMTA claim, GRAM/PTRM-style breadth is mandatory and must be wired
as:

```text
selected boundary/core latent state
-> K internal recurrent trajectories
-> per-trajectory latent adapters
-> internal speaker-space selector/aggregation over latent states
-> hnet one-body bridge
-> hnet_causal_speaker
-> same LM head
```

Detached candidate answer generation, external verifier-only selection, and
oracle selected accuracy do not satisfy this SSOT.

## Enforcement

The BLT trainer now emits `active_runtime_contract` in `model_summary` and uses
`wgram_lm.architecture.blt_runtime_contract.validate_active_blt_runtime_contract`
to reject ambiguous BLT runs that claim:

- QTRM core-world-model / LeWM is active,
- LeWM loss is active,
- `OneBodyParallelHybridBlock` is wired into the BLT runtime.

## Promotion Gates

This architecture is not promoted by loss alone. A checkpoint must pass these
gates before it can be called a reasoning-capable RI checkpoint:

| Gate | Required evidence |
|---|---|
| Free generation | Exact-answer accuracy and decoded samples from `scripts/565_eval_blt_generation_gate.py` |
| RI-1 depth scaling | `think_steps` sweep where decoded free-generation answers improve |
| IMTA breadth scaling | K=1 vs K>1 same-body GRAM/PTRM sweep measured by free-generation answers only |
| Own-latent prediction | latent-predictor-off ablation plus same-LM-head answer/free-generation lift |
| Loop control | Low repeated-token loop rate without relying only on decoding penalties |
| Causal path | Ablation showing byte residual gate, latent gate, and recurrent depth affect logits/answers |
| Runtime contract | `active_runtime_contract` in checkpoint report matches this SSOT |

## Evaluation Policy

As of 2026-05-31, promotion evaluation is free-generation-only:

```text
allowed:
  prompt -> autoregressive decoded answer -> exact/normalized answer check
  decoded sample review
  repetition/EOS/length statistics from the generated answer

not allowed for promotion:
  forced-choice logprob ranking
  oracle candidate coverage / pass@K
  selected-vs-oracle accuracy
  candidate reranking as the main score
  teacher-forced first-token/top-k as the main score
```

The old candidate rerank gate `scripts/566_eval_blt_candidate_rerank_gate.py`
is now disabled for active evaluation. Historical selected/oracle numbers may
be read only to recover architecture ideas, not to claim current model ability.
