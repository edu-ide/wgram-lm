# Multimodal QTRM-MemoryOS Architecture

## 1. Final concept

```text
Qwen3.5-Multimodal-QTRM-MemoryOS

Inputs:
  text
  images
  video frames
  OCR/layout tokens
  retrieved text evidence
  retrieved visual evidence
  tool outputs

Donor:
  Qwen3.5-2B-Base for 2.7B-ish route
  Qwen3.5-4B-Base for performance-first route

Core:
  TRM-style z_L / z_H recursive latent workspace
  Gated DeltaNet / Qwen-style hybrid recurrent/attention blocks
  Parcae-style stable injection
  LeWM-style future latent prediction

Memory:
  LLM Wiki / Visual Wiki
  Harrier text embeddings
  visual embeddings
  BM25 / code search
  GraphRAG / RAPTOR-style hierarchy

Controller:
  THINK, LOOK, READ_OCR, RETRIEVE_TEXT, RETRIEVE_IMAGE, TOOL, VERIFY, REVISE, ANSWER
```

## 2. Why Qwen3.5 donor

Qwen3.5-2B/4B style models are preferred donors because they are already multimodal image-text-to-text models with a vision encoder and a hybrid Gated DeltaNet / Gated Attention language backbone. This matches our desired 3:1 recurrent-to-exact-attention block pattern.

## 3. Core data path

```text
Text tokens ───────────────┐
Image / visual tokens ─────┤
OCR / layout tokens ───────┤
Retrieved evidence ────────┤
Tool observations ─────────┘
        ↓
Multimodal Projector / Resampler
        ↓
Fixed latent workspace
        ↓
QTRM recursive core
  z_L = fast local refinement
  z_H = slow global state
        ↓
Controller + verifier + LeWM future predictor
        ↓
Causal decoder
```

## 4. Key design rule

Never run the recurrent core over all image patches, all video frames, or all 1M context tokens. Always compress/read them into a bounded workspace first.

```text
Bad:
  all tokens / patches / frames → recursive core

Good:
  all tokens / patches / frames → context reader / resampler → 256 workspace tokens → recursive core
```

## 4.1 LeWM path

The JEPA path is no longer a pooled-vector smoke head. It follows the newer
LeWorldModel contract in token-latent space:

```text
text tokens
  -> causal JEPA encoder
  -> online latent sequence
  -> action-conditioned causal future predictor with AdaLN-zero
  -> predicted latent[t+1]

online latent[t+1]
  -> future latent target

loss = MSE(predicted latent, future latent) + SIGReg(latent sequence)
```

See `docs/LEWM_INTEGRATION.md` for the exact reference mapping and remaining
gaps versus full LeWorldModel parity.

## 4.2 Delta mixer path

The delta/recurrent mixer path must be checked against official Gated DeltaNet
or FLA GatedDeltaNet, not our PyTorch fallback.

```text
Official target:
  q/k/v projections
  short convolution on q/k/v
  gated delta rule in chunk mode
  beta and decay gates
  output RMSNorm + swish gate
  hybrid placement with attention layers

Current fallback:
  simple bounded recurrent mixer
  useful only for smoke/debug training
```

## 5. Multimodal MemoryOS

```text
Raw media store:
  documents, PDFs, screenshots, images, videos, audio transcripts

LLM Wiki / Visual Wiki:
  entity pages, concept pages, visual scene pages, OCR claims, chart claims, bounding boxes, timestamps

Text memory:
  Harrier embeddings over raw chunks, wiki pages, claims, summaries

Visual memory:
  Qwen donor vision embeddings, SigLIP-style embeddings, or reference CLIP embeddings

Graph memory:
  entities, relations, supports, contradicts, appears-in, spatial links, temporal links

Verifier:
  checks answer support against text spans, OCR spans, image regions, and tool results
```

## 6. Recommended staged implementation

```text
Stage 0: text-only smoke QTRM
Stage 1: random visual features + multimodal projector smoke test
Stage 2: Qwen3.5 vision encoder frozen, train projector/core/head only
Stage 3: image QA / OCR / chart SFT
Stage 4: multimodal MemoryOS retrieval
Stage 5: verifier-grounded multimodal RAG
Stage 6: video frames + temporal memory
```
