# QTRM Reasoning LM V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a clean V2 canonical path for RI-1~RI-7 reasoning-language work.

**Architecture:** V2 narrows the answer path to byte input, causal BLT chunk
summaries, a 3:1 recurrent/attention core interface, same-body IMTA latent
trajectories, same-body own-latent prediction, and one causal byte speaker with
one LM head. Promotion evaluation is free generation only.

**Tech Stack:** Python, PyTorch, `unittest`, existing `BLTDLocalDecoder`.

---

### Task 1: Contract-First V2 Scaffold

**Files:**
- Create: `tests/test_qtrm_v2_canonical.py`
- Create: `src/wgram_lm/v2/config.py`
- Create: `src/wgram_lm/v2/contracts.py`
- Create: `src/wgram_lm/v2/__init__.py`

- [x] **Step 1: Write failing contract tests**

Run: `python -m unittest tests.test_qtrm_v2_canonical`

Expected before implementation: import failure for `wgram_lm.v2`.

- [x] **Step 2: Implement config and contract validation**

The contract must reject forced-choice promotion, candidate rerank promotion,
external GRAM/PTRM answer selection, LeWM answer path, boundary-byte-only
state, and multiple answer heads.

### Task 2: Canonical V2 Forward Path

**Files:**
- Create: `src/wgram_lm/v2/chunk_encoder.py`
- Create: `src/wgram_lm/v2/recurrent_core.py`
- Create: `src/wgram_lm/v2/imta.py`
- Create: `src/wgram_lm/v2/latent_prediction.py`
- Create: `src/wgram_lm/v2/speaker.py`
- Create: `src/wgram_lm/v2/model.py`

- [x] **Step 1: Add tests for causal chunk summaries and same-mouth IMTA**

The tests assert non-boundary bytes affect current chunk summaries, future
bytes do not affect earlier chunks, IMTA uses K latent trajectories, and
own-latent loss is auxiliary.

- [x] **Step 2: Implement minimal smoke path**

The smoke path uses a small torch recurrent core to exercise interfaces. It is
not promotion-ready and must be rejected by promotion contract validation.

### Task 3: Free Generation API And SSOT Docs

**Files:**
- Create: `src/wgram_lm/v2/generation.py`
- Create: `src/wgram_lm/v2/README.md`
- Create: `docs/wiki/decisions/2026-05-31-qtrm-reasoning-lm-v2-ssot.md`

- [x] **Step 1: Add generation API test**

`generate_free` must not accept candidate choices and must expose
`free_generation_only` as promotion policy.

- [x] **Step 2: Document V2 promotion gap**

The SSOT must state that V2 is a canonical path scaffold, not yet a promoted
reasoning model.

### Task 4: Minimal V2 Train/Eval Loop

**Files:**
- Create: `scripts/590_train_qtrm_v2_prefixlm.py`
- Create: `tests/test_qtrm_v2_training_gate.py`

- [x] **Step 1: Add a toy DataIO training gate test**

The test writes a temporary sampled PrefixLM dataset, runs one V2 training
step, saves `last_model.pt`, reloads it, and runs a free-generation-only gate.

- [x] **Step 2: Implement the minimal trainer and gate**

The script trains `QTRMReasoningLMV2` on DataIO PrefixLM tensors, saves a V2
contract-bearing checkpoint, and reports decoded free-generation samples.

### Task 5: Single-Recipe Fastlane

**Files:**
- Create: `scripts/591_qtrm_v2_fastlane.py`
- Create: `tests/test_qtrm_v2_fastlane.py`
- Modify: `docs/wiki/decisions/2026-05-31-qtrm-reasoning-lm-v2-ssot.md`

- [x] **Step 1: Add fastlane policy tests**

The tests assert that the fastlane builds one primary V2 recipe and defers
non-core comparison sweeps.

- [x] **Step 2: Implement the fastlane runner**

The runner fixes `imta_trajectories=3`, `own_latent_prediction_weight > 0`,
`imta_diversity_weight > 0`, and `free_generation_only`; it supports smoke
mode only as a non-promotion wiring check.

- [x] **Step 3: Document comparison deferral**

The V2 SSOT now says K sweeps, own-latent-off sweeps, candidate rerankers, and
forced-choice gates are debug-only follow-ups, not the next priority.
