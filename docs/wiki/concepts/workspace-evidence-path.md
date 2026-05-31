# Workspace Evidence Path

Status: implemented as an eval/model path on 2026-04-30; corrected to a
single-source-of-truth architecture on 2026-05-02.

This is the next causality gate after the gated-workspace ablation.
It is also the first implemented step toward
[Internalized Context Engineering](internalized-context-engineering.md): the
retriever stays outside the model, but retrieved evidence can influence QTRM
through one canonical prompt token stream, with workspace-only and dual-view
paths kept only as causality probes.

Boundary correction: this page describes runtime/eval evidence paths around the
model. MemoryOS is not part of the QTRM model architecture. The QTRM model
starts after the Context Compiler emits canonical token ids.

## Problem

The previous MemoryOS eval put retrieved evidence directly into the visible
prompt. That is the standard RAG shape, but it did not prove latent-workspace
reasoning. If evidence is already in the normal prompt path, a residual head or
coda path can learn useful corrections without depending on workspace state.

The later workspace-only probe hid evidence from the visible prompt. That was
useful for causality, but it is not a good final LLM architecture by itself.
Most RAG systems let the model read retrieved evidence as visible context. The
correct product-facing architecture must therefore not create two independent
semantic contexts.

## Change

The eval script now supports:

```text
--evidence-injection ssot
--evidence-injection prompt
--evidence-injection workspace
--evidence-injection dual
```

`ssot` is the canonical path:

```text
user prompt
-> retrieval query
-> MemoryOS retrieval/rerank
-> Context Compiler / chat-template builder
-> one canonical donor-visible token stream
-> Frozen Qwen donor + QTRM core
-> answer channel
```

Source selectors and source masks annotate this token stream. They do not create
a second evidence reality. In code, `ssot` compiles retrieved evidence into the
prompt and sets `evidence_span_reader_context="input"` so the evidence span
reader scores token-aligned canonical prompt states.

When no MemoryOS retrieval is used, the same model path still exists:

```text
user prompt
-> Context Compiler / chat-template builder
-> one canonical donor-visible token stream
-> Frozen Qwen donor + QTRM core
-> answer channel
```

`prompt` is kept as a legacy visible-evidence alias. It is not the preferred
name for new architecture claims because it does not state the SSoT contract.

`workspace` is a strict causality probe:

1. The visible donor prompt contains the task and question, but not the retrieved
   evidence text.
2. Retrieved MemoryOS evidence is separately encoded by the frozen Qwen donor.
3. The resulting hidden states are passed as `workspace_text_states`.
4. QTRM prepends those states only to the workspace cross-attention context.
5. The coda/text path still sees only the normal prompt tokens.

This makes the evidence route:

```text
MemoryOS retrieval -> donor evidence encoder -> workspace memory context
-> latent workspace/core -> QTRM residual logits -> final answer
```

and prevents the easier route:

```text
MemoryOS retrieval -> normal donor-visible prompt -> direct text/coda answer
```

`dual` is now a legacy/probe path, not the canonical architecture:

```text
MemoryOS retrieval
-> shared evidence context (single source of truth)
-> view A: visible RAG context in the donor prompt
-> view B: donor-encoded workspace memory states
-> latent workspace/core
-> residual/verifier/answer channel
```

In `dual`, the two views are deterministic views of one retrieved evidence
context, so it is acceptable as an experiment. It should not be drawn as the
main LLM architecture because that invites the mistaken interpretation that
MemoryOS and the donor are separate semantic input paths.

SSoT rule: retrieval/reranking produces one ordered evidence context, and
semantic evidence enters the model as one canonical chat-template token stream.
Workspace tensors may be used for ablation or auxiliary readers, but they must
be deterministic annotations/views of the same compiled context.

## Code

- `src/wgram_lm/wgram_model.py`
  - Adds `workspace_text_states`, `workspace_attention_mask`, and
    `disable_workspace_memory_context`.
  - Returns `workspace_memory_token_count` telemetry.
- `src/wgram_lm/multimodal_projector.py`
  - Adds `feature_mask` so padded evidence states do not become active memory
    tokens.
- `src/wgram_lm/eval/memory_retrieval.py`
  - Adds `build_shared_evidence_context` and
    `build_case_prompt_and_workspace_memory`.
- `scripts/95_eval_memory_retrieval.py`
  - Defaults to `--evidence-injection ssot`.
  - Adds `evidence_span_reader_context="input"` for SSOT span-copy.
  - Keeps `--evidence-injection workspace` and `--evidence-injection dual` as
    probe modes.
  - Adds `qtrm_workspace_memory_off_with_evidence`.
- `src/wgram_lm/wgram_model.py`
  - Adds `evidence_span_reader_context="input"` so the span reader can score
    canonical prompt tokens without a separate workspace evidence text.
- `src/wgram_lm/data/jsonl_dataset.py` and `src/wgram_lm/training/train.py`
  - Add `workspace_evidence_injection_mode: ssot` for span-reader training
    against canonical prompt-token indices.
- `scripts/117_run_workspace_evidence_path_probe.sh`
  - Runs the 72-case MemoryOS gate with workspace-only evidence injection.
- `src/wgram_lm/data/jsonl_dataset.py`
  - Can split `MemoryOS evidence ... User prompt:` rows into visible prompt
    tokens plus `workspace_input_ids`.
  - Supports `workspace_evidence_injection_mode: dual`, which keeps evidence in
    the visible prompt while also emitting workspace evidence tensors.
- `src/wgram_lm/training/train.py`
  - Encodes `workspace_input_ids` with the frozen donor and forwards them as
    `workspace_text_states`.
- `configs/qwen35_2b_4090_workspace_evidence_path_s050.yaml`
  - Enables `train.workspace_evidence_injection: true` and
    `model.core_context_enabled: true`.
- `scripts/118_run_workspace_evidence_path_train.sh`
  - Trains from the gated-workspace checkpoint and then runs the evidence-path
    causality probe.

## Required Ablations

The evidence-path probe must compare:

```text
donor_only_with_evidence
qtrm_residual_with_evidence
qtrm_workspace_off_with_evidence
qtrm_core_off_with_evidence
qtrm_core_context_off_with_evidence
qtrm_coda_off_with_evidence
qtrm_residual_head_off_with_evidence
qtrm_workspace_gate_off_with_evidence
qtrm_workspace_memory_off_with_evidence
```

Interpretation:

- `ssot` must beat or match donor-only/prompt legacy baselines without
  increasing false positives.
- If `qtrm_residual_with_evidence` improves but `qtrm_workspace_off` does not
  drop, the model is still not using workspace memory.
- If `qtrm_workspace_memory_off` drops, retrieved evidence is flowing through
  the workspace evidence path.
- If `qtrm_core_context_off` drops, the recursive core is using the direct
  gated prompt/evidence context path.
- If `qtrm_workspace_gate_off` drops, the gated update is contributing.
- If only `qtrm_residual_head_off` drops, the model is still mainly a residual
  adapter.

## Current Boundary

Existing workspace-only checkpoints are causality probes, not the final RAG
shape. The corrected training/eval sequence is:

1. Train/evaluate `ssot` as the product-facing architecture.
2. Use `workspace` only to test whether a latent memory route is causal.
3. Use `dual` only when testing deterministic-view consistency.
4. Require `ssot`, `workspace-only`, and component-off comparisons before making
   broad architecture claims.

Gated core context injection is now implemented. The next stricter step is to
train with `core_context_enabled: true` and require `qtrm_core_context_off` to
drop on the held-out MemoryOS gate before making behavioral claims.
