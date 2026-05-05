# Internalized Context Engineering

Status: concept added on 2026-04-30.

## Definition

Internalized context engineering means making the model better at using a
compiled context. It does not mean putting MemoryOS itself inside the model.
The runtime still performs retrieval/rerank/context compilation; the model
learns trainable mechanisms that select, compress, preserve, update, and route
the resulting canonical token stream.

This is not the same as putting every retrieved document into the prompt.
It is also not the same as storing all knowledge permanently inside model
weights. The target is a middle path:

```text
external memory remains auditable
model-internal routes decide how evidence affects reasoning
```

## QTRM Mapping

The current QTRM direction can be read as a staged internalization path.

```text
Prompt-only baseline
  -> MemoryOS retrieval and rerank
  -> SSOT context compiler
  -> evidence-aware canonical token stream
  -> workspace-only evidence injection as an ablation probe
  -> gated core context injection
  -> internal gated memory lane
  -> sparse memory routing over huge memory pools
```

Current implemented boundary:

- MemoryOS retrieval still happens outside the model.
- Reranking and evidence selection are still external context engineering.
- Retrieved evidence should normally be compiled into the canonical prompt.
  It can be hidden from the visible prompt only in workspace-only causality
  probes.
- The cognitive core can now read prelude prompt/evidence context through
  [Gated Core Context Injection](gated-core-context-injection.md).
- `disable_core_context` is the required ablation before claiming that the
  cognitive core is using this direct context route.

## Why This Matters

External RAG can find evidence but still fail if the model does not use it.
The QTRM hypothesis is that context use should become a trainable behavior:

```text
find evidence -> compress into latent workspace -> decide what matters
-> update memory -> answer through residual logits
```

That is closer to "internal context engineering" than plain prompt stuffing.
It is still not full end-to-end retrieval training until the retrieval/rerank
stage itself receives model-training feedback.

## Required Ablations

Every claim must survive mode comparisons.

| Claim | Required ablation signal |
| --- | --- |
| Retrieved evidence flows through workspace | `workspace_memory_off` score drops. |
| Latent workspace is necessary | `workspace_off` score drops. |
| Gated memory update matters | `workspace_gate_off` score drops. |
| Core is reading context directly | `core_context_off` score drops. |
| Residual head is not doing all the work | `residual_head_off` is not the only failing mode. |

## Similar Research Families

Retrieval-augmented LMs:

- REALM trains a latent retrieval path with the language-model objective.
- RETRO feeds retrieved chunks into a retrieval-enhanced Transformer.
- Atlas couples retrieval and generation for knowledge-intensive tasks.
- DSI explores parameterized retrieval by mapping queries to document ids.

Trainable memory LMs:

- LM2 adds a gated auxiliary memory module around a Transformer.
- G-MemLLM adds a gated latent memory bank to a frozen backbone.
- MSA routes over sparse latent memory states for extreme long-memory scaling.

Agent memory systems:

- ACE, LightMem, MemCoT, MemGPT, Self-RAG, and related systems keep memory or
  context outside the base model, but improve selection, consolidation,
  verification, iterative search, and reusable task strategies.

## Design Rule

Keep external context engineering measurable while moving only proven pieces
inside QTRM.

The order should be:

1. prove retrieval/rerank recall;
2. prove workspace evidence causality;
3. prove core context causality;
4. prove gated memory causality;
5. scale memory routing only after the smaller gates pass.
