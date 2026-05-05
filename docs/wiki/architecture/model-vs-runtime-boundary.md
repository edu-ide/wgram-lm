# QTRM Model vs Runtime Boundary

Status: canonical boundary, 2026-05-02.

This page fixes the architecture split:

```text
QTRM model architecture != QTRM runtime system with MemoryOS
```

MemoryOS must not be drawn as an internal model block. It is an external
runtime/context layer that may prepare the model input.

## Model Architecture

The model starts after a canonical token stream already exists.

```text
canonical token stream
-> tokenizer ids
-> Frozen Qwen donor hidden states and optional donor logits
-> QTRM token embedding
-> donor-state projector
-> latent workspace
-> recurrent z_L / z_H core
-> coda / residual head / verifier heads
-> donor-logit fusion or QTRM logits
-> answer channel
```

This path must work without MemoryOS. If a plain prompt collapses into
repetition or `UNKNOWN`, the failure belongs to the donor/QTRM language and
answer path, not to MemoryOS.

## Runtime System

The runtime may optionally retrieve evidence before the model call.

```text
user prompt
-> optional MemoryOS retrieval
-> optional rerank / source selection
-> Context Compiler / chat-template builder
-> one canonical token stream
-> QTRM model architecture
-> answer
```

The Context Compiler is the SSoT boundary. All semantic information that the
model should use must be represented in that one token stream. Source masks,
trust scores, and selected-source metadata are annotations over the stream, not
independent evidence realities.

## Canonical Evaluation Contract

The user-facing QTRM answer architecture is measured with:

```text
--evidence-injection ssot
--answer-channel greedy
--require-canonical-ssot
```

This means retrieved evidence, if any, is first compiled into the same
donor-visible prompt. The model must then answer autoregressively from that
single stream. Span-copy, hidden workspace evidence, and dual evidence paths are
diagnostic probes; they are not accepted as the main model answer path.

## Accepted Terms

Use these terms:

- `QTRM model`: donor-backed residual cognitive adapter.
- `QTRM runtime`: model plus context compiler, tools, history, and optional
  MemoryOS retrieval.
- `QTRM + MemoryOS`: memory-augmented runtime system.
- `MemoryOS`: external memory/RAG layer.

Avoid these terms:

- `MemoryOS inside QTRM model`;
- `MemoryOS model architecture block`;
- `workspace evidence as the canonical model input`;
- `two semantic paths: prompt path and memory path`.

## Probe Exception

`workspace` and `dual` evidence modes are allowed only as controlled probes:

- `workspace`: hide evidence from the prompt to test whether the latent
  workspace path can causally carry evidence.
- `dual`: feed deterministic visible and hidden views of the same evidence to
  test consistency and ablations.

These modes are not the user-facing model architecture.

`evidence_span_copy` is also probe-only. In SSoT mode it can test whether a
span reader can locate evidence tokens inside the canonical prompt, but it is
not a substitute for fluent autoregressive answer generation.

## Diagram Rule

For papers or wiki figures, draw two diagrams:

1. `QTRM Model Architecture`: no MemoryOS block inside the model.
2. `QTRM Runtime With MemoryOS`: MemoryOS appears before the Context Compiler,
   outside the model boundary.

If one figure must include both, draw a thick vertical boundary:

```text
Runtime / System Layer | Model Layer
```

MemoryOS stays on the runtime side.
