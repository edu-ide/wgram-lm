# Long-Horizon Agent Architecture

This concept maps long-horizon agent papers into the QTRM/MemoryOS design.

Core claim:

Long-running agentic behavior should be built mostly as externalized runtime
architecture around the model. For QTRM, the model should not be expected to
"think for 9 hours" inside one forward pass or one giant context window. The
system should preserve state, evidence, failures, skills, budgets, and
verification results outside the model, then route compact relevant state back
into the donor/QTRM generation path.

Primary source map: [Long-Horizon Agent References](../sources/long-horizon-agent-references.md).

## QTRM Positioning

QTRM should be positioned as:

- a Qwen-backed residual adapter,
- a latent workspace over donor states,
- a memory/evidence-conditioned answer improver,
- and later a learned memory-use module.

QTRM should not currently be positioned as:

- a standalone loop LM,
- a proven latent-reasoning LM,
- a replacement for Qwen donor language ability,
- or an end-to-end autonomous agent by itself.

## Runtime Layers

Recommended layers:

1. `TaskRouter`
   - Classifies the request by context size, tool need, evidence need, and risk.
   - Selects one of the inference modes below.

2. `MemoryOS`
   - Owns durable evidence, embeddings, reranking, traces, summaries, and
     reflective memories.
   - Uses write-manage-read memory lifecycle.

3. `AgentHarness`
   - Owns long-running jobs, step budgets, tool calls, sandbox boundaries,
     retries, checkpoints, and verification gates.

4. `QTRMResponder`
   - Receives a compact prompt plus selected evidence and donor states.
   - Generates through donor logits plus bounded QTRM residual logits.

## Inference Modes

Use explicit modes rather than one universal agent path:

| Mode | Use When | Execution Shape |
| --- | --- | --- |
| `non_rlm_direct` | Short prompt, no external evidence needed | Donor/QTRM generation only. |
| `memory_rag` | Answer depends on stored/web evidence | Retrieve, rerank, evidence pack, answer, verify citations/aliases. |
| `rlm_no_subcalls` | Context is huge but can be inspected by code/search | Store context outside prompt; controller uses search/slicing/aggregation tools. |
| `rlm_recursive` | Large context plus semantic decomposition requires subquestions | Same as above, but allows bounded recursive model calls over snippets. |

Default should be `non_rlm_direct` or `memory_rag`. Recursive execution is an
advanced path with cost and reliability risks.

## Memory Types

Keep these stores separate:

| Memory Type | Stores | Write Gate |
| --- | --- | --- |
| Factual evidence | Source chunks, documents, web captures, citations | Source metadata and retrieval index. |
| Trace memory | Actions, observations, generated answers, eval outcomes | Task id, mode, checkpoint, exact prompt/evidence. |
| Reflection memory | Failure analysis and correction notes | Must be linked to a failed/passed verification event. |
| Skill memory | Reusable procedures/scripts/prompts | Must pass repeated execution tests before promotion. |
| Summary/index memory | Compact handles to full-fidelity evidence | Must preserve dereference path to original evidence. |

This separation prevents reflective guesses from contaminating factual memory.

## Verification Gates

Long-running agent work should not be judged by whether the model produced a
plausible answer. Use gates:

- Retrieval recall: target evidence appears before rerank.
- Rerank recall: target evidence remains after rerank.
- Evidence sufficiency: answerable versus unanswerable is classified correctly.
- Answer accuracy: aliases match expected answer without prompt leakage.
- Evidence insufficiency: missing-answer cases return `UNKNOWN` only in
  closed-evidence evals; in agentic mode they route to `NEEDS_SEARCH`.
- Fact verification: fact-seeking and fake-info tasks preserve `SUPPORTED`,
  `REFUTED`, `NOT_ENOUGH_INFO`, `CONFLICT`, and `STALE_OR_TIME_DEPENDENT`
  instead of collapsing every failure into one unknown state.
- Trace reproducibility: prompt, evidence, mode, checkpoint, and output are
  persisted.
- Budget safety: max steps, max subcalls, timeout, and tool allowlist are
  enforced.

## RLM Design Decision

RLM-style execution is useful, but it belongs in the harness:

- Long prompts become external context variables or indexed stores.
- The controller can inspect, search, slice, and aggregate those stores.
- Recursive calls are bounded and traced.
- REPL/code execution must run in a sandbox.
- QTRM can later learn to choose evidence or compress latent state, but it
  should not be the first controller for RLM recursion.

## Current Implementation Consequences

Immediate:

- Keep fixing the small held-out MemoryOS probe before scaling to 100M tokens.
- Preserve exact evidence and trace JSONL for each memory evaluation run.
- Add a documented mode router before implementing recursive subcalls.
- Add retrieval-quality judging before relying on generated answers.
- Treat `UNKNOWN` as an internal search trigger in open-world agent mode, not as
  the final user-facing answer.

Later:

- Add hierarchical summary/index memory for large corpora.
- Add reflection memory only after failed/passed verification events exist.
- Add a skill library for reusable MemoryOS workflows.
- Consider RLM recursive mode once non-recursive RAG and no-subcall inspection
  are stable.
