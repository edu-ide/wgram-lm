# Fact Verification Reasoning

Primary source map:
[Fact Verification And Fake Info References](../sources/fact-verification-and-fake-info.md).

## Core Position

Human-like factual intuition is not the same thing as latent world modeling.
For QTRM/MemoryOS, "intuition" should be split into multiple trainable and
runtime abilities:

| Axis | Mechanism | Main Risk If Missing |
| --- | --- | --- |
| Predictive intuition | LeWorldModel/JEPA-style latent next-state prediction | Weak physical/temporal priors. |
| Factual grounding | Retrieval, reranking, evidence packing | Fluent unsupported answers. |
| Verification | FEVER/FActScore/SAFE-style claim checking | Cannot distinguish supported from false. |
| Conflict reasoning | CONFACT/CARE-RAG/ArbGraph-style arbitration | Follows whichever source is most salient. |
| Temporal judgment | LiveFact-style evidence slices and dates | Treats stale or early evidence as final truth. |
| Source judgment | Source metadata and credibility features | Treats unreliable and authoritative sources equally. |
| Self-improvement | Verified preference traces and failure memory | Learns from its own unverified guesses. |

LeWorldModel is therefore useful but not sufficient. It can strengthen
predictive latent priors, especially for embodied, temporal, or multimodal
state transitions. It should not be treated as the source of factual truth.

## Wisdom Loop

The target is not a model that already knows everything. The target is a system
that becomes wiser through a repeatable loop:

```text
if it does not know, search;
compare the evidence it found;
notice contradictions;
weigh time and source quality;
record failures;
use verified failures to make the next run less wrong.
```

This loop belongs mostly in MemoryOS plus the agent/verifier harness. QTRM can
learn to use the selected evidence and residual-correct answers, but it should
not be treated as a self-contained source of all truth.

For value, religion, and spiritual-synthesis questions, the loop must continue
past doubt into constructive synthesis:
[Critical Synthesis Reasoning](critical-synthesis-reasoning.md).

## Runtime Contract

For fact-seeking or fake-info tasks:

```text
claim/question
-> retrieve local MemoryOS evidence
-> rerank and pack evidence
-> atomize claim or generated answer
-> classify each atomic claim:
   SUPPORTED / REFUTED / NOT_ENOUGH_INFO / CONFLICT / STALE_OR_TIME_DEPENDENT
-> if insufficient: NEEDS_SEARCH
-> web/search expansion under budget
-> conflict/source/time arbitration
-> generate final answer with citations or bounded non-verification
-> store trace and verification labels
```

The answer generator should be downstream of evidence arbitration. It should
not be asked to both resolve conflicts and write prose in the same opaque pass
when evidence is noisy.

## MemoryOS Fields

Evidence chunks should carry enough metadata for verification:

| Field | Why It Matters |
| --- | --- |
| `source_url` or `source_id` | Allows citation and deduplication. |
| `published_at` / `captured_at` | Required for time-sensitive claims. |
| `source_type` | Paper, official docs, news, social, user note, generated trace. |
| `credibility_tier` | Lets arbitration prefer stronger sources under conflict. |
| `claim_atoms` | Enables support/refute/contradict edges. |
| `supports` / `refutes` edges | Enables ArbGraph-style conflict graph reasoning. |
| `retrieval_score` and `rerank_score` | Separates search failure from reasoning failure. |
| `verdict` | Stores final local label without losing the evidence path. |

## Training Data Contract

Training rows for this axis should prefer structured traces over free-form
corrections:

```json
{
  "claim": "...",
  "evidence": [{"text": "...", "source_id": "...", "published_at": "..."}],
  "atomic_claims": ["..."],
  "verdict": "SUPPORTED|REFUTED|NOT_ENOUGH_INFO|CONFLICT|STALE_OR_TIME_DEPENDENT",
  "action": "ANSWER|NEEDS_SEARCH",
  "chosen": "...",
  "rejected": "...",
  "failure_tags": ["..."]
}
```

Chosen answers must be grounded in evidence or explicitly route to
`NEEDS_SEARCH`. Rejected answers should include common failure modes:
unsupported claim, contradicted claim, stale answer, source-conflict
overconfidence, and citation mismatch.

## Where LeWorldModel Fits

Use LeWorldModel for:

- predicting latent next states;
- surprise or inconsistency signals in sequential/multimodal inputs;
- compact world-state priors before planning;
- future multimodal MemoryOS traces.

Do not use it as:

- a factual verifier;
- a source credibility model;
- a substitute for retrieval;
- proof that QTRM has human-like intuition.

The practical target is not "LeWorldModel alone creates intuition." The target
is a composite system: latent prediction gives priors, retrieval supplies
evidence, verification assigns labels, and MemoryOS stores the outcomes.
