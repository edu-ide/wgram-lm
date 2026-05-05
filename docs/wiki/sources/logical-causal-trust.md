# Logical Causal Trust References

Date accessed: 2026-04-30

Purpose:

Track prior work for logical causality, evidence trust, and fake-evidence
resistance. The direct QTRM design question is: when retrieved MemoryOS
evidence is present, can the model distinguish supported, refuted, missing,
and counterfactual evidence before its residual writes answer logits?

## Downloaded Papers

- `references/papers/logical-causal-trust/proofver-2108.11357.pdf`
- `references/papers/logical-causal-trust/folio-2209.00840.pdf`
- `references/papers/logical-causal-trust/linc-2310.15164.pdf`
- `references/papers/logical-causal-trust/nl2logic-2602.13237.pdf`
- `references/papers/logical-causal-trust/insufficient-evidence-2022.tacl-1.43.pdf`
- `references/papers/logical-causal-trust/ragonite-2412.10571.pdf`
- `references/papers/logical-causal-trust/cf-rag-openreview-9U51rOnGko.pdf`
- `references/papers/logical-causal-trust/faithfulrag-2506.08938.pdf`

## Cloned Repositories

| Repo | Local Path | Commit | Use |
| --- | --- | --- | --- |
| LINC | `references/official/linc` | `718dfe8fae34` | LLM as semantic parser plus first-order logic prover. |
| FOLIO | `references/official/folio` | `5d7bb84c7eda` | Natural-language premises paired with first-order logic annotations. |
| NL2LOGIC | `references/official/nl2logic` | `3d625c5b33ae` | AST-guided NL-to-FOL translation with solver-compatible output. |

## Source Map

| Source | Link | QTRM Reading |
| --- | --- | --- |
| ProoFVer | https://aclanthology.org/2022.tacl-1.59/ | Natural-logic proof path maps to `support/refute/missing` heads rather than a direct opaque classifier. |
| FOLIO | https://arxiv.org/abs/2209.00840 | Keep logic examples with explicit premises and labels; use as a future data source for strict support/refute/missing supervision. |
| LINC | https://arxiv.org/abs/2310.15164 | For high-stakes logic, model should parse to a structured representation and defer final validity to a prover when possible. |
| LINC code | https://github.com/benlipkin/linc | Local reference for modular NL-to-logic/prover pipeline. |
| NL2LOGIC | https://arxiv.org/abs/2602.13237 | AST-guided translation is safer than unconstrained single-pass symbolic generation. |
| NL2LOGIC code | https://github.com/peng-gao-lab/nl2logic | Local reference for structured logic emission. |
| Fact Checking with Insufficient Evidence | https://aclanthology.org/2022.tacl-1.43/ | `NOT_ENOUGH_INFO` is not a failure token; it is a separate epistemic state requiring search or abstention. |
| RAGONITE | https://arxiv.org/abs/2412.10571 | Counterfactual attribution checks whether removing evidence changes the answer. |
| CF-RAG | https://openreview.net/forum?id=9U51rOnGko | Counterfactual queries and parallel arbitration are useful for distinguishing causally relevant evidence from plausible context. |
| FaithfulRAG | https://arxiv.org/abs/2506.08938 | Model parametric knowledge and retrieved context can conflict; suppressing one blindly is not robust. |
| RAGChecker | https://arxiv.org/abs/2408.08067 | Separate retrieval failure from generation/evidence-use failure. |

## Design Consequence

The current QTRM failure is not "retrieval cannot find evidence." The quick
probe retrieved the target but generation still failed. Therefore the next
architecture should expose evidence trust as trainable, ablatable tensors:

```text
workspace evidence
-> support / refute / missing heads
-> causal evidence gate
-> bounded QTRM residual logits
```

This is weaker than a full theorem prover but stronger than plain RAG. It gives
the model a measurable contract: evidence must be judged before the residual can
substantially change the answer.

## Future Upgrade Path

- Short term: supervise support/refute/missing and counterfactual evidence gate.
- Medium term: add shuffled-evidence and no-evidence ablations to every
  MemoryOS proof.
- Long term: add a symbolic verifier sidecar for logic-heavy rows using
  LINC/NL2LOGIC-style parsing and solver checks.
