# Fact Verification And Fake Info References

Purpose:

Track the fact-verification, fake-info, and hallucination-detection sources
that should shape QTRM/MemoryOS. The important distinction is that factual
truth is not learned reliably by latent prediction alone. It needs evidence
retrieval, source metadata, conflict handling, temporal state, and a verifier.

## Downloaded Papers

- `references/papers/fact_verification/fever_naacl2018_N18-1074.pdf`
- `references/papers/fact_verification/factscore_2305.14251.pdf`
- `references/papers/fact_verification/longform_factuality_safe_2403.18802.pdf`
- `references/papers/fact_verification/ragtruth_2401.00396.pdf`
- `references/papers/fact_verification/openfactscore_2507.05965.pdf`
- `references/papers/fact_verification/ciber_scientific_claim_verification_2503.07937.pdf`
- `references/papers/fact_verification/confact_conflicting_evidence_2505.17762.pdf`
- `references/papers/fact_verification/care_rag_2507.01281.pdf`
- `references/papers/fact_verification/verifiable_misinformation_agent_2508.03092.pdf`
- `references/papers/fact_verification/livefact_2604.04815.pdf`
- `references/papers/fact_verification/arbgraph_2604.18362.pdf`

## Cloned Official Repositories

| Repo | Local Path | Commit | Use |
| --- | --- | --- | --- |
| FEVER baseline | `references/official/fever-naacl-2018` | `72abdead388e` | Classic claim retrieval plus entailment baseline. |
| FActScore | `references/official/factscore` | `f28272deffcf` | Atomic fact extraction and validation metric. |
| Long-form factuality / SAFE | `references/official/long-form-factuality` | `9d27158d198c` | Search-augmented factuality evaluator and LongFact prompts. |
| RAGTruth | `references/official/ragtruth` | `c103204b9ce2` | Word-level hallucination corpus for RAG outputs. |
| OpenFActScore | `references/official/openfactscore` | `35c08f1f6137` | Open-model implementation of FActScore. |
| CONFACT | `references/official/confact` | `05a1227b4821` | Conflicting evidence and source-credibility dataset. |
| ArbGraph | `references/official/arbgraph` | `75305a15be44` | Pre-generation evidence graph and arbitration reference. |

## Source Map

| Source | Link | QTRM Reading |
| --- | --- | --- |
| FEVER | https://aclanthology.org/N18-1074/ | Use `SUPPORTED`, `REFUTED`, and `NOT_ENOUGH_INFO` as the base verification states, with evidence retrieval as part of the task. |
| FEVER baseline code | https://github.com/sheffieldnlp/naacl2018-fever | Keep retrieval and entailment/verdict separated. The old DrQA/TF-IDF stack is not the target, but the pipeline split still matters. |
| FEVER workshop/datasets | https://fever.ai/ | Treat fact-checking as an evolving benchmark family, including web evidence and multimodal claims. |
| FActScore | https://arxiv.org/abs/2305.14251 | Break long answers into atomic facts, then score support against reliable knowledge. |
| FActScore code | https://github.com/shmsw25/factscore | Reference implementation for atomic factuality evaluation. |
| Long-form factuality / SAFE | https://arxiv.org/abs/2403.18802 | Use search-augmented verification loops for long-form answers instead of trusting one generator pass. |
| SAFE code | https://github.com/google-deepmind/long-form-factuality | Reference for LongFact, SAFE, and F1@K-style factuality evaluation. |
| RAGTruth | https://arxiv.org/abs/2401.00396 | RAG can still hallucinate; train/evaluate hallucination detection at response and word levels. |
| RAGTruth code/data | https://github.com/ParticleMedia/RAGTruth | Reference corpus for unsupported or contradictory RAG claims. |
| OpenFActScore | https://arxiv.org/abs/2507.05965 | Prefer open-model atomic factuality tooling for local QTRM evaluation. |
| OpenFActScore code | https://github.com/lflage/OpenFActScore | Open Hugging Face-compatible path for atomic fact generation and validation. |
| CIBER | https://arxiv.org/abs/2503.07937 | Retrieve both corroborating and refuting evidence; use probe consistency when model internals are unavailable. |
| CONFACT | https://arxiv.org/abs/2505.17762 | Conflicting sources are a first-class failure mode; source credibility must enter retrieval/generation. |
| CONFACT code/data | https://github.com/zoeyyes/CONFACT | Dataset and implementation for conflict-aware fact-checking experiments. |
| CARE-RAG | https://arxiv.org/abs/2507.01281 | Compare internal parametric answers with retrieved evidence, then summarize conflicts before generation. |
| Verifiable misinformation agent | https://arxiv.org/abs/2508.03092 | For fake-info tasks, use multi-tool verification, source credibility checks, numeric checks, and evidence logs. |
| LiveFact | https://arxiv.org/abs/2604.04815 | Static benchmarks miss time uncertainty; evaluate temporal evidence slices and epistemic humility. |
| ArbGraph | https://arxiv.org/abs/2604.18362 | Resolve evidence conflicts before generation using atomic claims, support/contradiction edges, and credibility propagation. |
| ArbGraph code | https://github.com/1212Judy/ArbGraph | Latest local reference for pre-generation evidence arbitration. |

## QTRM Design Consequence

The verification stack should expose these labels to MemoryOS and training data:

```text
SUPPORTED
REFUTED
NOT_ENOUGH_INFO
CONFLICT
NEEDS_SEARCH
STALE_OR_TIME_DEPENDENT
```

Do not collapse them into one `UNKNOWN` token. `UNKNOWN` is acceptable for a
closed benchmark answer field, but MemoryOS should preserve the reason:
insufficient evidence, direct contradiction, source conflict, stale evidence, or
budget exhaustion.

## Repositories Not Cloned

The following were kept as paper-only references because a clearly official
GitHub link was not visible from the primary source during this pass:

- CIBER: `https://arxiv.org/abs/2503.07937`
- CARE-RAG: `https://arxiv.org/abs/2507.01281`
- Verifiable misinformation agent: `https://arxiv.org/abs/2508.03092`
- LiveFact: `https://arxiv.org/abs/2604.04815`

