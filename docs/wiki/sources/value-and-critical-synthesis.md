# Value And Critical Synthesis References

Purpose:

Track sources for value salience, tradition-aware reasoning, and constructive
critical synthesis. This axis is different from fact verification. It handles
questions where the answer depends on value hierarchy, religious tradition,
spiritual practice, symbolic interpretation, and constructive reframing.

## Local Source Material

| Source | Local Path | QTRM Reading |
| --- | --- | --- |
| 본각교 매뉴얼 | `/mnt/nvme0n1p2/workspace/monorepo/services/sajug/saju_data/본각교_매뉴얼.md` | Local doctrine-style source for criticizing fear, guilt, authority dependence, and external savior dependence. |
| 본각교 요약 | `/mnt/nvme0n1p2/workspace/monorepo/services/sajug/saju_data/본각교_요약.md` | Compact source for breath, observation, inner freedom, and self-sovereignty as a positive practice frame. |

## Paper Map

| Source | Link | QTRM Reading |
| --- | --- | --- |
| Wide Reflective Equilibrium in LLM Alignment | https://arxiv.org/abs/2506.00415 | Use judgment, principles, and background theories in a coherence loop instead of one rigid rule. |
| The Staircase of Ethics | https://arxiv.org/abs/2505.18154 | Value priorities shift by context; evaluate whether a model preserves the correct priority structure. |
| Structured Moral Reasoning | https://arxiv.org/abs/2506.14948 | Separate value identification, conflict, justification, and final judgment. |
| Sacred or Synthetic? | https://arxiv.org/abs/2508.08287 | Religious QA needs refusal/uncertainty and tradition awareness, not only factual accuracy. |
| IslamicLegalBench | https://arxiv.org/abs/2602.21226 | Religious reasoning is tradition-specific and cannot be flattened into generic moral advice. |
| Six Llamas | https://arxiv.org/abs/2604.18404 | Religious corpora can shift ethical reasoning; keep tradition identity explicit. |
| Credibility Assessment Survey | https://arxiv.org/abs/2410.21360 | Use textual credibility signals when deciding which evidence should constrain synthesis. |

## Design Consequence

QTRM/MemoryOS should not train a model to merely attack existing traditions.
The target is constructive criticism:

```text
question
-> classify as fact / value / tradition / synthesis
-> identify controlling or weakly grounded claims
-> preserve liberating and coherent values
-> mark unverifiable metaphysics as symbolic or hypothesis-level
-> warn about risks in both old authority and new overclaim
-> produce a positive practical conclusion
```

This prevents two failure modes:

- `authority_collapse`: accepting established tradition only because it is old.
- `cynicism_collapse`: rejecting everything and never forming a useful answer.

