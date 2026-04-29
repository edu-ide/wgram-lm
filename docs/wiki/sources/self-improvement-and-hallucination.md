# Self-Improvement And Hallucination References

Purpose:

Track recent references for QTRM/MemoryOS self-improvement and hallucination
control. The stable interpretation is that no current paper removes
hallucination in general. Reliable systems reduce ungrounded answers by
separating answer generation, evidence search, verification, fallback search,
and training updates.

For fact-checking-specific sources, see
[Fact Verification And Fake Info](fact-verification-and-fake-info.md).

## Downloaded References

- `references/papers/self_improvement/self_improvement_llm_overview_2603.25681.pdf`
- `references/papers/self_improvement/self_improvement_multimodal_survey_2510.02665.pdf`
- `references/papers/self_improvement/score_self_correction_2409.12917.pdf`
- `references/papers/self_improvement/dpo_2305.18290.pdf`

## Source Map

| Source | Link | QTRM Reading |
| --- | --- | --- |
| Self-Improvement of LLMs: Technical Overview and Future Outlook | https://arxiv.org/abs/2603.25681 | Use a closed loop: data acquisition, data selection, model optimization, inference refinement, and autonomous evaluation. |
| Self-Improvement in Multimodal Large Language Models: A Survey | https://arxiv.org/abs/2510.02665 | Multimodal self-improvement needs data collection, data organization, and model optimization gates. |
| SCoRe: Training Language Models to Self-Correct via Reinforcement Learning | https://arxiv.org/abs/2409.12917 | Avoid naive offline correction SFT; train/evaluate on the model's own correction distribution. |
| Direct Preference Optimization | https://arxiv.org/abs/2305.18290 | Convert verified chosen/rejected pairs into a simpler preference-learning target before heavier RL. |
| Why Language Models Hallucinate | https://arxiv.org/abs/2509.04664 | Treat hallucination as an evaluation/training incentive problem as well as a model-capability problem. |
| Why Language Models Hallucinate, Nature 2026 | https://www.nature.com/articles/s41586-026-10549-w | Do not claim full hallucination removal; reward abstention/uncertainty when evidence is insufficient. |
| In-Place Test-Time Training | https://arxiv.org/abs/2604.06169 | Test-time adaptation is a later donor-side ablation, not the primary anti-hallucination mechanism. |

## Design Consequence

For QTRM, "I do not know" is not the product end state. It is an internal
evidence-sufficiency state:

1. If current evidence supports an answer, answer with source-grounded support.
2. If current evidence is insufficient, emit `NEEDS_SEARCH` internally.
3. Agentic retrieval/search expands the evidence set under a budget.
4. A verifier checks whether the answer is now grounded.
5. Only after search budget exhaustion should the user-facing response say that
   the system could not verify the answer.

This preserves hallucination control while avoiding premature `UNKNOWN` as the
final user experience.
