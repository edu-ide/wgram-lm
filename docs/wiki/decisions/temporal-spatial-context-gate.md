# Raw Intelligence Gate

## Verdict

Gate type: `temporal_spatial_context`

Status: `rejected`

Claim: SSOT-derived temporal/spatial context tokens should causally improve held-out temporal and spatial reasoning over the same model with the context path ablated.

Recommendation: Do not claim temporal/spatial conditioning yet. The context-on path must beat the context-off ablation with no retrieval or MemoryOS shortcut.

## Checks

- Passed: `context_available_on_full_mode, context_disabled_on_ablation_mode, no_retrieval_or_memoryos_shortcut`
- Failed: `context_on_does_not_beat_context_off`
- Missing modes: `none`
- Shortcut records: `0`

## Mode Metrics

| Label | Mode | Hits | Accuracy |
| --- | --- | ---: | ---: |
| context_on | qtrm_core_steps_8_no_evidence | 4/24 | 0.167 |
| context_off | qtrm_core_steps_8_temporal_spatial_off_no_evidence | 4/24 | 0.167 |

## Expected-Paradigm Metrics

| Expected paradigm | Mode | Hits | Accuracy |
| --- | --- | ---: | ---: |
| context_conditioned_latent_reasoning | qtrm_core_steps_8_no_evidence | 4/24 | 0.167 |
| context_conditioned_latent_reasoning | qtrm_core_steps_8_temporal_spatial_off_no_evidence | 4/24 | 0.167 |

## Interpretation Rule

- `accepted` means the tested raw-intelligence component was causally useful on this eval.
- `rejected` means the component did not beat the simpler ablation/baseline or a shortcut contaminated the run.
- `inconclusive` means required modes are missing or empty.
- This gate is not a RAG, MemoryOS, answer-format, or SSOT score.
