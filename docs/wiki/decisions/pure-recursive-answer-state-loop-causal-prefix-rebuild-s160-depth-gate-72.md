# Raw Intelligence Gate

## Verdict

Gate type: `pure_recursive_reasoning`

Status: `rejected`

Claim: QTRM recursive core depth should improve held-out reasoning without retrieval, MemoryOS, or hidden evidence shortcuts.

Recommendation: Do not tune answer formatting. Redesign or retrain the recursive core so deeper latent steps beat donor-only and core-off on no-evidence tasks.

## Checks

- Passed: `deep_core_beats_core_off, depth_outputs_not_all_identical, no_retrieval_or_memoryos_shortcut`
- Failed: `deep_core_does_not_beat_donor, no_depth_scaling_gain`
- Missing modes: `none`
- Shortcut records: `0`

## Mode Metrics

| Label | Mode | Hits | Accuracy |
| --- | --- | ---: | ---: |
| donor | donor_only_no_evidence | 22/72 | 0.306 |
| core_off | qtrm_core_off_no_evidence | 0/72 | 0.000 |
| deepest_core | qtrm_core_steps_8_no_evidence | 18/72 | 0.250 |

## Depth Ladder

| Mode | Hits | Accuracy |
| --- | ---: | ---: |
| qtrm_core_steps_1_no_evidence | 19/72 | 0.264 |
| qtrm_core_steps_2_no_evidence | 18/72 | 0.250 |
| qtrm_core_steps_4_no_evidence | 18/72 | 0.250 |
| qtrm_core_steps_8_no_evidence | 18/72 | 0.250 |

## Depth Output Diversity

- Comparable cases: `72`
- Identical across all depth modes: `69`
- Changed by depth: `3`
- All depth outputs identical: `False`

## Expected-Paradigm Metrics

| Expected paradigm | Mode | Hits | Accuracy |
| --- | --- | ---: | ---: |
| hybrid_or_cot | donor_only_no_evidence | 8/36 | 0.222 |
| hybrid_or_cot | qtrm_core_off_no_evidence | 0/36 | 0.000 |
| hybrid_or_cot | qtrm_core_steps_1_no_evidence | 3/36 | 0.083 |
| hybrid_or_cot | qtrm_core_steps_2_no_evidence | 2/36 | 0.056 |
| hybrid_or_cot | qtrm_core_steps_4_no_evidence | 2/36 | 0.056 |
| hybrid_or_cot | qtrm_core_steps_8_no_evidence | 2/36 | 0.056 |
| latent_parallel | donor_only_no_evidence | 8/18 | 0.444 |
| latent_parallel | qtrm_core_off_no_evidence | 0/18 | 0.000 |
| latent_parallel | qtrm_core_steps_1_no_evidence | 10/18 | 0.556 |
| latent_parallel | qtrm_core_steps_2_no_evidence | 10/18 | 0.556 |
| latent_parallel | qtrm_core_steps_4_no_evidence | 10/18 | 0.556 |
| latent_parallel | qtrm_core_steps_8_no_evidence | 10/18 | 0.556 |
| latent_recurrent | donor_only_no_evidence | 6/18 | 0.333 |
| latent_recurrent | qtrm_core_off_no_evidence | 0/18 | 0.000 |
| latent_recurrent | qtrm_core_steps_1_no_evidence | 6/18 | 0.333 |
| latent_recurrent | qtrm_core_steps_2_no_evidence | 6/18 | 0.333 |
| latent_recurrent | qtrm_core_steps_4_no_evidence | 6/18 | 0.333 |
| latent_recurrent | qtrm_core_steps_8_no_evidence | 6/18 | 0.333 |

## Interpretation Rule

- `accepted` means the tested raw-intelligence component was causally useful on this eval.
- `rejected` means the component did not beat the simpler ablation/baseline or a shortcut contaminated the run.
- `inconclusive` means required modes are missing or empty.
- This gate is not a RAG, MemoryOS, answer-format, or SSOT score.
