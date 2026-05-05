# Raw Intelligence Gate

## Verdict

Gate type: `pure_recursive_reasoning`

Status: `rejected`

Claim: QTRM recursive core depth should improve held-out reasoning without retrieval, MemoryOS, or hidden evidence shortcuts.

Recommendation: Do not tune answer formatting. Redesign or retrain the recursive core so deeper latent steps beat donor-only and core-off on no-evidence tasks.

## Checks

- Passed: `deep_core_beats_core_off, depth_scaling_gain_present, depth_outputs_not_all_identical, no_retrieval_or_memoryos_shortcut`
- Failed: `deep_core_does_not_beat_donor, deep_core_does_not_beat_transition_state_off`
- Missing modes: `none`
- Shortcut records: `0`

## Mode Metrics

| Label | Mode | Hits | Accuracy |
| --- | --- | ---: | ---: |
| donor | donor_only_no_evidence | 5/16 | 0.312 |
| core_off | qtrm_core_off_no_evidence | 0/16 | 0.000 |
| deepest_core | qtrm_core_steps_8_no_evidence | 5/16 | 0.312 |
| transition_state_off | qtrm_core_steps_8_transition_state_off_no_evidence | 5/16 | 0.312 |

## Depth Ladder

| Mode | Hits | Accuracy |
| --- | ---: | ---: |
| qtrm_core_steps_1_no_evidence | 5/16 | 0.312 |
| qtrm_core_steps_2_no_evidence | 4/16 | 0.250 |
| qtrm_core_steps_4_no_evidence | 5/16 | 0.312 |
| qtrm_core_steps_8_no_evidence | 5/16 | 0.312 |

## Depth Output Diversity

- Comparable cases: `16`
- Identical across all depth modes: `13`
- Changed by depth: `3`
- All depth outputs identical: `False`

## Expected-Paradigm Metrics

| Expected paradigm | Mode | Hits | Accuracy |
| --- | --- | ---: | ---: |
| hybrid_or_cot | donor_only_no_evidence | 2/8 | 0.250 |
| hybrid_or_cot | qtrm_core_off_no_evidence | 0/8 | 0.000 |
| hybrid_or_cot | qtrm_core_steps_1_no_evidence | 2/8 | 0.250 |
| hybrid_or_cot | qtrm_core_steps_2_no_evidence | 2/8 | 0.250 |
| hybrid_or_cot | qtrm_core_steps_4_no_evidence | 3/8 | 0.375 |
| hybrid_or_cot | qtrm_core_steps_8_no_evidence | 3/8 | 0.375 |
| hybrid_or_cot | qtrm_core_steps_8_transition_state_off_no_evidence | 3/8 | 0.375 |
| latent_parallel | donor_only_no_evidence | 2/4 | 0.500 |
| latent_parallel | qtrm_core_off_no_evidence | 0/4 | 0.000 |
| latent_parallel | qtrm_core_steps_1_no_evidence | 2/4 | 0.500 |
| latent_parallel | qtrm_core_steps_2_no_evidence | 2/4 | 0.500 |
| latent_parallel | qtrm_core_steps_4_no_evidence | 2/4 | 0.500 |
| latent_parallel | qtrm_core_steps_8_no_evidence | 2/4 | 0.500 |
| latent_parallel | qtrm_core_steps_8_transition_state_off_no_evidence | 2/4 | 0.500 |
| latent_recurrent | donor_only_no_evidence | 1/4 | 0.250 |
| latent_recurrent | qtrm_core_off_no_evidence | 0/4 | 0.000 |
| latent_recurrent | qtrm_core_steps_1_no_evidence | 1/4 | 0.250 |
| latent_recurrent | qtrm_core_steps_2_no_evidence | 0/4 | 0.000 |
| latent_recurrent | qtrm_core_steps_4_no_evidence | 0/4 | 0.000 |
| latent_recurrent | qtrm_core_steps_8_no_evidence | 0/4 | 0.000 |
| latent_recurrent | qtrm_core_steps_8_transition_state_off_no_evidence | 0/4 | 0.000 |

## Interpretation Rule

- `accepted` means the tested raw-intelligence component was causally useful on this eval.
- `rejected` means the component did not beat the simpler ablation/baseline or a shortcut contaminated the run.
- `inconclusive` means required modes are missing or empty.
- This gate is not a RAG, MemoryOS, answer-format, or SSOT score.
