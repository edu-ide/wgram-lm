# Raw Intelligence Gate

## Verdict

Gate type: `pure_recursive_reasoning`

Status: `rejected`

Claim: QTRM recursive core depth should improve held-out reasoning without retrieval, MemoryOS, or hidden evidence shortcuts.

Recommendation: Do not tune answer formatting. Redesign or retrain the recursive core so deeper latent steps beat donor-only and core-off on no-evidence tasks.

## Checks

- Passed: `deep_core_beats_core_off, deep_core_beats_donor, no_retrieval_or_memoryos_shortcut`
- Failed: `no_depth_scaling_gain, depth_outputs_identical_across_steps`
- Missing modes: `none`
- Shortcut records: `0`

## Mode Metrics

| Label | Mode | Hits | Accuracy |
| --- | --- | ---: | ---: |
| donor | donor_only_no_evidence | 2/8 | 0.250 |
| core_off | qtrm_core_off_no_evidence | 0/8 | 0.000 |
| deepest_core | qtrm_core_steps_8_no_evidence | 4/8 | 0.500 |

## Depth Ladder

| Mode | Hits | Accuracy |
| --- | ---: | ---: |
| qtrm_core_steps_1_no_evidence | 4/8 | 0.500 |
| qtrm_core_steps_2_no_evidence | 4/8 | 0.500 |
| qtrm_core_steps_4_no_evidence | 4/8 | 0.500 |
| qtrm_core_steps_8_no_evidence | 4/8 | 0.500 |

## Depth Output Diversity

- Comparable cases: `8`
- Identical across all depth modes: `8`
- Changed by depth: `0`
- All depth outputs identical: `True`

## Interpretation Rule

- `accepted` means the tested raw-intelligence component was causally useful on this eval.
- `rejected` means the component did not beat the simpler ablation/baseline or a shortcut contaminated the run.
- `inconclusive` means required modes are missing or empty.
- This gate is not a RAG, MemoryOS, answer-format, or SSOT score.
