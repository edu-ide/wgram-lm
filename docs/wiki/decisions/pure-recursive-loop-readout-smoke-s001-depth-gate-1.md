# Raw Intelligence Gate

## Verdict

Gate type: `pure_recursive_reasoning`

Status: `rejected`

Claim: QTRM recursive core depth should improve held-out reasoning without retrieval, MemoryOS, or hidden evidence shortcuts.

Recommendation: Do not tune answer formatting. Redesign or retrain the recursive core so deeper latent steps beat donor-only and core-off on no-evidence tasks.

## Checks

- Passed: `depth_scaling_gain_present, depth_outputs_not_all_identical, no_retrieval_or_memoryos_shortcut`
- Failed: `deep_core_does_not_beat_core_off, deep_core_does_not_beat_donor`
- Missing modes: `none`
- Shortcut records: `0`

## Mode Metrics

| Label | Mode | Hits | Accuracy |
| --- | --- | ---: | ---: |
| donor | donor_only_no_evidence | 0/1 | 0.000 |
| core_off | qtrm_core_off_no_evidence | 1/1 | 1.000 |
| deepest_core | qtrm_core_steps_8_no_evidence | 0/1 | 0.000 |

## Depth Ladder

| Mode | Hits | Accuracy |
| --- | ---: | ---: |
| qtrm_core_steps_1_no_evidence | 1/1 | 1.000 |
| qtrm_core_steps_2_no_evidence | 0/1 | 0.000 |
| qtrm_core_steps_4_no_evidence | 1/1 | 1.000 |
| qtrm_core_steps_8_no_evidence | 0/1 | 0.000 |

## Depth Output Diversity

- Comparable cases: `1`
- Identical across all depth modes: `0`
- Changed by depth: `1`
- All depth outputs identical: `False`

## Interpretation Rule

- `accepted` means the tested raw-intelligence component was causally useful on this eval.
- `rejected` means the component did not beat the simpler ablation/baseline or a shortcut contaminated the run.
- `inconclusive` means required modes are missing or empty.
- This gate is not a RAG, MemoryOS, answer-format, or SSOT score.
