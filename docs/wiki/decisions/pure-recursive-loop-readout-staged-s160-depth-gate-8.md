# Raw Intelligence Gate

## Verdict

Gate type: `pure_recursive_reasoning`

Status: `accepted`

Claim: QTRM recursive core depth should improve held-out reasoning without retrieval, MemoryOS, or hidden evidence shortcuts.

Recommendation: Promote this result as raw-intelligence evidence only for the tested axis, then rerun on harder held-out cases before changing architecture.

## Checks

- Passed: `deep_core_beats_core_off, deep_core_beats_donor, depth_scaling_gain_present, depth_outputs_not_all_identical, no_retrieval_or_memoryos_shortcut`
- Failed: `none`
- Missing modes: `none`
- Shortcut records: `0`

## Mode Metrics

| Label | Mode | Hits | Accuracy |
| --- | --- | ---: | ---: |
| donor | donor_only_no_evidence | 2/8 | 0.250 |
| core_off | qtrm_core_off_no_evidence | 0/8 | 0.000 |
| deepest_core | qtrm_core_steps_8_no_evidence | 3/8 | 0.375 |

## Depth Ladder

| Mode | Hits | Accuracy |
| --- | ---: | ---: |
| qtrm_core_steps_1_no_evidence | 3/8 | 0.375 |
| qtrm_core_steps_2_no_evidence | 2/8 | 0.250 |
| qtrm_core_steps_4_no_evidence | 3/8 | 0.375 |
| qtrm_core_steps_8_no_evidence | 3/8 | 0.375 |

## Depth Output Diversity

- Comparable cases: `8`
- Identical across all depth modes: `6`
- Changed by depth: `2`
- All depth outputs identical: `False`

## Interpretation Rule

- `accepted` means the tested raw-intelligence component was causally useful on this eval.
- `rejected` means the component did not beat the simpler ablation/baseline or a shortcut contaminated the run.
- `inconclusive` means required modes are missing or empty.
- This gate is not a RAG, MemoryOS, answer-format, or SSOT score.
