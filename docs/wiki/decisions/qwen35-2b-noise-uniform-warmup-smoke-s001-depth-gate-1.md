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
| donor | donor_only_no_evidence | 0/1 | 0.000 |
| core_off | qtrm_core_off_no_evidence | 0/1 | 0.000 |
| deepest_core | qtrm_core_steps_8_no_evidence | 1/1 | 1.000 |

## Depth Ladder

| Mode | Hits | Accuracy |
| --- | ---: | ---: |
| qtrm_core_steps_1_no_evidence | 0/1 | 0.000 |
| qtrm_core_steps_2_no_evidence | 1/1 | 1.000 |
| qtrm_core_steps_4_no_evidence | 1/1 | 1.000 |
| qtrm_core_steps_8_no_evidence | 1/1 | 1.000 |

## Depth Output Diversity

- Comparable cases: `1`
- Identical across all depth modes: `0`
- Changed by depth: `1`
- All depth outputs identical: `False`

## Expected-Paradigm Metrics

| Expected paradigm | Mode | Hits | Accuracy |
| --- | --- | ---: | ---: |
| hybrid_or_cot | donor_only_no_evidence | 0/1 | 0.000 |
| hybrid_or_cot | qtrm_core_off_no_evidence | 0/1 | 0.000 |
| hybrid_or_cot | qtrm_core_steps_1_no_evidence | 0/1 | 0.000 |
| hybrid_or_cot | qtrm_core_steps_2_no_evidence | 1/1 | 1.000 |
| hybrid_or_cot | qtrm_core_steps_4_no_evidence | 1/1 | 1.000 |
| hybrid_or_cot | qtrm_core_steps_8_no_evidence | 1/1 | 1.000 |

## Interpretation Rule

- `accepted` means the tested raw-intelligence component was causally useful on this eval.
- `rejected` means the component did not beat the simpler ablation/baseline or a shortcut contaminated the run.
- `inconclusive` means required modes are missing or empty.
- This gate is not a RAG, MemoryOS, answer-format, or SSOT score.
