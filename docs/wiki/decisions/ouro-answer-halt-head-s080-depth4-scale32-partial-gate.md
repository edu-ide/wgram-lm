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

## Mode Semantics

| Label | Meaning |
| --- | --- |
| donor | Donor baseline. QTRM residual logits are forced off and donor logits are used as the scoring policy; this is the real donor-only comparison. |
| core_off | Internal QTRM ablation. The model still runs through the QTRM forward path with disable_core=True; donor fallback is not forced, so this is not equivalent to donor_only. |
| deepest_core | QTRM candidate with recursive core enabled at the deepest evaluated core_steps value. |
| transition_state_off | QTRM candidate with the recursive core still enabled but the explicit transition-state/code path disabled; this tests whether that state path is answer-causal. |

## Eval Contract

- Scoring: `causal_forced_choice`
- Choice score normalization: `mean`

## Mode Metrics

| Label | Mode | Hits | Accuracy |
| --- | --- | ---: | ---: |
| donor | donor_only_no_evidence | 0/32 | 0.000 |
| core_off | qtrm_core_off_no_evidence | 0/32 | 0.000 |
| deepest_core | qtrm_core_steps_4_no_evidence | 16/32 | 0.500 |

## Depth Ladder

| Mode | Hits | Accuracy |
| --- | ---: | ---: |
| qtrm_core_steps_1_no_evidence | 4/32 | 0.125 |
| qtrm_core_steps_2_no_evidence | 4/32 | 0.125 |
| qtrm_core_steps_4_no_evidence | 16/32 | 0.500 |

## Depth Output Diversity

- Comparable cases: `32`
- Identical across all depth modes: `6`
- Changed by depth: `26`
- All depth outputs identical: `False`

## Expected-Paradigm Metrics

| Expected paradigm | Mode | Hits | Accuracy |
| --- | --- | ---: | ---: |
| hybrid_or_cot | donor_only_no_evidence | 0/32 | 0.000 |
| hybrid_or_cot | qtrm_core_off_no_evidence | 0/32 | 0.000 |
| hybrid_or_cot | qtrm_core_steps_1_no_evidence | 4/32 | 0.125 |
| hybrid_or_cot | qtrm_core_steps_2_no_evidence | 4/32 | 0.125 |
| hybrid_or_cot | qtrm_core_steps_4_no_evidence | 16/32 | 0.500 |

## Interpretation Rule

- `accepted` means the tested raw-intelligence component was causally useful on this eval.
- `rejected` means the component did not beat the simpler ablation/baseline or a shortcut contaminated the run.
- `inconclusive` means required modes are missing or empty.
- This gate is not a RAG, MemoryOS, answer-format, or SSOT score.
