# Raw Intelligence Gate

## Verdict

Gate type: `pure_recursive_reasoning`

Status: `rejected`

Claim: QTRM recursive core depth should improve held-out reasoning without retrieval, MemoryOS, or hidden evidence shortcuts.

Recommendation: Do not tune answer formatting. Redesign or retrain the recursive core so deeper latent steps beat donor-only and core-off on no-evidence tasks.

## Checks

- Passed: `deep_core_beats_core_off, deep_core_beats_transition_state_off, depth_outputs_not_all_identical, no_retrieval_or_memoryos_shortcut`
- Failed: `deep_core_does_not_beat_donor, no_depth_scaling_gain`
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
- Choice score normalization: `unknown`

## Mode Metrics

| Label | Mode | Hits | Accuracy |
| --- | --- | ---: | ---: |
| donor | donor_only_no_evidence | 1/8 | 0.125 |
| core_off | qtrm_core_off_no_evidence | 0/8 | 0.000 |
| deepest_core | qtrm_core_steps_8_no_evidence | 1/8 | 0.125 |
| transition_state_off | qtrm_core_steps_8_transition_state_off_no_evidence | 0/8 | 0.000 |

## Depth Ladder

| Mode | Hits | Accuracy |
| --- | ---: | ---: |
| qtrm_core_steps_1_no_evidence | 1/8 | 0.125 |
| qtrm_core_steps_2_no_evidence | 1/8 | 0.125 |
| qtrm_core_steps_4_no_evidence | 1/8 | 0.125 |
| qtrm_core_steps_8_no_evidence | 1/8 | 0.125 |

## Depth Output Diversity

- Comparable cases: `8`
- Identical across all depth modes: `7`
- Changed by depth: `1`
- All depth outputs identical: `False`

## Expected-Paradigm Metrics

| Expected paradigm | Mode | Hits | Accuracy |
| --- | --- | ---: | ---: |
| hybrid_or_cot | donor_only_no_evidence | 1/8 | 0.125 |
| hybrid_or_cot | qtrm_core_off_no_evidence | 0/8 | 0.000 |
| hybrid_or_cot | qtrm_core_steps_1_no_evidence | 1/8 | 0.125 |
| hybrid_or_cot | qtrm_core_steps_2_no_evidence | 1/8 | 0.125 |
| hybrid_or_cot | qtrm_core_steps_4_no_evidence | 1/8 | 0.125 |
| hybrid_or_cot | qtrm_core_steps_8_no_evidence | 1/8 | 0.125 |
| hybrid_or_cot | qtrm_core_steps_8_transition_state_off_no_evidence | 0/8 | 0.000 |

## Interpretation Rule

- `accepted` means the tested raw-intelligence component was causally useful on this eval.
- `rejected` means the component did not beat the simpler ablation/baseline or a shortcut contaminated the run.
- `inconclusive` means required modes are missing or empty.
- This gate is not a RAG, MemoryOS, answer-format, or SSOT score.
