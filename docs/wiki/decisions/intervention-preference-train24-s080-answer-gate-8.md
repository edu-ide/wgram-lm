# Root Architecture Causality Gate

## Verdict

Status: `rejected`

Causal gate status: `rejected`

Strict promotion required: `True`

Claim: QTRM canonical answer path should improve over donor-only and lose when critical causal components are disabled.

Recommendation: Do not spend another local loss/threshold on this checkpoint. Move the answer signal onto a forced workspace/evidence bottleneck, then rerun workspace/core/memory-off ablations.

## Baseline

| Mode | Hits | Accuracy |
| --- | ---: | ---: |
| qtrm_residual_with_evidence | 5/8 | 0.625 |

## Comparison Checks

| Mode | Hits | Hit advantage | Accuracy advantage | Gate |
| --- | ---: | ---: | ---: | --- |
| donor_only_with_evidence | 5/8 | +0 | +0.000 | not-beaten |

## Critical Mode Checks

| Mode | Hits | Hit drop | Accuracy drop | Same completions | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| qtrm_workspace_off_with_evidence | 5/8 | +0 | +0.000 | 4/8 (0.500) | no-drop |
| qtrm_core_off_with_evidence | 6/8 | -1 | -0.125 | 4/8 (0.500) | no-drop |
| qtrm_workspace_memory_off_with_evidence | 5/8 | +0 | +0.000 | 8/8 (1.000) | no-drop, same-output |
| qtrm_core_context_off_with_evidence | 5/8 | +0 | +0.000 | 8/8 (1.000) | no-drop, same-output |
| qtrm_core_to_text_off_with_evidence | 5/8 | +0 | +0.000 | 8/8 (1.000) | no-drop, same-output |
| qtrm_evidence_bottleneck_off_with_evidence | 5/8 | +0 | +0.000 | 8/8 (1.000) | no-drop, same-output |
| qtrm_evidence_span_reader_off_with_evidence | 5/8 | +0 | +0.000 | 8/8 (1.000) | no-drop, same-output |
| qtrm_answer_residual_governor_off_with_evidence | 6/8 | -1 | -0.125 | 3/8 (0.375) | no-drop |

## Checks

- Passed: `none`
- Failed: `no_critical_causal_drop, critical_ablations_match_baseline_identity, baseline_does_not_beat_comparison, critical_ablation_beats_baseline`
- Missing modes: `none`
- Missing comparison modes: `none`
- Weak comparison modes: `donor_only_with_evidence`
- Improving critical modes: `qtrm_core_off_with_evidence, qtrm_answer_residual_governor_off_with_evidence`

## Interpretation Rule

- `causal_gate_status=accepted` means at least one critical ablation worsened a successful baseline.
- `status=accepted` means the causal gate passed and any enabled strict promotion checks also passed.
- `rejected` means the checkpoint cannot support the current root architecture claim on this eval.
- `inconclusive` means the required baseline or ablation rows are missing.
