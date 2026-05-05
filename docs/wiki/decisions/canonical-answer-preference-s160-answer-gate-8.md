# Root Architecture Causality Gate

## Verdict

Status: `rejected`

Causal gate status: `accepted`

Strict promotion required: `True`

Claim: QTRM canonical answer path should improve over donor-only and lose when critical causal components are disabled.

Recommendation: Keep the causal component signal as a diagnostic result, but do not promote the full architecture until it beats donor-only and critical ablations no longer outperform the full path.

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
| qtrm_workspace_off_with_evidence | 5/8 | +0 | +0.000 | 1/8 (0.125) | no-drop |
| qtrm_core_off_with_evidence | 6/8 | -1 | -0.125 | 3/8 (0.375) | no-drop |
| qtrm_workspace_memory_off_with_evidence | 5/8 | +0 | +0.000 | 8/8 (1.000) | no-drop, same-output |
| qtrm_core_context_off_with_evidence | 5/8 | +0 | +0.000 | 8/8 (1.000) | no-drop, same-output |
| qtrm_core_to_text_off_with_evidence | 5/8 | +0 | +0.000 | 8/8 (1.000) | no-drop, same-output |
| qtrm_evidence_bottleneck_off_with_evidence | 5/8 | +0 | +0.000 | 8/8 (1.000) | no-drop, same-output |
| qtrm_evidence_span_reader_off_with_evidence | 5/8 | +0 | +0.000 | 8/8 (1.000) | no-drop, same-output |
| qtrm_answer_residual_governor_off_with_evidence | 4/8 | +1 | +0.125 | 3/8 (0.375) | causal-drop |

## Checks

- Passed: `critical_causal_drop_present`
- Failed: `critical_ablations_match_baseline_identity, baseline_does_not_beat_comparison, critical_ablation_beats_baseline`
- Missing modes: `none`
- Missing comparison modes: `none`
- Weak comparison modes: `donor_only_with_evidence`
- Improving critical modes: `qtrm_core_off_with_evidence`

## Interpretation Rule

- `causal_gate_status=accepted` means at least one critical ablation worsened a successful baseline.
- `status=accepted` means the causal gate passed and any enabled strict promotion checks also passed.
- `rejected` means the checkpoint cannot support the current root architecture claim on this eval.
- `inconclusive` means the required baseline or ablation rows are missing.
