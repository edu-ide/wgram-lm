# Root Architecture Causality Gate

## Verdict

Status: `accepted`

Causal gate status: `accepted`

Strict promotion required: `True`

Claim: QTRM canonical answer path should improve over donor-only and lose when critical causal components are disabled.

Recommendation: Keep this architecture candidate under test. It has at least one causal component drop and passes the enabled promotion checks.

## Baseline

| Mode | Hits | Accuracy |
| --- | ---: | ---: |
| qtrm_residual_with_evidence | 6/8 | 0.750 |

## Comparison Checks

| Mode | Hits | Hit advantage | Accuracy advantage | Gate |
| --- | ---: | ---: | ---: | --- |
| donor_only_with_evidence | 5/8 | +1 | +0.125 | baseline-beats |

## Critical Mode Checks

| Mode | Hits | Hit drop | Accuracy drop | Same completions | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| qtrm_core_off_with_evidence | 6/8 | +0 | +0.000 | 8/8 (1.000) | no-drop, same-output |
| qtrm_workspace_off_with_evidence | 5/8 | +1 | +0.125 | 7/8 (0.875) | causal-drop |

## Checks

- Passed: `critical_causal_drop_present`
- Failed: `critical_ablations_match_baseline_identity`
- Missing modes: `none`
- Missing comparison modes: `none`
- Weak comparison modes: `none`
- Improving critical modes: `none`

## Interpretation Rule

- `causal_gate_status=accepted` means at least one critical ablation worsened a successful baseline.
- `status=accepted` means the causal gate passed and any enabled strict promotion checks also passed.
- `rejected` means the checkpoint cannot support the current root architecture claim on this eval.
- `inconclusive` means the required baseline or ablation rows are missing.
