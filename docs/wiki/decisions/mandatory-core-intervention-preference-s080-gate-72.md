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
| qtrm_residual_with_evidence | 50/72 | 0.694 |

## Comparison Checks

| Mode | Hits | Hit advantage | Accuracy advantage | Gate |
| --- | ---: | ---: | ---: | --- |
| donor_only_with_evidence | 39/72 | +11 | +0.153 | baseline-beats |

## Critical Mode Checks

| Mode | Hits | Hit drop | Accuracy drop | Same completions | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| qtrm_core_off_with_evidence | 39/72 | +11 | +0.153 | 18/72 (0.250) | causal-drop |
| qtrm_workspace_off_with_evidence | 39/72 | +11 | +0.153 | 18/72 (0.250) | causal-drop |

## Checks

- Passed: `critical_causal_drop_present`
- Failed: `none`
- Missing modes: `none`
- Missing comparison modes: `none`
- Weak comparison modes: `none`
- Improving critical modes: `none`

## Interpretation Rule

- `causal_gate_status=accepted` means at least one critical ablation worsened a successful baseline.
- `status=accepted` means the causal gate passed and any enabled strict promotion checks also passed.
- `rejected` means the checkpoint cannot support the current root architecture claim on this eval.
- `inconclusive` means the required baseline or ablation rows are missing.
