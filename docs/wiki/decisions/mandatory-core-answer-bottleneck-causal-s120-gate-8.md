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
| qtrm_core_off_with_evidence | 5/8 | +0 | +0.000 | 1/8 (0.125) | no-drop |
| qtrm_workspace_off_with_evidence | 5/8 | +0 | +0.000 | 1/8 (0.125) | no-drop |

## Checks

- Passed: `none`
- Failed: `no_critical_causal_drop, baseline_does_not_beat_comparison`
- Missing modes: `none`
- Missing comparison modes: `none`
- Weak comparison modes: `donor_only_with_evidence`
- Improving critical modes: `none`

## Interpretation Rule

- `causal_gate_status=accepted` means at least one critical ablation worsened a successful baseline.
- `status=accepted` means the causal gate passed and any enabled strict promotion checks also passed.
- `rejected` means the checkpoint cannot support the current root architecture claim on this eval.
- `inconclusive` means the required baseline or ablation rows are missing.
