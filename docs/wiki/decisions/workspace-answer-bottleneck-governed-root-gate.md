# Root Architecture Causality Gate

## Verdict

Status: `accepted`

Claim: QTRM latent workspace/core/evidence paths are causally necessary for the residual answer behavior on this eval.

Recommendation: Keep this architecture candidate under test, but require held-out generation quality and donor-only comparison before treating it as solved.

## Baseline

| Mode | Hits | Accuracy |
| --- | ---: | ---: |
| qtrm_residual_with_evidence | 1/4 | 0.250 |

## Critical Mode Checks

| Mode | Hits | Hit drop | Accuracy drop | Same completions | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| qtrm_workspace_off_with_evidence | 0/4 | +1 | +0.250 | 1/4 (0.250) | causal-drop |
| qtrm_core_off_with_evidence | 0/4 | +1 | +0.250 | 0/4 (0.000) | causal-drop |
| qtrm_workspace_memory_off_with_evidence | 0/4 | +1 | +0.250 | 1/4 (0.250) | causal-drop |
| qtrm_core_context_off_with_evidence | 1/4 | +0 | +0.000 | 4/4 (1.000) | no-drop, same-output |
| qtrm_evidence_bottleneck_off_with_evidence | 1/4 | +0 | +0.000 | 4/4 (1.000) | no-drop, same-output |

## Checks

- Passed: `critical_causal_drop_present`
- Failed: `critical_ablations_match_baseline_identity`
- Missing modes: `none`

## Interpretation Rule

- `accepted` means at least one critical workspace/core/evidence ablation worsened a successful baseline.
- `rejected` means the checkpoint cannot support the current root architecture claim on this eval.
- `inconclusive` means the required baseline or ablation rows are missing.
