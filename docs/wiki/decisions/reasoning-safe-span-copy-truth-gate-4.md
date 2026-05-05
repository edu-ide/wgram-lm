# Root Architecture Causality Gate

## Verdict

Status: `rejected`

Claim: QTRM latent workspace/core/evidence paths are causally necessary for the residual answer behavior on this eval.

Recommendation: Do not spend another local loss/threshold on this checkpoint. Move the answer signal onto a forced workspace/evidence bottleneck, then rerun workspace/core/memory-off ablations.

## Baseline

| Mode | Hits | Accuracy |
| --- | ---: | ---: |
| qtrm_residual_with_evidence | 2/4 | 0.500 |

## Critical Mode Checks

| Mode | Hits | Hit drop | Accuracy drop | Same completions | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| qtrm_workspace_off_with_evidence | 2/4 | +0 | +0.000 | 4/4 (1.000) | no-drop, same-output |
| qtrm_core_off_with_evidence | 2/4 | +0 | +0.000 | 4/4 (1.000) | no-drop, same-output |
| qtrm_workspace_memory_off_with_evidence | 2/4 | +0 | +0.000 | 4/4 (1.000) | no-drop, same-output |
| qtrm_core_context_off_with_evidence | 3/4 | -1 | -0.250 | 1/4 (0.250) | no-drop |
| qtrm_evidence_bottleneck_off_with_evidence | 2/4 | +0 | +0.000 | 4/4 (1.000) | no-drop, same-output |
| qtrm_evidence_span_reader_off_with_evidence | 2/4 | +0 | +0.000 | 4/4 (1.000) | no-drop, same-output |

## Checks

- Passed: `none`
- Failed: `no_critical_causal_drop, critical_ablations_match_baseline_identity`
- Missing modes: `none`

## Interpretation Rule

- `accepted` means at least one critical workspace/core/evidence ablation worsened a successful baseline.
- `rejected` means the checkpoint cannot support the current root architecture claim on this eval.
- `inconclusive` means the required baseline or ablation rows are missing.
