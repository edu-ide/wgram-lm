# ASI Controller Causal Loop Eval

## Verdict

Status: `rejected`

This is an action-policy gate, not an answer-quality or ASI proof.

## Controller Metrics

| Mode | Accuracy | Samples |
| --- | ---: | ---: |
| qtrm_core_to_text_off | 1.0000 | 216 |
| qtrm_harness | 1.0000 | 216 |
| qtrm_latent_core_off | 0.7037 | 216 |
| qtrm_workspace_memory_off | 1.0000 | 216 |
| qtrm_workspace_off | 0.6667 | 216 |

## ASI Gate Metrics

| Metric | Value |
| --- | ---: |
| donor_harness | 1.0000 |
| qtrm_harness | 1.0000 |
| qtrm_latent_core_off | 0.7037 |
| qtrm_verifier_off | 1.0000 |
| qtrm_world_model_off | 1.0000 |
| scripted_harness | 1.0000 |

## Failed Checks

- `qtrm_does_not_beat_donor_harness`
- `qtrm_does_not_beat_scripted_harness`
- `world_model_not_causal`
- `verifier_not_causal`

## Interpretation

The QTRM controller can imitate the explicit retrieve-verify-answer trace policy. The ASI gate is expected to reject here because the scripted harness already performs the same action ordering and the world-model/verifier paths are not yet causal for this metric.
