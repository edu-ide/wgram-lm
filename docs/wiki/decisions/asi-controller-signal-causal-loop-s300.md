# ASI Controller Causal Loop Eval

## Verdict

Status: `rejected`

This is an action-policy gate, not an answer-quality or ASI proof.

When `controller_signal` is present, it is an oracle scaffold for wiring future learned world-model and verifier outputs into the controller. It is not itself proof that those heads are learned.

## Controller Metrics

| Mode | Accuracy | Samples |
| --- | ---: | ---: |
| qtrm_controller_signal_off | 0.3333 | 216 |
| qtrm_core_to_text_off | 0.9167 | 216 |
| qtrm_harness | 0.9444 | 216 |
| qtrm_latent_core_off | 1.0000 | 216 |
| qtrm_verifier_off | 0.6111 | 216 |
| qtrm_workspace_memory_off | 0.9444 | 216 |
| qtrm_workspace_off | 1.0000 | 216 |
| qtrm_world_model_off | 0.3333 | 216 |

## ASI Gate Metrics

| Metric | Value |
| --- | ---: |
| donor_harness | 1.0000 |
| qtrm_harness | 0.9444 |
| qtrm_latent_core_off | 1.0000 |
| qtrm_verifier_off | 0.6111 |
| qtrm_world_model_off | 0.3333 |
| scripted_harness | 1.0000 |

## Failed Checks

- `qtrm_does_not_beat_donor_harness`
- `qtrm_does_not_beat_scripted_harness`
- `latent_core_not_causal`

## Interpretation

The QTRM controller can imitate the explicit retrieve-verify-answer trace policy. If controller signals are present, world-model-off zeros signal dimension 0 and verifier-off zeros signal dimension 1. The ASI gate should still reject unless QTRM beats the scripted and donor harness baselines and all required ablation drops are present.
