# ASI Controller Causal Loop Eval

## Verdict

Status: `rejected`

This is an action-policy gate, not an answer-quality or ASI proof.

This run used `controller_signal_source=learned_core`, so external oracle
`controller_signal` was not passed into the model. The signal was predicted
from the current latent core, then projected into the action controller.

The result is rejected because the policy collapses to one action on held-out
rows and the required core/world/verifier ablation drops are absent.

## Controller Metrics

| Mode | Accuracy | Samples |
| --- | ---: | ---: |
| qtrm_controller_signal_off | 0.3333 | 216 |
| qtrm_core_to_text_off | 0.3333 | 216 |
| qtrm_harness | 0.3333 | 216 |
| qtrm_latent_core_off | 0.3333 | 216 |
| qtrm_verifier_off | 0.3333 | 216 |
| qtrm_workspace_memory_off | 0.3333 | 216 |
| qtrm_workspace_off | 0.3333 | 216 |
| qtrm_world_model_off | 0.3333 | 216 |

## ASI Gate Metrics

| Metric | Value |
| --- | ---: |
| donor_harness | 1.0000 |
| qtrm_harness | 0.3333 |
| qtrm_latent_core_off | 0.3333 |
| qtrm_verifier_off | 0.3333 |
| qtrm_world_model_off | 0.3333 |
| scripted_harness | 1.0000 |

## Failed Checks

- `qtrm_does_not_beat_donor_harness`
- `qtrm_does_not_beat_scripted_harness`
- `latent_core_not_causal`
- `world_model_not_causal`
- `verifier_not_causal`

## Interpretation

The oracle controller-signal scaffold can drive the action policy, but this
head-only learned replacement does not recover the scaffold. The learned signal
collapses to an `ANSWER`-favoring path on held-out rows. World-model-off masks
controller signal dimension 0 and verifier-off masks dimension 1, either on an
external oracle signal or on the learned core-derived signal. The ASI gate
correctly rejects because QTRM does not beat scripted/donor harness baselines
and because the latent core, world-model, and verifier paths are not causally
necessary under this learned-signal variant.
