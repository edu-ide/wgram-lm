# ASI Controller Causal Loop Eval

## Verdict

Status: `rejected`

This is an action-policy gate, not an answer-quality or ASI proof.

This run used `controller_signal_source=learned_readout`. External oracle
`controller_signal` was not passed into the model; the signal was predicted
from the final generation/coda readout, then projected into the frozen action
mapping. This is a diagnostic, not the final ASI path.

## Controller Metrics

| Mode | Accuracy | Samples |
| --- | ---: | ---: |
| qtrm_controller_signal_off | 0.3333 | 216 |
| qtrm_core_to_text_off | 0.3333 | 216 |
| qtrm_harness | 0.3704 | 216 |
| qtrm_latent_core_off | 0.5926 | 216 |
| qtrm_verifier_off | 0.3333 | 216 |
| qtrm_workspace_memory_off | 0.3704 | 216 |
| qtrm_workspace_off | 0.6296 | 216 |
| qtrm_world_model_off | 0.3333 | 216 |

## ASI Gate Metrics

| Metric | Value |
| --- | ---: |
| donor_harness | 1.0000 |
| qtrm_harness | 0.3704 |
| qtrm_latent_core_off | 0.5926 |
| qtrm_verifier_off | 0.3333 |
| qtrm_world_model_off | 0.3333 |
| scripted_harness | 1.0000 |

## Failed Checks

- `qtrm_does_not_beat_donor_harness`
- `qtrm_does_not_beat_scripted_harness`
- `latent_core_not_causal`

## Interpretation

The readout diagnostic improves slightly over the `learned_core` variants but
still fails held-out action selection. It mostly predicts `ANSWER`; only 8 of
72 `VERIFY_EVIDENCE` rows are correct and no `RETRIEVE_MEMORY` rows are
correct. `latent_core_off` and `workspace_off` score higher than the full
model, which means the current latent/workspace path is not a reliable causal
planner in this setup. The ASI gate correctly rejects because QTRM does not
beat scripted/donor harness baselines and the latent core is not causally
beneficial.
