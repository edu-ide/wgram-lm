# Learned-State Answer Loop Gate

## Verdict

Status: `rejected`

This is a task-level answer reward gate for the learned transition-state controller.
The runtime rows hide trace-step and phase-specific state-summary text.

## Metrics

| Metric | Value |
| --- | ---: |
| learned_state_qtrm_accuracy | 0.6250 |
| scripted_qtrm_accuracy | 0.5000 |
| scripted_donor_accuracy | 0.5000 |
| state_off_accuracy | 0.2500 |
| gain_over_scripted_qtrm | 0.1250 |
| gain_over_scripted_donor | 0.1250 |
| transition_state_drop | 0.3750 |
| action_success_rate | 0.8750 |

## Mode Summary

| Mode | Hits | Count | Accuracy |
| --- | ---: | ---: | ---: |
| learned_state_qtrm | 5 | 8 | 0.6250 |
| learned_state_qtrm_state_off | 2 | 8 | 0.2500 |
| scripted_donor_answer_channel | 4 | 8 | 0.5000 |
| scripted_qtrm_answer_channel | 4 | 8 | 0.5000 |

## Failed Checks

- `learned_action_loop_not_stable`

## Interpretation

This is a near-miss, not an acceptance: answer accuracy improved over both scripted baselines and the state-off ablation dropped, but at least one strict gate still failed.

Action-loop failures:
- `synthetic-negative-authority-ko-location-0102`: expected_RETRIEVE_MEMORY_got_ANSWER; sequence=`ANSWER`

## Boundary

An accepted action loop is not enough. This gate only accepts if the learned-state loop improves answer reward over both scripted QTRM and scripted donor answer-channel baselines while dropping under transition-state ablation.

The controller input uses strict runtime rows: no trace-step oracle and no phase-specific state summary. Evidence is hidden from the controller prompt until the predicted `RETRIEVE_MEMORY` action places it into the previous-observation path.
