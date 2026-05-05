# Learned-State Answer Loop Gate

## Verdict

Status: `rejected`

This is a task-level answer reward gate for the learned transition-state controller.
The runtime rows hide trace-step and phase-specific state-summary text.

## Metrics

| Metric | Value |
| --- | ---: |
| learned_state_qtrm_accuracy | 0.2500 |
| scripted_qtrm_accuracy | 0.5000 |
| scripted_donor_accuracy | 0.5000 |
| state_off_accuracy | 0.2500 |
| gain_over_scripted_qtrm | -0.2500 |
| gain_over_scripted_donor | -0.2500 |
| transition_state_drop | 0.0000 |
| action_success_rate | 0.0000 |

## Mode Summary

| Mode | Hits | Count | Accuracy |
| --- | ---: | ---: | ---: |
| learned_state_qtrm | 2 | 8 | 0.2500 |
| learned_state_qtrm_state_off | 2 | 8 | 0.2500 |
| scripted_donor_answer_channel | 4 | 8 | 0.5000 |
| scripted_qtrm_answer_channel | 4 | 8 | 0.5000 |

## Failed Checks

- `learned_state_does_not_beat_scripted_qtrm`
- `learned_state_does_not_beat_scripted_donor`
- `transition_state_not_causal_for_answer_reward`
- `learned_action_loop_not_stable`

## Interpretation

This is a rejection: the learned transition-state loop did not prove task-level value over the simpler baselines.

Action-loop failures:
- `synthetic-negative-authority-redacted-0102`: expected_RETRIEVE_MEMORY_got_VERIFY_EVIDENCE; sequence=`VERIFY_EVIDENCE`
- `synthetic-multihop-ko-owner-0102`: expected_RETRIEVE_MEMORY_got_VERIFY_EVIDENCE; sequence=`VERIFY_EVIDENCE`
- `synthetic-multihop-ko-location-3hop-0103`: expected_RETRIEVE_MEMORY_got_VERIFY_EVIDENCE; sequence=`VERIFY_EVIDENCE`
- `synthetic-negative-authority-ko-location-0102`: expected_RETRIEVE_MEMORY_got_VERIFY_EVIDENCE; sequence=`VERIFY_EVIDENCE`
- `synthetic-multihop-maintainer-3hop-0100`: expected_RETRIEVE_MEMORY_got_VERIFY_EVIDENCE; sequence=`VERIFY_EVIDENCE`
- `synthetic-authority-vault-0100`: expected_RETRIEVE_MEMORY_got_VERIFY_EVIDENCE; sequence=`VERIFY_EVIDENCE`
- `synthetic-temporal-code-0101`: expected_RETRIEVE_MEMORY_got_VERIFY_EVIDENCE; sequence=`VERIFY_EVIDENCE`
- `synthetic-temporal-ko-location-0101`: expected_RETRIEVE_MEMORY_got_VERIFY_EVIDENCE; sequence=`VERIFY_EVIDENCE`

## Boundary

An accepted action loop is not enough. This gate only accepts if the learned-state loop improves answer reward over both scripted QTRM and scripted donor answer-channel baselines while dropping under transition-state ablation.

The controller input uses strict runtime rows: no trace-step oracle and no phase-specific state summary. Evidence is hidden from the controller prompt until the predicted `RETRIEVE_MEMORY` action places it into the previous-observation path.
