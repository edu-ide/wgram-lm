# Learned Answer Decision Head

## Verdict

Status: `rejected`

## Setup

- train records: `docs/wiki/decisions/evidence-span-truthcal-72-answer-channel-records.jsonl`
- eval records: `docs/wiki/decisions/evidence-span-truthcal-72-answer-channel-records.jsonl`
- selected threshold: `0.57`
- include task-family features: `False`

## Metrics

| Split | Baseline Acc | Learned Acc | Baseline FP | Learned FP | Block Improved | Block Harmed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 0.6111 | 0.9167 | 11 | 0 | 11 | 0 |
| eval | 0.7500 | 0.6667 | 2 | 0 | 2 | 5 |

## Failed Checks

- `eval_accuracy_gain_too_small`

## Boundary

This is a post-hoc learned decision head over recorded answer-channel telemetry. It is not yet wired into QTRM forward or trained end-to-end. It is a falsification gate for whether verifier telemetry contains enough signal to justify an integrated answer-decision module.
