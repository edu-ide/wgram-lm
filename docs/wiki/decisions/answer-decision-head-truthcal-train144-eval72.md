# Learned Answer Decision Head

## Verdict

Status: `accepted`

## Setup

- train records: `docs/wiki/decisions/evidence-span-truthcal-train144-answer-channel-records.jsonl`
- eval records: `docs/wiki/decisions/evidence-span-truthcal-72-answer-channel-records.jsonl`
- selected threshold: `0.63`
- include task-family features: `False`

## Metrics

| Split | Baseline Acc | Learned Acc | Baseline FP | Learned FP | Block Improved | Block Harmed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 0.7569 | 0.9444 | 27 | 0 | 27 | 0 |
| eval | 0.6806 | 0.8611 | 13 | 0 | 13 | 0 |

## Failed Checks

- none

## Boundary

This is a post-hoc learned decision head over recorded answer-channel telemetry. It is not yet wired into QTRM forward or trained end-to-end. It is a falsification gate for whether verifier telemetry contains enough signal to justify an integrated answer-decision module.

## Interpretation

This passes the minimal falsification gate that the static threshold failed.
Using train records from `memory_reasoning_synth_train_cases` and evaluating on
the separate 72-case heldout set:

```text
baseline heldout: 49 / 72 = 0.6806
learned decision heldout: 62 / 72 = 0.8611
false positives: 13 -> 0
block improved: 13
block harmed: 0
block neutral: 1
```

The result says the verifier/span telemetry contains learnable signal for
answer blocking. It does not yet prove an integrated QTRM verifier head, because
the MLP is trained post-hoc on recorded telemetry. The next implementation step
is to wire this decision into the answer loop and then train an in-model
answer-decision head with the same labels.

## Runtime Integration

The learned head is now wired into `scripts/95_eval_memory_retrieval.py` as an
optional runtime decision stage:

```text
--answer-decision-checkpoint runs/qwen35_2b_4090_answer_decision_head_truthcal_train144_eval72/last.pt
```

Runtime held-out result:

```text
report: docs/wiki/decisions/evidence-span-truthcal-72-answer-decision.md
records: docs/wiki/decisions/evidence-span-truthcal-72-answer-decision-records.jsonl
baseline span/truth: 49 / 72 = 0.6806
runtime decision: 62 / 72 = 0.8611
blocked candidates: 14
expected-unknown false positives: 0
```

This confirms that the gain survives the real eval path. The remaining
limitation is unchanged: it is still a post-hoc MLP, not an in-model QTRM head.
