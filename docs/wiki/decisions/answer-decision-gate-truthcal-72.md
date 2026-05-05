# Answer Decision Gate Calibration

## Verdict

Status: `rejected`

## Thresholds

| Threshold | Value |
| --- | ---: |
| support_min | 0.500 |
| causal_min | 0.750 |
| refute_max | 0.250 |
| missing_max | 0.550 |

## Metrics

| Split | Baseline Acc | Gated Acc | Baseline FP | Gated FP | Blocked Positive |
| --- | ---: | ---: | ---: | ---: | ---: |
| calibration | 0.6111 | 0.8056 | 11 | 0 | 4 |
| heldout | 0.7500 | 0.4167 | 2 | 0 | 16 |
| full | 0.6806 | 0.6111 | 13 | 0 | 20 |

## Boundary

This is not a learned verifier. It is a calibration probe over existing truth-gate probabilities. It can justify adding an answer-decision head only if held-out false positives drop without destroying positive answer recall.

## Interpretation

The static gate overfits the calibration split. It removes false positives, but
the heldout split loses 16 positive answers and falls from `0.7500` to `0.4167`.

Decision:
do not ship a threshold-only verifier. The next architecture needs a learned
answer-decision head trained with positive, negative, conflict, redaction, and
temporal examples, plus held-out false-positive and positive-recall gates.
