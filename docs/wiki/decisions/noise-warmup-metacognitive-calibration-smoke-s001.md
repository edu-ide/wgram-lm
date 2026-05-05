# Metacognitive Calibration Gate

Status: `accepted`

Baseline: `no_warmup_s001`
Candidate: `noise_warmup_s001`

## Global Comparison

| Metric | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| `accuracy` | 0.500000 | 0.500000 | +0.000000 |
| `ece` | 0.180124 | 0.176691 | -0.003433 |
| `brier` | 0.247725 | 0.243216 | -0.004509 |
| `mean_confidence` | 0.403210 | 0.406643 | +0.003433 |
| `avg_confidence_when_wrong` | 0.386081 | 0.385566 | -0.000514 |

## Checks

Passed: `candidate_accuracy_not_lower, candidate_ece_not_worse, candidate_brier_not_worse, candidate_calibration_improved`

Failed: ``

## Mode Comparisons

| Mode | Acc Delta | ECE Delta | Brier Delta | Wrong-Conf Delta |
| --- | ---: | ---: | ---: | ---: |
| `donor_only_no_evidence` | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `qtrm_core_off_no_evidence` | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `qtrm_core_steps_1_no_evidence` | +0.000000 | -0.001543 | -0.001524 | -0.001543 |
| `qtrm_core_steps_2_no_evidence` | +0.000000 | -0.006227 | -0.007095 | +0.000000 |
| `qtrm_core_steps_4_no_evidence` | +0.000000 | -0.007663 | -0.008838 | +0.000000 |
| `qtrm_core_steps_8_no_evidence` | +0.000000 | -0.008251 | -0.009597 | +0.000000 |

## Notes

- Confidence is softmax over forced-choice logprob scores.
- This gate uses choice-score calibration; it is not a full generative calibration proof.
- Promotion still needs donor-only, fused, low-donor/QTRM-only, and core-off comparisons.
