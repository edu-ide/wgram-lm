# Metacognitive Calibration Gate

Status: `rejected`

Baseline: `no_warmup_s001_40`
Candidate: `noise_warmup_s001_40`

## Global Comparison

| Metric | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| `accuracy` | 0.466667 | 0.466667 | +0.000000 |
| `ece` | 0.333939 | 0.337422 | +0.003483 |
| `brier` | 0.286778 | 0.286384 | -0.000394 |
| `mean_confidence` | 0.762159 | 0.766607 | +0.004448 |
| `avg_confidence_when_wrong` | 0.635804 | 0.635978 | +0.000174 |

## Checks

Passed: `candidate_accuracy_not_lower, candidate_brier_not_worse, candidate_calibration_improved, critical_mode_qtrm_core_steps_8_no_evidence_not_worse, critical_mode_qtrm_core_steps_8_qtrm_only_no_evidence_not_worse, critical_qtrm_mode_calibration_improved`

Failed: `candidate_ece_worse, critical_mode_qtrm_core_steps_8_low_donor_no_evidence_ece_worse`

## Mode Comparisons

| Mode | Acc Delta | ECE Delta | Brier Delta | Wrong-Conf Delta |
| --- | ---: | ---: | ---: | ---: |
| `donor_only_no_evidence` | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `qtrm_core_off_no_evidence` | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `qtrm_core_off_qtrm_only_no_evidence` | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `qtrm_core_steps_8_low_donor_no_evidence` | +0.000000 | +0.003209 | -0.002329 | -0.004076 |
| `qtrm_core_steps_8_no_evidence` | +0.000000 | -0.016408 | -0.000016 | +0.002733 |
| `qtrm_core_steps_8_qtrm_only_no_evidence` | +0.000000 | -0.016408 | -0.000016 | +0.002733 |

Critical modes: `qtrm_core_steps_8_no_evidence, qtrm_core_steps_8_low_donor_no_evidence, qtrm_core_steps_8_qtrm_only_no_evidence`

## Category Comparisons

| Category | Acc Delta | ECE Delta | Brier Delta | Wrong-Conf Delta |
| --- | ---: | ---: | ---: | ---: |
| `answerable_arithmetic` | +0.000000 | -0.003015 | -0.004881 | -0.003618 |
| `answerable_boolean` | +0.000000 | +0.003479 | +0.006817 | +0.004174 |
| `contradiction` | +0.000000 | -0.012540 | -0.003377 | +0.000000 |
| `ood_random_token` | +0.000000 | -0.003655 | -0.000152 | +0.000000 |
| `unknown_missing` | +0.000000 | -0.005583 | -0.000376 | +0.000000 |

## Expected Unknown Comparisons

| Expected Unknown | Acc Delta | ECE Delta | Brier Delta | Wrong-Conf Delta |
| --- | ---: | ---: | ---: | ---: |
| `False` | +0.000000 | +0.000232 | +0.000968 | +0.000278 |
| `True` | +0.000000 | -0.007259 | -0.001301 | +0.000000 |

## Notes

- Confidence is softmax over forced-choice logprob scores.
- This gate uses choice-score calibration; it is not a full generative calibration proof.
- Promotion still needs donor-only, fused, low-donor/QTRM-only, and core-off comparisons.
