# Metacognitive Calibration Gate

Status: `rejected`

Baseline: `no_warmup_s001_40`
Candidate: `metacog_forced_choice_s080_40`

## Global Comparison

| Metric | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| `accuracy` | 0.466667 | 0.433333 | -0.033333 |
| `ece` | 0.333939 | 0.295489 | -0.038450 |
| `brier` | 0.286778 | 0.293834 | +0.007056 |
| `mean_confidence` | 0.762159 | 0.582662 | -0.179497 |
| `avg_confidence_when_wrong` | 0.635804 | 0.534910 | -0.100895 |

## Checks

Passed: `candidate_ece_not_worse, candidate_calibration_improved, critical_qtrm_mode_calibration_improved`

Failed: `candidate_accuracy_dropped, candidate_brier_worse, critical_mode_qtrm_core_steps_8_no_evidence_accuracy_dropped, critical_mode_qtrm_core_steps_8_low_donor_no_evidence_ece_worse, critical_mode_qtrm_core_steps_8_low_donor_no_evidence_brier_worse, critical_mode_qtrm_core_steps_8_qtrm_only_no_evidence_accuracy_dropped`

## Mode Comparisons

| Mode | Acc Delta | ECE Delta | Brier Delta | Wrong-Conf Delta |
| --- | ---: | ---: | ---: | ---: |
| `donor_only_no_evidence` | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `qtrm_core_off_no_evidence` | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `qtrm_core_off_qtrm_only_no_evidence` | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `qtrm_core_steps_8_low_donor_no_evidence` | +0.150000 | +0.224569 | +0.055008 | +0.040645 |
| `qtrm_core_steps_8_no_evidence` | -0.175000 | -0.142373 | -0.006337 | -0.380628 |
| `qtrm_core_steps_8_qtrm_only_no_evidence` | -0.175000 | -0.142373 | -0.006337 | -0.380628 |

Critical modes: `qtrm_core_steps_8_no_evidence, qtrm_core_steps_8_low_donor_no_evidence, qtrm_core_steps_8_qtrm_only_no_evidence`

## Category Comparisons

| Category | Acc Delta | ECE Delta | Brier Delta | Wrong-Conf Delta |
| --- | ---: | ---: | ---: | ---: |
| `answerable_arithmetic` | +0.000000 | +0.025842 | +0.046820 | +0.031010 |
| `answerable_boolean` | +0.375000 | -0.441756 | -0.388177 | -0.277607 |
| `contradiction` | -0.208333 | -0.023308 | +0.123619 | +0.040395 |
| `ood_random_token` | +0.000000 | +0.050227 | +0.162993 | +0.000000 |
| `unknown_missing` | -0.333333 | +0.193261 | +0.090024 | +0.025698 |

## Expected Unknown Comparisons

| Expected Unknown | Acc Delta | ECE Delta | Brier Delta | Wrong-Conf Delta |
| --- | ---: | ---: | ---: | ---: |
| `False` | +0.187500 | -0.203275 | -0.170679 | -0.042742 |
| `True` | -0.180556 | +0.012569 | +0.125545 | +0.022903 |

## Notes

- Confidence is softmax over forced-choice logprob scores.
- This gate uses choice-score calibration; it is not a full generative calibration proof.
- Promotion still needs donor-only, fused, low-donor/QTRM-only, and core-off comparisons.
