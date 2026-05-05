# Metacognitive Calibration Gate

Status: `rejected`

Baseline: `no_warmup_s001_40`
Candidate: `metacog_unknown_teacher_kl_s080_40`

## Global Comparison

| Metric | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| `accuracy` | 0.466667 | 0.466667 | +0.000000 |
| `ece` | 0.333939 | 0.363077 | +0.029138 |
| `brier` | 0.286778 | 0.298315 | +0.011536 |
| `mean_confidence` | 0.762159 | 0.787036 | +0.024878 |
| `avg_confidence_when_wrong` | 0.635804 | 0.649499 | +0.013695 |

## Checks

Passed: `candidate_accuracy_not_lower, critical_qtrm_mode_calibration_improved`

Failed: `candidate_ece_worse, candidate_brier_worse, candidate_no_calibration_gain, critical_mode_qtrm_core_steps_8_no_evidence_brier_worse, critical_mode_qtrm_core_steps_8_low_donor_no_evidence_ece_worse, critical_mode_qtrm_core_steps_8_low_donor_no_evidence_brier_worse, critical_mode_qtrm_core_steps_8_qtrm_only_no_evidence_brier_worse`

## Mode Comparisons

| Mode | Acc Delta | ECE Delta | Brier Delta | Wrong-Conf Delta |
| --- | ---: | ---: | ---: | ---: |
| `donor_only_no_evidence` | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `qtrm_core_off_no_evidence` | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `qtrm_core_off_qtrm_only_no_evidence` | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `qtrm_core_steps_8_low_donor_no_evidence` | +0.000000 | +0.054939 | +0.067679 | +0.093112 |
| `qtrm_core_steps_8_no_evidence` | +0.000000 | -0.008579 | +0.000769 | +0.008222 |
| `qtrm_core_steps_8_qtrm_only_no_evidence` | +0.000000 | -0.008579 | +0.000769 | +0.008222 |

Critical modes: `qtrm_core_steps_8_no_evidence, qtrm_core_steps_8_low_donor_no_evidence, qtrm_core_steps_8_qtrm_only_no_evidence`

## Category Comparisons

| Category | Acc Delta | ECE Delta | Brier Delta | Wrong-Conf Delta |
| --- | ---: | ---: | ---: | ---: |
| `answerable_arithmetic` | +0.000000 | +0.026085 | +0.047302 | +0.031302 |
| `answerable_boolean` | +0.000000 | +0.010434 | +0.020612 | +0.012520 |
| `contradiction` | +0.000000 | -0.057868 | -0.009220 | +0.000000 |
| `ood_random_token` | +0.000000 | -0.011979 | -0.000293 | +0.000000 |
| `unknown_missing` | +0.000000 | -0.018023 | -0.000719 | +0.000000 |

## Expected Unknown Comparisons

| Expected Unknown | Acc Delta | ECE Delta | Brier Delta | Wrong-Conf Delta |
| --- | ---: | ---: | ---: | ---: |
| `False` | +0.000000 | +0.022942 | +0.033957 | +0.021911 |
| `True` | +0.000000 | -0.029290 | -0.003411 | +0.000000 |

## Notes

- Confidence is softmax over forced-choice logprob scores.
- This gate uses choice-score calibration; it is not a full generative calibration proof.
- Promotion still needs donor-only, fused, low-donor/QTRM-only, and core-off comparisons.
