# Metacognitive Calibration Gate

Status: `rejected`

Baseline: `no_warmup_s001_40`
Candidate: `metacog_unknown_teacher_kl_conservative_s040_40`

## Global Comparison

| Metric | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| `accuracy` | 0.466667 | 0.466667 | +0.000000 |
| `ece` | 0.333939 | 0.340911 | +0.006972 |
| `brier` | 0.286778 | 0.287452 | +0.000674 |
| `mean_confidence` | 0.762159 | 0.770096 | +0.007937 |
| `avg_confidence_when_wrong` | 0.635804 | 0.637578 | +0.001773 |

## Checks

Passed: `candidate_accuracy_not_lower, critical_mode_qtrm_core_steps_8_no_evidence_not_worse, critical_mode_qtrm_core_steps_8_qtrm_only_no_evidence_not_worse, critical_qtrm_mode_calibration_improved`

Failed: `candidate_ece_worse, candidate_brier_worse, candidate_no_calibration_gain, critical_mode_qtrm_core_steps_8_low_donor_no_evidence_ece_worse, critical_mode_qtrm_core_steps_8_low_donor_no_evidence_brier_worse`

## Mode Comparisons

| Mode | Acc Delta | ECE Delta | Brier Delta | Wrong-Conf Delta |
| --- | ---: | ---: | ---: | ---: |
| `donor_only_no_evidence` | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `qtrm_core_off_no_evidence` | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `qtrm_core_off_qtrm_only_no_evidence` | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `qtrm_core_steps_8_low_donor_no_evidence` | +0.000000 | +0.010174 | +0.004743 | +0.006535 |
| `qtrm_core_steps_8_no_evidence` | +0.000000 | -0.014239 | -0.000351 | +0.003826 |
| `qtrm_core_steps_8_qtrm_only_no_evidence` | +0.000000 | -0.014239 | -0.000351 | +0.003826 |

Critical modes: `qtrm_core_steps_8_no_evidence, qtrm_core_steps_8_low_donor_no_evidence, qtrm_core_steps_8_qtrm_only_no_evidence`

## Category Comparisons

| Category | Acc Delta | ECE Delta | Brier Delta | Wrong-Conf Delta |
| --- | ---: | ---: | ---: | ---: |
| `answerable_arithmetic` | +0.000000 | -0.000134 | -0.000142 | -0.000160 |
| `answerable_boolean` | +0.000000 | +0.004863 | +0.009545 | +0.005835 |
| `contradiction` | +0.000000 | -0.021275 | -0.005324 | +0.000000 |
| `ood_random_token` | +0.000000 | -0.005562 | -0.000209 | +0.000000 |
| `unknown_missing` | +0.000000 | -0.008120 | -0.000501 | +0.000000 |

## Expected Unknown Comparisons

| Expected Unknown | Acc Delta | ECE Delta | Brier Delta | Wrong-Conf Delta |
| --- | ---: | ---: | ---: | ---: |
| `False` | +0.000000 | +0.002365 | +0.004701 | +0.002838 |
| `True` | +0.000000 | -0.011653 | -0.002011 | +0.000000 |

## Notes

- Confidence is softmax over forced-choice logprob scores.
- This gate uses choice-score calibration; it is not a full generative calibration proof.
- Promotion still needs donor-only, fused, low-donor/QTRM-only, and core-off comparisons.
