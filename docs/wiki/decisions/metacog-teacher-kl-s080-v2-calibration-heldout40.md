# Metacognitive Calibration Gate

Status: `rejected`

Baseline: `no_warmup_s001_40`
Candidate: `metacog_teacher_kl_s080_v2_40`

## Global Comparison

| Metric | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| `accuracy` | 0.466667 | 0.466667 | +0.000000 |
| `ece` | 0.333939 | 0.352499 | +0.018560 |
| `brier` | 0.286778 | 0.292267 | +0.005489 |
| `mean_confidence` | 0.762159 | 0.779332 | +0.017173 |
| `avg_confidence_when_wrong` | 0.635804 | 0.643463 | +0.007658 |

## Checks

Passed: `candidate_accuracy_not_lower, critical_mode_qtrm_core_steps_8_no_evidence_not_worse, critical_mode_qtrm_core_steps_8_qtrm_only_no_evidence_not_worse, critical_qtrm_mode_calibration_improved`

Failed: `candidate_ece_worse, candidate_brier_worse, candidate_no_calibration_gain, critical_mode_qtrm_core_steps_8_low_donor_no_evidence_ece_worse, critical_mode_qtrm_core_steps_8_low_donor_no_evidence_brier_worse`

## Mode Comparisons

| Mode | Acc Delta | ECE Delta | Brier Delta | Wrong-Conf Delta |
| --- | ---: | ---: | ---: | ---: |
| `donor_only_no_evidence` | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `qtrm_core_off_no_evidence` | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `qtrm_core_off_qtrm_only_no_evidence` | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `qtrm_core_steps_8_low_donor_no_evidence` | +0.000000 | +0.033534 | +0.035042 | +0.050329 |
| `qtrm_core_steps_8_no_evidence` | +0.000000 | -0.020989 | -0.001054 | +0.005468 |
| `qtrm_core_steps_8_qtrm_only_no_evidence` | +0.000000 | -0.020989 | -0.001054 | +0.005468 |

Critical modes: `qtrm_core_steps_8_no_evidence, qtrm_core_steps_8_low_donor_no_evidence, qtrm_core_steps_8_qtrm_only_no_evidence`

## Category Comparisons

| Category | Acc Delta | ECE Delta | Brier Delta | Wrong-Conf Delta |
| --- | ---: | ---: | ---: | ---: |
| `answerable_arithmetic` | +0.000000 | +0.013497 | +0.023395 | +0.016197 |
| `answerable_boolean` | +0.000000 | +0.006924 | +0.013622 | +0.008309 |
| `contradiction` | +0.000000 | -0.043172 | -0.008629 | +0.000000 |
| `ood_random_token` | +0.000000 | -0.008923 | -0.000274 | +0.000000 |
| `unknown_missing` | +0.000000 | -0.013351 | -0.000670 | +0.000000 |

## Expected Unknown Comparisons

| Expected Unknown | Acc Delta | ECE Delta | Brier Delta | Wrong-Conf Delta |
| --- | ---: | ---: | ---: | ---: |
| `False` | +0.000000 | +0.010211 | +0.018509 | +0.012253 |
| `True` | +0.000000 | -0.021815 | -0.003191 | +0.000000 |

## Notes

- Confidence is softmax over forced-choice logprob scores.
- This gate uses choice-score calibration; it is not a full generative calibration proof.
- Promotion still needs donor-only, fused, low-donor/QTRM-only, and core-off comparisons.
