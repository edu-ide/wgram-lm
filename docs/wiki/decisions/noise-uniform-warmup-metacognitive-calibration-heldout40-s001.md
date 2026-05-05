# Metacognitive Calibration Gate

Status: `rejected`

Baseline: `no_warmup_s001_40`
Candidate: `noise_uniform_warmup_s001_40`

## Global Comparison

| Metric | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| `accuracy` | 0.466667 | 0.466667 | +0.000000 |
| `ece` | 0.333939 | 0.332757 | -0.001182 |
| `brier` | 0.286778 | 0.286715 | -0.000063 |
| `mean_confidence` | 0.762159 | 0.756753 | -0.005405 |
| `avg_confidence_when_wrong` | 0.635804 | 0.634379 | -0.001426 |

## Checks

Passed: `candidate_accuracy_not_lower, candidate_ece_not_worse, candidate_brier_not_worse, candidate_calibration_improved, critical_mode_qtrm_core_steps_8_low_donor_no_evidence_not_worse, critical_qtrm_mode_calibration_improved`

Failed: `critical_mode_qtrm_core_steps_8_no_evidence_ece_worse, critical_mode_qtrm_core_steps_8_no_evidence_brier_worse, critical_mode_qtrm_core_steps_8_qtrm_only_no_evidence_ece_worse, critical_mode_qtrm_core_steps_8_qtrm_only_no_evidence_brier_worse`

## Mode Comparisons

| Mode | Acc Delta | ECE Delta | Brier Delta | Wrong-Conf Delta |
| --- | ---: | ---: | ---: | ---: |
| `donor_only_no_evidence` | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `qtrm_core_off_no_evidence` | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `qtrm_core_off_qtrm_only_no_evidence` | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `qtrm_core_steps_8_low_donor_no_evidence` | +0.000000 | -0.007945 | -0.004724 | -0.007416 |
| `qtrm_core_steps_8_no_evidence` | +0.000000 | +0.008010 | +0.002172 | -0.001995 |
| `qtrm_core_steps_8_qtrm_only_no_evidence` | +0.000000 | +0.008010 | +0.002172 | -0.001995 |

Critical modes: `qtrm_core_steps_8_no_evidence, qtrm_core_steps_8_low_donor_no_evidence, qtrm_core_steps_8_qtrm_only_no_evidence`

## Category Comparisons

| Category | Acc Delta | ECE Delta | Brier Delta | Wrong-Conf Delta |
| --- | ---: | ---: | ---: | ---: |
| `answerable_arithmetic` | +0.000000 | -0.001228 | -0.002018 | -0.001473 |
| `answerable_boolean` | +0.000000 | -0.002574 | -0.005008 | -0.003089 |
| `contradiction` | +0.000000 | +0.015986 | +0.006168 | +0.000000 |
| `ood_random_token` | +0.000000 | +0.003020 | +0.000166 | +0.000000 |
| `unknown_missing` | +0.000000 | +0.004217 | +0.000373 | +0.000000 |

## Expected Unknown Comparisons

| Expected Unknown | Acc Delta | ECE Delta | Brier Delta | Wrong-Conf Delta |
| --- | ---: | ---: | ---: | ---: |
| `False` | +0.000000 | -0.001901 | -0.003513 | -0.002281 |
| `True` | +0.000000 | +0.007741 | +0.002236 | +0.000000 |

## Notes

- Confidence is softmax over forced-choice logprob scores.
- This gate uses choice-score calibration; it is not a full generative calibration proof.
- Promotion still needs donor-only, fused, low-donor/QTRM-only, and core-off comparisons.
