# Metacognitive Calibration Gate

Status: `rejected`

Baseline: `no_warmup_s001`
Candidate: `unknown_teacher_kl_conservative_s040`
Gate profile: `fused`
Profile records: `160` / source `480`
Included modes: `qtrm_core_steps_8_no_evidence, qtrm_core_steps_8_low_donor_no_evidence`

## Global Comparison

| Metric | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| `accuracy` | 0.600000 | 0.600000 | +0.000000 |
| `ece` | 0.348036 | 0.362485 | +0.014450 |
| `brier` | 0.364187 | 0.366383 | +0.002196 |
| `mean_confidence` | 0.948036 | 0.962485 | +0.014450 |
| `avg_confidence_when_wrong` | 0.947337 | 0.952518 | +0.005181 |

## Checks

Passed: `candidate_accuracy_not_lower, critical_mode_qtrm_core_steps_8_no_evidence_not_worse, critical_qtrm_mode_calibration_improved`

Failed: `candidate_ece_worse, candidate_brier_worse, candidate_no_calibration_gain, critical_mode_qtrm_core_steps_8_low_donor_no_evidence_ece_worse, critical_mode_qtrm_core_steps_8_low_donor_no_evidence_brier_worse`

## Mode Comparisons

| Mode | Acc Delta | ECE Delta | Brier Delta | Wrong-Conf Delta |
| --- | ---: | ---: | ---: | ---: |
| `qtrm_core_steps_8_low_donor_no_evidence` | +0.000000 | +0.010174 | +0.004743 | +0.006535 |
| `qtrm_core_steps_8_no_evidence` | +0.000000 | -0.014239 | -0.000351 | +0.003826 |

Critical modes: `qtrm_core_steps_8_no_evidence, qtrm_core_steps_8_low_donor_no_evidence`

## Category Comparisons

| Category | Acc Delta | ECE Delta | Brier Delta | Wrong-Conf Delta |
| --- | ---: | ---: | ---: | ---: |
| `answerable_arithmetic` | +0.000000 | -0.000404 | -0.000432 | -0.000404 |
| `answerable_boolean` | +0.000000 | +0.010766 | +0.021083 | +0.010766 |
| `contradiction` | +0.000000 | -0.036899 | -0.008469 | +0.000000 |
| `ood_random_token` | +0.000000 | -0.010572 | -0.000380 | +0.000000 |
| `unknown_missing` | +0.000000 | -0.014416 | -0.000820 | +0.000000 |

## Expected Unknown Comparisons

| Expected Unknown | Acc Delta | ECE Delta | Brier Delta | Wrong-Conf Delta |
| --- | ---: | ---: | ---: | ---: |
| `False` | +0.000000 | +0.005181 | +0.010325 | +0.005181 |
| `True` | +0.000000 | -0.020629 | -0.003223 | +0.000000 |

## Notes

- Confidence is softmax over forced-choice logprob scores.
- This gate uses choice-score calibration; it is not a full generative calibration proof.
- Gate profiles filter records before global metrics; strict keeps all modes.
- Promotion still needs donor-only, fused, low-donor/QTRM-only, and core-off comparisons.
