# Metacognitive Calibration Gate

Status: `accepted`

Baseline: `no_warmup_s001`
Candidate: `unknown_teacher_kl_conservative_s040`
Gate profile: `qtrm_core`
Profile records: `160` / source `480`
Included modes: `qtrm_core_steps_8_no_evidence, qtrm_core_steps_8_qtrm_only_no_evidence`

## Global Comparison

| Metric | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| `accuracy` | 0.600000 | 0.600000 | +0.000000 |
| `ece` | 0.408397 | 0.394158 | -0.014239 |
| `brier` | 0.399219 | 0.398868 | -0.000351 |
| `mean_confidence` | 0.952655 | 0.971380 | +0.018725 |
| `avg_confidence_when_wrong` | 0.991762 | 0.995589 | +0.003826 |

## Checks

Passed: `candidate_accuracy_not_lower, candidate_ece_not_worse, candidate_brier_not_worse, candidate_calibration_improved, critical_mode_qtrm_core_steps_8_no_evidence_not_worse, critical_mode_qtrm_core_steps_8_qtrm_only_no_evidence_not_worse, critical_qtrm_mode_calibration_improved`

Failed: ``

## Mode Comparisons

| Mode | Acc Delta | ECE Delta | Brier Delta | Wrong-Conf Delta |
| --- | ---: | ---: | ---: | ---: |
| `qtrm_core_steps_8_no_evidence` | +0.000000 | -0.014239 | -0.000351 | +0.003826 |
| `qtrm_core_steps_8_qtrm_only_no_evidence` | +0.000000 | -0.014239 | -0.000351 | +0.003826 |

Critical modes: `qtrm_core_steps_8_no_evidence, qtrm_core_steps_8_qtrm_only_no_evidence`

## Category Comparisons

| Category | Acc Delta | ECE Delta | Brier Delta | Wrong-Conf Delta |
| --- | ---: | ---: | ---: | ---: |
| `answerable_arithmetic` | +0.000000 | +0.000006 | +0.000012 | +0.000006 |
| `answerable_boolean` | +0.000000 | +0.007646 | +0.015102 | +0.007646 |
| `contradiction` | +0.000000 | -0.053854 | -0.015008 | +0.000000 |
| `ood_random_token` | +0.000000 | -0.012229 | -0.000494 | +0.000000 |
| `unknown_missing` | +0.000000 | -0.019890 | -0.001366 | +0.000000 |

## Expected Unknown Comparisons

| Expected Unknown | Acc Delta | ECE Delta | Brier Delta | Wrong-Conf Delta |
| --- | ---: | ---: | ---: | ---: |
| `False` | +0.000000 | +0.003826 | +0.007557 | +0.003826 |
| `True` | +0.000000 | -0.028658 | -0.005623 | +0.000000 |

## Notes

- Confidence is softmax over forced-choice logprob scores.
- This gate uses choice-score calibration; it is not a full generative calibration proof.
- Gate profiles filter records before global metrics; strict keeps all modes.
- Promotion still needs donor-only, fused, low-donor/QTRM-only, and core-off comparisons.
