# Metacognitive Calibration Gate

Status: `rejected`

Baseline: `no_warmup_s001_smoke8`
Candidate: `unknown_teacher_kl_conservative_s040_smoke8`
Gate profile: `strict`
Profile records: `64` / source `64`
Included modes: `all`

## Global Comparison

| Metric | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| `accuracy` | 0.500000 | 0.500000 | +0.000000 |
| `ece` | 0.498750 | 0.499257 | +0.000507 |
| `brier` | 0.497586 | 0.498591 | +0.001005 |
| `mean_confidence` | 0.998750 | 0.999257 | +0.000507 |
| `avg_confidence_when_wrong` | 0.997569 | 0.998585 | +0.001016 |

## Checks

Passed: `candidate_accuracy_not_lower, critical_qtrm_mode_calibration_improved`

Failed: `candidate_ece_worse, candidate_brier_worse, candidate_no_calibration_gain, critical_mode_qtrm_core_steps_8_donor_scale_0p25_no_evidence_ece_worse, critical_mode_qtrm_core_steps_8_donor_scale_0p50_no_evidence_ece_worse, critical_mode_qtrm_core_steps_8_donor_scale_0p50_no_evidence_brier_worse, critical_mode_qtrm_core_steps_8_donor_scale_0p75_no_evidence_ece_worse, critical_mode_qtrm_core_steps_8_donor_scale_0p75_no_evidence_brier_worse, critical_mode_qtrm_core_steps_8_donor_scale_1p0_no_evidence_ece_worse, critical_mode_qtrm_core_steps_8_donor_scale_1p0_no_evidence_brier_worse`

## Mode Comparisons

| Mode | Acc Delta | ECE Delta | Brier Delta | Wrong-Conf Delta |
| --- | ---: | ---: | ---: | ---: |
| `qtrm_core_steps_8_donor_scale_0p25_no_evidence` | +0.000000 | +0.000002 | -0.000001 | -0.000001 |
| `qtrm_core_steps_8_donor_scale_0p50_no_evidence` | +0.000000 | +0.000007 | +0.000015 | +0.000015 |
| `qtrm_core_steps_8_donor_scale_0p75_no_evidence` | +0.000000 | +0.000419 | +0.000841 | +0.000843 |
| `qtrm_core_steps_8_donor_scale_1p0_no_evidence` | +0.000000 | +0.001601 | +0.003165 | +0.003206 |

Critical modes: `qtrm_core_steps_8_donor_scale_0p25_no_evidence, qtrm_core_steps_8_donor_scale_0p50_no_evidence, qtrm_core_steps_8_donor_scale_0p75_no_evidence, qtrm_core_steps_8_donor_scale_1p0_no_evidence`

## Category Comparisons

| Category | Acc Delta | ECE Delta | Brier Delta | Wrong-Conf Delta |
| --- | ---: | ---: | ---: | ---: |
| `answerable_arithmetic` | +0.000000 | +0.002019 | +0.003994 | +0.002019 |
| `answerable_boolean` | +0.000000 | +0.000013 | +0.000026 | +0.000013 |
| `contradiction` | +0.000000 | -0.000036 | -0.000000 | +0.000000 |
| `ood_random_token` | +0.000000 | +0.000029 | +0.000000 | +0.000000 |
| `unknown_missing` | +0.000000 | +0.000006 | +0.000000 | +0.000000 |

## Expected Unknown Comparisons

| Expected Unknown | Acc Delta | ECE Delta | Brier Delta | Wrong-Conf Delta |
| --- | ---: | ---: | ---: | ---: |
| `False` | +0.000000 | +0.001016 | +0.002010 | +0.001016 |
| `True` | +0.000000 | +0.000002 | -0.000000 | +0.000000 |

## Notes

- Confidence is softmax over forced-choice logprob scores.
- This gate uses choice-score calibration; it is not a full generative calibration proof.
- Gate profiles filter records before global metrics; strict keeps all modes.
- Promotion still needs donor-only, fused, low-donor/QTRM-only, and core-off comparisons.
