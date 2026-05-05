# ASI Causal Loop Gate

## Verdict

Status: `rejected`

## Baseline Gains

| Check | Value |
| --- | ---: |
| gain_over_donor_harness | -0.0556 |
| gain_over_scripted_harness | -0.0556 |
| min_gain | 0.0200 |

## Causal Drops

| Ablation | Drop |
| --- | ---: |
| qtrm_latent_core_off | -0.0556 |
| qtrm_verifier_off | 0.3333 |
| qtrm_world_model_off | 0.6111 |

## Failed Checks

- `qtrm_does_not_beat_donor_harness`
- `qtrm_does_not_beat_scripted_harness`
- `latent_core_not_causal`
