# G-Stage Matrix Probe Results Summary (2026-05-30)

**Matrix**: local_556_real642_full_gstage_matrix_fixed_20260527_1256
**Probe run on completed variants**: 01, 02, 03, 04 (100 steps each on real 642 gold path)

**Probe settings**: ablation=zero, 40 steps, 6 trials

## Results

| Variant                              | Mean Degradation after State Ablation (zero) |
|--------------------------------------|----------------------------------------------|
| 01_full_556_real_gold_stoch_on       | 0.0000 |
| 02_stochastic_ablation_zero          | 0.0000 |
| 03_no_attractor_protection           | 0.0000 |
| 04_no_scheduled_decay_fixed_high     | 0.0000 |

## Analysis

All variants returned exactly 0.0000 degradation.

This is consistent with previous probe runs on the 180-step and 100-step checkpoints.

**Conclusion**: The current implementation of `probe_state_ablation_robustness.py` (simple recurrent state norm difference on random synthetic workspaces) is not sensitive enough to detect differences in this small-scale synthetic training regime.

**Positive note**: The probe runs without errors and loads all curriculum checkpoints correctly. The infrastructure is ready.

**Next required action for meaningful data**:
- Improve the probe (better input sequences that mimic "reasoning", measure trajectory consistency or multi-step prediction error after ablation, etc.).
- Or move the best checkpoints to a larger model with real hard-family evaluation.

**Recommendation**: Treat the current probe results as "baseline / infrastructure validated" rather than scientific findings. Focus next effort on improving the probe metric before running it on the full set of matrix variants again.
