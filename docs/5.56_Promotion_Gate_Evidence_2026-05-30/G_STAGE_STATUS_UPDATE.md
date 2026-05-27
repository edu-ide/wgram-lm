# G-Stage Status Update — 2026-05-30

**Critical Generalization Experiment Launched**

**Variant**: Stochastic Ablation Zero on Real 642 Gold (100 steps)
- Directory: local_556_gstage_stoch_zero_real_fixed_20260527_1255
- Status: Running cleanly (confirmed at step ~30+)
- Expected behavior observed: stochastic_diversity = 0.0000 (ablation contract holding under real gold attempt conditions)
- Gold handling hardening active
- Scheduled decay progressing normally

This is the key G-stage test for causal contribution of stochastic breadth when the 642 gold structural bias path is provided.

**Previous Context**:
- 180-step full recipe run completed with stable high diversity.
- Launcher fixed (PYTHONPATH + venv python).
- G-Stage Completion Report created.

**Next**:
- Monitor this run to completion.
- Analyze contrast with previous full (stoch ON) 180-step run.
- Re-launch full matrix now that launcher is reliable.

**Overall G-Stage**:
Training dynamics generalization is progressing well. The critical ablation test is now live on real gold data.
