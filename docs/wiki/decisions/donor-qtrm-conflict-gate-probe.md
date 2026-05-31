# Donor-QTRM Conflict Gate Probe

Status: implemented as an ablation probe, not promoted as canonical.

## Problem

The conservative unknown-only teacher-KL checkpoint showed a narrow QTRM-core
metacognition gain, but the fused donor/QTRM path still worsened calibration:

```text
qtrm_core profile:
  status: accepted
  accuracy_delta: +0.000000
  ece_delta: -0.014239
  brier_delta: -0.000351

fused profile:
  status: rejected
  accuracy_delta: +0.000000
  ece_delta: +0.014450
  brier_delta: +0.002196
```

This points to a fusion-calibration failure, not only a recursive-core
calibration failure.

## Probe

Add an optional model-forward gate:

```text
if donor top token != QTRM residual top token:
  qtrm_text_residual *= donor_qtrm_conflict_qtrm_scale
else:
  qtrm_text_residual *= 1.0

fused_logits = donor_logits * donor_logits_scale + gated_qtrm_residual
```

The default is off:

```text
donor_qtrm_conflict_gate_enabled: false
donor_qtrm_conflict_qtrm_scale: 0.0
```

Eval CLI:

```bash
PYTHONPATH=src .venv/bin/python scripts/192_eval_raw_intelligence.py \
  --donor-qtrm-conflict-gate \
  --donor-qtrm-conflict-qtrm-scale 0.0 \
  ...
```

Safe full-sweep runner:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm \
PYTHONPATH=src \
bash scripts/204_run_metacog_fusion_conflict_gate_sweep.sh
```

The runner writes eval JSONL files to `local_eval/` instead of `runs/eval/`.
This avoids the `/mnt/sdb1` `runs` symlink for outputs. The checkpoints still
must be readable; if `/mnt/sdb1` remains in I/O error state, copy the two
checkpoint files to a healthy disk and set:

```bash
BASELINE_CHECKPOINT=/healthy/path/no_warmup/last.pt \
CANDIDATE_CHECKPOINT=/healthy/path/conservative/last.pt \
bash scripts/204_run_metacog_fusion_conflict_gate_sweep.sh
```

Checkpoint localization helper:

```bash
bash scripts/205_localize_metacog_checkpoints.sh
```

Default destination:

```text
/mnt/nvme1n1p2/qtrm-local-checkpoints/metacog_fusion_sweep
```

The helper:

```text
1. checks destination writability with preflight_write_test
2. checks each source checkpoint with open(src, 'rb').read(1)
3. copies with shutil.copy2
4. verifies source/destination sha256
5. prints BASELINE_CHECKPOINT=... and CANDIDATE_CHECKPOINT=... command lines
```

Current status:

```text
Failed to localize baseline:
  OSError: [Errno 5] Input/output error
  source: runs/qwen35_2b_4090_noise_warmup_matched_no_warmup_s001/last.pt
```

So the active blocker is source checkpoint readability, not destination space or
the eval runner.

If no readable backup exists, use the rebuild runner:

```bash
ALLOW_RANDOM_INIT=1 \
bash scripts/206_run_metacog_pair_rebuild.sh
```

or, preferably, seed it from a healthy prior checkpoint:

```bash
INIT_CHECKPOINT=/healthy/path/last.pt \
bash scripts/206_run_metacog_pair_rebuild.sh
```

This produces a new matched pair on `/mnt/nvme1n1p2`. It is valid for a new
within-pair fusion sweep, but it is not exact recovery of the old unreadable
checkpoint result.

The runner now performs real preflight checks:

```text
checkpoint: open(path, 'rb').read(1)
output dir: create/delete preflight_write_test
```

This catches the observed `/mnt/sdb1` I/O failure before loading Qwen or
starting a long forced-choice sweep.

## Telemetry

Forced-choice eval rows now preserve conflict-gate telemetry inside
`choice_scores`:

```json
{
  "choice": "UNKNOWN",
  "logprob": -1.23,
  "donor_qtrm_conflict_gate_mean": 0.42,
  "donor_qtrm_conflict_gate_observations": 1
}
```

The calibration gate summary also aggregates:

```text
mean_predicted_conflict_gate
mean_choice_conflict_gate
```

These fields make the conflict-gate probe falsifiable: a report can show
whether the gate actually suppressed QTRM residuals on the predicted choices,
instead of only saying that the CLI flag was enabled.

## Why This Is A Probe

This is not raw ASI progress by itself. It is a KISS/YAGNI diagnostic for the
fusion boundary:

- If it improves fused calibration without hurting QTRM-core-only metrics, the
  next candidate should be a learned fusion calibration/router.
- If it does not improve fused calibration, the bottleneck is deeper than
  top-token conflict and should move to trained reliability targets or
  residual-scale calibration.

## Files

```text
src/wgram_lm/config.py
src/wgram_lm/wgram_model.py
scripts/192_eval_raw_intelligence.py
scripts/204_run_metacog_fusion_conflict_gate_sweep.sh
scripts/205_localize_metacog_checkpoints.sh
scripts/206_run_metacog_pair_rebuild.sh
tests/test_model_config.py
tests/test_raw_intelligence_eval_script.py
tests/test_metacog_fusion_sweep_runner.py
tests/test_metacog_checkpoint_localize_script.py
tests/test_metacog_pair_rebuild_runner.py
tests/test_metacognitive_calibration_gate_script.py
```

## Verification

```text
PYTHONPATH=src .venv/bin/python -m unittest \
  tests.test_model_config \
  tests.test_raw_intelligence_eval_script \
  tests.test_metacognitive_calibration_gate_script \
  tests.test_pure_recursive_reasoning_preferences \
  tests.test_metacognitive_calibration_cases \
  tests.test_pure_recursive_depth_supervised_train_script \
  tests.test_symbolic_transition_gate_script \
  tests.test_core_world_model_transition_eval

Result: 101 tests OK
```
