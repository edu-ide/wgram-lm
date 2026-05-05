# Metacognitive Checkpoint Rebuild Recovery

Status: implemented as an artifact recovery path, not a promoted model result.

## Problem

The metacognitive fusion full sweep needs two checkpoints:

```text
baseline:
  runs/qwen35_2b_4090_noise_warmup_matched_no_warmup_s001/last.pt

candidate:
  runs/qwen35_2b_4090_metacog_unknown_teacher_kl_conservative_s040/last.pt
```

Both live under `runs -> /mnt/sdb1/ws-sky-data/qtrm-runs`. That mount is in an
`emergency_ro` state and returns `OSError: [Errno 5] Input/output error` on
checkpoint reads. The localization helper cannot copy bytes that the source
disk cannot read.

## Decision

There are two distinct recovery modes:

```text
exact recovery:
  requires readable old baseline/candidate checkpoints or backups
  preserves the old reports' meaning

new matched-pair rebuild:
  creates a fresh baseline/candidate pair on a healthy disk
  is comparable only within that newly rebuilt pair
  must not be reported as the old s001/conservative-s040 result
```

The new runner is:

```bash
bash scripts/206_run_metacog_pair_rebuild.sh
```

Default checkpoint destination:

```text
/mnt/nvme1n1p2/qtrm-local-checkpoints/metacog_pair_rebuild
```

The runner can start from either a healthy checkpoint:

```bash
INIT_CHECKPOINT=/healthy/path/last.pt \
bash scripts/206_run_metacog_pair_rebuild.sh
```

or, explicitly, a fresh random QTRM init:

```bash
ALLOW_RANDOM_INIT=1 \
bash scripts/206_run_metacog_pair_rebuild.sh
```

The random-init mode is only for restoring the evaluation loop when all old
checkpoints are unreadable. It is not evidence that the prior candidate was
recovered.

## Conservative Candidate Settings

The rebuilt candidate preserves the same conservative teacher-KL recipe:

```text
steps: 40
lr: 2.0e-6
teacher: rebuilt baseline checkpoint
teacher_first_token_depth_kl_weight: 5.0
all_depth_ce_weight: 0.10
choice_margin_weight: 0.25
```

The trainer now has an explicit random-init opt-in:

```bash
--allow-random-init
```

Without `--init-checkpoint` or `--allow-random-init`, it fails. This prevents
accidentally treating random initialization as an intended continuation run.

## Next Gate

After the matched pair is rebuilt, run:

```bash
BASELINE_CHECKPOINT=/mnt/nvme1n1p2/qtrm-local-checkpoints/metacog_pair_rebuild/no_warmup_rebuilt_s001/last.pt \
CANDIDATE_CHECKPOINT=/mnt/nvme1n1p2/qtrm-local-checkpoints/metacog_pair_rebuild/unknown_teacher_kl_conservative_rebuilt_s040/last.pt \
CONFIG=configs/qwen35_2b_4090.yaml \
PYTHONPATH=src \
bash scripts/204_run_metacog_fusion_conflict_gate_sweep.sh
```

Acceptance remains unchanged: the candidate must improve metacognitive
calibration without lowering matched forced-choice accuracy, and any donor/QTRM
fusion gain must beat the rebuilt baseline on the same held-out 40 cases.

## Files

```text
scripts/196_train_pure_recursive_depth_supervised.py
scripts/197_run_pure_recursive_depth_supervised_train.sh
scripts/206_run_metacog_pair_rebuild.sh
tests/test_pure_recursive_depth_supervised_train_script.py
tests/test_metacog_pair_rebuild_runner.py
```
