#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi
export PYTHONPATH="${PYTHONPATH:-$PWD/src}"
export HF_HOME="${HF_HOME:-/mnt/nvme1n1p2/hf-cache-qtrm}"

PYTHON="${PYTHON:-$PWD/.venv/bin/python}"
CONFIG="${CONFIG:-configs/qwen35_2b_4090.yaml}"
TRAIN_DATA="${TRAIN_DATA:-data/filtered/metacognitive_calibration_unknown_preferences_train.jsonl}"
CASES="${CASES:-data/eval/metacognitive_calibration_heldout_40.jsonl}"
LOCAL_CKPT_ROOT="${LOCAL_CKPT_ROOT:-/mnt/nvme1n1p2/qtrm-local-checkpoints/metacog_pair_rebuild}"
BASELINE_DIR="${BASELINE_DIR:-$LOCAL_CKPT_ROOT/no_warmup_rebuilt_s001}"
CANDIDATE_DIR="${CANDIDATE_DIR:-$LOCAL_CKPT_ROOT/unknown_teacher_kl_conservative_rebuilt_s040}"
BASELINE_CHECKPOINT="$BASELINE_DIR/last.pt"
CANDIDATE_CHECKPOINT="$CANDIDATE_DIR/last.pt"

INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
ALLOW_RANDOM_INIT="${ALLOW_RANDOM_INIT:-0}"
BASELINE_STEPS="${BASELINE_STEPS:-0}"
DEPTH_STEPS="${DEPTH_STEPS:-1,2,4,8}"
TARGET_MODE="${TARGET_MODE:-final}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
LOG_EVERY="${LOG_EVERY:-10}"

STEPS="${CANDIDATE_STEPS:-40}"
LR="${CANDIDATE_LR:-2.0e-6}"
FINAL_LOGIT_CE_WEIGHT="${FINAL_LOGIT_CE_WEIGHT:-1.0}"
TEACHER_CHECKPOINT="$BASELINE_CHECKPOINT"
TEACHER_FIRST_TOKEN_DEPTH_KL_WEIGHT=5.0
ALL_DEPTH_CE_WEIGHT=0.10
CHOICE_MARGIN_WEIGHT=0.25
CHOICE_MARGIN="${CHOICE_MARGIN:-0.10}"
PROGRESS_MARGIN_WEIGHT="${PROGRESS_MARGIN_WEIGHT:-0.25}"
PROGRESS_MARGIN="${PROGRESS_MARGIN:-0.10}"

check_readable_file() {
  local label="$1"
  local path="$2"
  local env_var="$3"
  "$PYTHON" - "$label" "$path" "$env_var" <<'PY'
import sys
from pathlib import Path

label, path, env_var = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    open(path, "rb").read(1)
except Exception as exc:
    raise SystemExit(
        f"Missing or unreadable {label}: {path}\n"
        f"{type(exc).__name__}: {exc}\n"
        f"Set {env_var} to a healthy readable path."
    )
PY
}

check_writable_dir() {
  local path="$1"
  mkdir -p "$path"
  "$PYTHON" - "$path" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
probe = path / "preflight_write_test"
try:
    probe.write_text("ok\n", encoding="utf-8")
    probe.unlink()
except Exception as exc:
    raise SystemExit(
        f"Output directory is not writable: {path}\n"
        f"{type(exc).__name__}: {exc}"
    )
PY
}

check_readable_file "config" "$CONFIG" "CONFIG"
check_readable_file "training data" "$TRAIN_DATA" "TRAIN_DATA"
check_readable_file "held-out cases" "$CASES" "CASES"
check_writable_dir "$LOCAL_CKPT_ROOT"

BASELINE_INIT_ARGS=()
if [[ -n "$INIT_CHECKPOINT" ]]; then
  check_readable_file "init checkpoint" "$INIT_CHECKPOINT" "INIT_CHECKPOINT"
  BASELINE_INIT_ARGS+=(--init-checkpoint "$INIT_CHECKPOINT")
elif [[ "$ALLOW_RANDOM_INIT" == "1" || "$ALLOW_RANDOM_INIT" == "true" ]]; then
  BASELINE_INIT_ARGS+=(--allow-random-init)
else
  cat >&2 <<EOF
No readable INIT_CHECKPOINT was provided.

The old /mnt/sdb1 checkpoints currently fail with I/O error, so exact rebuild is impossible.
Provide INIT_CHECKPOINT=/healthy/path/last.pt or explicitly set ALLOW_RANDOM_INIT=1
to create a new matched random-init baseline/candidate pair.
EOF
  exit 1
fi

echo "============================================================"
echo "Rebuilding metacognitive matched pair on healthy disk"
echo "config=$CONFIG"
echo "train_data=$TRAIN_DATA"
echo "local_ckpt_root=$LOCAL_CKPT_ROOT"
echo "baseline=$BASELINE_CHECKPOINT"
echo "candidate=$CANDIDATE_CHECKPOINT"
echo "init=${INIT_CHECKPOINT:-random_init}"
echo "============================================================"

"$PYTHON" scripts/196_train_pure_recursive_depth_supervised.py \
  --config "$CONFIG" \
  --data-jsonl "$TRAIN_DATA" \
  "${BASELINE_INIT_ARGS[@]}" \
  --tokenizer-model-id Qwen/Qwen3.5-2B-Base \
  --max-length "$MAX_LENGTH" \
  --steps "$BASELINE_STEPS" \
  --depth-steps "$DEPTH_STEPS" \
  --target-mode "$TARGET_MODE" \
  --out-dir "$BASELINE_DIR" \
  --final-logit-ce-weight "$FINAL_LOGIT_CE_WEIGHT" \
  --all-depth-ce-weight 0.0 \
  --progress-margin-weight 0.0 \
  --choice-margin-weight 0.0 \
  --log-every "$LOG_EVERY"

check_readable_file "rebuilt baseline checkpoint" "$BASELINE_CHECKPOINT" "BASELINE_CHECKPOINT"

"$PYTHON" scripts/196_train_pure_recursive_depth_supervised.py \
  --config "$CONFIG" \
  --data-jsonl "$TRAIN_DATA" \
  --init-checkpoint "$BASELINE_CHECKPOINT" \
  --tokenizer-model-id Qwen/Qwen3.5-2B-Base \
  --max-length "$MAX_LENGTH" \
  --steps "$STEPS" \
  --lr "$LR" \
  --depth-steps "$DEPTH_STEPS" \
  --target-mode "$TARGET_MODE" \
  --out-dir "$CANDIDATE_DIR" \
  --final-logit-ce-weight "$FINAL_LOGIT_CE_WEIGHT" \
  --all-depth-ce-weight "$ALL_DEPTH_CE_WEIGHT" \
  --progress-margin-weight "$PROGRESS_MARGIN_WEIGHT" \
  --progress-margin "$PROGRESS_MARGIN" \
  --choice-margin-weight "$CHOICE_MARGIN_WEIGHT" \
  --choice-margin "$CHOICE_MARGIN" \
  --teacher-checkpoint "$TEACHER_CHECKPOINT" \
  --teacher-first-token-depth-kl-weight "$TEACHER_FIRST_TOKEN_DEPTH_KL_WEIGHT" \
  --log-every "$LOG_EVERY"

check_readable_file "rebuilt candidate checkpoint" "$CANDIDATE_CHECKPOINT" "CANDIDATE_CHECKPOINT"

cat <<EOF
============================================================
Rebuilt matched pair is ready.

Run the full fusion sweep:

BASELINE_CHECKPOINT=$BASELINE_CHECKPOINT \\
CANDIDATE_CHECKPOINT=$CANDIDATE_CHECKPOINT \\
CONFIG=$CONFIG \\
CASES=$CASES \\
HF_HOME=$HF_HOME \\
PYTHONPATH=src \\
bash scripts/204_run_metacog_fusion_conflict_gate_sweep.sh
============================================================
EOF
