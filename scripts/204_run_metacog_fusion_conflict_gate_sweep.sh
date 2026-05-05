#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi
export PYTHONPATH=${PYTHONPATH:-$PWD/src}
export HF_HOME=${HF_HOME:-/mnt/nvme1n1p2/hf-cache-qtrm}

CONFIG="${CONFIG:-configs/qwen35_2b_4090.yaml}"
CASES="${CASES:-data/eval/metacognitive_calibration_heldout_40.jsonl}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-runs/qwen35_2b_4090_noise_warmup_matched_no_warmup_s001/last.pt}"
CANDIDATE_CHECKPOINT="${CANDIDATE_CHECKPOINT:-runs/qwen35_2b_4090_metacog_unknown_teacher_kl_conservative_s040/last.pt}"
OUT_ROOT="${OUT_ROOT:-local_eval}"
MAX_CASES="${MAX_CASES:-40}"
CONFLICT_QTRM_SCALE="${CONFLICT_QTRM_SCALE:-0.0}"

BASELINE_OUT="$OUT_ROOT/metacognitive_fusion_scale_sweep_baseline_40.jsonl"
CANDIDATE_OUT="$OUT_ROOT/metacognitive_fusion_scale_sweep_candidate_40.jsonl"
CANDIDATE_CONFLICT_OUT="$OUT_ROOT/metacognitive_fusion_scale_sweep_candidate_conflict_gate_40.jsonl"

PLAIN_MD="docs/wiki/decisions/metacog-fusion-scale-sweep-conservative-s040-full40.md"
PLAIN_JSON="docs/wiki/decisions/metacog-fusion-scale-sweep-conservative-s040-full40-summary.json"
CONFLICT_MD="docs/wiki/decisions/metacog-fusion-conflict-gate-conservative-s040-full40.md"
CONFLICT_JSON="docs/wiki/decisions/metacog-fusion-conflict-gate-conservative-s040-full40-summary.json"

SCALE_MODES=(
  qtrm_core_steps_8_donor_scale_1p0_no_evidence
  qtrm_core_steps_8_donor_scale_0p75_no_evidence
  qtrm_core_steps_8_donor_scale_0p50_no_evidence
  qtrm_core_steps_8_donor_scale_0p25_no_evidence
)

MODE_ARGS=()
CRITICAL_ARGS=()
for mode in "${SCALE_MODES[@]}"; do
  MODE_ARGS+=(--mode "$mode")
  CRITICAL_ARGS+=(--critical-mode "$mode")
done

check_readable_file() {
  local label="$1"
  local path="$2"
  local env_var="$3"
  python - "$label" "$path" "$env_var" <<'PY'
import sys
from pathlib import Path

label, path, env_var = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    open(path, 'rb').read(1)
except Exception as exc:
    raise SystemExit(
        f"Missing or unreadable {label}: {path}\n"
        f"{type(exc).__name__}: {exc}\n"
        "If this is an /mnt/sdb1 I/O error, copy the checkpoint to a healthy disk "
        f"and set {env_var}."
    )
PY
}

check_writable_dir() {
  local path="$1"
  mkdir -p "$path"
  python - "$path" <<'PY'
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

check_readable_file "baseline checkpoint" "$BASELINE_CHECKPOINT" "BASELINE_CHECKPOINT"
check_readable_file "candidate checkpoint" "$CANDIDATE_CHECKPOINT" "CANDIDATE_CHECKPOINT"
check_readable_file "cases" "$CASES" "CASES"
check_writable_dir "$OUT_ROOT"

run_eval() {
  local checkpoint="$1"
  local out="$2"
  shift 2
  HF_HOME="$HF_HOME" PYTHONPATH="$PYTHONPATH" python scripts/192_eval_raw_intelligence.py \
    --config "$CONFIG" \
    --checkpoint "$checkpoint" \
    --cases "$CASES" \
    --max-cases "$MAX_CASES" \
    --scoring forced_choice \
    "${MODE_ARGS[@]}" \
    --out "$out" \
    "$@"
}

echo "============================================================"
echo "Metacognitive fusion scale sweep"
echo "config=$CONFIG"
echo "cases=$CASES"
echo "max_cases=$MAX_CASES"
echo "out_root=$OUT_ROOT"
echo "baseline=$BASELINE_CHECKPOINT"
echo "candidate=$CANDIDATE_CHECKPOINT"
echo "conflict_qtrm_scale=$CONFLICT_QTRM_SCALE"
echo "============================================================"

run_eval "$BASELINE_CHECKPOINT" "$BASELINE_OUT"
run_eval "$CANDIDATE_CHECKPOINT" "$CANDIDATE_OUT"
run_eval \
  "$CANDIDATE_CHECKPOINT" \
  "$CANDIDATE_CONFLICT_OUT" \
  --donor-qtrm-conflict-gate \
  --donor-qtrm-conflict-qtrm-scale "$CONFLICT_QTRM_SCALE"

python scripts/202_build_metacognitive_calibration_gate.py \
  --baseline-jsonl "$BASELINE_OUT" \
  --candidate-jsonl "$CANDIDATE_OUT" \
  --baseline-label no_warmup_s001_full40 \
  --candidate-label unknown_teacher_kl_conservative_s040_full40 \
  "${CRITICAL_ARGS[@]}" \
  --markdown-out "$PLAIN_MD" \
  --json-out "$PLAIN_JSON"

python scripts/202_build_metacognitive_calibration_gate.py \
  --baseline-jsonl "$BASELINE_OUT" \
  --candidate-jsonl "$CANDIDATE_CONFLICT_OUT" \
  --baseline-label no_warmup_s001_full40 \
  --candidate-label unknown_teacher_kl_conservative_s040_conflict_gate_full40 \
  "${CRITICAL_ARGS[@]}" \
  --markdown-out "$CONFLICT_MD" \
  --json-out "$CONFLICT_JSON"

echo "wrote $PLAIN_MD"
echo "wrote $CONFLICT_MD"
