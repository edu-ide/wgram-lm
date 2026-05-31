#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="${PYTHONPATH:-src}"
export HF_HOME="${HF_HOME:-/mnt/nvme1n1p2/hf-cache-qtrm}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-2B-Base}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-80}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TRAIN_CASES="${TRAIN_CASES:-192}"
EVAL_CASES="${EVAL_CASES:-192}"
SEED="${SEED:-20260515}"
FORCE="${FORCE:-0}"

run_gate() {
  local name="$1"
  local steps="$2"
  local h_cycles="$3"
  local l_cycles="$4"
  local min_gain="$5"
  local min_family_gain="$6"
  local min_family_acc="$7"
  local transitions_per_step=$((h_cycles * (l_cycles + 1)))
  local approx_core_transitions=$((steps * transitions_per_step))
  local out_dir="local_eval/${name}_seed${SEED}_s${steps}_t${approx_core_transitions}_20260515"
  local report="${out_dir}/report.json"
  if [[ "$FORCE" != "1" && -f "$report" ]]; then
    echo "skip existing ${report}"
    return 0
  fi
  echo "=== ${name} ==="
  echo "steps=${steps} h=${h_cycles} l=${l_cycles} approx_core_transitions=${approx_core_transitions}"
  set +e
  .venv/bin/python scripts/362_train_qwen_backbone_wgram_core_gate.py \
    --model-id "$MODEL_ID" \
    --out-dir "$out_dir" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --steps "$steps" \
    --batch-size "$BATCH_SIZE" \
    --train-cases "$TRAIN_CASES" \
    --eval-cases "$EVAL_CASES" \
    --seed "$SEED" \
    --log-every 50 \
    --core-impl qwen_layer_wrapped \
    --qwen-core-layer-indices 3 \
    --core-adapter-dim 128 \
    --core-gate-init -2.0 \
    --residual-scale 0.5 \
    --h-cycles "$h_cycles" \
    --l-cycles "$l_cycles" \
    --outer-steps 1 \
    --case-mode hard_v1 \
    --min-reasoning-gain "$min_gain" \
    --min-language-top1-agreement 0.50 \
    --min-family-gain "$min_family_gain" \
    --min-family-core-accuracy "$min_family_acc"
  local code="$?"
  set -e
  echo "exit_code=${code}"
}

# h=1,l=1 -> 2 Qwen-layer transition calls per optimizer step.
# h=3,l=6 -> 21 Qwen-layer transition calls per optimizer step.
# 210*2 and 20*21 both equal 420 approximate core transition calls.
run_gate \
  "qwen_backbone_wgram_qwen_transition_gateopen_nonnested_compare" \
  "${NONNESTED_STEPS:-210}" \
  1 \
  1 \
  "${MIN_GAIN:-0.01}" \
  "${MIN_FAMILY_GAIN:--1.0}" \
  "${MIN_FAMILY_ACC:-0.0}"

run_gate \
  "qwen_backbone_wgram_qwen_transition_gateopen_nested_h3_l6_compare" \
  "${NESTED_STEPS:-20}" \
  3 \
  6 \
  "${MIN_GAIN:-0.01}" \
  "${MIN_FAMILY_GAIN:--1.0}" \
  "${MIN_FAMILY_ACC:-0.0}"

.venv/bin/python - <<'PY'
import json
from pathlib import Path

paths = sorted(Path("local_eval").glob("qwen_backbone_wgram_qwen_transition_gateopen_*_compare_seed*_s*_t*_20260515/report.json"))
rows = []
for path in paths:
    report = json.loads(path.read_text())
    after = report.get("after_eval", {})
    family = report.get("after_family_summary", {})
    steps = int(report.get("steps", 0))
    h = int(report.get("h_cycles", 0))
    l = int(report.get("l_cycles", 0))
    rows.append(
        {
            "path": str(path),
            "accepted": report.get("accepted"),
            "steps": steps,
            "h_cycles": h,
            "l_cycles": l,
            "approx_core_transitions": steps * h * (l + 1),
            "gain": after.get("gain"),
            "core_accuracy": after.get("core_accuracy"),
            "min_family_gain": family.get("min_gain"),
            "min_family_core_accuracy": family.get("min_core_accuracy"),
            "core_gate_value": report.get("core_gate_value"),
            "language_top1": report.get("after_language", {}).get("top1_agreement"),
        }
    )
print(json.dumps(rows, ensure_ascii=False, indent=2))
PY
