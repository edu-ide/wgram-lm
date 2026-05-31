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
SEED="${SEED:-20260515}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TRAIN_CASES="${TRAIN_CASES:-192}"
EVAL_CASES="${EVAL_CASES:-192}"
BASE_DIR="${BASE_DIR:-local_eval/qwen_backbone_wgram_nested_curriculum_20260515}"
FORCE="${FORCE:-0}"
STEP_CONDITIONING="${STEP_CONDITIONING:-1}"
STEP_CONDITIONING_MAX_STEPS="${STEP_CONDITIONING_MAX_STEPS:-64}"
STEP_CONDITIONING_SCALE="${STEP_CONDITIONING_SCALE:-1.0}"
CORE_DELTA_ADAPTER_MODE="${CORE_DELTA_ADAPTER_MODE:-adapter_only}"

run_stage() {
  local stage="$1"
  local steps="$2"
  local h_cycles="$3"
  local l_cycles="$4"
  local outer_steps="$5"
  local residual_scale="$6"
  local halt_enabled="$7"
  local halt_threshold="$8"
  local init_checkpoint="$9"
  local out_dir="${BASE_DIR}/${stage}"
  local report="${out_dir}/report.json"
  if [[ "$FORCE" != "1" && -f "$report" ]]; then
    echo "skip existing ${report}"
    return 0
  fi
  mkdir -p "$out_dir"
  echo "=== ${stage} ==="
  echo "steps=${steps} h=${h_cycles} l=${l_cycles} outer=${outer_steps} residual=${residual_scale} halt=${halt_enabled} threshold=${halt_threshold}"
  local halt_args=()
  if [[ "$halt_enabled" == "1" ]]; then
    halt_args+=(--core-convergence-halt-enabled --core-convergence-halt-threshold "$halt_threshold" --core-convergence-halt-min-outer 1)
  fi
  local step_args=()
  if [[ "$STEP_CONDITIONING" == "1" ]]; then
    step_args+=(
      --core-step-conditioning-enabled
      --core-step-conditioning-max-steps "$STEP_CONDITIONING_MAX_STEPS"
      --core-step-conditioning-scale "$STEP_CONDITIONING_SCALE"
    )
  fi
  local init_args=()
  if [[ -n "$init_checkpoint" ]]; then
    init_args+=(--init-checkpoint "$init_checkpoint")
  fi
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
    --core-delta-adapter-mode "$CORE_DELTA_ADAPTER_MODE" \
    --core-gate-init -2.0 \
    --residual-scale "$residual_scale" \
    --h-cycles "$h_cycles" \
    --l-cycles "$l_cycles" \
    --outer-steps "$outer_steps" \
    --case-mode hard_v1 \
    --eval-every-steps 50 \
    --restore-best-checkpoint \
    --min-reasoning-gain "${MIN_GAIN:-0.01}" \
    --min-language-top1-agreement 0.50 \
    --min-family-gain "${MIN_FAMILY_GAIN:--1.0}" \
    --min-family-core-accuracy "${MIN_FAMILY_ACC:-0.0}" \
    "${halt_args[@]}" \
    "${step_args[@]}" \
    "${init_args[@]}"
  local code="$?"
  set -e
  echo "exit_code=${code}"
}

stage1="${BASE_DIR}/stage1_h1_l1/last_core.pt"
stage2="${BASE_DIR}/stage2_h1_l3/last_core.pt"
stage3="${BASE_DIR}/stage3_h3_l6/last_core.pt"

run_stage "stage1_h1_l1" "${STAGE1_STEPS:-120}" 1 1 1 0.5 0 0.0 ""
run_stage "stage2_h1_l3" "${STAGE2_STEPS:-80}" 1 3 1 0.25 0 0.0 "$stage1"
run_stage "stage3_h3_l6" "${STAGE3_STEPS:-40}" 3 6 1 0.1 0 0.0 "$stage2"
run_stage "stage4_h3_l6_halt" "${STAGE4_STEPS:-20}" 3 6 3 0.1 1 "${HALT_THRESHOLD:-0.5}" "$stage3"

.venv/bin/python - "$BASE_DIR" <<'PY'
import json
import sys
from pathlib import Path

base = Path(sys.argv[1])
rows = []
for report_path in sorted(base.glob("stage*/report.json")):
    report = json.loads(report_path.read_text())
    after = report.get("after_eval", {})
    family = report.get("after_family_summary", {})
    rows.append(
        {
            "stage": report_path.parent.name,
            "accepted": report.get("accepted"),
            "h_cycles": report.get("h_cycles"),
            "l_cycles": report.get("l_cycles"),
            "outer_steps": report.get("outer_steps"),
            "halt_enabled": report.get("core_convergence_halt_enabled"),
            "step_conditioning": report.get("core_step_conditioning_enabled"),
            "core_delta_adapter_mode": report.get("core_delta_adapter_mode"),
            "gain": after.get("gain"),
            "core_accuracy": after.get("core_accuracy"),
            "min_family_gain": family.get("min_gain"),
            "min_family_core_accuracy": family.get("min_core_accuracy"),
            "mean_core_outer_iterations": after.get("mean_core_outer_iterations"),
            "core_converged_fraction": after.get("core_converged_fraction"),
            "language_top1": report.get("after_language", {}).get("top1_agreement"),
            "best_periodic_eval": report.get("train", {}).get("best_periodic_eval"),
        }
    )
print(json.dumps(rows, ensure_ascii=False, indent=2))
PY
