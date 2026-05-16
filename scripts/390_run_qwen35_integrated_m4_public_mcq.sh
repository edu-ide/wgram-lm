#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
SUITE_JSONL="${SUITE_JSONL:-local_eval/m7_public_reasoning_suite/mmlu_pro_validation_64.jsonl}"
SUITE_REPORT="${SUITE_REPORT:-local_eval/m7_public_reasoning_suite/report.json}"
CHECKPOINT="${CHECKPOINT:-local_eval/qwen35_integrated_healing_l23_langkl_s100_20260516/last_core.pt}"
OUT_DIR="${OUT_DIR:-local_eval/qwen35_integrated_m4_mmlu_pro64_20260516}"
MAX_CASES="${MAX_CASES:-64}"
MIN_CASES="${MIN_CASES:-64}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-2B-Base}"
HF_HOME="${HF_HOME:-/mnt/nvme1n1p2/hf-cache-qtrm}"
QWEN_CORE_LAYER_INDICES="${QWEN_CORE_LAYER_INDICES:-3}"
CORE_INSERTION_MODE="${CORE_INSERTION_MODE:-final_residual}"
CORE_INSERT_AFTER_LAYER="${CORE_INSERT_AFTER_LAYER:--1}"
CORE_ADAPTER_DIM="${CORE_ADAPTER_DIM:-128}"
CORE_DELTA_ADAPTER_MODE="${CORE_DELTA_ADAPTER_MODE:-add}"
CORE_RESIDUAL_GATE_MODE="${CORE_RESIDUAL_GATE_MODE:-constant}"
CORE_RESIDUAL_GATE_DIM="${CORE_RESIDUAL_GATE_DIM:-128}"
CORE_RESIDUAL_GATE_INIT="${CORE_RESIDUAL_GATE_INIT:--2.0}"
RESIDUAL_SCALE="${RESIDUAL_SCALE:-0.05}"
N_CORE_LAYERS="${N_CORE_LAYERS:-1}"
H_CYCLES="${H_CYCLES:-1}"
L_CYCLES="${L_CYCLES:-1}"
OUTER_STEPS="${OUTER_STEPS:-1}"
CORE_CONVERGENCE_HALT_ENABLED="${CORE_CONVERGENCE_HALT_ENABLED:-0}"
CORE_CONVERGENCE_HALT_THRESHOLD="${CORE_CONVERGENCE_HALT_THRESHOLD:-0.001}"
CORE_CONVERGENCE_HALT_MIN_OUTER="${CORE_CONVERGENCE_HALT_MIN_OUTER:-1}"
CORE_STEP_CONDITIONING_ENABLED="${CORE_STEP_CONDITIONING_ENABLED:-0}"
CORE_STEP_CONDITIONING_MAX_STEPS="${CORE_STEP_CONDITIONING_MAX_STEPS:-64}"
CORE_STEP_CONDITIONING_SCALE="${CORE_STEP_CONDITIONING_SCALE:-1.0}"

export HF_HOME

HALT_ARGS=()
if [[ "${CORE_CONVERGENCE_HALT_ENABLED}" == "1" ]]; then
  HALT_ARGS+=(--core-convergence-halt-enabled)
else
  HALT_ARGS+=(--no-core-convergence-halt)
fi
STEP_CONDITIONING_ARGS=()
if [[ "${CORE_STEP_CONDITIONING_ENABLED}" == "1" ]]; then
  STEP_CONDITIONING_ARGS+=(--core-step-conditioning-enabled)
else
  STEP_CONDITIONING_ARGS+=(--no-core-step-conditioning)
fi

if [[ ! -f "${SUITE_JSONL}" ]]; then
  PYTHONPATH="${PYTHONPATH:-src}" "${PYTHON_BIN}" scripts/383_materialize_m7_public_reasoning_suite.py \
    --max-cases "${MAX_CASES}" \
    --min-cases "${MIN_CASES}" \
    --out-jsonl "${SUITE_JSONL}" \
    --out-report "${SUITE_REPORT}"
fi

PYTHONPATH="${PYTHONPATH:-src}" "${PYTHON_BIN}" scripts/390_eval_qwen35_integrated_public_mcq.py \
  --suite-jsonl "${SUITE_JSONL}" \
  --checkpoint "${CHECKPOINT}" \
  --model-id "${MODEL_ID}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --mandatory-core \
  --core-impl qwen_layer_wrapped \
  --core-insertion-mode "${CORE_INSERTION_MODE}" \
  --core-insert-after-layer "${CORE_INSERT_AFTER_LAYER}" \
  --qwen-core-layer-indices "${QWEN_CORE_LAYER_INDICES}" \
  --core-adapter-dim "${CORE_ADAPTER_DIM}" \
  --core-delta-adapter-mode "${CORE_DELTA_ADAPTER_MODE}" \
  --core-residual-gate-mode "${CORE_RESIDUAL_GATE_MODE}" \
  --core-residual-gate-dim "${CORE_RESIDUAL_GATE_DIM}" \
  --core-residual-gate-init "${CORE_RESIDUAL_GATE_INIT}" \
  --n-core-layers "${N_CORE_LAYERS}" \
  --h-cycles "${H_CYCLES}" \
  --l-cycles "${L_CYCLES}" \
  --outer-steps "${OUTER_STEPS}" \
  "${HALT_ARGS[@]}" \
  --core-convergence-halt-threshold "${CORE_CONVERGENCE_HALT_THRESHOLD}" \
  --core-convergence-halt-min-outer "${CORE_CONVERGENCE_HALT_MIN_OUTER}" \
  "${STEP_CONDITIONING_ARGS[@]}" \
  --core-step-conditioning-max-steps "${CORE_STEP_CONDITIONING_MAX_STEPS}" \
  --core-step-conditioning-scale "${CORE_STEP_CONDITIONING_SCALE}" \
  --residual-scale "${RESIDUAL_SCALE}" \
  --max-cases "${MAX_CASES}" \
  --min-cases "${MIN_CASES}" \
  --out-dir "${OUT_DIR}" \
  --out-json "${OUT_DIR}/report.json" \
  --out-jsonl "${OUT_DIR}/predictions.jsonl"
