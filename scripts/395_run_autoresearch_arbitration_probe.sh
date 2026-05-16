#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
HF_HOME="${HF_HOME:-/mnt/nvme1n1p2/hf-cache-qtrm}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-2B-Base}"
CHECKPOINT="${CHECKPOINT:-local_eval/qwen35_integrated_midlayer_suffix_adapteronly_coretrain_langanchor_s160_20260516/last_core.pt}"
CHECKPOINT_LOAD_MODE="${CHECKPOINT_LOAD_MODE:-strict_shapes}"
SUITE_JSONL="${SUITE_JSONL:-local_eval/m7_public_reasoning_suite/mmlu_pro_validation_64.jsonl}"
OUT_DIR="${OUT_DIR:-local_eval/qwen35_integrated_autoresearch_arbitration_probe_mmlupro64_20260516}"
LEDGER_PATH="${LEDGER_PATH:-local_eval/qwen35_integrated_autoresearch_arbitration_probe/results.tsv}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-224}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_CASES="${MAX_CASES:-64}"
FIT_FRACTION="${FIT_FRACTION:-0.5}"
CORE_INSERTION_MODE="${CORE_INSERTION_MODE:-mid_layer_suffix}"
CORE_INSERT_AFTER_LAYER="${CORE_INSERT_AFTER_LAYER:-11}"
QWEN_CORE_LAYER_INDICES="${QWEN_CORE_LAYER_INDICES:-3}"
CORE_ADAPTER_DIM="${CORE_ADAPTER_DIM:-512}"
CORE_DELTA_ADAPTER_MODE="${CORE_DELTA_ADAPTER_MODE:-adapter_only}"
CORE_RESIDUAL_GATE_MODE="${CORE_RESIDUAL_GATE_MODE:-constant}"
CORE_RESIDUAL_GATE_DIM="${CORE_RESIDUAL_GATE_DIM:-128}"
CORE_RESIDUAL_GATE_INIT="${CORE_RESIDUAL_GATE_INIT:--2.0}"
CLONE_QWEN_CORE_LAYERS="${CLONE_QWEN_CORE_LAYERS:-0}"
H_CYCLES="${H_CYCLES:-3}"
L_CYCLES="${L_CYCLES:-6}"
OUTER_STEPS="${OUTER_STEPS:-3}"
CORE_CONVERGENCE_HALT_THRESHOLD="${CORE_CONVERGENCE_HALT_THRESHOLD:-0.2}"
CORE_CONVERGENCE_HALT_MIN_OUTER="${CORE_CONVERGENCE_HALT_MIN_OUTER:-1}"
CORE_STEP_CONDITIONING_MAX_STEPS="${CORE_STEP_CONDITIONING_MAX_STEPS:-64}"
CORE_STEP_CONDITIONING_SCALE="${CORE_STEP_CONDITIONING_SCALE:-1.0}"
CORE_GATE_INIT="${CORE_GATE_INIT:--4.0}"
RESIDUAL_SCALE="${RESIDUAL_SCALE:-1.0}"
POLICY="${POLICY:-threshold}"
BASE_MARGIN_GRID="${BASE_MARGIN_GRID:-0,0.1,0.25,0.5,0.75,1,1.5,2,3,5}"
CORE_MARGIN_GRID="${CORE_MARGIN_GRID:-0,0.1,0.25,0.5,0.75,1,1.5,2,3,5}"
SWITCH_ADV_GRID="${SWITCH_ADV_GRID:--1,-0.5,-0.25,0,0.25,0.5,0.75,1,1.5,2}"
LINEAR_STEPS="${LINEAR_STEPS:-300}"
LINEAR_LR="${LINEAR_LR:-0.05}"
LINEAR_WEIGHT_DECAY="${LINEAR_WEIGHT_DECAY:-0.01}"
LINEAR_THRESHOLD_GRID="${LINEAR_THRESHOLD_GRID:-0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}"
SEED="${SEED:-20260522}"

export HF_HOME

CLONE_ARGS=()
if [[ "${CLONE_QWEN_CORE_LAYERS}" == "1" ]]; then
  CLONE_ARGS+=(--clone-qwen-core-layers)
fi

AUTORESEARCH_COMMIT=""
if [[ -d references/official/autoresearch/.git ]]; then
  AUTORESEARCH_COMMIT="$(git -C references/official/autoresearch rev-parse HEAD)"
fi

PYTHONPATH="${PYTHONPATH:-src}" "${PYTHON_BIN}" scripts/395_autoresearch_arbitration_probe.py \
  --model-id "${MODEL_ID}" \
  --checkpoint "${CHECKPOINT}" \
  --checkpoint-load-mode "${CHECKPOINT_LOAD_MODE}" \
  --suite-jsonl "${SUITE_JSONL}" \
  --out-dir "${OUT_DIR}" \
  --ledger-path "${LEDGER_PATH}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --max-seq-len "${MAX_SEQ_LEN}" \
  --batch-size "${BATCH_SIZE}" \
  --max-cases "${MAX_CASES}" \
  --fit-fraction "${FIT_FRACTION}" \
  --core-insertion-mode "${CORE_INSERTION_MODE}" \
  --core-insert-after-layer "${CORE_INSERT_AFTER_LAYER}" \
  --qwen-core-layer-indices "${QWEN_CORE_LAYER_INDICES}" \
  --core-adapter-dim "${CORE_ADAPTER_DIM}" \
  --core-delta-adapter-mode "${CORE_DELTA_ADAPTER_MODE}" \
  --core-residual-gate-mode "${CORE_RESIDUAL_GATE_MODE}" \
  --core-residual-gate-dim "${CORE_RESIDUAL_GATE_DIM}" \
  --core-residual-gate-init "${CORE_RESIDUAL_GATE_INIT}" \
  "${CLONE_ARGS[@]}" \
  --h-cycles "${H_CYCLES}" \
  --l-cycles "${L_CYCLES}" \
  --outer-steps "${OUTER_STEPS}" \
  --core-convergence-halt-enabled \
  --core-convergence-halt-threshold "${CORE_CONVERGENCE_HALT_THRESHOLD}" \
  --core-convergence-halt-min-outer "${CORE_CONVERGENCE_HALT_MIN_OUTER}" \
  --core-step-conditioning-enabled \
  --core-step-conditioning-max-steps "${CORE_STEP_CONDITIONING_MAX_STEPS}" \
  --core-step-conditioning-scale "${CORE_STEP_CONDITIONING_SCALE}" \
  --core-gate-init "${CORE_GATE_INIT}" \
  --residual-scale "${RESIDUAL_SCALE}" \
  --policy "${POLICY}" \
  --base-margin-grid="${BASE_MARGIN_GRID}" \
  --core-margin-grid="${CORE_MARGIN_GRID}" \
  --switch-adv-grid="${SWITCH_ADV_GRID}" \
  --linear-steps "${LINEAR_STEPS}" \
  --linear-lr "${LINEAR_LR}" \
  --linear-weight-decay "${LINEAR_WEIGHT_DECAY}" \
  --linear-threshold-grid="${LINEAR_THRESHOLD_GRID}" \
  --autoresearch-commit "${AUTORESEARCH_COMMIT}" \
  --seed "${SEED}"
