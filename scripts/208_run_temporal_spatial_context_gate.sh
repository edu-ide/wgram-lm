#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/qwen35_2b_4090_temporal_spatial_context_probe.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_DATA="${TRAIN_DATA:-data/train/temporal_spatial_context_train_120.jsonl}"
EVAL_DATA="${EVAL_DATA:-data/eval/temporal_spatial_context_heldout_24.jsonl}"
OUT_DIR="${OUT_DIR:-/mnt/nvme1n1p2/qtrm-local-checkpoints/temporal_spatial_context_probe/s240}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-/mnt/nvme1n1p2/qtrm-local-checkpoints/metacog_pair_rebuild/no_warmup_rebuilt_s001/last.pt}"
EVAL_OUT="${EVAL_OUT:-local_eval/temporal_spatial_context_gate.jsonl}"
GATE_MD="${GATE_MD:-docs/wiki/decisions/temporal-spatial-context-gate.md}"
GATE_JSON="${GATE_JSON:-docs/wiki/decisions/temporal-spatial-context-gate-summary.json}"
TRAIN_CASES_PER_FAMILY="${TRAIN_CASES_PER_FAMILY:-60}"
EVAL_CASES_PER_FAMILY="${EVAL_CASES_PER_FAMILY:-12}"
STEPS="${STEPS:-240}"
DEPTH_STEPS="${DEPTH_STEPS:-1,2,4,8}"
MAX_LENGTH="${MAX_LENGTH:-512}"
LR="${LR:-5.0e-5}"
FINAL_LOGIT_CE_WEIGHT="${FINAL_LOGIT_CE_WEIGHT:-1.0}"
ALL_DEPTH_CE_WEIGHT="${ALL_DEPTH_CE_WEIGHT:-0.5}"
PROGRESS_MARGIN_WEIGHT="${PROGRESS_MARGIN_WEIGHT:-0.25}"
PROGRESS_MARGIN="${PROGRESS_MARGIN:-0.10}"
CAUSAL_PREFIX_MAX_TARGET_TOKENS="${CAUSAL_PREFIX_MAX_TARGET_TOKENS:-2}"
CAUSAL_PREFIX_LATER_TOKEN_WEIGHT="${CAUSAL_PREFIX_LATER_TOKEN_WEIGHT:-1.0}"
TEMPORAL_SPATIAL_CONTEXT_CONTRAST_WEIGHT="${TEMPORAL_SPATIAL_CONTEXT_CONTRAST_WEIGHT:-0.5}"
TEMPORAL_SPATIAL_CONTEXT_CONTRAST_MARGIN="${TEMPORAL_SPATIAL_CONTEXT_CONTRAST_MARGIN:-0.10}"

echo "=== Temporal-Spatial Context Gate ==="
echo "Python: ${PYTHON_BIN}"
echo "Config: ${CONFIG}"
echo "Train data: ${TRAIN_DATA}"
echo "Eval data: ${EVAL_DATA}"
echo "Init checkpoint: ${INIT_CHECKPOINT}"
echo "Out dir: ${OUT_DIR}"
echo "Eval out: ${EVAL_OUT}"
echo "Gate markdown: ${GATE_MD}"
echo "Gate json: ${GATE_JSON}"
echo "Context contrast weight: ${TEMPORAL_SPATIAL_CONTEXT_CONTRAST_WEIGHT}"

PYTHONPATH="${PYTHONPATH:-src}" "${PYTHON_BIN}" scripts/207_build_temporal_spatial_context_cases.py \
  --out "${TRAIN_DATA}" \
  --cases-per-family "${TRAIN_CASES_PER_FAMILY}" \
  --start-index 1000

PYTHONPATH="${PYTHONPATH:-src}" "${PYTHON_BIN}" scripts/207_build_temporal_spatial_context_cases.py \
  --out "${EVAL_DATA}" \
  --cases-per-family "${EVAL_CASES_PER_FAMILY}" \
  --start-index 0

PYTHONPATH="${PYTHONPATH:-src}" "${PYTHON_BIN}" scripts/196_train_pure_recursive_depth_supervised.py \
  --config "${CONFIG}" \
  --data-jsonl "${TRAIN_DATA}" \
  --init-checkpoint "${INIT_CHECKPOINT}" \
  --max-length "${MAX_LENGTH}" \
  --steps "${STEPS}" \
  --depth-steps "${DEPTH_STEPS}" \
  --target-mode final \
  --lr "${LR}" \
  --out-dir "${OUT_DIR}" \
  --final-logit-ce-weight "${FINAL_LOGIT_CE_WEIGHT}" \
  --all-depth-ce-weight "${ALL_DEPTH_CE_WEIGHT}" \
  --progress-margin-weight "${PROGRESS_MARGIN_WEIGHT}" \
  --progress-margin "${PROGRESS_MARGIN}" \
  --causal-prefix-supervision \
  --causal-prefix-max-target-tokens "${CAUSAL_PREFIX_MAX_TARGET_TOKENS}" \
  --causal-prefix-later-token-weight "${CAUSAL_PREFIX_LATER_TOKEN_WEIGHT}" \
  --temporal-spatial-context-contrast-weight "${TEMPORAL_SPATIAL_CONTEXT_CONTRAST_WEIGHT}" \
  --temporal-spatial-context-contrast-margin "${TEMPORAL_SPATIAL_CONTEXT_CONTRAST_MARGIN}"

PYTHONPATH="${PYTHONPATH:-src}" "${PYTHON_BIN}" scripts/192_eval_raw_intelligence.py \
  --config "${CONFIG}" \
  --checkpoint "${OUT_DIR}/last.pt" \
  --cases "${EVAL_DATA}" \
  --out "${EVAL_OUT}" \
  --scoring forced_choice \
  --mode qtrm_core_steps_8_no_evidence \
  --mode qtrm_core_steps_8_temporal_spatial_off_no_evidence

PYTHONPATH="${PYTHONPATH:-src}" "${PYTHON_BIN}" scripts/191_build_raw_intelligence_gate.py \
  --eval-jsonl "${EVAL_OUT}" \
  --gate-type temporal_spatial_context \
  --markdown-out "${GATE_MD}" \
  --json-out "${GATE_JSON}"

echo "=== Gate complete ==="
echo "Wrote ${GATE_MD}"
echo "Wrote ${GATE_JSON}"
