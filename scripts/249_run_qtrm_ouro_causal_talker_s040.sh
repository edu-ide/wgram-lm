#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-src}"
export HF_HOME="${HF_HOME:-/mnt/nvme1n1p2/hf-cache-qtrm}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

CONFIG="${CONFIG:-configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_causal_talker_s040.yaml}"
TRAIN_DATA="${TRAIN_DATA:-data/filtered/pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_len579_train40000_v0to5.jsonl}"
EVAL_CASES="${EVAL_CASES:-data/eval/pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_len579_eval50000_v6to7_len7_9.jsonl}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_halt_head_s080_from_tail_s020/last.pt}"
OUT_DIR="${OUT_DIR:-local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_causal_talker_s040_from_halt_s080}"
STEPS="${STEPS:-40}"
LR="${LR:-5.0e-5}"
DEPTH_FINAL_CE_WEIGHT="${DEPTH_FINAL_CE_WEIGHT:-0.5}"
HALT_CE_WEIGHT="${HALT_CE_WEIGHT:-0.25}"
TEACHER_DEPTH_KL_WEIGHT="${TEACHER_DEPTH_KL_WEIGHT:-0.10}"

mkdir -p "${OUT_DIR}"

echo "=== QTRM Ouro causal Talker S040 ==="
echo "config: ${CONFIG}"
echo "init:   ${INIT_CHECKPOINT}"
echo "data:   ${TRAIN_DATA}"
echo "eval:   ${EVAL_CASES}"
echo "out:    ${OUT_DIR}"
echo

"${PYTHON_BIN}" scripts/196_train_pure_recursive_depth_supervised.py \
  --config "${CONFIG}" \
  --data-jsonl "${TRAIN_DATA}" \
  --init-checkpoint "${INIT_CHECKPOINT}" \
  --tokenizer-model-id Qwen/Qwen3.5-2B-Base \
  --steps "${STEPS}" \
  --lr "${LR}" \
  --depth-steps 4,8 \
  --target-mode staged \
  --out-dir "${OUT_DIR}" \
  --final-logit-ce-weight 0.0 \
  --depth-final-ce-weight "${DEPTH_FINAL_CE_WEIGHT}" \
  --all-depth-ce-weight 0.0 \
  --progress-margin-weight 0.0 \
  --answer-state-loop-halt-ce-weight "${HALT_CE_WEIGHT}" \
  --teacher-checkpoint "${INIT_CHECKPOINT}" \
  --teacher-first-token-depth-kl-weight "${TEACHER_DEPTH_KL_WEIGHT}" \
  --teacher-depth-kl-temperature 2.0 \
  --causal-prefix-supervision \
  --causal-prefix-max-target-tokens 8 \
  --causal-prefix-later-token-weight 0.65

"${PYTHON_BIN}" scripts/192_eval_raw_intelligence.py \
  --config "${CONFIG}" \
  --checkpoint "${OUT_DIR}/last.pt" \
  --cases "${EVAL_CASES}" \
  --max-cases 8 \
  --max-length 512 \
  --max-new-tokens 8 \
  --scoring generation \
  --choice-score-normalization mean \
  --suppress-visible-reasoning-tokens \
  --mode donor_only_no_evidence \
  --mode qtrm_core_off_no_evidence \
  --mode qtrm_core_steps_4_no_evidence \
  --mode qtrm_core_steps_8_no_evidence \
  --out "${OUT_DIR}/generation_smoke8.jsonl"

"${PYTHON_BIN}" scripts/192_eval_raw_intelligence.py \
  --config "${CONFIG}" \
  --checkpoint "${OUT_DIR}/last.pt" \
  --cases "${EVAL_CASES}" \
  --max-cases 4 \
  --max-length 512 \
  --scoring causal_forced_choice \
  --choice-score-normalization mean \
  --mode donor_only_no_evidence \
  --mode qtrm_core_off_no_evidence \
  --mode qtrm_core_steps_4_no_evidence \
  --mode qtrm_core_steps_8_no_evidence \
  --mode qtrm_core_steps_8_answer_halt_gate_off_no_evidence \
  --out "${OUT_DIR}/causal_forced_choice_smoke4.jsonl"
