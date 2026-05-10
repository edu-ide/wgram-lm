#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-src}"
export HF_HOME="${HF_HOME:-/mnt/nvme1n1p2/hf-cache-qtrm}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

CONFIG="${CONFIG:-configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_donor_guided_adapter_s060.yaml}"
TRAIN_DATA="${TRAIN_DATA:-data/filtered/pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_len579_train40000_v0to5.jsonl}"
EVAL_CASES="${EVAL_CASES:-data/eval/pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_len579_eval50000_v6to7_len7_9.jsonl}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_halt_head_s080_from_tail_s020/last.pt}"
OUT_DIR="${OUT_DIR:-local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_donor_guided_hardneg_s080_from_halt_s080}"
STEPS="${STEPS:-80}"
LR="${LR:-2.0e-4}"
FINAL_LOGIT_CE_WEIGHT="${FINAL_LOGIT_CE_WEIGHT:-0.8}"
FINAL_GREEDY_MARGIN_WEIGHT="${FINAL_GREEDY_MARGIN_WEIGHT:-0.25}"
CHOICE_MARGIN_WEIGHT="${CHOICE_MARGIN_WEIGHT:-0.40}"
TAIL_NEGATIVE_MARGIN_WEIGHT="${TAIL_NEGATIVE_MARGIN_WEIGHT:-0.50}"
SUBTRACT_TAIL_COUNTERFACTUAL_MARGIN_WEIGHT="${SUBTRACT_TAIL_COUNTERFACTUAL_MARGIN_WEIGHT:-0.50}"
TEACHER_FINAL_KL_WEIGHT="${TEACHER_FINAL_KL_WEIGHT:-0.05}"

mkdir -p "${OUT_DIR}"

echo "=== QTRM Ouro donor-guided final-answer hard-negative S080 ==="
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
  --depth-steps 8 \
  --target-mode final \
  --out-dir "${OUT_DIR}" \
  --final-logit-ce-weight "${FINAL_LOGIT_CE_WEIGHT}" \
  --depth-final-ce-weight 0.0 \
  --all-depth-ce-weight 0.0 \
  --progress-margin-weight 0.0 \
  --final-greedy-token-margin-weight "${FINAL_GREEDY_MARGIN_WEIGHT}" \
  --greedy-token-margin 0.12 \
  --choice-margin-weight "${CHOICE_MARGIN_WEIGHT}" \
  --choice-margin 0.15 \
  --choice-margin-mode sequence \
  --tail-negative-margin-weight "${TAIL_NEGATIVE_MARGIN_WEIGHT}" \
  --tail-negative-margin 0.12 \
  --tail-negative-family-filter mixed_list_arithmetic \
  --subtract-tail-counterfactual-margin-weight "${SUBTRACT_TAIL_COUNTERFACTUAL_MARGIN_WEIGHT}" \
  --subtract-tail-counterfactual-margin 0.08 \
  --subtract-tail-counterfactual-family-filter mixed_list_arithmetic \
  --teacher-checkpoint "${INIT_CHECKPOINT}" \
  --teacher-final-logit-kl-weight "${TEACHER_FINAL_KL_WEIGHT}" \
  --teacher-depth-kl-temperature 2.0 \
  --causal-prefix-supervision \
  --causal-prefix-max-target-tokens 8 \
  --causal-prefix-later-token-weight 0.85

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
  --mode qtrm_core_steps_8_no_evidence \
  --mode qtrm_core_steps_8_delta_off_no_evidence \
  --mode qtrm_core_steps_8_answer_halt_gate_off_no_evidence \
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
  --mode qtrm_core_steps_8_no_evidence \
  --mode qtrm_core_steps_8_delta_off_no_evidence \
  --mode qtrm_core_steps_8_answer_halt_gate_off_no_evidence \
  --out "${OUT_DIR}/causal_forced_choice_smoke4.jsonl"
