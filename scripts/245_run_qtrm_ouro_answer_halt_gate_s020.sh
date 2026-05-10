#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

export PYTHONPATH=${PYTHONPATH:-$PWD/src}
export HF_HOME=${HF_HOME:-/mnt/nvme1n1p2/hf-cache-qtrm}

CONFIG="${CONFIG:-configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_halt_gate_s020.yaml}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_tailneg_s020_from_tail_s020/last.pt}"
TRAIN_DATA="${TRAIN_DATA:-data/filtered/pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_len579_train40000_v0to5.jsonl}"
EVAL_CASES="${EVAL_CASES:-data/eval/pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_len579_eval50000_v6to7_len7_9.jsonl}"
OUT_DIR="${OUT_DIR:-local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_halt_gate_s020_from_tail_s020}"
STEPS="${STEPS:-20}"
LR="${LR:-1.0e-5}"
ANSWER_STATE_LOOP_HALT_CE_WEIGHT="${ANSWER_STATE_LOOP_HALT_CE_WEIGHT:-0.25}"

echo "=== QTRM Ouro answer-state halt gate S020 ==="
echo "config: ${CONFIG}"
echo "init:   ${INIT_CHECKPOINT}"
echo "data:   ${TRAIN_DATA}"
echo "eval:   ${EVAL_CASES}"
echo "out:    ${OUT_DIR}"
echo

python scripts/196_train_pure_recursive_depth_supervised.py \
  --config "${CONFIG}" \
  --data-jsonl "${TRAIN_DATA}" \
  --init-checkpoint "${INIT_CHECKPOINT}" \
  --tokenizer-model-id Qwen/Qwen3.5-2B-Base \
  --steps "${STEPS}" \
  --lr "${LR}" \
  --depth-steps 1,2,4,8 \
  --target-mode staged \
  --out-dir "${OUT_DIR}" \
  --final-logit-ce-weight 0.0 \
  --depth-final-ce-weight 0.0 \
  --all-depth-ce-weight 0.0 \
  --progress-margin-weight 0.0 \
  --answer-state-loop-halt-ce-weight "${ANSWER_STATE_LOOP_HALT_CE_WEIGHT}" \
  --causal-prefix-supervision \
  --causal-prefix-max-target-tokens 8 \
  --causal-prefix-later-token-weight 0.35 \
  --tail-negative-margin-weight 0.25 \
  --tail-negative-margin 0.07 \
  --tail-negative-family-filter mixed_list_arithmetic \
  --transition-joint-answer-bridge-contrast-weight 0.5 \
  --transition-joint-answer-bridge-contrast-margin 0.05

python scripts/192_eval_raw_intelligence.py \
  --config "${CONFIG}" \
  --checkpoint "${OUT_DIR}/last.pt" \
  --cases "${EVAL_CASES}" \
  --max-cases 8 \
  --scoring causal_forced_choice \
  --choice-score-normalization mean \
  --mode donor_only_no_evidence \
  --mode qtrm_core_off_no_evidence \
  --mode qtrm_core_steps_1_no_evidence \
  --mode qtrm_core_steps_2_no_evidence \
  --mode qtrm_core_steps_4_no_evidence \
  --mode qtrm_core_steps_8_no_evidence \
  --mode qtrm_core_steps_8_answer_halt_gate_off_no_evidence \
  --mode qtrm_core_steps_8_transition_joint_answer_bridge_off_no_evidence \
  --out "${OUT_DIR}/lm_causal_forced_choice_smoke8_with_baselines.jsonl"

python scripts/230_eval_qtrm_latent_action_codebook.py \
  --config "${CONFIG}" \
  --checkpoint "${OUT_DIR}/last.pt" \
  --data-jsonl "${EVAL_CASES}" \
  --core-steps 8 \
  --max-cases 32 \
  --out-json "${OUT_DIR}/action_code_eval32.json"

python scripts/241_summarize_mixed_tail_errors.py \
  --eval-jsonl "${OUT_DIR}/lm_causal_forced_choice_smoke8_with_baselines.jsonl" \
  --out-json "${OUT_DIR}/tail_error_summary_smoke8.json"
