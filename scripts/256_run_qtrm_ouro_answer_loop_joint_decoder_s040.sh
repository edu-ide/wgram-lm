#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-src}"
export HF_HOME="${HF_HOME:-/mnt/nvme0n1p2/hf-cache-qtrm}"
export TMPDIR="${TMPDIR:-/mnt/nvme0n1p2/tmp}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/mnt/nvme0n1p2/torchinductor-cache}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

CONFIG="${CONFIG:-configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_loop_joint_decoder_s040.yaml}"
TRAIN_DATA="${TRAIN_DATA:-data/filtered/pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_len579_train40000_v0to5.jsonl}"
EVAL_CASES="${EVAL_CASES:-data/eval/pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_len579_eval50000_v6to7_len7_9.jsonl}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-/mnt/nvme1n1p2/qtrm-runs/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_next_token_decoder_selfrollout_s040_from_s080/last.pt}"
OUT_DIR="${OUT_DIR:-/mnt/nvme0n1p2/qtrm-runs/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_loop_joint_decoder_s040_from_selfrollout}"
STEPS="${STEPS:-40}"
LR="${LR:-5.0e-5}"
SELF_ROLLOUT_WEIGHT="${SELF_ROLLOUT_WEIGHT:-0.35}"

mkdir -p "${OUT_DIR}"
mkdir -p "${TMPDIR}" "${TORCHINDUCTOR_CACHE_DIR}"

echo "=== QTRM Ouro answer-loop joint decoder S040 ==="
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
  --final-logit-ce-weight 1.0 \
  --depth-final-ce-weight 0.0 \
  --all-depth-ce-weight 0.0 \
  --progress-margin-weight 0.0 \
  --final-greedy-token-margin-weight 0.20 \
  --greedy-token-margin 0.10 \
  --choice-margin-weight 0.30 \
  --choice-margin 0.12 \
  --choice-margin-mode sequence \
  --tail-negative-margin-weight 0.30 \
  --tail-negative-margin 0.08 \
  --tail-negative-family-filter mixed_list_arithmetic \
  --subtract-tail-counterfactual-margin-weight 0.30 \
  --subtract-tail-counterfactual-margin 0.05 \
  --subtract-tail-counterfactual-family-filter mixed_list_arithmetic \
  --causal-prefix-supervision \
  --causal-prefix-max-target-tokens 8 \
  --causal-prefix-later-token-weight 0.85 \
  --causal-prefix-self-rollout-weight "${SELF_ROLLOUT_WEIGHT}" \
  --causal-prefix-self-rollout-max-target-tokens 8

"${PYTHON_BIN}" scripts/247_probe_qtrm_gold_token_ranks.py \
  --config "${CONFIG}" \
  --checkpoint "${OUT_DIR}/last.pt" \
  --cases "${EVAL_CASES}" \
  --max-cases 4 \
  --max-length 512 \
  --mode donor_only_no_evidence \
  --mode qtrm_core_off_no_evidence \
  --mode qtrm_core_steps_8_no_evidence \
  --mode qtrm_core_steps_8_answer_next_token_decoder_off_no_evidence \
  --mode qtrm_core_steps_8_answer_halt_gate_off_no_evidence \
  --out "${OUT_DIR}/gold_token_rank_probe4.jsonl"

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
  --mode qtrm_core_steps_8_answer_next_token_decoder_off_no_evidence \
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
  --mode qtrm_core_steps_8_answer_next_token_decoder_off_no_evidence \
  --mode qtrm_core_steps_8_answer_halt_gate_off_no_evidence \
  --out "${OUT_DIR}/causal_forced_choice_smoke4.jsonl"
