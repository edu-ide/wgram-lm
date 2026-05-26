#!/usr/bin/env bash
set -euo pipefail

# Stage61C-D: local from-scratch one-body with official NVlabs GatedDeltaNet-2.
#
# Plain-language contract:
#   One model reads tokens, edits working memory with official GDN2 delta
#   blocks, periodically re-reads with attention, thinks through the recurrent
#   core, and speaks through the same LM logits. There is no Qwen donor, no side
#   renderer, and no external calculator.
#
# Architecture:
#   Qwen3.5-style 3:1 backbone:
#     official GDN2 -> official GDN2 -> official GDN2 -> attention
#   repeated inside the reader/thought/speaker path.

OUT_DIR="${OUT_DIR:-/tmp/stage61c_official_gdn2_3to1_$(date +%Y%m%d_%H%M%S)}"
STEPS="${STEPS:-400}"
TRAIN_CASES="${TRAIN_CASES:-2048}"
EVAL_CASES="${EVAL_CASES:-128}"
PROGRAM_LEN="${PROGRAM_LEN:-4}"
THINK_STEPS="${THINK_STEPS:-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"
D_MODEL="${D_MODEL:-128}"
N_HEADS="${N_HEADS:-8}"
D_FF="${D_FF:-256}"
LR="${LR:-3e-4}"
DEVICE="${DEVICE:-cuda}"
LOG_EVERY="${LOG_EVERY:-100}"
EVAL_EVERY="${EVAL_EVERY:-200}"
ACCEPT_MIN_EXACT="${ACCEPT_MIN_EXACT:-0.20}"
ACCEPT_MIN_DEPTH_GAIN="${ACCEPT_MIN_DEPTH_GAIN:-0.06}"
ACCEPT_MIN_ABLATION_DROP="${ACCEPT_MIN_ABLATION_DROP:-0.06}"
ACCEPT_MIN_FAMILY_EXACT="${ACCEPT_MIN_FAMILY_EXACT:-0.03}"

PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py \
  --out-dir "${OUT_DIR}" \
  --target-level "Stage61C-D official GDN2-backed 3:1 single-thought from scratch" \
  --steps "${STEPS}" \
  --train-cases "${TRAIN_CASES}" \
  --eval-cases "${EVAL_CASES}" \
  --task-families 'modchain,revchain,modchain,revchain,checksum' \
  --eval-task-families 'modchain,revchain,checksum' \
  --eval-family-order-invariant \
  --include-family-tag \
  --tokenizer-mode number \
  --number-tokenizer-max-value 99 \
  --number-tokenizer-op-role-tokens \
  --value-codec circular \
  --program-len "${PROGRAM_LEN}" \
  --modulus 32 \
  --d-model "${D_MODEL}" \
  --n-heads "${N_HEADS}" \
  --n-kv-heads 4 \
  --d-ff "${D_FF}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --device "${DEVICE}" \
  --train-think-steps "${THINK_STEPS}" \
  --eval-think-steps "${THINK_STEPS}" \
  --backbone trm_qwen35_3to1 \
  --encode-backbone trm_qwen35_3to1 \
  --think-backbone trm_qwen35_3to1 \
  --decode-backbone trm_qwen35_3to1 \
  --think-structure single \
  --delta-backend official_gated_delta2 \
  --delta-head-dim "$((D_MODEL / N_HEADS))" \
  --delta-num-v-heads "${N_HEADS}" \
  --delta-expand-v 1.0 \
  --delta-no-short-conv \
  --strict-backends \
  --position-embedding-mode randomized \
  --model-max-seq-len 128 \
  --active-len-curriculum \
  --active-len-curriculum-min 1 \
  --active-len-curriculum-warmup-frac 0.35 \
  --depth-intermediate-loss-weight 0.30 \
  --depth-intermediate-min-depth 1 \
  --answer-margin-loss-weight 0.30 \
  --eval-seed 9338 \
  --eval-during-training-every "${EVAL_EVERY}" \
  --eval-during-training-cases "${EVAL_CASES}" \
  --periodic-eval-score-mode family_floor \
  --restore-best-eval-checkpoint \
  --save-best-periodic-checkpoint \
  --eval-state-trace \
  --eval-core-answer-probe \
  --eval-core-step-probe \
  --eval-operation-breakdown \
  --accept-min-exact "${ACCEPT_MIN_EXACT}" \
  --accept-min-depth-gain "${ACCEPT_MIN_DEPTH_GAIN}" \
  --accept-min-ablation-drop "${ACCEPT_MIN_ABLATION_DROP}" \
  --accept-min-family-exact "${ACCEPT_MIN_FAMILY_EXACT}" \
  --accepted-decision "accepted_stage61c_official_gdn2_3to1" \
  --log-every "${LOG_EVERY}" \
  ${EXTRA_ARGS:-}

echo "Stage61C-D official GDN2-backed 3:1 run wrote: ${OUT_DIR}"
