#!/usr/bin/env bash
set -euo pipefail

# Stage62A: local born-one-body PV-GRAM with official GatedDeltaNet-2 3:1.
#
# Plain-language contract:
#   One student is born with one nervous system:
#     text reader -> recurrent thought/search -> same LM-token speaker
#   There is no Qwen donor, no side renderer, and no external calculator.

ACTION="${1:-plan}"
ROOT="${ROOT:-/home/tripleyoung/qtrm-workspace/qtrm_multimodal_memoryos}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-/tmp/stage62a_born_onebody_pvgram_gdn2_3to1_${TS}}"
LOG="${LOG:-/tmp/stage62a_born_onebody_pvgram_gdn2_3to1_${TS}.log}"
TARGET_LEVEL="${TARGET_LEVEL:-Stage62A born-one-body PV-GRAM official GDN2 3:1}"
ACCEPTED_DECISION="${ACCEPTED_DECISION:-accepted_stage62a_born_onebody_pvgram}"
PYTHON="${PYTHON:-.venv/bin/python}"

STEPS="${STEPS:-1200}"
TRAIN_CASES="${TRAIN_CASES:-4096}"
EVAL_CASES="${EVAL_CASES:-256}"
PROGRAM_LEN="${PROGRAM_LEN:-4}"
THINK_STEPS="${THINK_STEPS:-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"
D_MODEL="${D_MODEL:-128}"
N_HEADS="${N_HEADS:-8}"
D_FF="${D_FF:-256}"
LR="${LR:-3e-4}"
DEVICE="${DEVICE:-cuda}"
LOG_EVERY="${LOG_EVERY:-25}"
EVAL_EVERY="${EVAL_EVERY:-300}"
DELTA_BACKEND="${DELTA_BACKEND:-official_gated_delta2}"
STRICT_BACKENDS="${STRICT_BACKENDS:-1}"
GRAM_TRAJECTORY_COUNT="${GRAM_TRAJECTORY_COUNT:-4}"
GRAM_NOISE_STD="${GRAM_NOISE_STD:-0.1}"
GRAM_LPRM_WEIGHT="${GRAM_LPRM_WEIGHT:-0.2}"
GRAM_LPRM_MAX_CASES="${GRAM_LPRM_MAX_CASES:-8}"
GRAM_LPRM_EVERY="${GRAM_LPRM_EVERY:-25}"
GRAM_LPRM_RANKING_WEIGHT="${GRAM_LPRM_RANKING_WEIGHT:-0.0}"
GRAM_CANDIDATE_TOPK_PER_TRAJECTORY="${GRAM_CANDIDATE_TOPK_PER_TRAJECTORY:-1}"
GRAM_CANDIDATE_SCORE_MODE="${GRAM_CANDIDATE_SCORE_MODE:-generated_state}"
GRAM_CANDIDATE_SCORE_TRAIN_BODY="${GRAM_CANDIDATE_SCORE_TRAIN_BODY:-0}"
GRAM_CANDIDATE_SELECTOR="${GRAM_CANDIDATE_SELECTOR:-lprm_head}"
GRAM_ATTRACTOR_ITERATIONS="${GRAM_ATTRACTOR_ITERATIONS:-3}"
GRAM_ATTRACTOR_STEP_SCALE="${GRAM_ATTRACTOR_STEP_SCALE:-0.5}"
GRAM_LCV_LATENT_DIM="${GRAM_LCV_LATENT_DIM:-0}"
GRAM_LCV_TEMPERATURE="${GRAM_LCV_TEMPERATURE:-1.0}"
GRAM_TRACE_MAX_LEN="${GRAM_TRACE_MAX_LEN:-0}"
GRAM_TRACE_CONSISTENCY_WEIGHT="${GRAM_TRACE_CONSISTENCY_WEIGHT:-0.0}"
ACCEPT_MIN_EXACT="${ACCEPT_MIN_EXACT:-0.12}"
ACCEPT_MIN_DEPTH_GAIN="${ACCEPT_MIN_DEPTH_GAIN:-0.04}"
ACCEPT_MIN_ABLATION_DROP="${ACCEPT_MIN_ABLATION_DROP:-0.04}"
ACCEPT_MIN_FAMILY_EXACT="${ACCEPT_MIN_FAMILY_EXACT:-0.03}"

run_foreground() {
  cd "${ROOT}"
  local train_body_args=()
  case "${GRAM_CANDIDATE_SCORE_TRAIN_BODY}" in
    1|true|TRUE|yes|YES|on|ON)
      train_body_args=(--gram-candidate-score-train-body)
      ;;
  esac
  local strict_backend_args=()
  case "${STRICT_BACKENDS}" in
    1|true|TRUE|yes|YES|on|ON)
      strict_backend_args=(--strict-backends)
      ;;
  esac
  PYTHONUNBUFFERED=1 PYTHONPATH=src "${PYTHON}" scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py \
    --out-dir "${OUT_DIR}" \
    --target-level "${TARGET_LEVEL}" \
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
    --delta-backend "${DELTA_BACKEND}" \
    --delta-head-dim "$((D_MODEL / N_HEADS))" \
    --delta-num-v-heads "${N_HEADS}" \
    --delta-expand-v 1.0 \
    --delta-no-short-conv \
    "${strict_backend_args[@]}" \
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
    --eval-gram-trajectory-search \
    --gram-trajectory-count "${GRAM_TRAJECTORY_COUNT}" \
    --gram-stochastic-noise-std "${GRAM_NOISE_STD}" \
    --gram-lprm-loss-weight "${GRAM_LPRM_WEIGHT}" \
    --gram-lprm-max-cases "${GRAM_LPRM_MAX_CASES}" \
    --gram-lprm-every "${GRAM_LPRM_EVERY}" \
    --gram-lprm-ranking-loss-weight "${GRAM_LPRM_RANKING_WEIGHT}" \
    --gram-candidate-topk-per-trajectory "${GRAM_CANDIDATE_TOPK_PER_TRAJECTORY}" \
    --gram-candidate-selector "${GRAM_CANDIDATE_SELECTOR}" \
    --gram-attractor-iterations "${GRAM_ATTRACTOR_ITERATIONS}" \
    --gram-attractor-step-scale "${GRAM_ATTRACTOR_STEP_SCALE}" \
    --gram-lcv-latent-dim "${GRAM_LCV_LATENT_DIM}" \
    --gram-lcv-temperature "${GRAM_LCV_TEMPERATURE}" \
    --gram-trace-max-len "${GRAM_TRACE_MAX_LEN}" \
    --gram-trace-consistency-weight "${GRAM_TRACE_CONSISTENCY_WEIGHT}" \
    --gram-candidate-score-mode "${GRAM_CANDIDATE_SCORE_MODE}" \
    "${train_body_args[@]}" \
    --accept-min-exact "${ACCEPT_MIN_EXACT}" \
    --accept-min-depth-gain "${ACCEPT_MIN_DEPTH_GAIN}" \
    --accept-min-ablation-drop "${ACCEPT_MIN_ABLATION_DROP}" \
    --accept-min-family-exact "${ACCEPT_MIN_FAMILY_EXACT}" \
    --accepted-decision "${ACCEPTED_DECISION}" \
    --log-every "${LOG_EVERY}" \
    ${EXTRA_ARGS:-}
}

case "${ACTION}" in
  plan)
    cat <<PLAN
Stage62A born-one-body PV-GRAM local run

Human contract:
  One model learns to read, think/search, verify paths, and speak through the
  same LM-token path from the beginning.

Run:
  bash scripts/launch_stage62a_local_born_onebody_pvgram.sh run

Foreground debug:
  bash scripts/launch_stage62a_local_born_onebody_pvgram.sh foreground

Status:
  bash scripts/launch_stage62a_local_born_onebody_pvgram.sh status

OUT_DIR=${OUT_DIR}
LOG=${LOG}
PLAN
    ;;
  foreground)
    run_foreground
    ;;
  run)
    mkdir -p "$(dirname "${LOG}")"
    setsid bash "${BASH_SOURCE[0]}" foreground > "${LOG}" 2>&1 < /dev/null &
    echo "PID:$!"
    echo "OUT_DIR:${OUT_DIR}"
    echo "LOG:${LOG}"
    ;;
  status)
    echo "Recent Stage62A processes:"
    ps -eo pid,ppid,sid,stat,etime,cmd | rg "launch_stage62a_local_born_onebody_pvgram|337_train_qtrm_native_mixed_text_reasoning_probe" || true
    echo
    echo "Latest log:"
    tail -n 40 "${LOG}" 2>/dev/null || true
    ;;
  *)
    echo "unknown action: ${ACTION}" >&2
    exit 2
    ;;
esac
