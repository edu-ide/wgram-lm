#!/usr/bin/env bash
set -euo pipefail

# Stage66 HRM-Text-style born-one-body PV-GRAM pretraining.
#
# Plain-language contract:
#   HRM-Text succeeds because the reader, recurrent thinker, and token speaker
#   are trained as one body. Stage66 keeps that contract, then adds PV-GRAM:
#   candidate paths are generated, scored through the same forward body, and
#   trace consistency is auxiliary evidence rather than a detached shortcut.

ACTION="${1:-plan}"
ROOT="${ROOT:-/home/tripleyoung/qtrm-workspace/qtrm_multimodal_memoryos}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-/tmp/stage66_hrmtext_pvgram_pretrain_${TS}}"
LOG="${LOG:-/tmp/stage66_hrmtext_pvgram_pretrain_${TS}.log}"
STAGE62_LAUNCHER="${STAGE62_LAUNCHER:-${ROOT}/scripts/launch_stage62a_local_born_onebody_pvgram.sh}"

TARGET_LEVEL="${TARGET_LEVEL:-Stage66 HRM-Text-style born-one-body PV-GRAM pretraining}"
ACCEPTED_DECISION="${ACCEPTED_DECISION:-accepted_stage66_hrmtext_pvgram_pretrain}"

# HRM-Text-like: train the normal answer path and candidate verifier together.
GRAM_CANDIDATE_SELECTOR="${GRAM_CANDIDATE_SELECTOR:-lprm_head}"
GRAM_CANDIDATE_SCORE_MODE="${GRAM_CANDIDATE_SCORE_MODE:-candidate_forward}"
GRAM_CANDIDATE_SCORE_TRAIN_BODY="${GRAM_CANDIDATE_SCORE_TRAIN_BODY:-1}"
GRAM_CANDIDATE_TOPK_PER_TRAJECTORY="${GRAM_CANDIDATE_TOPK_PER_TRAJECTORY:-2}"
GRAM_TRAJECTORY_COUNT="${GRAM_TRAJECTORY_COUNT:-4}"
GRAM_LPRM_WEIGHT="${GRAM_LPRM_WEIGHT:-0.2}"
GRAM_LPRM_RANKING_WEIGHT="${GRAM_LPRM_RANKING_WEIGHT:-0.25}"
GRAM_TRACE_MAX_LEN="${GRAM_TRACE_MAX_LEN:-4}"
GRAM_TRACE_CONSISTENCY_WEIGHT="${GRAM_TRACE_CONSISTENCY_WEIGHT:-0.2}"

# DGX GB10 + official GatedDeltaNet-2 needs an explicitly pinned ptxas.
# Do not fall back to Triton's bundled ptxas or auto-discovered CUDA paths.
REQUIRED_TRITON_PTXAS_PATH="${REQUIRED_TRITON_PTXAS_PATH:-}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-}"

require_triton_ptxas() {
  if [[ -z "${REQUIRED_TRITON_PTXAS_PATH:-}" ]]; then
    echo "missing required ptxas contract: set REQUIRED_TRITON_PTXAS_PATH explicitly" >&2
    exit 5
  fi
  if [[ -z "${TRITON_PTXAS_PATH:-}" ]]; then
    echo "missing required ptxas: set TRITON_PTXAS_PATH=${REQUIRED_TRITON_PTXAS_PATH}" >&2
    exit 5
  fi
  if [[ "${TRITON_PTXAS_PATH}" != "${REQUIRED_TRITON_PTXAS_PATH}" ]]; then
    echo "wrong ptxas: TRITON_PTXAS_PATH=${TRITON_PTXAS_PATH}, required=${REQUIRED_TRITON_PTXAS_PATH}" >&2
    exit 5
  fi
  if [[ ! -x "${TRITON_PTXAS_PATH}" ]]; then
    echo "missing required ptxas: ${TRITON_PTXAS_PATH}" >&2
    exit 5
  fi
}

# Learning-efficiency claim gate. This is intentionally explicit: no baseline
# report means no 10x claim.
LEARNING_EFFICIENCY_FACTOR="${LEARNING_EFFICIENCY_FACTOR:-10}"
HRM_TEXT_BASELINE_REPORT="${HRM_TEXT_BASELINE_REPORT:-}"
CANDIDATE_REPORT="${CANDIDATE_REPORT:-${OUT_DIR}/report.json}"
EFFICIENCY_REPORT="${EFFICIENCY_REPORT:-${OUT_DIR}/learning_efficiency_${LEARNING_EFFICIENCY_FACTOR}x.json}"

run_stage62_with_stage66_defaults() {
  require_triton_ptxas
  local triton_env=()
  triton_env+=(TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH}")
  if [[ -n "${TRITON_CACHE_DIR}" ]]; then
    triton_env+=(TRITON_CACHE_DIR="${TRITON_CACHE_DIR}")
  fi
  env \
    "${triton_env[@]}" \
    ROOT="${ROOT}" \
    OUT_DIR="${OUT_DIR}" \
    LOG="${LOG}" \
    TARGET_LEVEL="${TARGET_LEVEL}" \
    ACCEPTED_DECISION="${ACCEPTED_DECISION}" \
    GRAM_CANDIDATE_SELECTOR="${GRAM_CANDIDATE_SELECTOR}" \
    GRAM_CANDIDATE_SCORE_MODE="${GRAM_CANDIDATE_SCORE_MODE}" \
    GRAM_CANDIDATE_SCORE_TRAIN_BODY="${GRAM_CANDIDATE_SCORE_TRAIN_BODY}" \
    GRAM_CANDIDATE_TOPK_PER_TRAJECTORY="${GRAM_CANDIDATE_TOPK_PER_TRAJECTORY}" \
    GRAM_TRAJECTORY_COUNT="${GRAM_TRAJECTORY_COUNT}" \
    GRAM_LPRM_WEIGHT="${GRAM_LPRM_WEIGHT}" \
    GRAM_LPRM_RANKING_WEIGHT="${GRAM_LPRM_RANKING_WEIGHT}" \
    GRAM_TRACE_MAX_LEN="${GRAM_TRACE_MAX_LEN}" \
    GRAM_TRACE_CONSISTENCY_WEIGHT="${GRAM_TRACE_CONSISTENCY_WEIGHT}" \
    bash "${STAGE62_LAUNCHER}" "$@"
}

compare_learning_efficiency() {
  if [[ -z "${HRM_TEXT_BASELINE_REPORT}" ]]; then
    echo "HRM_TEXT_BASELINE_REPORT is required for a ${LEARNING_EFFICIENCY_FACTOR}x claim." >&2
    echo "Run without compare, or provide a comparable baseline report." >&2
    exit 2
  fi
  cd "${ROOT}"
  PYTHONPATH=src .venv/bin/python scripts/532_compare_learning_efficiency_claim.py \
    --baseline-report "${HRM_TEXT_BASELINE_REPORT}" \
    --candidate-report "${CANDIDATE_REPORT}" \
    --factor "${LEARNING_EFFICIENCY_FACTOR}" \
    --output "${EFFICIENCY_REPORT}"
}

case "${ACTION}" in
  plan)
    cat <<PLAN
Stage66 HRM-Text-style PV-GRAM pretraining

Human contract:
  One born model learns reader -> recurrent thinker -> candidate/value judge ->
  token speaker as one body. Candidate scoring uses candidate_forward so the
  normal answer path is trained, not a side-only verifier.

Run:
  bash scripts/launch_stage66_local_hrmtext_pvgram_pretrain.sh run

Foreground debug:
  bash scripts/launch_stage66_local_hrmtext_pvgram_pretrain.sh foreground

Status:
  bash scripts/launch_stage66_local_hrmtext_pvgram_pretrain.sh status

10x learning-efficiency gate:
  HRM_TEXT_BASELINE_REPORT=/path/to/baseline/report.json \\
    bash scripts/launch_stage66_local_hrmtext_pvgram_pretrain.sh compare

OUT_DIR=${OUT_DIR}
LOG=${LOG}
TARGET_LEVEL=${TARGET_LEVEL}
TRITON_PTXAS_PATH=${TRITON_PTXAS_PATH:-<unset>}
REQUIRED_TRITON_PTXAS_PATH=${REQUIRED_TRITON_PTXAS_PATH}
TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-<unset>}
LEARNING_EFFICIENCY_FACTOR=${LEARNING_EFFICIENCY_FACTOR}
PLAN
    ;;
  foreground)
    run_stage62_with_stage66_defaults foreground
    ;;
  run)
    run_stage62_with_stage66_defaults run
    ;;
  status)
    run_stage62_with_stage66_defaults status
    ;;
  compare)
    compare_learning_efficiency
    ;;
  *)
    echo "unknown action: ${ACTION}" >&2
    exit 2
    ;;
esac
