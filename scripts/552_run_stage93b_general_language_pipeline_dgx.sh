#!/usr/bin/env bash
set -euo pipefail

# Build and train the first HRM-Text-style general-language continuation.
#
# Plain-language contract:
#   Stage93A proved that the one-body model can memorize the tiny math/reasoning
#   handout, but it still talks like that handout. Stage93B changes the book:
#   ordinary instruction/dialogue/QA/translation shelves are sampled through the
#   same PrefixLM answer path before we judge language generation again.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${ROOT}"

DATA_RUN_NAME="${DATA_RUN_NAME:-stage93_hrm_text_general_language_curriculum}"
DATA_WORK_DIR="${DATA_WORK_DIR:-${ROOT}/local_eval/${DATA_RUN_NAME}_dataio}"
SAMPLED_DATA="${SAMPLED_DATA:-${DATA_WORK_DIR}/sampled}"

OUT_RUN_NAME="${OUT_RUN_NAME:-20260524_STAGE93B_DGX913M_GENERAL_LANGUAGE_CONTINUE}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/local_eval/${OUT_RUN_NAME}}"
TRAIN_LOG="${TRAIN_LOG:-/tmp/${OUT_RUN_NAME}.log}"
PIPELINE_LOG="${PIPELINE_LOG:-/tmp/${OUT_RUN_NAME}_pipeline.log}"
PREP_LOG="${PREP_LOG:-/tmp/${DATA_RUN_NAME}_dataio.log}"

RESUME="${RESUME:-${ROOT}/local_eval/20260524_STAGE93A00_DGX913M_MICRO_HARDLINK_TO24500/last_model.pt}"
TARGET_STEPS="${TARGET_STEPS:-60000}"
POLL_SECONDS="${POLL_SECONDS:-120}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-86400}"
WAIT_FOR_OLD_REASONING_SAMPLERS="${WAIT_FOR_OLD_REASONING_SAMPLERS:-0}"
DATA_PREP_RELAUNCH_GRACE_SECONDS="${DATA_PREP_RELAUNCH_GRACE_SECONDS:-300}"
LAST_DATA_PREP_LAUNCH=0

# Targeted broad-language FLAN shelves. This is intentionally not full_literal:
# it restores HRM-Text-style instruction/language coverage without letting the
# 264GB FLAN bulk dominate the first continuation.
INCLUDE_CLUSTERS="${INCLUDE_CLUSTERS:-tasksource textbookreasoning acereason openthoughts2}"
FLAN_INCLUDE_REGEX="${FLAN_INCLUDE_REGEX:-dialog|wiki_dialog|qrecc|samsum|squad|natural_questions|trivia_qa|quac|coqa|bool_q|openbookqa|piqa|hellaswag|story_cloze|common_gen|gem_|web_nlg|e2e_nlg|wiki_lingua|aeslc|xsum|cnn_dailymail|multi_news|ag_news|imdb|yelp|wmt|translate|translation|xquad|tydi|mlqa|xnli}"
FLAN_MAX_FILES="${FLAN_MAX_FILES:-192}"
EPOCHS="${EPOCHS:-4}"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${PIPELINE_LOG}" >/dev/null
}

active_prefixlm_train() {
  ps -eo args | awk '/[p]ython .*scripts\/534_train_native_prefixlm_dataio/ {found=1} END {exit !found}'
}

active_stage93_sampler() {
  ps -eo args | awk '/[s]ample_tokenized.py .*stage93_hrm_text_reasoning_nonflan_dataio/ {found=1} END {exit !found}'
}

active_data_prep() {
  ps -eo comm,args | awk -v data_dir="${DATA_WORK_DIR}" '
    $1 == "bash" && $3 ~ /535_prepare_stage93_hrm_text_large_dataio.sh$/ && $4 == "all" { found = 1 }
    $1 == "tokenizer" && index($0, data_dir) { found = 1 }
    $1 ~ /python/ && index($0, "sample_tokenized.py") && index($0, data_dir) { found = 1 }
    END { exit !found }
  '
}

recent_data_prep_launch() {
  local now age
  [[ "${LAST_DATA_PREP_LAUNCH}" =~ ^[0-9]+$ ]] || return 1
  (( LAST_DATA_PREP_LAUNCH > 0 )) || return 1
  now="$(date +%s)"
  age="$((now - LAST_DATA_PREP_LAUNCH))"
  (( age < DATA_PREP_RELAUNCH_GRACE_SECONDS ))
}

sample_ready() {
  [[ -s "${SAMPLED_DATA}/metadata.json" ]] \
    && [[ -s "${SAMPLED_DATA}/tokens.npy" ]] \
    && [[ -s "${SAMPLED_DATA}/epoch_0/inst_start.npy" ]] \
    && [[ -s "${SAMPLED_DATA}/epoch_0/resp_start.npy" ]]
}

wait_until_quiet() {
  local start now elapsed
  start="$(date +%s)"
  while active_prefixlm_train || { [[ "${WAIT_FOR_OLD_REASONING_SAMPLERS}" == "1" ]] && active_stage93_sampler; }; do
    now="$(date +%s)"
    elapsed="$((now - start))"
    if (( elapsed >= MAX_WAIT_SECONDS )); then
      log "max wait reached while waiting for existing Stage93 jobs"
      return 3
    fi
    if active_prefixlm_train; then
      log "waiting for existing PrefixLM trainer to finish; elapsed=${elapsed}s"
    else
      log "waiting for old reasoning-only sampler because WAIT_FOR_OLD_REASONING_SAMPLERS=1; elapsed=${elapsed}s"
    fi
    sleep "${POLL_SECONDS}"
  done
}

launch_data_prep() {
  if sample_ready; then
    log "sample already ready: ${SAMPLED_DATA}"
    return 0
  fi

  if active_data_prep; then
    log "data prep already active for ${DATA_RUN_NAME}"
    return 0
  fi

  log "launching general-language curriculum data prep"
  LAST_DATA_PREP_LAUNCH="$(date +%s)"
  PROFILE=full_curriculum \
    RUN_NAME="${DATA_RUN_NAME}" \
    WORK_DIR="${DATA_WORK_DIR}" \
    INCLUDE_DATA_DIR=1 \
    INCLUDE_CLUSTERS="${INCLUDE_CLUSTERS}" \
    INCLUDE_FLAN=1 \
    FLAN_MAX_FILES="${FLAN_MAX_FILES}" \
    FLAN_INCLUDE_REGEX="${FLAN_INCLUDE_REGEX}" \
    EPOCHS="${EPOCHS}" \
    LOG="${PREP_LOG}" \
    bash "${ROOT}/scripts/535_prepare_stage93_hrm_text_large_dataio.sh" launch \
      >> "${PIPELINE_LOG}" 2>&1
}

wait_for_sample() {
  local start now elapsed size1 size2
  start="$(date +%s)"
  while true; do
    if sample_ready; then
      size1="$(stat -c '%s' "${SAMPLED_DATA}/tokens.npy")"
      sleep 10
      size2="$(stat -c '%s' "${SAMPLED_DATA}/tokens.npy")"
      if [[ "${size1}" == "${size2}" ]]; then
        log "sample ready and stable: ${SAMPLED_DATA}"
        return 0
      fi
      log "sample exists but tokens.npy is still changing"
    fi

    if ! active_data_prep; then
      if recent_data_prep_launch; then
        log "waiting for recently launched data prep to enter tokenize/sample phase"
        sleep "${POLL_SECONDS}"
        continue
      fi
      log "data prep is not active and sample is not ready; relaunching data prep"
      launch_data_prep
    fi

    now="$(date +%s)"
    elapsed="$((now - start))"
    if (( elapsed >= MAX_WAIT_SECONDS )); then
      log "max wait reached without sampled data"
      return 4
    fi
    log "waiting for sampled data; elapsed=${elapsed}s"
    sleep "${POLL_SECONDS}"
  done
}

launch_training() {
  if active_prefixlm_train; then
    log "PrefixLM trainer already active; not launching Stage93B"
    return 0
  fi

  if [[ ! -s "${RESUME}" ]]; then
    log "resume checkpoint missing: ${RESUME}"
    return 2
  fi

  log "launching Stage93B general-language continuation"
  RUN_NAME="${OUT_RUN_NAME}" \
    OUT_ROOT="${OUT_ROOT}" \
    LOG_FILE="${TRAIN_LOG}" \
    SAMPLED_DATA="${SAMPLED_DATA}" \
    RESUME="${RESUME}" \
    STEPS="${TARGET_STEPS}" \
    EVAL_EVERY="${EVAL_EVERY:-500}" \
    CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-4000}" \
    MODEL_CHECKPOINT_EVERY="${MODEL_CHECKPOINT_EVERY:-1000}" \
    bash "${ROOT}/scripts/536_launch_stage93_dgx_continue_prefixlm.sh" \
      >> "${PIPELINE_LOG}" 2>&1
}

launch_language_gate_watcher() {
  local gate_out gate_watch gate_log
  gate_out="${ROOT}/local_eval/${OUT_RUN_NAME}_LANGUAGE_GATES"
  gate_watch="/tmp/${OUT_RUN_NAME}_language_gates_on_target.log"
  gate_log="/tmp/${OUT_RUN_NAME}_language_gates.log"
  if pgrep -af "${OUT_RUN_NAME}_language_gates_on_target|549_run_stage93_micro_language_gates_on_target.sh.*${OUT_RUN_NAME}" >/dev/null 2>&1; then
    log "language gate watcher already active"
    return 0
  fi
  log "launching language gate watcher target_steps=${TARGET_STEPS}"
  nohup env \
    ROOT="${ROOT}" \
    TRAIN_LOG="${TRAIN_LOG}" \
    TARGET_STEPS="${TARGET_STEPS}" \
    CHECKPOINT="${OUT_ROOT}/last_model.pt" \
    TENSORBOARD_DIR="${OUT_ROOT}/tensorboard" \
    OUT_DIR="${gate_out}" \
    WATCH_LOG="${gate_watch}" \
    GATES_LOG="${gate_log}" \
    POLL_SECONDS="${POLL_SECONDS}" \
    bash "${ROOT}/scripts/549_run_stage93_micro_language_gates_on_target.sh" \
      >> "${PIPELINE_LOG}" 2>&1 &
  log "language gate watcher pid=${!}"
}

main() {
  : > "${PIPELINE_LOG}"
  log "Stage93B general-language pipeline started"
  log "WAIT_FOR_OLD_REASONING_SAMPLERS=${WAIT_FOR_OLD_REASONING_SAMPLERS}"
  wait_until_quiet
  launch_data_prep
  wait_for_sample
  launch_training
  launch_language_gate_watcher
  log "Stage93B pipeline handoff complete"
}

main "$@"
