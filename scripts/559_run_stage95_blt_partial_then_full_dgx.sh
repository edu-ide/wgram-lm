#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [[ -n "${PYTHON:-}" ]]; then
  PYTHON="${PYTHON}"
elif [[ -x "/mnt/data4tb/venv_sglang_pr23000/bin/python" ]]; then
  PYTHON="/mnt/data4tb/venv_sglang_pr23000/bin/python"
elif [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/.venv/bin/python"
else
  PYTHON="python3"
fi
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
REQUIRED_TRITON_PTXAS_PATH="${REQUIRED_TRITON_PTXAS_PATH:-}"
OFFICIAL_GDN2_PREFLIGHT_SMOKE="${OFFICIAL_GDN2_PREFLIGHT_SMOKE:-forward_auto}"
SELECTION_MODE="${SELECTION_MODE:-utility}"
PARTIAL_SELECTION_MODE="${PARTIAL_SELECTION_MODE:-first}"
FULL_SELECTION_MODE="${FULL_SELECTION_MODE:-${SELECTION_MODE}}"

WORK_BASE="${WORK_BASE:-${ROOT}/local_eval}"
PARTIAL_WORK_DIR="${PARTIAL_WORK_DIR:-${WORK_BASE}/stage95_blt_foundation_byte_curriculum_partial_20k}"
if [[ "${FULL_SELECTION_MODE}" == "utility" ]]; then
  FULL_WORK_DIR="${FULL_WORK_DIR:-${WORK_BASE}/stage95_blt_foundation_byte_curriculum_broad_240k_opus_gd}"
else
  FULL_WORK_DIR="${FULL_WORK_DIR:-${WORK_BASE}/stage95_blt_foundation_byte_curriculum_broad_240k}"
fi
PARTIAL_SAMPLE="${PARTIAL_SAMPLE:-${PARTIAL_WORK_DIR}/sampled}"
FULL_SAMPLE="${FULL_SAMPLE:-${FULL_WORK_DIR}/sampled}"
PARTIAL_OUT="${PARTIAL_OUT:-${WORK_BASE}/20260525_STAGE95G_DGX_1B_OFFICIAL_GDN2_ONEBODY_PARTIAL_CLEAN}"
if [[ "${FULL_SELECTION_MODE}" == "utility" ]]; then
  FULL_OUT="${FULL_OUT:-${WORK_BASE}/20260525_STAGE95I_DGX_1B_OPUS_GD_OFFICIAL_GDN2_ONEBODY_FULL}"
else
  FULL_OUT="${FULL_OUT:-${WORK_BASE}/20260525_STAGE95H_DGX_1B_OFFICIAL_GDN2_ONEBODY_FULL_CLEAN}"
fi
FULL_RESUME_CKPT="${FULL_RESUME_CKPT:-}"
LOG_DIR="${LOG_DIR:-/tmp}"
SUPERVISOR_LOG="${SUPERVISOR_LOG:-${LOG_DIR}/20260525_STAGE95G_OFFICIAL_GDN2_ONEBODY_PARTIAL_THEN_FULL_SUPERVISOR.log}"
PARTIAL_LOG="${PARTIAL_LOG:-${LOG_DIR}/20260525_STAGE95G_DGX_1B_OFFICIAL_GDN2_ONEBODY_PARTIAL_CLEAN.log}"
FULL_LOG="${FULL_LOG:-${LOG_DIR}/$(basename "${FULL_OUT}").log}"
FULL_BUILD_LOG="${FULL_BUILD_LOG:-${LOG_DIR}/$(basename "${FULL_WORK_DIR}")_build.log}"
LOCK_FILE="${LOCK_FILE:-${WORK_BASE}/stage95_official_gdn2_onebody_partial_then_full.lock}"
FULL_BUILD_LOCK="${FULL_BUILD_LOCK:-${FULL_WORK_DIR}/sample_build.lock}"

PARTIAL_MAX_ROWS="${PARTIAL_MAX_ROWS:-20000}"
PARTIAL_MAX_ROWS_PER_FILE="${PARTIAL_MAX_ROWS_PER_FILE:-200}"
PARTIAL_MAX_SCAN_ROWS_PER_FILE="${PARTIAL_MAX_SCAN_ROWS_PER_FILE:-3000}"
FULL_MAX_ROWS="${FULL_MAX_ROWS:-240000}"
FULL_MAX_ROWS_PER_FILE="${FULL_MAX_ROWS_PER_FILE:-200}"
FULL_MAX_SCAN_ROWS_PER_FILE="${FULL_MAX_SCAN_ROWS_PER_FILE:-5000}"
EPOCHS="${EPOCHS:-3}"

PARTIAL_STEPS="${PARTIAL_STEPS:-1200}"
FULL_STEPS="${FULL_STEPS:-10000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-200}"
OPTIMIZER_CHECKPOINT_EVERY="${OPTIMIZER_CHECKPOINT_EVERY:-0}"
SEQ_LEN="${SEQ_LEN:-384}"
EVAL_EVERY="${EVAL_EVERY:-100}"
EVAL_MAX_ROWS="${EVAL_MAX_ROWS:-256}"
BATCH_SIZE="${BATCH_SIZE:-}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-}"
PARTIAL_BATCH_SIZE="${PARTIAL_BATCH_SIZE:-${BATCH_SIZE:-8}}"
FULL_BATCH_SIZE="${FULL_BATCH_SIZE:-${BATCH_SIZE:-8}}"
PARTIAL_EVAL_BATCH_SIZE="${PARTIAL_EVAL_BATCH_SIZE:-${EVAL_BATCH_SIZE:-1}}"
FULL_EVAL_BATCH_SIZE="${FULL_EVAL_BATCH_SIZE:-${EVAL_BATCH_SIZE:-1}}"
LR="${LR:-2.2e-4}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-500}"
SEED="${SEED:-9595}"
RESUME_STRICT="${RESUME_STRICT:-1}"
RESUME_LOAD_OPTIMIZER="${RESUME_LOAD_OPTIMIZER:-auto}"
PATCH_SIZE="${PATCH_SIZE:-8}"
PATCH_BOUNDARY_MODE="${PATCH_BOUNDARY_MODE:-hnet_dechunk}"
DYNAMIC_MIN_PATCH_SIZE="${DYNAMIC_MIN_PATCH_SIZE:-3}"
DYNAMIC_SOFT_PATCH_SIZE="${DYNAMIC_SOFT_PATCH_SIZE:-0}"
HBF_BOUNDARY_THRESHOLD="${HBF_BOUNDARY_THRESHOLD:-0.35}"
BOUNDARY_PRIOR_WEIGHT="${BOUNDARY_PRIOR_WEIGHT:-0.02}"
BOUNDARY_TARGET_RATIO="${BOUNDARY_TARGET_RATIO:-0.25}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-${WORK_BASE}/20260524_STAGE94C_LOCAL_BYTEFREE82M_LANGSAMPLE_RETRY/last_model.pt}"
TEACHER_DISTILL_WEIGHT="${TEACHER_DISTILL_WEIGHT:-0.0}"
TEACHER_DISTILL_TEMPERATURE="${TEACHER_DISTILL_TEMPERATURE:-1.0}"
TEACHER_DISTILL_MAX_TARGETS="${TEACHER_DISTILL_MAX_TARGETS:-512}"
TEACHER_SEQ_LEN="${TEACHER_SEQ_LEN:-0}"
QWEN_BOUNDARY_PRIOR_WEIGHT="${QWEN_BOUNDARY_PRIOR_WEIGHT:-0.0}"
QWEN_BOUNDARY_TOKENIZER_MODEL_ID="${QWEN_BOUNDARY_TOKENIZER_MODEL_ID:-Qwen/Qwen3.5-0.8B-Base}"
DECODER_LATENT_MODE="${DECODER_LATENT_MODE:-one_body}"
PAST_SUCCESS_REPORT_JSON="${PAST_SUCCESS_REPORT_JSON:-${ROOT}/docs/wiki/decisions/past-success-doubt-loop-stage56-stage58.report.json}"
PAST_SUCCESS_RESTORATION_GATE_JSON="${PAST_SUCCESS_RESTORATION_GATE_JSON:-}"
ALLOW_MISSING_PAST_SUCCESS_PREFLIGHT="${ALLOW_MISSING_PAST_SUCCESS_PREFLIGHT:-0}"
ACKNOWLEDGE_PAST_SUCCESS_RESTORATION_GAP="${ACKNOWLEDGE_PAST_SUCCESS_RESTORATION_GAP:-0}"
SOURCE_BUCKET_QUOTAS="${SOURCE_BUCKET_QUOTAS:-general_instruction=30000 tasksource=50000 flan_instruction_qa=50000 flan_multilingual_translation=30000 flan_other=20000 reasoning=30000 math=20000 synthetic_math_like=10000}"
SOURCE_BUCKET_MAX_ROWS_PER_FILE="${SOURCE_BUCKET_MAX_ROWS_PER_FILE:-general_instruction=10000 tasksource=800 flan_instruction_qa=500 flan_multilingual_translation=500 flan_other=300 reasoning=5000 math=3000 synthetic_math_like=20}"
PARTIAL_SOURCE_BUCKET_QUOTAS="${PARTIAL_SOURCE_BUCKET_QUOTAS:-general_instruction=3000 tasksource=4000 flan_instruction_qa=4000 flan_multilingual_translation=2500 flan_other=1500 reasoning=2500 math=1500 synthetic_math_like=1000}"
FULL_SOURCE_BUCKET_QUOTAS="${FULL_SOURCE_BUCKET_QUOTAS:-${SOURCE_BUCKET_QUOTAS}}"
START_FULL_BUILD_EARLY="${START_FULL_BUILD_EARLY:-0}"
UTILITY_SCORE_JSONL="${UTILITY_SCORE_JSONL:-}"
UTILITY_TEMPERATURE="${UTILITY_TEMPERATURE:-0.0}"
OPUS_CHECKPOINT="${OPUS_CHECKPOINT:-}"
OPUS_PROXY_JSONL="${OPUS_PROXY_JSONL:-${ROOT}/data/eval/prefixlm_language_heldout.jsonl ${ROOT}/data/eval/official_gdsuite_choice_probe.jsonl}"
OPUS_SCORE_OUT="${OPUS_SCORE_OUT:-}"
OPUS_REPORT_OUT="${OPUS_REPORT_OUT:-}"
OPUS_CANDIDATE_MAX_ROWS="${OPUS_CANDIDATE_MAX_ROWS:-4096}"
OPUS_CANDIDATE_MAX_SCAN_ROWS_PER_FILE="${OPUS_CANDIDATE_MAX_SCAN_ROWS_PER_FILE:-5000}"
OPUS_PROXY_MAX_ROWS="${OPUS_PROXY_MAX_ROWS:-0}"
OPUS_PROXY_MAX_ROWS_PER_GROUP="${OPUS_PROXY_MAX_ROWS_PER_GROUP:-8}"
OPUS_PROXY_GROUPING="${OPUS_PROXY_GROUPING:-source_file_bucket}"
OPUS_PROXY_SCORE_MODE="${OPUS_PROXY_SCORE_MODE:-minimax_mean}"
OPUS_PROXY_MEAN_WEIGHT="${OPUS_PROXY_MEAN_WEIGHT:-0.25}"
OPUS_PROJECTION_DIM="${OPUS_PROJECTION_DIM:-2048}"
OPUS_PRECONDITIONER="${OPUS_PRECONDITIONER:-adamw_state}"
OPUS_PARAM_NAME_REGEX="${OPUS_PARAM_NAME_REGEX:-^(byte_embed|byte_pos_embed|patch_len_embed|bos_latent|patch_proj|semantic_boundary_scorer|semantic_chunk_proj|hierarchical_chunk_proj|hierarchical_chunk_gate|clean_decoder|hnet_byte_speaker)}"
OPUS_REDUNDANCY_WEIGHT="${OPUS_REDUNDANCY_WEIGHT:-1.0}"
OPUS_DEVICE="${OPUS_DEVICE:-cuda}"
ONLINE_OPUS_ENABLED="${ONLINE_OPUS_ENABLED:-1}"
ONLINE_OPUS_CANDIDATE_BATCHES="${ONLINE_OPUS_CANDIDATE_BATCHES:-2}"
ONLINE_OPUS_PROXY_BATCHES="${ONLINE_OPUS_PROXY_BATCHES:-1}"
ONLINE_OPUS_EVERY="${ONLINE_OPUS_EVERY:-1}"
ONLINE_OPUS_START_STEP="${ONLINE_OPUS_START_STEP:-1}"
ONLINE_OPUS_PROXY_SOURCE="${ONLINE_OPUS_PROXY_SOURCE:-eval}"
ONLINE_OPUS_PROJECTION_DIM="${ONLINE_OPUS_PROJECTION_DIM:-1024}"
ONLINE_OPUS_PRECONDITIONER="${ONLINE_OPUS_PRECONDITIONER:-adamw_state}"
ONLINE_OPUS_PARAM_NAME_REGEX="${ONLINE_OPUS_PARAM_NAME_REGEX:-${OPUS_PARAM_NAME_REGEX}}"
SAVE_OPTIMIZER_CHECKPOINT="${SAVE_OPTIMIZER_CHECKPOINT:-auto}"
TRAIN_THINK_STEPS="${TRAIN_THINK_STEPS:-2}"
GD_LITE_ENABLED="${GD_LITE_ENABLED:-1}"
GD_LITE_PROBE_JSONL="${GD_LITE_PROBE_JSONL:-${ROOT}/data/eval/official_gdsuite_choice_probe.jsonl}"
GD_LITE_OUT="${GD_LITE_OUT:-${FULL_OUT}/generalization_dynamics_official_choice_report.json}"
GD_LITE_DEVICE="${GD_LITE_DEVICE:-cuda}"
GD_LITE_MAX_ROWS="${GD_LITE_MAX_ROWS:-0}"
GD_LITE_THINK_STEPS="${GD_LITE_THINK_STEPS:-${TRAIN_THINK_STEPS}}"
GD_LITE_REQUIRE_ACCEPT="${GD_LITE_REQUIRE_ACCEPT:-0}"

usage() {
  cat <<'USAGE'
Stage95 BLT partial-then-full automation

Plain-language contract:
  Do not wait for the full textbook to finish binding before the model starts
  reading. Build a small tokenizer-free byte partial, train the same 1B BLT
  body on it, keep optimizer-bearing checkpoints when OPUS needs AdamW state,
  then automatically select and continue on the broad/full byte sample.

Actions:
  plan    Show the partial -> full continuation contract.
  status  Print sample/checkpoint/process state.
  preflight
          Explain whether ptxas/checkpoint/runtime evidence is clean enough for
          an official GDN2 continuation.
  run-full
          Train directly on an already materialized FULL_SAMPLE without waiting
          for partial/full sample automation.
  launch-full
          Start run-full with nohup so SSH can disconnect.
  run     Run the supervisor in the foreground.
  launch  Start the supervisor with nohup so SSH can disconnect.
USAGE
}

sample_ready() {
  local sample_dir="$1"
  [[ -f "${sample_dir}/metadata.json" && -f "${sample_dir}/tokens.npy" ]]
}

sample_ready_for_selection() {
  local sample_dir="$1"
  local selection_mode="$2"
  sample_ready "${sample_dir}" || return 1
  if [[ "${selection_mode}" != "utility" ]]; then
    return 0
  fi
  "${PYTHON}" - "${sample_dir}" <<'PY'
import json
import sys
from pathlib import Path

metadata = Path(sys.argv[1]) / "metadata.json"
payload = json.loads(metadata.read_text(encoding="utf-8"))
contract = payload.get("data_selection_contract") or {}
selection_mode = str(contract.get("selection_mode", ""))
scores_loaded = int(contract.get("utility_scores_loaded", 0) or 0)
raise SystemExit(0 if selection_mode == "utility" and scores_loaded > 0 else 1)
PY
}

process_matching() {
  local pattern="$1"
  pgrep -af "${pattern}" 2>/dev/null || true
}

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
  export PATH="$(dirname "${TRITON_PTXAS_PATH}"):${PATH}"
}

teacher_distill_enabled() {
  [[ -n "${TEACHER_CHECKPOINT}" ]] || return 1
  [[ "${TEACHER_DISTILL_WEIGHT}" != "0" && "${TEACHER_DISTILL_WEIGHT}" != "0.0" && "${TEACHER_DISTILL_WEIGHT}" != "0.00" ]]
}

ensure_teacher_distill_ready() {
  if ! teacher_distill_enabled; then
    echo "teacher_distill=disabled"
    return 0
  fi
  if [[ ! -f "${TEACHER_CHECKPOINT}" ]]; then
    echo "missing_teacher_checkpoint=${TEACHER_CHECKPOINT}" >&2
    echo "Set TEACHER_CHECKPOINT to a raw byte-free checkpoint or TEACHER_DISTILL_WEIGHT=0 to disable distillation." >&2
    exit 4
  fi
  echo "teacher_distill=enabled checkpoint=${TEACHER_CHECKPOINT} weight=${TEACHER_DISTILL_WEIGHT} max_targets=${TEACHER_DISTILL_MAX_TARGETS}"
}

resume_matching_processes() {
  local pattern="$1"
  local line pid ppid stat parent_stat
  while IFS= read -r line; do
    [[ -z "${line}" ]] && continue
    pid="${line%% *}"
    ppid="$(ps -o ppid= -p "${pid}" 2>/dev/null | tr -d ' ')"
    stat="$(ps -o stat= -p "${pid}" 2>/dev/null | tr -d ' ')"
    parent_stat="$(ps -o stat= -p "${ppid}" 2>/dev/null | tr -d ' ' || true)"
    if [[ -n "${ppid}" && "${parent_stat}" == *T* ]]; then
      kill -CONT "${ppid}" 2>/dev/null || true
      echo "resumed_stopped_parent pid=${ppid} child=${pid} pattern=${pattern}"
    fi
    if [[ "${stat}" == *T* ]]; then
      kill -CONT "${pid}" 2>/dev/null || true
      echo "resumed_stopped_process pid=${pid} pattern=${pattern}"
    fi
  done < <(process_matching "${pattern}")
}

truthy() {
  [[ "$1" == "1" || "$1" == "true" || "$1" == "yes" ]]
}

optimizer_checkpoint_enabled() {
  local run_name="${TRAINING_NAME_FOR_ARGS:-}"
  if truthy "${SAVE_OPTIMIZER_CHECKPOINT}"; then
    return 0
  fi
  if [[ "${SAVE_OPTIMIZER_CHECKPOINT}" == "auto" && "${FULL_SELECTION_MODE}" == "utility" && "${run_name}" == "partial_training" ]]; then
    return 0
  fi
  return 1
}

resume_load_optimizer_for_path() {
  local path="$1"
  local out_dir="$2"
  if truthy "${RESUME_LOAD_OPTIMIZER}"; then
    return 0
  fi
  if [[ "${RESUME_LOAD_OPTIMIZER}" == "auto" && "${path}" == "${out_dir}/last.pt" ]]; then
    return 0
  fi
  return 1
}

resolve_full_opus_checkpoint() {
  if [[ -n "${OPUS_CHECKPOINT}" ]]; then
    echo "${OPUS_CHECKPOINT}"
    return 0
  fi
  if [[ -f "${PARTIAL_OUT}/last.pt" ]]; then
    echo "${PARTIAL_OUT}/last.pt"
    return 0
  fi
  return 1
}

ensure_full_sample_building() {
  if sample_ready_for_selection "${FULL_SAMPLE}" "${FULL_SELECTION_MODE}"; then
    echo "full_sample=ready path=${FULL_SAMPLE}"
    return 0
  fi
  if sample_ready "${FULL_SAMPLE}" && [[ "${FULL_SELECTION_MODE}" == "utility" ]]; then
    echo "full_sample=present_but_not_opus_selected path=${FULL_SAMPLE}; rebuilding"
  fi
  local effective_opus_checkpoint="${OPUS_CHECKPOINT}"
  if [[ "${FULL_SELECTION_MODE}" == "utility" ]]; then
    if ! effective_opus_checkpoint="$(resolve_full_opus_checkpoint)"; then
      echo "full_sample_builder=waiting_for_partial_opus_checkpoint path=${PARTIAL_OUT}/last.pt"
      return 0
    fi
  fi
  mkdir -p "$(dirname "${FULL_SAMPLE}")"
  local pattern="555_prepare_byte_prefixlm_sample.py .*${FULL_SAMPLE}"
  local matches
  matches="$(process_matching "${pattern}")"
  if [[ -n "${matches}" ]]; then
    resume_matching_processes "${pattern}"
    echo "full_sample_builder=already_running_or_resumed path=${FULL_SAMPLE}"
    return 0
  fi
  echo "full_sample_builder=starting path=${FULL_SAMPLE}"
  mkdir -p "$(dirname "${FULL_BUILD_LOCK}")"
  (
    if ! flock -n 8; then
      echo "full_sample_builder=already_locked lock=${FULL_BUILD_LOCK}"
      exit 0
    fi
    cd "${ROOT}"
    env \
      WORK_DIR="${FULL_WORK_DIR}" \
      SAMPLED_OUT="${FULL_SAMPLE}" \
      MAX_ROWS="${FULL_MAX_ROWS}" \
      MAX_ROWS_PER_FILE="${FULL_MAX_ROWS_PER_FILE}" \
      MAX_SCAN_ROWS_PER_FILE="${FULL_MAX_SCAN_ROWS_PER_FILE}" \
      EPOCHS="${EPOCHS}" \
      SOURCE_BUCKET_QUOTAS="${FULL_SOURCE_BUCKET_QUOTAS}" \
      SOURCE_BUCKET_MAX_ROWS_PER_FILE="${SOURCE_BUCKET_MAX_ROWS_PER_FILE}" \
      SELECTION_MODE="${FULL_SELECTION_MODE}" \
      UTILITY_SCORE_JSONL="${UTILITY_SCORE_JSONL}" \
      UTILITY_TEMPERATURE="${UTILITY_TEMPERATURE}" \
      OPUS_CHECKPOINT="${effective_opus_checkpoint}" \
      OPUS_PROXY_JSONL="${OPUS_PROXY_JSONL}" \
      OPUS_SCORE_OUT="${OPUS_SCORE_OUT:-${FULL_WORK_DIR}/opus_projected_utility_scores.jsonl}" \
      OPUS_REPORT_OUT="${OPUS_REPORT_OUT:-${FULL_WORK_DIR}/opus_projected_utility_report.json}" \
      OPUS_CANDIDATE_MAX_ROWS="${OPUS_CANDIDATE_MAX_ROWS}" \
      OPUS_CANDIDATE_MAX_SCAN_ROWS_PER_FILE="${OPUS_CANDIDATE_MAX_SCAN_ROWS_PER_FILE}" \
      OPUS_PROXY_MAX_ROWS="${OPUS_PROXY_MAX_ROWS}" \
      OPUS_PROXY_MAX_ROWS_PER_GROUP="${OPUS_PROXY_MAX_ROWS_PER_GROUP}" \
      OPUS_PROXY_GROUPING="${OPUS_PROXY_GROUPING}" \
      OPUS_PROXY_SCORE_MODE="${OPUS_PROXY_SCORE_MODE}" \
      OPUS_PROXY_MEAN_WEIGHT="${OPUS_PROXY_MEAN_WEIGHT}" \
      OPUS_PROJECTION_DIM="${OPUS_PROJECTION_DIM}" \
      OPUS_PRECONDITIONER="${OPUS_PRECONDITIONER}" \
      OPUS_PARAM_NAME_REGEX="${OPUS_PARAM_NAME_REGEX}" \
      OPUS_REDUNDANCY_WEIGHT="${OPUS_REDUNDANCY_WEIGHT}" \
      OPUS_DEVICE="${OPUS_DEVICE}" \
      PYTHON="${PYTHON}" \
      bash scripts/558_prepare_stage95_blt_foundation_byte_sample_dgx.sh build
  ) 8>"${FULL_BUILD_LOCK}" >>"${FULL_BUILD_LOG}" 2>&1 &
  echo "full_sample_builder_pid=$!"
}

build_partial_sample_if_needed() {
  if sample_ready "${PARTIAL_SAMPLE}"; then
    echo "partial_sample=ready path=${PARTIAL_SAMPLE}"
    return 0
  fi
  echo "partial_sample=building path=${PARTIAL_SAMPLE}"
  (
    cd "${ROOT}"
    env \
      WORK_DIR="${PARTIAL_WORK_DIR}" \
      SAMPLED_OUT="${PARTIAL_SAMPLE}" \
      MAX_ROWS="${PARTIAL_MAX_ROWS}" \
      MAX_ROWS_PER_FILE="${PARTIAL_MAX_ROWS_PER_FILE}" \
      MAX_SCAN_ROWS_PER_FILE="${PARTIAL_MAX_SCAN_ROWS_PER_FILE}" \
      EPOCHS="${EPOCHS}" \
      SOURCE_BUCKET_QUOTAS="${PARTIAL_SOURCE_BUCKET_QUOTAS}" \
      SOURCE_BUCKET_MAX_ROWS_PER_FILE="${SOURCE_BUCKET_MAX_ROWS_PER_FILE}" \
      SELECTION_MODE="${PARTIAL_SELECTION_MODE}" \
      UTILITY_SCORE_JSONL="${UTILITY_SCORE_JSONL}" \
      UTILITY_TEMPERATURE="${UTILITY_TEMPERATURE}" \
      OPUS_CHECKPOINT="${OPUS_CHECKPOINT}" \
      OPUS_PROXY_JSONL="${OPUS_PROXY_JSONL}" \
      OPUS_SCORE_OUT="${OPUS_SCORE_OUT:-${PARTIAL_WORK_DIR}/opus_projected_utility_scores.jsonl}" \
      OPUS_REPORT_OUT="${OPUS_REPORT_OUT:-${PARTIAL_WORK_DIR}/opus_projected_utility_report.json}" \
      OPUS_CANDIDATE_MAX_ROWS="${OPUS_CANDIDATE_MAX_ROWS}" \
      OPUS_CANDIDATE_MAX_SCAN_ROWS_PER_FILE="${OPUS_CANDIDATE_MAX_SCAN_ROWS_PER_FILE}" \
      OPUS_PROXY_MAX_ROWS="${OPUS_PROXY_MAX_ROWS}" \
      OPUS_PROXY_MAX_ROWS_PER_GROUP="${OPUS_PROXY_MAX_ROWS_PER_GROUP}" \
      OPUS_PROXY_GROUPING="${OPUS_PROXY_GROUPING}" \
      OPUS_PROXY_SCORE_MODE="${OPUS_PROXY_SCORE_MODE}" \
      OPUS_PROXY_MEAN_WEIGHT="${OPUS_PROXY_MEAN_WEIGHT}" \
      OPUS_PROJECTION_DIM="${OPUS_PROJECTION_DIM}" \
      OPUS_PRECONDITIONER="${OPUS_PRECONDITIONER}" \
      OPUS_PARAM_NAME_REGEX="${OPUS_PARAM_NAME_REGEX}" \
      OPUS_REDUNDANCY_WEIGHT="${OPUS_REDUNDANCY_WEIGHT}" \
      OPUS_DEVICE="${OPUS_DEVICE}" \
      PYTHON="${PYTHON}" \
      bash scripts/558_prepare_stage95_blt_foundation_byte_sample_dgx.sh build
  )
}

train_args_common() {
  local sampled_data="$1"
  local out_dir="$2"
  local steps="$3"
  local batch_size="$4"
  local eval_batch_size="$5"
  shift 5
  local teacher_args=()
  local qwen_boundary_args=()
  if teacher_distill_enabled; then
    teacher_args=(
      --teacher-checkpoint "${TEACHER_CHECKPOINT}"
      --teacher-distill-weight "${TEACHER_DISTILL_WEIGHT}"
      --teacher-distill-temperature "${TEACHER_DISTILL_TEMPERATURE}"
      --teacher-distill-max-targets "${TEACHER_DISTILL_MAX_TARGETS}"
      --teacher-seq-len "${TEACHER_SEQ_LEN}"
    )
  fi
  if [[ "${QWEN_BOUNDARY_PRIOR_WEIGHT}" != "0" && "${QWEN_BOUNDARY_PRIOR_WEIGHT}" != "0.0" && "${QWEN_BOUNDARY_PRIOR_WEIGHT}" != "0.00" ]]; then
    qwen_boundary_args=(
      --qwen-boundary-prior-weight "${QWEN_BOUNDARY_PRIOR_WEIGHT}"
      --qwen-boundary-tokenizer-model-id "${QWEN_BOUNDARY_TOKENIZER_MODEL_ID}"
    )
  fi
  local preflight_args=()
  if [[ -n "${PAST_SUCCESS_REPORT_JSON}" ]]; then
    preflight_args+=(--past-success-report-json "${PAST_SUCCESS_REPORT_JSON}")
  fi
  if [[ -n "${PAST_SUCCESS_RESTORATION_GATE_JSON}" ]]; then
    preflight_args+=(--past-success-restoration-gate-json "${PAST_SUCCESS_RESTORATION_GATE_JSON}")
  fi
  if [[ "${ALLOW_MISSING_PAST_SUCCESS_PREFLIGHT}" == "1" || "${ALLOW_MISSING_PAST_SUCCESS_PREFLIGHT}" == "true" || "${ALLOW_MISSING_PAST_SUCCESS_PREFLIGHT}" == "yes" ]]; then
    preflight_args+=(--allow-missing-past-success-preflight)
  fi
  if [[ "${ACKNOWLEDGE_PAST_SUCCESS_RESTORATION_GAP}" == "1" || "${ACKNOWLEDGE_PAST_SUCCESS_RESTORATION_GAP}" == "true" || "${ACKNOWLEDGE_PAST_SUCCESS_RESTORATION_GAP}" == "yes" ]]; then
    preflight_args+=(--acknowledge-past-success-restoration-gap)
  fi
  local online_opus_args=()
  if truthy "${ONLINE_OPUS_ENABLED}"; then
    online_opus_args=(
      --online-opus-enabled
      --online-opus-candidate-batches "${ONLINE_OPUS_CANDIDATE_BATCHES}"
      --online-opus-proxy-batches "${ONLINE_OPUS_PROXY_BATCHES}"
      --online-opus-every "${ONLINE_OPUS_EVERY}"
      --online-opus-start-step "${ONLINE_OPUS_START_STEP}"
      --online-opus-proxy-source "${ONLINE_OPUS_PROXY_SOURCE}"
      --online-opus-projection-dim "${ONLINE_OPUS_PROJECTION_DIM}"
      --online-opus-preconditioner "${ONLINE_OPUS_PRECONDITIONER}"
      --online-opus-param-name-regex "${ONLINE_OPUS_PARAM_NAME_REGEX}"
    )
  fi
  local optimizer_checkpoint_args=()
  if optimizer_checkpoint_enabled; then
    optimizer_checkpoint_args=(--save-optimizer-checkpoint --optimizer-checkpoint-every "${OPTIMIZER_CHECKPOINT_EVERY}")
  else
    optimizer_checkpoint_args=(--no-save-optimizer-checkpoint)
  fi
  echo \
    --sampled-data "${sampled_data}" \
    --out-dir "${out_dir}" \
    --steps "${steps}" \
    --checkpoint-every "${CHECKPOINT_EVERY}" \
    --batch-size "${batch_size}" \
    --seq-len "${SEQ_LEN}" \
    --eval-sampled-data "${sampled_data}" \
    --eval-every "${EVAL_EVERY}" \
    --eval-max-rows "${EVAL_MAX_ROWS}" \
    --eval-batch-size "${eval_batch_size}" \
    --eval-max-batches 0 \
    --lr "${LR}" \
    --lr-warmup-steps "${LR_WARMUP_STEPS}" \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --weight-decay 0.1 \
    --grad-clip 1.0 \
    --amp-dtype bf16 \
    --matmul-precision high \
    --log-every 25 \
    --tensorboard-dir "${out_dir}/tensorboard" \
    --patch-size "${PATCH_SIZE}" \
    --patch-boundary-mode "${PATCH_BOUNDARY_MODE}" \
    --dynamic-min-patch-size "${DYNAMIC_MIN_PATCH_SIZE}" \
    --dynamic-soft-patch-size "${DYNAMIC_SOFT_PATCH_SIZE}" \
    --hbf-boundary-threshold "${HBF_BOUNDARY_THRESHOLD}" \
    --boundary-prior-weight "${BOUNDARY_PRIOR_WEIGHT}" \
    --boundary-target-ratio "${BOUNDARY_TARGET_RATIO}" \
    --decoder-latent-mode "${DECODER_LATENT_MODE}" \
    --diffusion-weight 0.0 \
    --diffusion-mask-prob 0.0 \
    --d-model 1792 \
    --n-heads 16 \
    --n-kv-heads 4 \
    --d-ff 4864 \
    --dropout 0.0 \
    --backbone trm_qwen35_3to1 \
    --think-structure trm_dual_z \
    --train-think-steps "${TRAIN_THINK_STEPS}" \
    --hybrid-layers 4 \
    --attn-every 4 \
    --delta-backend official_gated_delta2 \
    --strict-backends \
    --attention-backend sdpa \
    --seed "${SEED}" \
    "${optimizer_checkpoint_args[@]}" \
    "${teacher_args[@]}" \
    "${qwen_boundary_args[@]}" \
    "${preflight_args[@]}" \
    "${online_opus_args[@]}" \
    "$@"
}

run_training_if_needed() {
  local name="$1"
  local sampled_data="$2"
  local out_dir="$3"
  local steps="$4"
  local log_path="$5"
  local batch_size="$6"
  local eval_batch_size="$7"
  shift 7
  local model_ckpt="${out_dir}/last_model.pt"
  local optimizer_ckpt="${out_dir}/last.pt"
  local ckpt="${model_ckpt}"
  if [[ -f "${optimizer_ckpt}" ]]; then
    if [[ ! -f "${model_ckpt}" || ! "${model_ckpt}" -nt "${optimizer_ckpt}" ]]; then
      ckpt="${optimizer_ckpt}"
    fi
  fi
  local report="${out_dir}/report.json"
  local pattern="557_train_blt_d_prefixlm_dataio.py .*${out_dir}"
  local matches
  while true; do
    if [[ -f "${report}" ]]; then
      echo "${name}=complete report=${report}"
      return 0
    fi
    matches="$(process_matching "${pattern}")"
    if [[ -z "${matches}" ]]; then
      break
    fi
    resume_matching_processes "${pattern}"
    echo "${name}=already_running_waiting_for_report out=${out_dir} report=${report}"
    sleep 60
  done
  require_triton_ptxas
  ensure_teacher_distill_ready
  local extra_args=("$@")
  local has_explicit_resume=0
  local has_resume_strict_arg=0
  local has_resume_load_optimizer_arg=0
  local resume_path=""
  local arg previous_arg
  previous_arg=""
  for arg in "$@"; do
    if [[ "${previous_arg}" == "--resume" ]]; then
      resume_path="${arg}"
      previous_arg=""
      continue
    fi
    if [[ "${arg}" == "--resume" ]]; then
      has_explicit_resume=1
      previous_arg="--resume"
    fi
    if [[ "${arg}" == "--resume-strict" || "${arg}" == "--no-resume-strict" ]]; then
      has_resume_strict_arg=1
    fi
    if [[ "${arg}" == "--resume-load-optimizer" ]]; then
      has_resume_load_optimizer_arg=1
    fi
  done
  if [[ -f "${ckpt}" && "${has_explicit_resume}" -eq 0 ]]; then
    extra_args+=(--resume "${ckpt}")
    has_explicit_resume=1
    resume_path="${ckpt}"
    echo "${name}=resuming_checkpoint path=${ckpt}"
  fi
  if [[ "${has_explicit_resume}" -eq 1 && "${has_resume_strict_arg}" -eq 0 ]]; then
    if [[ "${RESUME_STRICT}" == "1" || "${RESUME_STRICT}" == "true" || "${RESUME_STRICT}" == "yes" ]]; then
      extra_args+=(--resume-strict)
    else
      extra_args+=(--no-resume-strict)
    fi
  fi
  if [[ "${has_explicit_resume}" -eq 1 && "${has_resume_load_optimizer_arg}" -eq 0 && -n "${resume_path}" ]]; then
    if resume_load_optimizer_for_path "${resume_path}" "${out_dir}"; then
      extra_args+=(--resume-load-optimizer)
      echo "${name}=resuming_optimizer_state path=${resume_path}"
    fi
  fi
  mkdir -p "${out_dir}" "$(dirname "${log_path}")"
  echo "${name}=starting out=${out_dir} log=${log_path}"
  (
    cd "${ROOT}"
    TRAINING_NAME_FOR_ARGS="${name}"
    "${PYTHON}" scripts/557_train_blt_d_prefixlm_dataio.py \
      $(train_args_common "${sampled_data}" "${out_dir}" "${steps}" "${batch_size}" "${eval_batch_size}" "${extra_args[@]}")
  ) >>"${log_path}" 2>&1
}

wait_for_full_sample() {
  while ! sample_ready "${FULL_SAMPLE}"; do
    ensure_full_sample_building
    echo "waiting_for_full_sample path=${FULL_SAMPLE}"
    sleep 60
  done
}

plan() {
  usage
  cat <<PLAN

Configuration:
  ROOT=${ROOT}
  PYTHON=${PYTHON}
  PARTIAL_SAMPLE=${PARTIAL_SAMPLE}
  FULL_SAMPLE=${FULL_SAMPLE}
  PARTIAL_OUT=${PARTIAL_OUT}
  FULL_OUT=${FULL_OUT}
  FULL_RESUME_CKPT=${FULL_RESUME_CKPT:-<auto>}
  PARTIAL_STEPS=${PARTIAL_STEPS}
  FULL_STEPS=${FULL_STEPS}
  CHECKPOINT_EVERY=${CHECKPOINT_EVERY}
  OPTIMIZER_CHECKPOINT_EVERY=${OPTIMIZER_CHECKPOINT_EVERY}
  SEQ_LEN=${SEQ_LEN}
  TRAIN_THINK_STEPS=${TRAIN_THINK_STEPS}
  PARTIAL_BATCH_SIZE=${PARTIAL_BATCH_SIZE}
  FULL_BATCH_SIZE=${FULL_BATCH_SIZE}
  PARTIAL_EVAL_BATCH_SIZE=${PARTIAL_EVAL_BATCH_SIZE}
  FULL_EVAL_BATCH_SIZE=${FULL_EVAL_BATCH_SIZE}
  RESUME_STRICT=${RESUME_STRICT}
  RESUME_LOAD_OPTIMIZER=${RESUME_LOAD_OPTIMIZER}
  PATCH_SIZE=${PATCH_SIZE}
  PATCH_BOUNDARY_MODE=${PATCH_BOUNDARY_MODE}
  DYNAMIC_MIN_PATCH_SIZE=${DYNAMIC_MIN_PATCH_SIZE}
  BOUNDARY_PRIOR_WEIGHT=${BOUNDARY_PRIOR_WEIGHT}
  BOUNDARY_TARGET_RATIO=${BOUNDARY_TARGET_RATIO}
  TEACHER_CHECKPOINT=${TEACHER_CHECKPOINT}
  TEACHER_DISTILL_WEIGHT=${TEACHER_DISTILL_WEIGHT}
  TEACHER_DISTILL_TEMPERATURE=${TEACHER_DISTILL_TEMPERATURE}
  TEACHER_DISTILL_MAX_TARGETS=${TEACHER_DISTILL_MAX_TARGETS}
  TEACHER_SEQ_LEN=${TEACHER_SEQ_LEN}
  QWEN_BOUNDARY_PRIOR_WEIGHT=${QWEN_BOUNDARY_PRIOR_WEIGHT}
  QWEN_BOUNDARY_TOKENIZER_MODEL_ID=${QWEN_BOUNDARY_TOKENIZER_MODEL_ID}
  DECODER_LATENT_MODE=${DECODER_LATENT_MODE}
  PAST_SUCCESS_REPORT_JSON=${PAST_SUCCESS_REPORT_JSON}
  PAST_SUCCESS_RESTORATION_GATE_JSON=${PAST_SUCCESS_RESTORATION_GATE_JSON:-<none>}
  ALLOW_MISSING_PAST_SUCCESS_PREFLIGHT=${ALLOW_MISSING_PAST_SUCCESS_PREFLIGHT}
  ACKNOWLEDGE_PAST_SUCCESS_RESTORATION_GAP=${ACKNOWLEDGE_PAST_SUCCESS_RESTORATION_GAP}
  PARTIAL_SOURCE_BUCKET_QUOTAS=${PARTIAL_SOURCE_BUCKET_QUOTAS}
  FULL_SOURCE_BUCKET_QUOTAS=${FULL_SOURCE_BUCKET_QUOTAS}
  SOURCE_BUCKET_MAX_ROWS_PER_FILE=${SOURCE_BUCKET_MAX_ROWS_PER_FILE}
  START_FULL_BUILD_EARLY=${START_FULL_BUILD_EARLY}
  FULL_BUILD_LOCK=${FULL_BUILD_LOCK}
  SELECTION_MODE=${SELECTION_MODE}
  PARTIAL_SELECTION_MODE=${PARTIAL_SELECTION_MODE}
  FULL_SELECTION_MODE=${FULL_SELECTION_MODE}
  UTILITY_SCORE_JSONL=${UTILITY_SCORE_JSONL}
  UTILITY_TEMPERATURE=${UTILITY_TEMPERATURE}
  OPUS_CHECKPOINT=${OPUS_CHECKPOINT:-<none>}
  OPUS_PROXY_JSONL=${OPUS_PROXY_JSONL}
  OPUS_CANDIDATE_MAX_ROWS=${OPUS_CANDIDATE_MAX_ROWS}
  OPUS_PROXY_MAX_ROWS=${OPUS_PROXY_MAX_ROWS}
  OPUS_PROXY_MAX_ROWS_PER_GROUP=${OPUS_PROXY_MAX_ROWS_PER_GROUP}
  OPUS_PROJECTION_DIM=${OPUS_PROJECTION_DIM}
  OPUS_PRECONDITIONER=${OPUS_PRECONDITIONER}
  OPUS_PARAM_NAME_REGEX=${OPUS_PARAM_NAME_REGEX}
  OPUS_DEVICE=${OPUS_DEVICE}
  ONLINE_OPUS_ENABLED=${ONLINE_OPUS_ENABLED}
  ONLINE_OPUS_CANDIDATE_BATCHES=${ONLINE_OPUS_CANDIDATE_BATCHES}
  ONLINE_OPUS_PROXY_BATCHES=${ONLINE_OPUS_PROXY_BATCHES}
  ONLINE_OPUS_EVERY=${ONLINE_OPUS_EVERY}
  ONLINE_OPUS_PROXY_SOURCE=${ONLINE_OPUS_PROXY_SOURCE}
  SAVE_OPTIMIZER_CHECKPOINT=${SAVE_OPTIMIZER_CHECKPOINT}
  GD_LITE_ENABLED=${GD_LITE_ENABLED}
  GD_LITE_PROBE_JSONL=${GD_LITE_PROBE_JSONL}
  GD_LITE_OUT=${GD_LITE_OUT}
  GD_LITE_THINK_STEPS=${GD_LITE_THINK_STEPS}
  GD_LITE_REQUIRE_ACCEPT=${GD_LITE_REQUIRE_ACCEPT}
  REQUIRED_TRITON_PTXAS_PATH=${REQUIRED_TRITON_PTXAS_PATH}
  TRITON_PTXAS_PATH=${TRITON_PTXAS_PATH:-<unset>}
  OFFICIAL_GDN2_PREFLIGHT_SMOKE=${OFFICIAL_GDN2_PREFLIGHT_SMOKE}

Automatic flow:
  1. Build PARTIAL_SAMPLE first. FULL_SAMPLE is built later by default to avoid
     parquet I/O contention; set START_FULL_BUILD_EARLY=1 to overlap it.
  2. Build PARTIAL_SAMPLE quickly from the same broad curriculum. By default
     PARTIAL_SELECTION_MODE=first, so OPUS can use this as the first
     optimizer-state-bearing anchor rather than needing a prior checkpoint.
  3. Train 1B BLT on PARTIAL_SAMPLE with the same batch-8 default used by
     the full continuation path.
  4. Save PARTIAL_OUT/last_model.pt and copy_last_model.pt.
  5. Wait until FULL_SAMPLE/metadata.json and tokens.npy exist.
  6. Continue training on FULL_SAMPLE with --resume FULL_RESUME_CKPT when set,
     otherwise the best available run-local checkpoint, and a larger full-run
     batch. If FULL_OUT/last.pt is at least as fresh as last_model.pt, it is
     preferred so optimizer state is not silently lost; if last_model.pt is
     newer, recovery keeps the newer weights rather than rewinding progress.
  7. Save FULL_OUT/last_model.pt and copy_last_model.pt.

Direct full action:
  If FULL_SAMPLE is already materialized, use run-full/launch-full to train on
  it immediately without building or waiting on PARTIAL_SAMPLE. This is the
  right path for a static full-sample baseline. It is not a proper OPUS window
  unless FULL_SAMPLE itself was materialized from OPUS utility scores.

Checkpoint policy:
  Static runs use --no-save-optimizer-checkpoint by default. Proper OPUS runs
  with FULL_SELECTION_MODE=utility use --save-optimizer-checkpoint for the
  partial phase under SAVE_OPTIMIZER_CHECKPOINT=auto so the scorer can read
  AdamW state from PARTIAL_OUT/last.pt. OPTIMIZER_CHECKPOINT_EVERY=0 writes
  that optimizer-bearing checkpoint only at partial finalization, avoiding
  repeated multi-GB mid-run I/O stalls. Full training returns to model-only
  checkpoints unless SAVE_OPTIMIZER_CHECKPOINT=true is set explicitly. When a
  run-local last.pt exists, RESUME_LOAD_OPTIMIZER=auto adds
  --resume-load-optimizer so crash recovery restores both weights and AdamW
  momentum.
  Training commands include --strict-backends and --resume-strict by default.
  Resume defaults to --resume-strict. official GatedDeltaNet-2 is fail-fast:
  if the official module, ptxas, or kernel path is not available, the run stops
  instead of writing an official-looking fallback checkpoint.
  The default OUT dirs are clean 20260525 Stage95G/I paths. They intentionally
  do not point at the legacy Stage95B/C fallback/add-mode checkpoints.

This is semantic byte/BLT partial -> full continuation, not Stage93 BPE and
not fixed BLT-2. The default boundary path is hnet_dechunk so the model learns
where meaningful byte chunks begin, compresses them through the recurrent core,
then de-chunks them back to byte logits.

Teacher distillation is disabled by default on the official OPUS+GD path. The
old Stage94 raw-byte teacher may contain legacy fallback mixer keys; using it
would make an official GDN2 run inherit a different engine's handwriting.
Enable TEACHER_DISTILL_WEIGHT only with a clean teacher checkpoint.

The promoted path defaults to DECODER_LATENT_MODE=one_body. Long one-body runs
must carry the past-success doubt report, and must not bypass a rejected
restoration gate unless ACKNOWLEDGE_PAST_SUCCESS_RESTORATION_GAP=1 marks the
launch as an explicit diagnostic continuation.

Qwen tokenizer boundary guidance is available as a small ablation via
QWEN_BOUNDARY_PRIOR_WEIGHT, but defaults to 0.0. It should be used as a weak
reader's underline, not as the main teacher.

OPUS projected-utility data-window selection is wired but opt-in. The default
SELECTION_MODE=utility makes OPUS+Generalization Dynamics the promoted path.
PARTIAL_SELECTION_MODE=first gives the model its first reading lesson and saves
PARTIAL_OUT/last.pt. FULL_SELECTION_MODE=utility then uses that AdamW state to
score the full data window. The OPUS proxy defaults to both normal language
heldout rows and Generalization Dynamics anti-parrot rows, so selected data must
push toward speaking and away from shortcut parroting. After full training,
GD-lite runs automatically; loss alone is not a promoted acceptance signal.
PLAN
}

status() {
  echo "partial_sample_ready=$(sample_ready "${PARTIAL_SAMPLE}" && echo yes || echo no) path=${PARTIAL_SAMPLE}"
  echo "full_sample_ready=$(sample_ready "${FULL_SAMPLE}" && echo yes || echo no) path=${FULL_SAMPLE}"
  echo "partial_checkpoint=$([[ -f "${PARTIAL_OUT}/last_model.pt" ]] && echo present || echo missing) path=${PARTIAL_OUT}/last_model.pt"
  echo "partial_report=$([[ -f "${PARTIAL_OUT}/report.json" ]] && echo present || echo missing) path=${PARTIAL_OUT}/report.json"
  echo "full_checkpoint=$([[ -f "${FULL_OUT}/last_model.pt" ]] && echo present || echo missing) path=${FULL_OUT}/last_model.pt"
  echo "full_report=$([[ -f "${FULL_OUT}/report.json" ]] && echo present || echo missing) path=${FULL_OUT}/report.json"
  echo "full_resume_ckpt=$([[ -n "${FULL_RESUME_CKPT}" && -f "${FULL_RESUME_CKPT}" ]] && echo present || ([[ -n "${FULL_RESUME_CKPT}" ]] && echo missing || echo auto)) path=${FULL_RESUME_CKPT:-<auto>}"
  echo "resume_load_optimizer=${RESUME_LOAD_OPTIMIZER}"
  echo "teacher_distill=$([[ -n "${TEACHER_CHECKPOINT}" && "${TEACHER_DISTILL_WEIGHT}" != "0" && "${TEACHER_DISTILL_WEIGHT}" != "0.0" && "${TEACHER_DISTILL_WEIGHT}" != "0.00" ]] && echo enabled || echo disabled) checkpoint=${TEACHER_CHECKPOINT:-<none>} weight=${TEACHER_DISTILL_WEIGHT}"
  echo "qwen_boundary_prior=$([[ "${QWEN_BOUNDARY_PRIOR_WEIGHT}" != "0" && "${QWEN_BOUNDARY_PRIOR_WEIGHT}" != "0.0" && "${QWEN_BOUNDARY_PRIOR_WEIGHT}" != "0.00" ]] && echo enabled || echo disabled) weight=${QWEN_BOUNDARY_PRIOR_WEIGHT} tokenizer=${QWEN_BOUNDARY_TOKENIZER_MODEL_ID}"
  echo "decoder_latent_mode=${DECODER_LATENT_MODE}"
  echo "past_success_report=$([[ -f "${PAST_SUCCESS_REPORT_JSON}" ]] && echo present || echo missing) path=${PAST_SUCCESS_REPORT_JSON}"
  echo "past_success_restoration_gate=$([[ -n "${PAST_SUCCESS_RESTORATION_GATE_JSON}" && -f "${PAST_SUCCESS_RESTORATION_GATE_JSON}" ]] && echo present || ([[ -n "${PAST_SUCCESS_RESTORATION_GATE_JSON}" ]] && echo missing || echo none)) path=${PAST_SUCCESS_RESTORATION_GATE_JSON:-<none>}"
  echo "allow_missing_past_success_preflight=${ALLOW_MISSING_PAST_SUCCESS_PREFLIGHT}"
  echo "acknowledge_past_success_restoration_gap=${ACKNOWLEDGE_PAST_SUCCESS_RESTORATION_GAP}"
  echo "start_full_build_early=${START_FULL_BUILD_EARLY}"
  echo "selection_mode=${SELECTION_MODE}"
  echo "partial_selection_mode=${PARTIAL_SELECTION_MODE}"
  echo "full_selection_mode=${FULL_SELECTION_MODE}"
  echo "utility_score_jsonl=${UTILITY_SCORE_JSONL:-<none>}"
  echo "utility_temperature=${UTILITY_TEMPERATURE}"
  echo "opus_checkpoint=${OPUS_CHECKPOINT:-<none>}"
  echo "opus_proxy_jsonl=${OPUS_PROXY_JSONL}"
  echo "opus_candidate_max_rows=${OPUS_CANDIDATE_MAX_ROWS}"
  echo "opus_proxy_max_rows=${OPUS_PROXY_MAX_ROWS}"
  echo "opus_proxy_max_rows_per_group=${OPUS_PROXY_MAX_ROWS_PER_GROUP}"
  echo "opus_projection_dim=${OPUS_PROJECTION_DIM}"
  echo "opus_preconditioner=${OPUS_PRECONDITIONER}"
  echo "opus_param_name_regex=${OPUS_PARAM_NAME_REGEX}"
  echo "opus_device=${OPUS_DEVICE}"
  echo "online_opus_enabled=${ONLINE_OPUS_ENABLED}"
  echo "online_opus_candidate_batches=${ONLINE_OPUS_CANDIDATE_BATCHES}"
  echo "online_opus_proxy_batches=${ONLINE_OPUS_PROXY_BATCHES}"
  echo "online_opus_every=${ONLINE_OPUS_EVERY}"
  echo "online_opus_proxy_source=${ONLINE_OPUS_PROXY_SOURCE}"
  echo "save_optimizer_checkpoint=${SAVE_OPTIMIZER_CHECKPOINT}"
  echo "optimizer_checkpoint_every=${OPTIMIZER_CHECKPOINT_EVERY}"
  echo "gd_lite_enabled=${GD_LITE_ENABLED}"
  echo "gd_lite_probe_jsonl=${GD_LITE_PROBE_JSONL}"
  echo "gd_lite_out=${GD_LITE_OUT}"
  echo "gd_lite_think_steps=${GD_LITE_THINK_STEPS}"
  echo "gd_lite_require_accept=${GD_LITE_REQUIRE_ACCEPT}"
  echo "required_triton_ptxas_path=${REQUIRED_TRITON_PTXAS_PATH}"
  echo "triton_ptxas_path=${TRITON_PTXAS_PATH:-<unset>}"
  echo "official_gdn2_preflight_smoke=${OFFICIAL_GDN2_PREFLIGHT_SMOKE}"
  echo "partial_training_processes:"
  process_matching "557_train_blt_d_prefixlm_dataio.py .*${PARTIAL_OUT}" || true
  echo "full_training_processes:"
  process_matching "557_train_blt_d_prefixlm_dataio.py .*${FULL_OUT}" || true
  echo "partial_sample_builder_processes:"
  process_matching "555_prepare_byte_prefixlm_sample.py .*${PARTIAL_SAMPLE}" || true
  echo "full_sample_builder_processes:"
  process_matching "555_prepare_byte_prefixlm_sample.py .*${FULL_SAMPLE}" || true
  echo "supervisor_processes:"
  process_matching "559_run_stage95_blt_partial_then_full_dgx.sh run" || true
}

preflight() {
  require_triton_ptxas
  local resume_ckpt="${FULL_RESUME_CKPT:-}"
  if [[ -z "${resume_ckpt}" && -f "${FULL_OUT}/last_model.pt" ]]; then
    resume_ckpt="${FULL_OUT}/last_model.pt"
  fi
  if [[ -z "${resume_ckpt}" && -f "${PARTIAL_OUT}/last_model.pt" ]]; then
    resume_ckpt="${PARTIAL_OUT}/last_model.pt"
  fi
  local report_json=""
  if [[ -f "${FULL_OUT}/report.json" ]]; then
    report_json="${FULL_OUT}/report.json"
  elif [[ -f "${PARTIAL_OUT}/report.json" ]]; then
    report_json="${PARTIAL_OUT}/report.json"
  fi

  local args=(
    scripts/613_preflight_official_gdn2_contract.py
    --required-ptxas "${REQUIRED_TRITON_PTXAS_PATH}"
    --triton-ptxas "${TRITON_PTXAS_PATH}"
    --target-backend official_gated_delta2
    --expect-decoder-latent-mode "${DECODER_LATENT_MODE}"
    --official-smoke "${OFFICIAL_GDN2_PREFLIGHT_SMOKE}"
  )
  if [[ -n "${resume_ckpt}" ]]; then
    args+=(--checkpoint "${resume_ckpt}")
  fi
  if [[ -n "${report_json}" ]]; then
    args+=(--report-json "${report_json}")
  fi
  "${PYTHON}" "${args[@]}"
}

run_gd_lite_gate_if_enabled() {
  if ! truthy "${GD_LITE_ENABLED}"; then
    echo "gd_lite_gate=disabled"
    return 0
  fi
  local checkpoint="${1:-${FULL_OUT}/last_model.pt}"
  local sampled_data="${2:-${FULL_SAMPLE}}"
  local out_path="${3:-${GD_LITE_OUT}}"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "gd_lite_gate=missing_checkpoint path=${checkpoint}" >&2
    exit 6
  fi
  if [[ ! -f "${sampled_data}/metadata.json" ]]; then
    echo "gd_lite_gate=missing_sample path=${sampled_data}" >&2
    exit 6
  fi
  mkdir -p "$(dirname "${out_path}")"
  echo "gd_lite_gate=starting checkpoint=${checkpoint} out=${out_path}"
  "${PYTHON}" scripts/567_eval_blt_generalization_dynamics_probe.py \
    --checkpoint "${checkpoint}" \
    --sampled-data "${sampled_data}" \
    --probe-jsonl "${GD_LITE_PROBE_JSONL}" \
    --out "${out_path}" \
    --device "${GD_LITE_DEVICE}" \
    --think-steps "${GD_LITE_THINK_STEPS}" \
    --max-rows "${GD_LITE_MAX_ROWS}" \
    --tensorboard-dir "$(dirname "${out_path}")/tensorboard" \
    --tensorboard-prefix "eval/generalization_dynamics_official"
  if truthy "${GD_LITE_REQUIRE_ACCEPT}"; then
    "${PYTHON}" - "${out_path}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(0 if payload.get("accepted") is True else 1)
PY
  fi
}

run() {
  mkdir -p "${WORK_BASE}"
  exec 9>"${LOCK_FILE}"
  if ! flock -n 9; then
    echo "stage95_supervisor=already_running lock=${LOCK_FILE}"
    exit 0
  fi
  echo "stage95_supervisor=start $(date -Is)"
  echo "stage95_preflight=initial"
  preflight
  if [[ "${START_FULL_BUILD_EARLY}" == "1" || "${START_FULL_BUILD_EARLY}" == "true" || "${START_FULL_BUILD_EARLY}" == "yes" ]]; then
    ensure_full_sample_building
  else
    echo "full_sample_builder=delayed_until_after_partial_training path=${FULL_SAMPLE}"
  fi
  build_partial_sample_if_needed
  run_training_if_needed \
    "partial_training" \
    "${PARTIAL_SAMPLE}" \
    "${PARTIAL_OUT}" \
    "${PARTIAL_STEPS}" \
    "${PARTIAL_LOG}" \
    "${PARTIAL_BATCH_SIZE}" \
    "${PARTIAL_EVAL_BATCH_SIZE}"
  wait_for_full_sample
  local resume_ckpt="${FULL_RESUME_CKPT:-}"
  if [[ -z "${resume_ckpt}" ]]; then
    resume_ckpt="${FULL_OUT}/last_model.pt"
  fi
  if [[ ! -f "${resume_ckpt}" ]]; then
    resume_ckpt="${PARTIAL_OUT}/last_model.pt"
  fi
  if [[ ! -f "${resume_ckpt}" ]]; then
    echo "missing_resume_checkpoint=${resume_ckpt}" >&2
    exit 3
  fi
  echo "stage95_preflight=before_full_resume checkpoint=${resume_ckpt}"
  FULL_RESUME_CKPT="${resume_ckpt}" preflight
  run_training_if_needed \
    "full_continue_training" \
    "${FULL_SAMPLE}" \
    "${FULL_OUT}" \
    "${FULL_STEPS}" \
    "${FULL_LOG}" \
    "${FULL_BATCH_SIZE}" \
    "${FULL_EVAL_BATCH_SIZE}" \
    --resume "${resume_ckpt}"
  run_gd_lite_gate_if_enabled "${FULL_OUT}/last_model.pt" "${FULL_SAMPLE}" "${GD_LITE_OUT}"
  echo "stage95_supervisor=done $(date -Is)"
}

run_full() {
  mkdir -p "${WORK_BASE}"
  if ! sample_ready_for_selection "${FULL_SAMPLE}" "${FULL_SELECTION_MODE}"; then
    echo "full_sample=missing_or_wrong_selection path=${FULL_SAMPLE} selection=${FULL_SELECTION_MODE}" >&2
    exit 4
  fi
  echo "stage95_full_direct=start $(date -Is)"
  echo "stage95_preflight=full_direct"
  preflight
  local extra_args=()
  if [[ -n "${FULL_RESUME_CKPT}" ]]; then
    if [[ ! -f "${FULL_RESUME_CKPT}" ]]; then
      echo "missing_full_resume_checkpoint=${FULL_RESUME_CKPT}" >&2
      exit 3
    fi
    extra_args+=(--resume "${FULL_RESUME_CKPT}")
  fi
  run_training_if_needed \
    "full_direct_training" \
    "${FULL_SAMPLE}" \
    "${FULL_OUT}" \
    "${FULL_STEPS}" \
    "${FULL_LOG}" \
    "${FULL_BATCH_SIZE}" \
    "${FULL_EVAL_BATCH_SIZE}" \
    "${extra_args[@]}"
  run_gd_lite_gate_if_enabled "${FULL_OUT}/last_model.pt" "${FULL_SAMPLE}" "${GD_LITE_OUT}"
  echo "stage95_full_direct=done $(date -Is)"
}

launch() {
  mkdir -p "$(dirname "${SUPERVISOR_LOG}")" "${WORK_BASE}"
  (
    cd "${ROOT}"
    nohup env \
      ROOT="${ROOT}" \
      PYTHON="${PYTHON}" \
      WORK_BASE="${WORK_BASE}" \
      PARTIAL_WORK_DIR="${PARTIAL_WORK_DIR}" \
      FULL_WORK_DIR="${FULL_WORK_DIR}" \
      PARTIAL_SAMPLE="${PARTIAL_SAMPLE}" \
      FULL_SAMPLE="${FULL_SAMPLE}" \
      PARTIAL_OUT="${PARTIAL_OUT}" \
      FULL_OUT="${FULL_OUT}" \
      FULL_RESUME_CKPT="${FULL_RESUME_CKPT}" \
      LOG_DIR="${LOG_DIR}" \
      SUPERVISOR_LOG="${SUPERVISOR_LOG}" \
      PARTIAL_LOG="${PARTIAL_LOG}" \
      FULL_LOG="${FULL_LOG}" \
      FULL_BUILD_LOG="${FULL_BUILD_LOG}" \
      LOCK_FILE="${LOCK_FILE}" \
      FULL_BUILD_LOCK="${FULL_BUILD_LOCK}" \
      PARTIAL_STEPS="${PARTIAL_STEPS}" \
      FULL_STEPS="${FULL_STEPS}" \
      CHECKPOINT_EVERY="${CHECKPOINT_EVERY}" \
      OPTIMIZER_CHECKPOINT_EVERY="${OPTIMIZER_CHECKPOINT_EVERY}" \
      SEQ_LEN="${SEQ_LEN}" \
      TRAIN_THINK_STEPS="${TRAIN_THINK_STEPS}" \
      EVAL_EVERY="${EVAL_EVERY}" \
      EVAL_MAX_ROWS="${EVAL_MAX_ROWS}" \
      PARTIAL_BATCH_SIZE="${PARTIAL_BATCH_SIZE}" \
      FULL_BATCH_SIZE="${FULL_BATCH_SIZE}" \
      PARTIAL_EVAL_BATCH_SIZE="${PARTIAL_EVAL_BATCH_SIZE}" \
      FULL_EVAL_BATCH_SIZE="${FULL_EVAL_BATCH_SIZE}" \
      LR="${LR}" \
      LR_WARMUP_STEPS="${LR_WARMUP_STEPS}" \
      RESUME_STRICT="${RESUME_STRICT}" \
      RESUME_LOAD_OPTIMIZER="${RESUME_LOAD_OPTIMIZER}" \
      PATCH_SIZE="${PATCH_SIZE}" \
      PATCH_BOUNDARY_MODE="${PATCH_BOUNDARY_MODE}" \
      DYNAMIC_MIN_PATCH_SIZE="${DYNAMIC_MIN_PATCH_SIZE}" \
      DYNAMIC_SOFT_PATCH_SIZE="${DYNAMIC_SOFT_PATCH_SIZE}" \
      HBF_BOUNDARY_THRESHOLD="${HBF_BOUNDARY_THRESHOLD}" \
      BOUNDARY_PRIOR_WEIGHT="${BOUNDARY_PRIOR_WEIGHT}" \
      BOUNDARY_TARGET_RATIO="${BOUNDARY_TARGET_RATIO}" \
      TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT}" \
      TEACHER_DISTILL_WEIGHT="${TEACHER_DISTILL_WEIGHT}" \
      TEACHER_DISTILL_TEMPERATURE="${TEACHER_DISTILL_TEMPERATURE}" \
      TEACHER_DISTILL_MAX_TARGETS="${TEACHER_DISTILL_MAX_TARGETS}" \
      TEACHER_SEQ_LEN="${TEACHER_SEQ_LEN}" \
      QWEN_BOUNDARY_PRIOR_WEIGHT="${QWEN_BOUNDARY_PRIOR_WEIGHT}" \
      QWEN_BOUNDARY_TOKENIZER_MODEL_ID="${QWEN_BOUNDARY_TOKENIZER_MODEL_ID}" \
      DECODER_LATENT_MODE="${DECODER_LATENT_MODE}" \
      PAST_SUCCESS_REPORT_JSON="${PAST_SUCCESS_REPORT_JSON}" \
      PAST_SUCCESS_RESTORATION_GATE_JSON="${PAST_SUCCESS_RESTORATION_GATE_JSON}" \
      ALLOW_MISSING_PAST_SUCCESS_PREFLIGHT="${ALLOW_MISSING_PAST_SUCCESS_PREFLIGHT}" \
      ACKNOWLEDGE_PAST_SUCCESS_RESTORATION_GAP="${ACKNOWLEDGE_PAST_SUCCESS_RESTORATION_GAP}" \
      SOURCE_BUCKET_QUOTAS="${SOURCE_BUCKET_QUOTAS}" \
      PARTIAL_SOURCE_BUCKET_QUOTAS="${PARTIAL_SOURCE_BUCKET_QUOTAS}" \
      FULL_SOURCE_BUCKET_QUOTAS="${FULL_SOURCE_BUCKET_QUOTAS}" \
      SOURCE_BUCKET_MAX_ROWS_PER_FILE="${SOURCE_BUCKET_MAX_ROWS_PER_FILE}" \
      START_FULL_BUILD_EARLY="${START_FULL_BUILD_EARLY}" \
      SELECTION_MODE="${SELECTION_MODE}" \
      PARTIAL_SELECTION_MODE="${PARTIAL_SELECTION_MODE}" \
      FULL_SELECTION_MODE="${FULL_SELECTION_MODE}" \
      UTILITY_SCORE_JSONL="${UTILITY_SCORE_JSONL}" \
      UTILITY_TEMPERATURE="${UTILITY_TEMPERATURE}" \
      OPUS_CHECKPOINT="${OPUS_CHECKPOINT}" \
      OPUS_PROXY_JSONL="${OPUS_PROXY_JSONL}" \
      OPUS_SCORE_OUT="${OPUS_SCORE_OUT}" \
      OPUS_REPORT_OUT="${OPUS_REPORT_OUT}" \
      OPUS_CANDIDATE_MAX_ROWS="${OPUS_CANDIDATE_MAX_ROWS}" \
      OPUS_CANDIDATE_MAX_SCAN_ROWS_PER_FILE="${OPUS_CANDIDATE_MAX_SCAN_ROWS_PER_FILE}" \
      OPUS_PROXY_MAX_ROWS="${OPUS_PROXY_MAX_ROWS}" \
      OPUS_PROXY_MAX_ROWS_PER_GROUP="${OPUS_PROXY_MAX_ROWS_PER_GROUP}" \
      OPUS_PROXY_GROUPING="${OPUS_PROXY_GROUPING}" \
      OPUS_PROXY_SCORE_MODE="${OPUS_PROXY_SCORE_MODE}" \
      OPUS_PROXY_MEAN_WEIGHT="${OPUS_PROXY_MEAN_WEIGHT}" \
      OPUS_PROJECTION_DIM="${OPUS_PROJECTION_DIM}" \
      OPUS_PRECONDITIONER="${OPUS_PRECONDITIONER}" \
      OPUS_PARAM_NAME_REGEX="${OPUS_PARAM_NAME_REGEX}" \
      OPUS_REDUNDANCY_WEIGHT="${OPUS_REDUNDANCY_WEIGHT}" \
      OPUS_DEVICE="${OPUS_DEVICE}" \
      ONLINE_OPUS_ENABLED="${ONLINE_OPUS_ENABLED}" \
      ONLINE_OPUS_CANDIDATE_BATCHES="${ONLINE_OPUS_CANDIDATE_BATCHES}" \
      ONLINE_OPUS_PROXY_BATCHES="${ONLINE_OPUS_PROXY_BATCHES}" \
      ONLINE_OPUS_EVERY="${ONLINE_OPUS_EVERY}" \
      ONLINE_OPUS_START_STEP="${ONLINE_OPUS_START_STEP}" \
      ONLINE_OPUS_PROXY_SOURCE="${ONLINE_OPUS_PROXY_SOURCE}" \
      ONLINE_OPUS_PROJECTION_DIM="${ONLINE_OPUS_PROJECTION_DIM}" \
      ONLINE_OPUS_PRECONDITIONER="${ONLINE_OPUS_PRECONDITIONER}" \
      ONLINE_OPUS_PARAM_NAME_REGEX="${ONLINE_OPUS_PARAM_NAME_REGEX}" \
      SAVE_OPTIMIZER_CHECKPOINT="${SAVE_OPTIMIZER_CHECKPOINT}" \
      OPTIMIZER_CHECKPOINT_EVERY="${OPTIMIZER_CHECKPOINT_EVERY}" \
      GD_LITE_ENABLED="${GD_LITE_ENABLED}" \
      GD_LITE_PROBE_JSONL="${GD_LITE_PROBE_JSONL}" \
      GD_LITE_OUT="${GD_LITE_OUT}" \
      GD_LITE_DEVICE="${GD_LITE_DEVICE}" \
      GD_LITE_MAX_ROWS="${GD_LITE_MAX_ROWS}" \
      GD_LITE_THINK_STEPS="${GD_LITE_THINK_STEPS}" \
      GD_LITE_REQUIRE_ACCEPT="${GD_LITE_REQUIRE_ACCEPT}" \
      REQUIRED_TRITON_PTXAS_PATH="${REQUIRED_TRITON_PTXAS_PATH}" \
      TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-}" \
      OFFICIAL_GDN2_PREFLIGHT_SMOKE="${OFFICIAL_GDN2_PREFLIGHT_SMOKE}" \
      bash scripts/559_run_stage95_blt_partial_then_full_dgx.sh run \
      >>"${SUPERVISOR_LOG}" 2>&1 &
    echo "stage95_supervisor_pid=$!"
    echo "stage95_supervisor_log=${SUPERVISOR_LOG}"
  )
}

launch_full() {
  mkdir -p "$(dirname "${SUPERVISOR_LOG}")" "${WORK_BASE}"
  (
    cd "${ROOT}"
    nohup env \
      ROOT="${ROOT}" \
      PYTHON="${PYTHON}" \
      WORK_BASE="${WORK_BASE}" \
      FULL_WORK_DIR="${FULL_WORK_DIR}" \
      FULL_SAMPLE="${FULL_SAMPLE}" \
      FULL_OUT="${FULL_OUT}" \
      FULL_RESUME_CKPT="${FULL_RESUME_CKPT}" \
      LOG_DIR="${LOG_DIR}" \
      SUPERVISOR_LOG="${SUPERVISOR_LOG}" \
      FULL_LOG="${FULL_LOG}" \
      FULL_STEPS="${FULL_STEPS}" \
      CHECKPOINT_EVERY="${CHECKPOINT_EVERY}" \
      OPTIMIZER_CHECKPOINT_EVERY="${OPTIMIZER_CHECKPOINT_EVERY}" \
      SEQ_LEN="${SEQ_LEN}" \
      TRAIN_THINK_STEPS="${TRAIN_THINK_STEPS}" \
      EVAL_EVERY="${EVAL_EVERY}" \
      EVAL_MAX_ROWS="${EVAL_MAX_ROWS}" \
      FULL_BATCH_SIZE="${FULL_BATCH_SIZE}" \
      FULL_EVAL_BATCH_SIZE="${FULL_EVAL_BATCH_SIZE}" \
      LR="${LR}" \
      LR_WARMUP_STEPS="${LR_WARMUP_STEPS}" \
      RESUME_STRICT="${RESUME_STRICT}" \
      RESUME_LOAD_OPTIMIZER="${RESUME_LOAD_OPTIMIZER}" \
      PATCH_SIZE="${PATCH_SIZE}" \
      PATCH_BOUNDARY_MODE="${PATCH_BOUNDARY_MODE}" \
      DYNAMIC_MIN_PATCH_SIZE="${DYNAMIC_MIN_PATCH_SIZE}" \
      DYNAMIC_SOFT_PATCH_SIZE="${DYNAMIC_SOFT_PATCH_SIZE}" \
      HBF_BOUNDARY_THRESHOLD="${HBF_BOUNDARY_THRESHOLD}" \
      BOUNDARY_PRIOR_WEIGHT="${BOUNDARY_PRIOR_WEIGHT}" \
      BOUNDARY_TARGET_RATIO="${BOUNDARY_TARGET_RATIO}" \
      TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT}" \
      TEACHER_DISTILL_WEIGHT="${TEACHER_DISTILL_WEIGHT}" \
      TEACHER_DISTILL_TEMPERATURE="${TEACHER_DISTILL_TEMPERATURE}" \
      TEACHER_DISTILL_MAX_TARGETS="${TEACHER_DISTILL_MAX_TARGETS}" \
      TEACHER_SEQ_LEN="${TEACHER_SEQ_LEN}" \
      QWEN_BOUNDARY_PRIOR_WEIGHT="${QWEN_BOUNDARY_PRIOR_WEIGHT}" \
      QWEN_BOUNDARY_TOKENIZER_MODEL_ID="${QWEN_BOUNDARY_TOKENIZER_MODEL_ID}" \
      DECODER_LATENT_MODE="${DECODER_LATENT_MODE}" \
      PAST_SUCCESS_REPORT_JSON="${PAST_SUCCESS_REPORT_JSON}" \
      PAST_SUCCESS_RESTORATION_GATE_JSON="${PAST_SUCCESS_RESTORATION_GATE_JSON}" \
      ALLOW_MISSING_PAST_SUCCESS_PREFLIGHT="${ALLOW_MISSING_PAST_SUCCESS_PREFLIGHT}" \
      ACKNOWLEDGE_PAST_SUCCESS_RESTORATION_GAP="${ACKNOWLEDGE_PAST_SUCCESS_RESTORATION_GAP}" \
      ONLINE_OPUS_ENABLED="${ONLINE_OPUS_ENABLED}" \
      ONLINE_OPUS_CANDIDATE_BATCHES="${ONLINE_OPUS_CANDIDATE_BATCHES}" \
      ONLINE_OPUS_PROXY_BATCHES="${ONLINE_OPUS_PROXY_BATCHES}" \
      ONLINE_OPUS_EVERY="${ONLINE_OPUS_EVERY}" \
      ONLINE_OPUS_START_STEP="${ONLINE_OPUS_START_STEP}" \
      ONLINE_OPUS_PROXY_SOURCE="${ONLINE_OPUS_PROXY_SOURCE}" \
      ONLINE_OPUS_PROJECTION_DIM="${ONLINE_OPUS_PROJECTION_DIM}" \
      ONLINE_OPUS_PRECONDITIONER="${ONLINE_OPUS_PRECONDITIONER}" \
      ONLINE_OPUS_PARAM_NAME_REGEX="${ONLINE_OPUS_PARAM_NAME_REGEX}" \
      SAVE_OPTIMIZER_CHECKPOINT="${SAVE_OPTIMIZER_CHECKPOINT}" \
      OPTIMIZER_CHECKPOINT_EVERY="${OPTIMIZER_CHECKPOINT_EVERY}" \
      GD_LITE_ENABLED="${GD_LITE_ENABLED}" \
      GD_LITE_PROBE_JSONL="${GD_LITE_PROBE_JSONL}" \
      GD_LITE_OUT="${GD_LITE_OUT}" \
      GD_LITE_DEVICE="${GD_LITE_DEVICE}" \
      GD_LITE_MAX_ROWS="${GD_LITE_MAX_ROWS}" \
      GD_LITE_THINK_STEPS="${GD_LITE_THINK_STEPS}" \
      GD_LITE_REQUIRE_ACCEPT="${GD_LITE_REQUIRE_ACCEPT}" \
      REQUIRED_TRITON_PTXAS_PATH="${REQUIRED_TRITON_PTXAS_PATH}" \
      TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-}" \
      OFFICIAL_GDN2_PREFLIGHT_SMOKE="${OFFICIAL_GDN2_PREFLIGHT_SMOKE}" \
      bash scripts/559_run_stage95_blt_partial_then_full_dgx.sh run-full \
      >>"${SUPERVISOR_LOG}" 2>&1 &
    echo "stage95_full_direct_pid=$!"
    echo "stage95_full_direct_log=${SUPERVISOR_LOG}"
  )
}

action="${1:-plan}"
case "${action}" in
  plan)
    plan
    ;;
  status)
    status
    ;;
  preflight)
    preflight
    ;;
  run)
    run
    ;;
  launch)
    launch
    ;;
  run-full)
    run_full
    ;;
  launch-full)
    launch_full
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
