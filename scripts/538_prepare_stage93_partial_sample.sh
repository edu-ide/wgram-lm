#!/usr/bin/env bash
set -euo pipefail

# Build a safe partial Stage93 sampled shard from tokenized task folders that
# are already complete, while the full tokenizer keeps running.
#
# Plain-language contract:
#   the full job is still printing the whole textbook;
#   this script gathers only already-finished pages into a thin temporary book;
#   Stage93A can study that book now, then Stage93B can continue on the full
#   bound book later.

ACTION="${1:-plan}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATA_IO_DIR="${DATA_IO_DIR:-${ROOT}/references/official/data_io}"
WORK_DIR="${WORK_DIR:-${ROOT}/local_eval/stage93_hrm_text_reasoning_nonflan_dataio}"
TOKENIZED_IN="${TOKENIZED_IN:-${WORK_DIR}/tokenized}"
PARTIAL_TOKENIZED_OUT="${PARTIAL_TOKENIZED_OUT:-${WORK_DIR}/tokenized_partial}"
PARTIAL_SAMPLED_OUT="${PARTIAL_SAMPLED_OUT:-${WORK_DIR}/sampled_partial}"
PARTIAL_ANALYTICS_OUT="${PARTIAL_ANALYTICS_OUT:-${WORK_DIR}/show_analytics_partial.md}"
PREFIX_CONFIG_PATH="${PREFIX_CONFIG_PATH:-${DATA_IO_DIR}/prefix_config.yaml}"
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"

MAX_TASKS="${MAX_TASKS:-256}"
MIN_TASKS="${MIN_TASKS:-64}"
STABLE_SECONDS="${STABLE_SECONDS:-120}"
EPOCHS="${EPOCHS:-2}"
CONTEXT_SIZE="${CONTEXT_SIZE:-1025}"
MIN_RESP_LENGTH="${MIN_RESP_LENGTH:-2}"

RUN_NAME="${RUN_NAME:-20260524_STAGE93A_DGX913M_PARTIAL_TO30K}"
PARTIAL_STEPS="${PARTIAL_STEPS:-30000}"
LAUNCH_LOG="${LAUNCH_LOG:-/tmp/${RUN_NAME}.log}"
RESUME="${RESUME:-${ROOT}/local_eval/20260523_STAGE92_DGX913M_BS8_CONTINUE_TO24K/last_model.pt}"

REQUIRED_FILES=(inst_start.npy inst_len.npy resp_start.npy resp_len.npy tokens.npy metadata.json)

print_plan() {
  cat <<PLAN
Stage93 partial binding plan

actions: plan status snapshot sample all launch

ROOT=${ROOT}
DATA_IO_DIR=${DATA_IO_DIR}
WORK_DIR=${WORK_DIR}
TOKENIZED_IN=${TOKENIZED_IN}
PARTIAL_TOKENIZED_OUT=${PARTIAL_TOKENIZED_OUT}
PARTIAL_SAMPLED_OUT=${PARTIAL_SAMPLED_OUT}
PARTIAL_ANALYTICS_OUT=${PARTIAL_ANALYTICS_OUT}
PREFIX_CONFIG_PATH=${PREFIX_CONFIG_PATH}
PYTHON=${PYTHON}

MAX_TASKS=${MAX_TASKS}
MIN_TASKS=${MIN_TASKS}
STABLE_SECONDS=${STABLE_SECONDS}
EPOCHS=${EPOCHS}
CONTEXT_SIZE=${CONTEXT_SIZE}
MIN_RESP_LENGTH=${MIN_RESP_LENGTH}

RUN_NAME=${RUN_NAME}
PARTIAL_STEPS=${PARTIAL_STEPS}
LAUNCH_LOG=${LAUNCH_LOG}
RESUME=${RESUME}
PLAN
}

task_complete_and_stable() {
  local dir="$1"
  local file
  for file in "${REQUIRED_FILES[@]}"; do
    [[ -s "${dir}/${file}" ]] || return 1
  done

  local newest now age
  newest="$(find "${dir}" -maxdepth 1 -type f -printf '%T@\n' | sort -nr | head -n 1)"
  now="$(date +%s)"
  newest="${newest%.*}"
  age="$(( now - newest ))"
  (( age >= STABLE_SECONDS ))
}

snapshot() {
  if [[ ! -d "${TOKENIZED_IN}" ]]; then
    echo "missing TOKENIZED_IN: ${TOKENIZED_IN}" >&2
    exit 2
  fi
  if [[ ! -f "${TOKENIZED_IN}/tokenizer_info.json" ]]; then
    echo "missing tokenizer_info.json in ${TOKENIZED_IN}" >&2
    exit 2
  fi

  rm -rf "${PARTIAL_TOKENIZED_OUT}.tmp" "${PARTIAL_TOKENIZED_OUT}"
  mkdir -p "${PARTIAL_TOKENIZED_OUT}.tmp"
  ln "${TOKENIZED_IN}/tokenizer_info.json" "${PARTIAL_TOKENIZED_OUT}.tmp/tokenizer_info.json"

  local count=0
  local dir name file dst
  while IFS= read -r -d '' dir; do
    if ! task_complete_and_stable "${dir}"; then
      continue
    fi
    name="$(basename "${dir}")"
    dst="${PARTIAL_TOKENIZED_OUT}.tmp/${name}"
    mkdir -p "${dst}"
    for file in "${REQUIRED_FILES[@]}"; do
      ln "${dir}/${file}" "${dst}/${file}"
    done
    printf '%s\n' "${name}" >> "${PARTIAL_TOKENIZED_OUT}.tmp/partial_manifest.txt"
    count="$((count + 1))"
    if (( count >= MAX_TASKS )); then
      break
    fi
  done < <(find "${TOKENIZED_IN}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

  if (( count < MIN_TASKS )); then
    echo "only ${count} stable tokenized tasks found; need MIN_TASKS=${MIN_TASKS}" >&2
    exit 4
  fi

  mv "${PARTIAL_TOKENIZED_OUT}.tmp" "${PARTIAL_TOKENIZED_OUT}"
  echo "PARTIAL_TOKENIZED_READY:${PARTIAL_TOKENIZED_OUT}"
  echo "TASKS:${count}"
}

sample() {
  if [[ ! -f "${PARTIAL_TOKENIZED_OUT}/tokenizer_info.json" ]]; then
    echo "partial tokenized snapshot missing; run snapshot first" >&2
    exit 2
  fi
  rm -rf "${PARTIAL_SAMPLED_OUT}"
  mkdir -p "$(dirname "${PARTIAL_SAMPLED_OUT}")"
  (
    cd "${DATA_IO_DIR}"
    "${PYTHON}" sample_tokenized.py \
      "tokenized_path=${PARTIAL_TOKENIZED_OUT}" \
      "output_path=${PARTIAL_SAMPLED_OUT}" \
      "prefix_config_path=${PREFIX_CONFIG_PATH}" \
      "epochs=${EPOCHS}" \
      "context_size=${CONTEXT_SIZE}" \
      "min_resp_length=${MIN_RESP_LENGTH}" \
      > "${PARTIAL_ANALYTICS_OUT}"
  )
  echo "PARTIAL_SAMPLED_READY:${PARTIAL_SAMPLED_OUT}"
  ls -lh "${PARTIAL_SAMPLED_OUT}/tokens.npy"
}

launch() {
  if [[ ! -f "${PARTIAL_SAMPLED_OUT}/tokens.npy" ]]; then
    echo "partial sampled data missing: ${PARTIAL_SAMPLED_OUT}/tokens.npy" >&2
    exit 2
  fi
  SAMPLED_DATA="${PARTIAL_SAMPLED_OUT}" \
    RESUME="${RESUME}" \
    RUN_NAME="${RUN_NAME}" \
    STEPS="${PARTIAL_STEPS}" \
    LOG_FILE="${LAUNCH_LOG}" \
    bash "${ROOT}/scripts/536_launch_stage93_dgx_continue_prefixlm.sh"
}

status() {
  print_plan
  echo
  echo "status:"
  [[ -d "${TOKENIZED_IN}" ]] && find "${TOKENIZED_IN}" -mindepth 2 -maxdepth 2 -name tokens.npy | wc -l | sed 's/^/tokenized_complete=/'
  [[ -d "${PARTIAL_TOKENIZED_OUT}" ]] && { find "${PARTIAL_TOKENIZED_OUT}" -mindepth 1 -maxdepth 1 -type d | wc -l | sed 's/^/partial_tasks=/'; du -sh "${PARTIAL_TOKENIZED_OUT}" || true; }
  [[ -f "${PARTIAL_SAMPLED_OUT}/tokens.npy" ]] && ls -lh "${PARTIAL_SAMPLED_OUT}/tokens.npy"
  [[ -f "${PARTIAL_ANALYTICS_OUT}" ]] && tail -n 30 "${PARTIAL_ANALYTICS_OUT}" || true
}

case "${ACTION}" in
  plan)
    print_plan
    ;;
  status)
    status
    ;;
  snapshot)
    snapshot
    ;;
  sample)
    sample
    ;;
  all)
    snapshot
    sample
    ;;
  launch)
    launch
    ;;
  *)
    echo "unknown action: ${ACTION}" >&2
    print_plan >&2
    exit 2
    ;;
esac
