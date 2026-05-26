#!/usr/bin/env bash
set -euo pipefail

# Prepare a larger HRM-Text/data_io PrefixLM training shard for Stage93.
#
# Plain-language contract:
#   the cleaned dataset is the bookcase;
#   this script selects the first serious study shelf, without copying the
#   whole bookcase;
#   tokenization and sampling turn that shelf into the actual training handout.

ACTION="${1:-plan}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATA_IO_DIR="${DATA_IO_DIR:-${ROOT}/references/official/data_io}"

if [[ -z "${CLEANED_DATA_PATH:-}" ]]; then
  if [[ -d /mnt/data4tb/datasets/hrm-text-data-io-cleaned-20260515 ]]; then
    CLEANED_DATA_PATH="/mnt/data4tb/datasets/hrm-text-data-io-cleaned-20260515"
  else
    CLEANED_DATA_PATH="/mnt/sdc1/datasets/hrm-text-data-io-cleaned-20260515"
  fi
fi

TOKENIZER_PATH="${TOKENIZER_PATH:-${DATA_IO_DIR}/trained_tokenizers/bpe/tokenizer.json}"
PREFIX_CONFIG_PATH="${PREFIX_CONFIG_PATH:-${DATA_IO_DIR}/prefix_config.yaml}"
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"

PROFILE="${PROFILE:-reasoning_nonflan}"
RUN_NAME="${RUN_NAME:-stage93_hrm_text_${PROFILE}}"
WORK_DIR="${WORK_DIR:-${ROOT}/local_eval/${RUN_NAME}_dataio}"
SUBSET_DIR="${SUBSET_DIR:-${WORK_DIR}/cleaned_subset}"
TOKENIZED_OUT="${TOKENIZED_OUT:-${WORK_DIR}/tokenized}"
SAMPLED_OUT="${SAMPLED_OUT:-${WORK_DIR}/sampled}"
ANALYTICS_OUT="${ANALYTICS_OUT:-${WORK_DIR}/show_analytics.md}"
LOG="${LOG:-/tmp/${RUN_NAME}_dataio.log}"

EPOCHS="${EPOCHS:-5}"
CONTEXT_SIZE="${CONTEXT_SIZE:-1025}"
MIN_RESP_LENGTH="${MIN_RESP_LENGTH:-2}"

# This is the first high-probability shelf: reasoning/math/instruction-heavy
# data without the 264GB FLAN bulk. FLAN can be added after this shard is proven.
case "${PROFILE}" in
  reasoning_nonflan)
    DEFAULT_INCLUDE_CLUSTERS="SYNTH acereason ampsmathematica dmmath openmathinstruct2 openthoughts2 sudoku_extreme tasksource textbookreasoning"
    DEFAULT_INCLUDE_FLAN="0"
    DEFAULT_FLAN_MAX_FILES="0"
    ;;
  full_curriculum)
    # "Full" means every major shelf is represented, not that repetitive FLAN
    # rows are allowed to dominate the first data build.
    DEFAULT_INCLUDE_CLUSTERS="SYNTH acereason ampsmathematica dmmath openmathinstruct2 openthoughts2 sudoku_extreme tasksource textbookreasoning"
    DEFAULT_INCLUDE_FLAN="1"
    DEFAULT_FLAN_MAX_FILES="64"
    DEFAULT_FLAN_INCLUDE_REGEX=""
    ;;
  multilingual_curriculum)
    # Add targeted multilingual/translation shelves without letting the 264GB
    # FLAN bulk dominate the first broad-data build.
    DEFAULT_INCLUDE_CLUSTERS="SYNTH acereason ampsmathematica dmmath openmathinstruct2 openthoughts2 sudoku_extreme tasksource textbookreasoning"
    DEFAULT_INCLUDE_FLAN="1"
    DEFAULT_FLAN_MAX_FILES="all"
    DEFAULT_FLAN_INCLUDE_REGEX="translate|translation|wmt|xquad|wiki_lingua|cc_alligned|xnli|xquad|tydi|mlqa|paws-x|xstory"
    ;;
  full_literal)
    # Use only when disk/time budget is explicitly accepted.
    DEFAULT_INCLUDE_CLUSTERS="SYNTH acereason ampsmathematica dmmath openmathinstruct2 openthoughts2 sudoku_extreme tasksource textbookreasoning"
    DEFAULT_INCLUDE_FLAN="1"
    DEFAULT_FLAN_MAX_FILES="all"
    DEFAULT_FLAN_INCLUDE_REGEX=""
    ;;
  *)
    echo "unknown PROFILE=${PROFILE}; expected reasoning_nonflan, full_curriculum, multilingual_curriculum, or full_literal" >&2
    exit 2
    ;;
esac

INCLUDE_DATA_DIR="${INCLUDE_DATA_DIR:-1}"
INCLUDE_CLUSTERS="${INCLUDE_CLUSTERS:-${DEFAULT_INCLUDE_CLUSTERS}}"
INCLUDE_FLAN="${INCLUDE_FLAN:-${DEFAULT_INCLUDE_FLAN}}"
FLAN_MAX_FILES="${FLAN_MAX_FILES:-${DEFAULT_FLAN_MAX_FILES}}"
FLAN_INCLUDE_REGEX="${FLAN_INCLUDE_REGEX:-${DEFAULT_FLAN_INCLUDE_REGEX:-}}"

TOKENIZER_DIR="${DATA_IO_DIR}/tokenizer"
TOKENIZER_BIN="${TOKENIZER_DIR}/target/release/tokenizer"

print_plan() {
  cat <<PLAN
Stage93 HRM-Text large Data-IO preparation

actions: plan status link tokenize sample all launch

ROOT=${ROOT}
DATA_IO_DIR=${DATA_IO_DIR}
CLEANED_DATA_PATH=${CLEANED_DATA_PATH}
TOKENIZER_PATH=${TOKENIZER_PATH}
PREFIX_CONFIG_PATH=${PREFIX_CONFIG_PATH}
PYTHON=${PYTHON}

PROFILE=${PROFILE}
RUN_NAME=${RUN_NAME}
WORK_DIR=${WORK_DIR}
SUBSET_DIR=${SUBSET_DIR}
TOKENIZED_OUT=${TOKENIZED_OUT}
SAMPLED_OUT=${SAMPLED_OUT}
ANALYTICS_OUT=${ANALYTICS_OUT}
LOG=${LOG}

EPOCHS=${EPOCHS}
CONTEXT_SIZE=${CONTEXT_SIZE}
MIN_RESP_LENGTH=${MIN_RESP_LENGTH}
INCLUDE_DATA_DIR=${INCLUDE_DATA_DIR}
INCLUDE_CLUSTERS=${INCLUDE_CLUSTERS}
INCLUDE_FLAN=${INCLUDE_FLAN}
FLAN_MAX_FILES=${FLAN_MAX_FILES}
FLAN_INCLUDE_REGEX=${FLAN_INCLUDE_REGEX}

Profiles:
  reasoning_nonflan:
    Build a large reasoning/instruction-heavy shard and exclude the 264GB FLAN
    cluster at first so disk and time stay controlled.
  full_curriculum:
    Represent every major shelf, including a capped FLAN slice. This is the
    default "use all data types" mode.
  multilingual_curriculum:
    Represent every major shelf plus targeted FLAN translation/multilingual
    files. This is the first "make multilingual measurable" mode.
  full_literal:
    Include all FLAN files too. Use only when disk/time budget is explicitly
    accepted.

Default mechanics:
  Use hardlinks, not copies, for selected source files.

Use on DGX:
  CLEANED_DATA_PATH=/mnt/data4tb/datasets/hrm-text-data-io-cleaned-20260515 \\
    bash scripts/535_prepare_stage93_hrm_text_large_dataio.sh launch
PLAN
}

require_paths() {
  if [[ ! -d "${CLEANED_DATA_PATH}" ]]; then
    echo "missing CLEANED_DATA_PATH: ${CLEANED_DATA_PATH}" >&2
    exit 2
  fi
  if [[ ! -f "${TOKENIZER_PATH}" ]]; then
    echo "missing TOKENIZER_PATH: ${TOKENIZER_PATH}" >&2
    exit 2
  fi
}

link_one_file() {
  local src="$1"
  local rel="${src#${CLEANED_DATA_PATH}/}"
  local dst="${SUBSET_DIR}/${rel}"
  mkdir -p "$(dirname "${dst}")"
  if [[ -e "${dst}" ]]; then
    return
  fi
  ln "${src}" "${dst}"
}

link_tree_files() {
  local tree="$1"
  if [[ ! -d "${tree}" ]]; then
    echo "skip missing tree: ${tree}" >&2
    return
  fi
  while IFS= read -r -d '' file; do
    link_one_file "${file}"
  done < <(find "${tree}" -type f \( -name '*.jsonl' -o -name '*.parquet' \) -print0 | sort -z)
}

link_flan_slim() {
  local flan_dir="${CLEANED_DATA_PATH}/data_clustered/flan"
  local count=0
  if [[ "${INCLUDE_FLAN}" != "1" ]]; then
    return
  fi
  while IFS= read -r -d '' file; do
    if [[ -n "${FLAN_INCLUDE_REGEX}" ]] && ! [[ "${file}" =~ ${FLAN_INCLUDE_REGEX} ]]; then
      continue
    fi
    link_one_file "${file}"
    count=$((count + 1))
    if [[ "${FLAN_MAX_FILES}" != "all" ]] && (( count >= FLAN_MAX_FILES )); then
      break
    fi
  done < <(find "${flan_dir}" -type f -name '*.parquet' -print0 | sort -z)
}

run_link() {
  require_paths
  mkdir -p "${SUBSET_DIR}"
  if [[ "${INCLUDE_DATA_DIR}" == "1" ]]; then
    link_tree_files "${CLEANED_DATA_PATH}/data"
  fi
  local cluster
  for cluster in ${INCLUDE_CLUSTERS}; do
    link_tree_files "${CLEANED_DATA_PATH}/data_clustered/${cluster}"
  done
  link_flan_slim
}

ensure_tokenizer() {
  if [[ -x "${TOKENIZER_BIN}" ]]; then
    return
  fi
  export PATH="${HOME}/.cargo/bin:${PATH}"
  if ! command -v cargo >/dev/null 2>&1; then
    echo "cargo not found; install Rust/Cargo or add it to PATH before tokenization" >&2
    exit 2
  fi
  (cd "${TOKENIZER_DIR}" && cargo build --release --bin tokenizer)
}

run_tokenize() {
  run_link
  ensure_tokenizer
  mkdir -p "${TOKENIZED_OUT}"
  "${TOKENIZER_BIN}" "${SUBSET_DIR}" \
    --output-dir "${TOKENIZED_OUT}" \
    --tokenizer-path "${TOKENIZER_PATH}"
}

run_sample() {
  if [[ ! -f "${TOKENIZED_OUT}/tokenizer_info.json" ]]; then
    echo "tokenized output missing: ${TOKENIZED_OUT}/tokenizer_info.json" >&2
    echo "run tokenize first, or use action all" >&2
    exit 2
  fi
  rm -rf "${SAMPLED_OUT}"
  mkdir -p "$(dirname "${SAMPLED_OUT}")"
  (
    cd "${DATA_IO_DIR}"
    "${PYTHON}" sample_tokenized.py \
      "tokenized_path=${TOKENIZED_OUT}" \
      "output_path=${SAMPLED_OUT}" \
      "prefix_config_path=${PREFIX_CONFIG_PATH}" \
      "epochs=${EPOCHS}" \
      "context_size=${CONTEXT_SIZE}" \
      "min_resp_length=${MIN_RESP_LENGTH}" \
      > "${ANALYTICS_OUT}"
  )
}

run_all() {
  run_tokenize
  run_sample
}

run_launch() {
  mkdir -p "${WORK_DIR}"
  nohup env \
    ROOT="${ROOT}" \
    DATA_IO_DIR="${DATA_IO_DIR}" \
    CLEANED_DATA_PATH="${CLEANED_DATA_PATH}" \
    TOKENIZER_PATH="${TOKENIZER_PATH}" \
    PREFIX_CONFIG_PATH="${PREFIX_CONFIG_PATH}" \
    PYTHON="${PYTHON}" \
    PROFILE="${PROFILE}" \
    RUN_NAME="${RUN_NAME}" \
    WORK_DIR="${WORK_DIR}" \
    SUBSET_DIR="${SUBSET_DIR}" \
    TOKENIZED_OUT="${TOKENIZED_OUT}" \
    SAMPLED_OUT="${SAMPLED_OUT}" \
    ANALYTICS_OUT="${ANALYTICS_OUT}" \
    LOG="${LOG}" \
    EPOCHS="${EPOCHS}" \
    CONTEXT_SIZE="${CONTEXT_SIZE}" \
    MIN_RESP_LENGTH="${MIN_RESP_LENGTH}" \
    INCLUDE_DATA_DIR="${INCLUDE_DATA_DIR}" \
    INCLUDE_CLUSTERS="${INCLUDE_CLUSTERS}" \
    INCLUDE_FLAN="${INCLUDE_FLAN}" \
    FLAN_MAX_FILES="${FLAN_MAX_FILES}" \
    FLAN_INCLUDE_REGEX="${FLAN_INCLUDE_REGEX}" \
    bash "${BASH_SOURCE[0]}" all > "${LOG}" 2>&1 &
  echo "STAGE93_DATAIO_LAUNCHED:$!"
  echo "LOG:${LOG}"
  echo "SAMPLED_OUT:${SAMPLED_OUT}"
}

print_status() {
  print_plan
  echo
  echo "status:"
  [[ -d "${CLEANED_DATA_PATH}" ]] && du -sh "${CLEANED_DATA_PATH}" || true
  [[ -d "${SUBSET_DIR}" ]] && { find "${SUBSET_DIR}" -type f | wc -l | sed 's/^/subset_files=/'; du -sh "${SUBSET_DIR}" || true; }
  [[ -d "${TOKENIZED_OUT}" ]] && { find "${TOKENIZED_OUT}" -type f -name 'tokens.npy' | wc -l | sed 's/^/tokenized_tasks=/'; du -sh "${TOKENIZED_OUT}" || true; }
  [[ -d "${SAMPLED_OUT}" ]] && { find "${SAMPLED_OUT}" -maxdepth 2 -type f | wc -l | sed 's/^/sampled_files=/'; du -sh "${SAMPLED_OUT}" || true; }
  [[ -f "${ANALYTICS_OUT}" ]] && tail -n 40 "${ANALYTICS_OUT}" || true
  [[ -f "${LOG}" ]] && tail -n 40 "${LOG}" || true
}

case "${ACTION}" in
  plan)
    print_plan
    ;;
  status)
    print_status
    ;;
  link)
    run_link
    ;;
  tokenize)
    run_tokenize
    ;;
  sample)
    run_sample
    ;;
  all)
    run_all
    ;;
  launch)
    run_launch
    ;;
  *)
    echo "unknown action: ${ACTION}" >&2
    print_plan >&2
    exit 2
    ;;
esac
