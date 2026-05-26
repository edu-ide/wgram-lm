#!/usr/bin/env bash
set -euo pipefail

# Prepare a small HRM-Text/data_io-compatible sampled PrefixLM dataset.
#
# Plain-language contract:
#   cleaned instruction/response rows are the manuscript;
#   tokenization turns them into token streams plus boundaries;
#   sampling packs those boundaries into the HRM-Text training tensor layout.

ACTION="${1:-plan}"
ROOT="${ROOT:-/home/tripleyoung/qtrm-workspace/qtrm_multimodal_memoryos}"
DATA_IO_DIR="${DATA_IO_DIR:-${ROOT}/references/official/data_io}"
CLEANED_DATA_PATH="${CLEANED_DATA_PATH:-/mnt/sdc1/datasets/hrm-text-data-io-cleaned-20260515}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${DATA_IO_DIR}/trained_tokenizers/bpe/tokenizer.json}"
PREFIX_CONFIG_PATH="${PREFIX_CONFIG_PATH:-${DATA_IO_DIR}/prefix_config.yaml}"
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
WORK_DIR="${WORK_DIR:-/tmp/hrm_text_dataio_sample_${TS}}"
SUBSET_DIR="${SUBSET_DIR:-${WORK_DIR}/cleaned_subset}"
TOKENIZED_OUT="${TOKENIZED_OUT:-${WORK_DIR}/tokenized}"
SAMPLED_OUT="${SAMPLED_OUT:-${WORK_DIR}/sampled}"

SOURCE_FILES="${SOURCE_FILES:-data/gsm8k_train.jsonl data/math_train.jsonl data/omnimath.jsonl data/Platypus/openbookqa.jsonl}"
EPOCHS="${EPOCHS:-2}"
CONTEXT_SIZE="${CONTEXT_SIZE:-1025}"
MIN_RESP_LENGTH="${MIN_RESP_LENGTH:-2}"

TOKENIZER_DIR="${DATA_IO_DIR}/tokenizer"
TOKENIZER_BIN="${TOKENIZER_DIR}/target/release/tokenizer"

print_plan() {
  cat <<PLAN
HRM-Text Data-IO sample preparation

actions: plan status tokenize sample all

CLEANED_DATA_PATH=${CLEANED_DATA_PATH}
DATA_IO_DIR=${DATA_IO_DIR}
TOKENIZER_PATH=${TOKENIZER_PATH}
PREFIX_CONFIG_PATH=${PREFIX_CONFIG_PATH}
PYTHON=${PYTHON}

WORK_DIR=${WORK_DIR}
SUBSET_DIR=${SUBSET_DIR}
TOKENIZED_OUT=${TOKENIZED_OUT}
SAMPLED_OUT=${SAMPLED_OUT}

SOURCE_FILES=${SOURCE_FILES}
EPOCHS=${EPOCHS}
CONTEXT_SIZE=${CONTEXT_SIZE}
MIN_RESP_LENGTH=${MIN_RESP_LENGTH}

Use:
  bash scripts/533_prepare_hrm_text_dataio_sample.sh all

The output expected by HRM-Text-style PrefixLM loaders is:
  \${SAMPLED_OUT}/tokens.npy
  \${SAMPLED_OUT}/epoch_0/inst_start.npy
  \${SAMPLED_OUT}/epoch_0/inst_len.npy
  \${SAMPLED_OUT}/epoch_0/resp_start.npy
  \${SAMPLED_OUT}/epoch_0/resp_len.npy
  \${SAMPLED_OUT}/metadata.json
PLAN
}

prepare_subset() {
  rm -rf "${SUBSET_DIR}"
  mkdir -p "${SUBSET_DIR}"
  local rel src dst parent
  for rel in ${SOURCE_FILES}; do
    src="${CLEANED_DATA_PATH}/${rel}"
    if [[ ! -f "${src}" ]]; then
      echo "missing source file: ${src}" >&2
      exit 2
    fi
    dst="${SUBSET_DIR}/${rel}"
    parent="$(dirname "${dst}")"
    mkdir -p "${parent}"
    cp -f "${src}" "${dst}"
  done
}

ensure_tokenizer() {
  if [[ -x "${TOKENIZER_BIN}" ]]; then
    return
  fi
  (cd "${TOKENIZER_DIR}" && cargo build --release --bin tokenizer)
}

run_tokenize() {
  prepare_subset
  ensure_tokenizer
  rm -rf "${TOKENIZED_OUT}"
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
  (
    cd "${DATA_IO_DIR}"
    "${PYTHON}" sample_tokenized.py \
      "tokenized_path=${TOKENIZED_OUT}" \
      "output_path=${SAMPLED_OUT}" \
      "prefix_config_path=${PREFIX_CONFIG_PATH}" \
      "epochs=${EPOCHS}" \
      "context_size=${CONTEXT_SIZE}" \
      "min_resp_length=${MIN_RESP_LENGTH}"
  )
}

print_status() {
  print_plan
  echo
  echo "status:"
  [[ -d "${CLEANED_DATA_PATH}" ]] && du -sh "${CLEANED_DATA_PATH}" || true
  [[ -d "${TOKENIZED_OUT}" ]] && find "${TOKENIZED_OUT}" -maxdepth 2 -type f | sort | sed 's/^/  tokenized: /' || true
  [[ -d "${SAMPLED_OUT}" ]] && find "${SAMPLED_OUT}" -maxdepth 2 -type f | sort | sed 's/^/  sampled: /' || true
}

case "${ACTION}" in
  plan)
    print_plan
    ;;
  status)
    print_status
    ;;
  tokenize)
    run_tokenize
    ;;
  sample)
    run_sample
    ;;
  all)
    run_tokenize
    run_sample
    ;;
  *)
    echo "unknown action: ${ACTION}" >&2
    print_plan >&2
    exit 2
    ;;
esac
