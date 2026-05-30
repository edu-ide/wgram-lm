#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [[ -n "${PYTHON:-}" ]]; then
  PYTHON="${PYTHON}"
elif [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/.venv/bin/python"
else
  PYTHON="python3"
fi
if ! "${PYTHON}" -c 'import pyarrow' >/dev/null 2>&1; then
  if [[ -x "/mnt/data4tb/venv_sglang_pr23000/bin/python" ]] \
    && /mnt/data4tb/venv_sglang_pr23000/bin/python -c 'import pyarrow' >/dev/null 2>&1; then
    PYTHON="/mnt/data4tb/venv_sglang_pr23000/bin/python"
  fi
fi
CLEANED_DATA_PATH="${CLEANED_DATA_PATH:-/mnt/data4tb/datasets/hrm-text-data-io-cleaned-20260515}"
WORK_DIR="${WORK_DIR:-${ROOT}/local_eval/stage95_blt_foundation_byte_curriculum}"
SAMPLED_OUT="${SAMPLED_OUT:-${WORK_DIR}/sampled}"
EPOCHS="${EPOCHS:-3}"
MAX_ROWS="${MAX_ROWS:-240000}"
MAX_ROWS_PER_FILE="${MAX_ROWS_PER_FILE:-20000}"
MAX_SCAN_ROWS_PER_FILE="${MAX_SCAN_ROWS_PER_FILE:-60000}"
MAX_INST_BYTES="${MAX_INST_BYTES:-2048}"
MAX_RESP_BYTES="${MAX_RESP_BYTES:-1536}"
SEED="${SEED:-9595}"

# Keep JSONL shelves explicit and broad.  These are stable across local/DGX
# cleaned HRM-Text snapshots and prevent Stage95 from becoming math-only.
SOURCE_FILES="${SOURCE_FILES:-data/no_robots.jsonl data/natural_reasoning.jsonl data/webinstruct_verified.jsonl data/gsm8k_train.jsonl data/math_train.jsonl data/numinamath.jsonl data/omnimath.jsonl data/Platypus/openbookqa.jsonl data/Platypus/arb_physics.jsonl data/Platypus/arb_law.jsonl data/raw/ultradata_sft_2605_math_train.jsonl data/raw/ultradata_sft_2605_multi_lang_math_train.jsonl data/raw/ultradata_sft_2605_if_train.jsonl data/raw/ultradata_sft_2605_code_train.jsonl data/raw/ultradata_sft_2605_knowledge_train.jsonl data/raw/ultradata_sft_2605_multi_lang_knowledge_train.jsonl}"

# Parquet shelves carry most of the DGX data.  FLAN is filtered by the builder's
# row caps; it is included here because Stage95 must learn ordinary language and
# multilingual text, not only reasoning traces.
SOURCE_GLOBS="${SOURCE_GLOBS:-data_clustered/SYNTH/*.parquet data_clustered/acereason/*.parquet data_clustered/textbookreasoning/*.parquet data_clustered/openmathinstruct2/*.parquet data_clustered/openthoughts2/*.parquet data_clustered/tasksource/*.parquet data_clustered/flan/*.parquet}"

# File count is not data quality. Stage95's broad sample must not let many-shard
# SYNTH/FLAN shelves crowd out ordinary language. These quotas make the first
# foundation diet closer to HRM-Text/Data-IO: read and speak first, then reason.
SOURCE_BUCKET_QUOTAS="${SOURCE_BUCKET_QUOTAS:-general_instruction=30000 tasksource=50000 flan_instruction_qa=50000 flan_multilingual_translation=30000 flan_other=20000 reasoning=30000 math=20000 synthetic_math_like=10000}"
SOURCE_BUCKET_MAX_ROWS_PER_FILE="${SOURCE_BUCKET_MAX_ROWS_PER_FILE:-general_instruction=10000 tasksource=800 flan_instruction_qa=500 flan_multilingual_translation=500 flan_other=300 reasoning=5000 math=3000 synthetic_math_like=20}"
SELECTION_MODE="${SELECTION_MODE:-first}"
UTILITY_SCORE_JSONL="${UTILITY_SCORE_JSONL:-}"
UTILITY_TEMPERATURE="${UTILITY_TEMPERATURE:-0.0}"
OPUS_CHECKPOINT="${OPUS_CHECKPOINT:-}"
OPUS_PROXY_JSONL="${OPUS_PROXY_JSONL:-${ROOT}/data/eval/prefixlm_language_heldout.jsonl ${ROOT}/data/eval/official_gdsuite_choice_probe.jsonl}"
OPUS_SCORE_OUT="${OPUS_SCORE_OUT:-${WORK_DIR}/opus_projected_utility_scores.jsonl}"
OPUS_REPORT_OUT="${OPUS_REPORT_OUT:-${WORK_DIR}/opus_projected_utility_report.json}"
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

usage() {
  cat <<'USAGE'
Stage95 BLT foundation byte sample

Actions:
  plan    Print the broad Stage95 data contract.
  status  Show expected paths and existing sample files.
  score-opus
          Build OPUS projected-utility scores for candidate source rows.
  build   Build a tokenizer-free UTF-8 byte PrefixLM sample.

Plain-language contract:
  Stage95 is not a math-only model and not an agentic/tool-trace model.
  It is a from-scratch BLT foundation sample:
    general language
    reasoning
    math
    multilingual
    memory/context

  agentic/tool traces are later.  First grow a model that can read, think, and
  speak ordinary Korean/English-centered text through the normal byte LM path.
USAGE
}

plan() {
  usage
  cat <<PLAN

Configuration:
  ROOT=${ROOT}
  PYTHON=${PYTHON}
  CLEANED_DATA_PATH=${CLEANED_DATA_PATH}
  WORK_DIR=${WORK_DIR}
  SAMPLED_OUT=${SAMPLED_OUT}
  EPOCHS=${EPOCHS}
  MAX_ROWS=${MAX_ROWS}
  MAX_ROWS_PER_FILE=${MAX_ROWS_PER_FILE}
  MAX_SCAN_ROWS_PER_FILE=${MAX_SCAN_ROWS_PER_FILE}
  MAX_INST_BYTES=${MAX_INST_BYTES}
  MAX_RESP_BYTES=${MAX_RESP_BYTES}
  SEED=${SEED}
  SOURCE_BUCKET_QUOTAS=${SOURCE_BUCKET_QUOTAS}
  SOURCE_BUCKET_MAX_ROWS_PER_FILE=${SOURCE_BUCKET_MAX_ROWS_PER_FILE}
  SELECTION_MODE=${SELECTION_MODE}
  UTILITY_SCORE_JSONL=${UTILITY_SCORE_JSONL}
  UTILITY_TEMPERATURE=${UTILITY_TEMPERATURE}
  OPUS_CHECKPOINT=${OPUS_CHECKPOINT:-<none>}
  OPUS_PROXY_JSONL=${OPUS_PROXY_JSONL}
  OPUS_SCORE_OUT=${OPUS_SCORE_OUT}
  OPUS_REPORT_OUT=${OPUS_REPORT_OUT}
  OPUS_CANDIDATE_MAX_ROWS=${OPUS_CANDIDATE_MAX_ROWS}
  OPUS_PROXY_MAX_ROWS=${OPUS_PROXY_MAX_ROWS}
  OPUS_PROXY_MAX_ROWS_PER_GROUP=${OPUS_PROXY_MAX_ROWS_PER_GROUP}
  OPUS_PROXY_GROUPING=${OPUS_PROXY_GROUPING}
  OPUS_PROXY_SCORE_MODE=${OPUS_PROXY_SCORE_MODE}
  OPUS_PROXY_MEAN_WEIGHT=${OPUS_PROXY_MEAN_WEIGHT}
	  OPUS_PROJECTION_DIM=${OPUS_PROJECTION_DIM}
	  OPUS_PRECONDITIONER=${OPUS_PRECONDITIONER}
	  OPUS_PARAM_NAME_REGEX=${OPUS_PARAM_NAME_REGEX}
	  OPUS_DEVICE=${OPUS_DEVICE}

Source files:
  ${SOURCE_FILES}

Source globs passed to scripts/555_prepare_byte_prefixlm_sample.py --source-globs:
  ${SOURCE_GLOBS}

Required shelves:
  general language:
    no_robots, webinstruct, tasksource, FLAN dialogue/QA/summarization rows
  reasoning:
    natural_reasoning, webinstruct_verified, acereason, textbookreasoning
  math:
    gsm8k, math, numinamath, omnimath, openmathinstruct2, SYNTH
  multilingual:
    FLAN translation/XQuAD/MLQA/TyDi/XNLI-style rows when present
  memory/context:
    long QA, summarization, evidence-like tasksource/FLAN rows

Balanced target:
  general/instruction/multilingual/tasksource dominate the language-body phase;
  synthetic math is kept as a small spice, not the whole meal.

OPUS projected-utility selection:
  By default SELECTION_MODE=first preserves the static HRM-Text/Data-IO sample.
  Set SELECTION_MODE=utility with UTILITY_SCORE_JSONL to materialize an
  OPUS selected window. If UTILITY_SCORE_JSONL is empty and OPUS_CHECKPOINT is
  set, build calls score-opus first. The scorer ranks rows by whether their
  AdamW-shaped projected update points toward the heldout proxy direction.
	  The default proxy combines ordinary language heldout rows with
	  Generalization Dynamics anti-parrot rows, so OPUS selection is not just
	  low-loss sampling.
	  By default OPUS scores the byte reader/chunker/speaker path instead of every
	  1B-scale global-core tensor. That keeps the data audition causal enough for
	  language-body selection while making the overnight automation finish.

Actions:
  bash scripts/558_prepare_stage95_blt_foundation_byte_sample_dgx.sh plan
  bash scripts/558_prepare_stage95_blt_foundation_byte_sample_dgx.sh status
  bash scripts/558_prepare_stage95_blt_foundation_byte_sample_dgx.sh score-opus
  bash scripts/558_prepare_stage95_blt_foundation_byte_sample_dgx.sh build
PLAN
}

status() {
  echo "CLEANED_DATA_PATH=${CLEANED_DATA_PATH}"
  echo "SAMPLED_OUT=${SAMPLED_OUT}"
  if [[ -d "${CLEANED_DATA_PATH}" ]]; then
    echo "cleaned_data=present"
  else
    echo "cleaned_data=missing"
  fi
  if [[ -f "${SAMPLED_OUT}/metadata.json" ]]; then
    echo "sample_metadata=present"
    "${PYTHON}" - <<PY
import json
from pathlib import Path
p = Path("${SAMPLED_OUT}") / "metadata.json"
data = json.loads(p.read_text())
print("rows=", data.get("rows"))
print("total_length=", data.get("total_length"))
print("source_files=", len(data.get("source_files", [])))
print("accepted_by_bucket=", data.get("accepted_by_bucket"))
PY
  else
    echo "sample_metadata=missing"
  fi
  echo "opus_checkpoint=${OPUS_CHECKPOINT:-<none>}"
  echo "opus_proxy_jsonl=${OPUS_PROXY_JSONL}"
  echo "opus_score_out=${OPUS_SCORE_OUT}"
  echo "opus_scores=$([[ -f "${OPUS_SCORE_OUT}" ]] && echo present || echo missing)"
  echo "opus_report=$([[ -f "${OPUS_REPORT_OUT}" ]] && echo present || echo missing)"
}

validate_opus_proxy_jsonl() {
  local proxy
  local missing=0
  for proxy in ${OPUS_PROXY_JSONL//,/ }; do
    if [[ -z "${proxy}" ]]; then
      continue
    fi
    if [[ ! -f "${proxy}" ]]; then
      echo "missing OPUS_PROXY_JSONL entry: ${proxy}" >&2
      missing=1
    fi
  done
  if [[ "${missing}" -ne 0 ]]; then
    return 1
  fi
}

score_opus() {
  if [[ -z "${OPUS_CHECKPOINT}" ]]; then
    echo "missing OPUS_CHECKPOINT: OPUS scoring needs a current checkpoint with optimizer state" >&2
    exit 3
  fi
  if [[ ! -f "${OPUS_CHECKPOINT}" ]]; then
    echo "missing OPUS_CHECKPOINT file: ${OPUS_CHECKPOINT}" >&2
    exit 3
  fi
  if ! validate_opus_proxy_jsonl; then
    echo "missing OPUS_PROXY_JSONL: ${OPUS_PROXY_JSONL}" >&2
    exit 3
  fi
  mkdir -p "$(dirname "${OPUS_SCORE_OUT}")"
  "${PYTHON}" "${ROOT}/scripts/614_score_opus_projected_utility.py" \
    --checkpoint "${OPUS_CHECKPOINT}" \
    --cleaned-data-root "${CLEANED_DATA_PATH}" \
    --source-files "${SOURCE_FILES}" \
    --source-globs "${SOURCE_GLOBS}" \
    --proxy-jsonl "${OPUS_PROXY_JSONL}" \
    --out "${OPUS_SCORE_OUT}" \
    --report-out "${OPUS_REPORT_OUT}" \
    --candidate-max-rows "${OPUS_CANDIDATE_MAX_ROWS}" \
    --candidate-max-scan-rows-per-file "${OPUS_CANDIDATE_MAX_SCAN_ROWS_PER_FILE}" \
    --proxy-max-rows "${OPUS_PROXY_MAX_ROWS}" \
    --proxy-max-rows-per-group "${OPUS_PROXY_MAX_ROWS_PER_GROUP}" \
    --proxy-grouping "${OPUS_PROXY_GROUPING}" \
    --proxy-score-mode "${OPUS_PROXY_SCORE_MODE}" \
    --proxy-mean-weight "${OPUS_PROXY_MEAN_WEIGHT}" \
    --max-inst-bytes "${MAX_INST_BYTES}" \
    --max-resp-bytes "${MAX_RESP_BYTES}" \
    --projection-dim "${OPUS_PROJECTION_DIM}" \
    --preconditioner "${OPUS_PRECONDITIONER}" \
    --param-name-regex "${OPUS_PARAM_NAME_REGEX}" \
    --redundancy-weight "${OPUS_REDUNDANCY_WEIGHT}" \
    --device "${OPUS_DEVICE}"
}

build() {
  if [[ ! -d "${CLEANED_DATA_PATH}" ]]; then
    echo "missing CLEANED_DATA_PATH: ${CLEANED_DATA_PATH}" >&2
    exit 2
  fi
  local effective_utility_score_jsonl="${UTILITY_SCORE_JSONL}"
  if [[ "${SELECTION_MODE}" == "utility" && -z "${effective_utility_score_jsonl}" ]]; then
    if [[ -z "${OPUS_CHECKPOINT}" ]]; then
      echo "SELECTION_MODE=utility requires UTILITY_SCORE_JSONL or OPUS_CHECKPOINT" >&2
      exit 3
    fi
    score_opus
    effective_utility_score_jsonl="${OPUS_SCORE_OUT}"
  fi
  mkdir -p "$(dirname "${SAMPLED_OUT}")"
  "${PYTHON}" "${ROOT}/scripts/555_prepare_byte_prefixlm_sample.py" \
    --cleaned-data-root "${CLEANED_DATA_PATH}" \
    --source-files "${SOURCE_FILES}" \
    --source-globs "${SOURCE_GLOBS}" \
    --out "${SAMPLED_OUT}" \
    --epochs "${EPOCHS}" \
    --max-rows "${MAX_ROWS}" \
    --max-rows-per-file "${MAX_ROWS_PER_FILE}" \
    --max-scan-rows-per-file "${MAX_SCAN_ROWS_PER_FILE}" \
    --max-inst-bytes "${MAX_INST_BYTES}" \
    --max-resp-bytes "${MAX_RESP_BYTES}" \
    --bucket-quotas "${SOURCE_BUCKET_QUOTAS}" \
    --bucket-max-rows-per-file "${SOURCE_BUCKET_MAX_ROWS_PER_FILE}" \
    --selection-mode "${SELECTION_MODE}" \
    --utility-score-jsonl "${effective_utility_score_jsonl}" \
    --utility-temperature "${UTILITY_TEMPERATURE}" \
    --shuffle-epochs \
    --seed "${SEED}"
}

action="${1:-plan}"
case "${action}" in
  plan)
    plan
    ;;
  status)
    status
    ;;
  score-opus)
    score_opus
    ;;
  build)
    build
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
