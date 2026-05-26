#!/usr/bin/env bash
set -euo pipefail

# Run the language gates for a native PrefixLM checkpoint.
#
# Plain-language contract:
#   - heldout loss asks whether the model can do unseen language "dictation";
#   - generation probe asks whether it can answer without echo/repetition;
#   - by default, do not steal the DGX GPU from an active training run.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${ROOT}"

RUN_NAME="${RUN_NAME:-20260524_PREFIXLM_LANGUAGE_GATES}"
OUT_DIR="${OUT_DIR:-${ROOT}/local_eval/${RUN_NAME}}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${OUT_DIR}/tensorboard}"
PYTHON="${PYTHON:-/mnt/data4tb/venv_sglang_pr23000/bin/python}"
CHECKPOINT="${CHECKPOINT:-${ROOT}/local_eval/20260524_STAGE93A00_DGX913M_MICRO_HARDLINK_TO24500/last_model.pt}"
DEVICE="${DEVICE:-cuda}"
REQUIRED_TRITON_PTXAS_PATH="${REQUIRED_TRITON_PTXAS_PATH:-}"
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
export TRITON_PTXAS_PATH
HELDOUT_JSONL="${HELDOUT_JSONL:-${ROOT}/data/eval/prefixlm_language_heldout.jsonl}"
PROBE_JSONL="${PROBE_JSONL:-${ROOT}/data/eval/prefixlm_multilingual_probe.jsonl}"
RAW_PROBE_JSONL="${RAW_PROBE_JSONL:-${ROOT}/data/eval/prefixlm_raw_intelligence_probe.jsonl}"
GENERAL_HELDOUT_JSONL="${GENERAL_HELDOUT_JSONL:-${ROOT}/data/eval/prefixlm_general_language_heldout.jsonl}"
GENERAL_PROBE_JSONL="${GENERAL_PROBE_JSONL:-${ROOT}/data/eval/prefixlm_general_language_generation_probe.jsonl}"
GENERAL_MAX_NEW_TOKENS="${GENERAL_MAX_NEW_TOKENS:-48}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "checkpoint missing: ${CHECKPOINT}" >&2
  exit 2
fi

if [[ "${FORCE:-0}" != "1" ]]; then
  if pgrep -af "scripts/534_train_native_prefixlm_dataio|534_train_native_prefixlm_dataio_stage90.py" >/dev/null; then
    echo "LANGUAGE_GATE_WAITING: PrefixLM training is active; not stealing the GPU." >&2
    pgrep -af "scripts/534_train_native_prefixlm_dataio|534_train_native_prefixlm_dataio_stage90.py" >&2 || true
    exit 3
  fi
fi

mkdir -p "${OUT_DIR}"

"${PYTHON}" scripts/544_eval_prefixlm_language_heldout_loss.py \
  --checkpoint "${CHECKPOINT}" \
  --heldout-jsonl "${HELDOUT_JSONL}" \
  --device "${DEVICE}" \
  --out "${OUT_DIR}/language_heldout_loss.json"

"${PYTHON}" scripts/544_eval_prefixlm_language_heldout_loss.py \
  --checkpoint "${CHECKPOINT}" \
  --heldout-jsonl "${GENERAL_HELDOUT_JSONL}" \
  --device "${DEVICE}" \
  --out "${OUT_DIR}/general_language_heldout_loss.json"

"${PYTHON}" scripts/542_eval_prefixlm_multilingual_probe.py \
  --checkpoint "${CHECKPOINT}" \
  --probe-jsonl "${GENERAL_PROBE_JSONL}" \
  --device "${DEVICE}" \
  --max-new-tokens "${GENERAL_MAX_NEW_TOKENS}" \
  --out "${OUT_DIR}/general_language_generation_probe.json"

"${PYTHON}" scripts/542_eval_prefixlm_multilingual_probe.py \
  --checkpoint "${CHECKPOINT}" \
  --probe-jsonl "${PROBE_JSONL}" \
  --device "${DEVICE}" \
  --out "${OUT_DIR}/multilingual_generation_probe.json"

"${PYTHON}" scripts/546_eval_prefixlm_raw_intelligence_suite.py \
  --checkpoint "${CHECKPOINT}" \
  --probe-jsonl "${RAW_PROBE_JSONL}" \
  --device "${DEVICE}" \
  --out "${OUT_DIR}/raw_intelligence_suite.json"

"${PYTHON}" scripts/547_write_prefixlm_raw_intelligence_tensorboard.py \
  --raw-json "${OUT_DIR}/language_heldout_loss.json" \
  --tensorboard-dir "${TENSORBOARD_DIR}" \
  --prefix eval/language_heldout

"${PYTHON}" scripts/547_write_prefixlm_raw_intelligence_tensorboard.py \
  --raw-json "${OUT_DIR}/general_language_heldout_loss.json" \
  --tensorboard-dir "${TENSORBOARD_DIR}" \
  --prefix eval/general_language_heldout

"${PYTHON}" scripts/547_write_prefixlm_raw_intelligence_tensorboard.py \
  --raw-json "${OUT_DIR}/general_language_generation_probe.json" \
  --tensorboard-dir "${TENSORBOARD_DIR}" \
  --prefix eval/general_language_generation

"${PYTHON}" scripts/547_write_prefixlm_raw_intelligence_tensorboard.py \
  --raw-json "${OUT_DIR}/multilingual_generation_probe.json" \
  --tensorboard-dir "${TENSORBOARD_DIR}" \
  --prefix eval/multilingual_generation

"${PYTHON}" scripts/547_write_prefixlm_raw_intelligence_tensorboard.py \
  --raw-json "${OUT_DIR}/raw_intelligence_suite.json" \
  --tensorboard-dir "${TENSORBOARD_DIR}" \
  --prefix eval/raw_intelligence

echo "LANGUAGE_GATES_DONE:${OUT_DIR}"
echo "HELDOUT_LOSS:${OUT_DIR}/language_heldout_loss.json"
echo "GENERAL_LANGUAGE_HELDOUT:${OUT_DIR}/general_language_heldout_loss.json"
echo "GENERAL_LANGUAGE_GENERATION:${OUT_DIR}/general_language_generation_probe.json"
echo "GENERATION_PROBE:${OUT_DIR}/multilingual_generation_probe.json"
echo "RAW_INTELLIGENCE_SUITE:${OUT_DIR}/raw_intelligence_suite.json"
echo "TENSORBOARD:${TENSORBOARD_DIR}"
