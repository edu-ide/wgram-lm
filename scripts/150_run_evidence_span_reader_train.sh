#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

export PYTHONPATH=${PYTHONPATH:-$PWD/src}
export HF_HOME=${HF_HOME:-~/.cache/huggingface}

CONFIG="${CONFIG:-configs/qwen35_2b_4090_evidence_span_reader_s050.yaml}"
SOURCE_JSONL="${SOURCE_JSONL:-data/filtered/memory_reasoning_synth_traces.jsonl}"
SPAN_JSONL="${SPAN_JSONL:-data/filtered/memory_reasoning_synth_span_reader.jsonl}"
OUT_DIR="${OUT_DIR:-runs/qwen35_2b_4090_evidence_span_reader_s050}"
CHECKPOINT="${CHECKPOINT:-$OUT_DIR/last.pt}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-runs/qwen35_2b_4090_workspace_answer_bottleneck_causal_s050/last.pt}"
SAVE_EVERY="${SAVE_EVERY:-100}"
DIAG_EVERY="${DIAG_EVERY:-0}"
REBUILD_SPAN_JSONL="${REBUILD_SPAN_JSONL:-1}"

echo "============================================================"
echo "Evidence span-reader training"
echo "config=${CONFIG}"
echo "source=${SOURCE_JSONL}"
echo "span_data=${SPAN_JSONL}"
echo "rebuild_span_data=${REBUILD_SPAN_JSONL}"
echo "init=${INIT_CHECKPOINT}"
echo "out_dir=${OUT_DIR}"
echo "============================================================"

if [[ "$REBUILD_SPAN_JSONL" == "1" && ! -f "$SOURCE_JSONL" ]]; then
  echo "Missing source data: $SOURCE_JSONL" >&2
  exit 1
fi
if [[ ! -f "$INIT_CHECKPOINT" ]]; then
  echo "Missing init checkpoint: $INIT_CHECKPOINT" >&2
  exit 1
fi

if [[ "$REBUILD_SPAN_JSONL" == "1" ]]; then
  python scripts/build_evidence_span_reader_dataset.py \
    --input-jsonl "$SOURCE_JSONL" \
    --output-jsonl "$SPAN_JSONL"
elif [[ ! -f "$SPAN_JSONL" ]]; then
  echo "Missing prebuilt span data: $SPAN_JSONL" >&2
  exit 1
else
  echo "Using prebuilt span data: $SPAN_JSONL"
fi

MULTIMODAL=0 DATA_JSONL="$SPAN_JSONL" bash scripts/08_train_donor_adapter.sh "$CONFIG" \
  --init-checkpoint "$INIT_CHECKPOINT" \
  --save-every "$SAVE_EVERY" \
  --diag-every "$DIAG_EVERY"

echo "wrote $CHECKPOINT"
