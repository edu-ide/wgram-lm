#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

export PYTHONPATH=${PYTHONPATH:-$PWD/src}
export HF_HOME=${HF_HOME:-~/.cache/huggingface}

CONFIG="${CONFIG:-configs/qwen35_2b_4090_controller_trace_s050.yaml}"
SOURCE_JSONL="${SOURCE_JSONL:-data/filtered/memory_reasoning_synth_traces.jsonl}"
TRACE_JSONL="${TRACE_JSONL:-data/filtered/asi_controller_trace_replay.jsonl}"
OUT_DIR="${OUT_DIR:-runs/qwen35_2b_4090_controller_trace_s050}"
CHECKPOINT="${CHECKPOINT:-$OUT_DIR/last.pt}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-runs/qwen35_2b_4090_evidence_span_reader_s050/last.pt}"
SAVE_EVERY="${SAVE_EVERY:-0}"
DIAG_EVERY="${DIAG_EVERY:-0}"
REBUILD_TRACE_JSONL="${REBUILD_TRACE_JSONL:-1}"

echo "============================================================"
echo "Controller trace-replay training"
echo "config=${CONFIG}"
echo "source=${SOURCE_JSONL}"
echo "trace_data=${TRACE_JSONL}"
echo "rebuild_trace_data=${REBUILD_TRACE_JSONL}"
echo "init=${INIT_CHECKPOINT}"
echo "out_dir=${OUT_DIR}"
echo "============================================================"

if [[ "$REBUILD_TRACE_JSONL" == "1" && ! -f "$SOURCE_JSONL" ]]; then
  echo "Missing source data: $SOURCE_JSONL" >&2
  exit 1
fi
if [[ ! -f "$INIT_CHECKPOINT" ]]; then
  echo "Missing init checkpoint: $INIT_CHECKPOINT" >&2
  exit 1
fi

if [[ "$REBUILD_TRACE_JSONL" == "1" ]]; then
  python scripts/155_build_controller_trace_replay.py \
    --input-jsonl "$SOURCE_JSONL" \
    --output-jsonl "$TRACE_JSONL"
elif [[ ! -f "$TRACE_JSONL" ]]; then
  echo "Missing prebuilt trace data: $TRACE_JSONL" >&2
  exit 1
else
  echo "Using prebuilt trace data: $TRACE_JSONL"
fi

MULTIMODAL=0 DATA_JSONL="$TRACE_JSONL" bash scripts/08_train_donor_adapter.sh "$CONFIG" \
  --init-checkpoint "$INIT_CHECKPOINT" \
  --save-every "$SAVE_EVERY" \
  --diag-every "$DIAG_EVERY"

echo "wrote $CHECKPOINT"
