#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

export PYTHONPATH=${PYTHONPATH:-$PWD/src}
export HF_HOME=${HF_HOME:-~/.cache/huggingface}

CONFIG="${CONFIG:-configs/qwen35_2b_4090_evidence_span_reader_truthcal_s300.yaml}"
SOURCE_JSONL="${SOURCE_JSONL:-data/filtered/memory_reasoning_synth_traces.jsonl}"
BASE_SPAN_JSONL="${BASE_SPAN_JSONL:-data/filtered/memory_reasoning_synth_span_reader.jsonl}"
MIX_JSONL="${MIX_JSONL:-data/filtered/memory_reasoning_synth_span_reader_truthcal.jsonl}"
HARD_NEGATIVE_CASES="${HARD_NEGATIVE_CASES:-data/eval/memory_reasoning_heldout_expanded_72.jsonl}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-runs/qwen35_2b_4090_evidence_span_reader_trainhardnegx2_s500/last.pt}"
OUT_DIR="${OUT_DIR:-runs/qwen35_2b_4090_evidence_span_reader_truthcal_s300}"
CHECKPOINT="${CHECKPOINT:-$OUT_DIR/last.pt}"
REBUILD_SPAN_JSONL="${REBUILD_SPAN_JSONL:-1}"
REBUILD_MIX_JSONL="${REBUILD_MIX_JSONL:-1}"
HARD_NEGATIVE_REPEAT="${HARD_NEGATIVE_REPEAT:-2}"
MAX_HARD_NEGATIVES="${MAX_HARD_NEGATIVES:-0}"
SAVE_EVERY="${SAVE_EVERY:-100}"
DIAG_EVERY="${DIAG_EVERY:-0}"
EVAL_AFTER_TRAIN="${EVAL_AFTER_TRAIN:-1}"
MAX_CASES="${MAX_CASES:-16}"

echo "============================================================"
echo "Truth-gate calibration training"
echo "config=${CONFIG}"
echo "source=${SOURCE_JSONL}"
echo "base_span=${BASE_SPAN_JSONL}"
echo "mix=${MIX_JSONL}"
echo "hard_negative_cases=${HARD_NEGATIVE_CASES}"
echo "init=${INIT_CHECKPOINT}"
echo "checkpoint=${CHECKPOINT}"
echo "============================================================"

if [[ ! -f "$INIT_CHECKPOINT" ]]; then
  echo "Missing init checkpoint: $INIT_CHECKPOINT" >&2
  exit 1
fi

if [[ "$REBUILD_SPAN_JSONL" == "1" ]]; then
  if [[ ! -f "$SOURCE_JSONL" ]]; then
    echo "Missing source data: $SOURCE_JSONL" >&2
    exit 1
  fi
  python scripts/build_evidence_span_reader_dataset.py \
    --input-jsonl "$SOURCE_JSONL" \
    --output-jsonl "$BASE_SPAN_JSONL"
elif [[ ! -f "$BASE_SPAN_JSONL" ]]; then
  echo "Missing prebuilt span data: $BASE_SPAN_JSONL" >&2
  exit 1
fi

if [[ "$REBUILD_MIX_JSONL" == "1" ]]; then
  if [[ ! -f "$HARD_NEGATIVE_CASES" ]]; then
    python scripts/110_build_expanded_memory_reasoning_heldout.py \
      --out "$HARD_NEGATIVE_CASES"
  fi
  python scripts/build_evidence_span_reader_training_mix.py \
    --base-span-jsonl "$BASE_SPAN_JSONL" \
    --output-jsonl "$MIX_JSONL" \
    --hard-negative-cases "$HARD_NEGATIVE_CASES" \
    --hard-negative-top-k 3 \
    --max-hard-negatives "$MAX_HARD_NEGATIVES" \
    --hard-negative-repeat "$HARD_NEGATIVE_REPEAT"
elif [[ ! -f "$MIX_JSONL" ]]; then
  echo "Missing prebuilt mix data: $MIX_JSONL" >&2
  exit 1
fi

MULTIMODAL=0 DATA_JSONL="$MIX_JSONL" bash scripts/08_train_donor_adapter.sh "$CONFIG" \
  --init-checkpoint "$INIT_CHECKPOINT" \
  --save-every "$SAVE_EVERY" \
  --diag-every "$DIAG_EVERY"

echo "wrote $CHECKPOINT"

if [[ "$EVAL_AFTER_TRAIN" == "1" ]]; then
  CONFIG="$CONFIG" \
    CHECKPOINT="$CHECKPOINT" \
    TRUTH_GATE=1 \
    MAX_CASES="$MAX_CASES" \
    OUT="runs/eval/reasoning_safe_span_copy_truthcal_${MAX_CASES}.jsonl" \
    AUDIT_OUT="runs/eval/reasoning_safe_span_copy_truthcal_${MAX_CASES}_audit.jsonl" \
    ROOT_MD="docs/wiki/decisions/reasoning-safe-span-copy-truthcal-${MAX_CASES}.md" \
    ROOT_JSON="docs/wiki/decisions/reasoning-safe-span-copy-truthcal-${MAX_CASES}-summary.json" \
	bash scripts/153_run_reasoning_safe_span_copy_gate.sh
fi
