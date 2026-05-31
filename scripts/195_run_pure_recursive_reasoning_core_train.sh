#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi
export PYTHONPATH=${PYTHONPATH:-$PWD/src}
export HF_HOME=${HF_HOME:-~/.cache/huggingface}

CONFIG="${CONFIG:-configs/qwen35_2b_4090_pure_recursive_reasoning_core_s160.yaml}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-runs/qwen35_2b_4090_mandatory_core_intervention_preference_s080/last.pt}"
TRAIN_CASES="${TRAIN_CASES:-data/filtered/pure_recursive_reasoning_train256_cases.jsonl}"
DATA="${DATA:-data/filtered/pure_recursive_reasoning_preferences_train.jsonl}"
OUT_DIR="${OUT_DIR:-runs/qwen35_2b_4090_pure_recursive_reasoning_core_s160}"
HELDOUT_CASES="${HELDOUT_CASES:-data/eval/pure_recursive_reasoning_heldout_72.jsonl}"
MAX_CASES="${MAX_CASES:-16}"
TRAIN_CASES_PER_FAMILY="${TRAIN_CASES_PER_FAMILY:-64}"
TRAIN_START_INDEX="${TRAIN_START_INDEX:-100}"
MAX_REJECTED_PER_CASE="${MAX_REJECTED_PER_CASE:-3}"

if [[ ! -f "$INIT_CHECKPOINT" ]]; then
  echo "Missing init checkpoint: $INIT_CHECKPOINT" >&2
  exit 1
fi

python scripts/190_build_pure_recursive_reasoning_cases.py \
  --out "$TRAIN_CASES" \
  --cases-per-family "$TRAIN_CASES_PER_FAMILY" \
  --start-index "$TRAIN_START_INDEX"

python scripts/194_build_pure_recursive_reasoning_preferences.py \
  --cases "$TRAIN_CASES" \
  --out "$DATA" \
  --max-rejected-per-case "$MAX_REJECTED_PER_CASE"

echo "============================================================"
echo "Pure recursive reasoning core training"
echo "config=$CONFIG"
echo "init=$INIT_CHECKPOINT"
echo "data=$DATA"
echo "out_dir=$OUT_DIR"
echo "============================================================"

python -m wgram_lm.training.train \
  --config "$CONFIG" \
  --use-donor \
  --data-jsonl "$DATA" \
  --tokenizer-model-id Qwen/Qwen3.5-2B-Base \
  --init-checkpoint "$INIT_CHECKPOINT"

CHECKPOINT="$OUT_DIR/last.pt" \
CONFIG="$CONFIG" \
CASES="$HELDOUT_CASES" \
MAX_CASES="$MAX_CASES" \
OUT="runs/eval/pure_recursive_reasoning_core_s160_depth_gate_${MAX_CASES}.jsonl" \
ROOT_MD="docs/wiki/decisions/pure-recursive-reasoning-core-s160-depth-gate-${MAX_CASES}.md" \
ROOT_JSON="docs/wiki/decisions/pure-recursive-reasoning-core-s160-depth-gate-${MAX_CASES}-summary.json" \
bash scripts/193_run_pure_recursive_reasoning_depth_gate.sh
