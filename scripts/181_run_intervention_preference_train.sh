#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi
export PYTHONPATH=${PYTHONPATH:-$PWD/src}
export HF_HOME=${HF_HOME:-~/.cache/huggingface}

DATA="${DATA:-data/filtered/memory_reasoning_intervention_preferences_train24.jsonl}"
TRAIN_CASES="${TRAIN_CASES:-data/filtered/memory_reasoning_synth_train_cases.jsonl}"
TRAIN_EVAL="${TRAIN_EVAL:-runs/eval/canonical_answer_preference_s160_train24_answer_gate.jsonl}"
CONFIG="${CONFIG:-configs/qwen35_2b_4090_intervention_preference_train24_s080.yaml}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-runs/qwen35_2b_4090_canonical_answer_preference_s160/last.pt}"
OUT_DIR="${OUT_DIR:-runs/qwen35_2b_4090_intervention_preference_train24_s080}"
CASES="${CASES:-data/eval/memory_reasoning_heldout_expanded_72.jsonl}"
MAX_CASES="${MAX_CASES:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-24}"
QTRM_LOGITS_SCALE="${QTRM_LOGITS_SCALE:-0.75}"
DONOR_LOGITS_SCALE="${DONOR_LOGITS_SCALE:-1.0}"

if [[ ! -f "$DATA" ]]; then
  if [[ ! -f "$TRAIN_EVAL" ]]; then
    echo "Missing intervention preference data: $DATA" >&2
    echo "Also missing train eval records needed to build it: $TRAIN_EVAL" >&2
    exit 1
  fi
  python scripts/180_build_intervention_preferences.py \
    --cases-jsonl "$TRAIN_CASES" \
    --eval-jsonl "$TRAIN_EVAL" \
    --out-jsonl "$DATA"
fi

if [[ ! -f "$INIT_CHECKPOINT" ]]; then
  echo "Missing init checkpoint: $INIT_CHECKPOINT" >&2
  exit 1
fi

echo "============================================================"
echo "Intervention preference training"
echo "config=$CONFIG"
echo "data=$DATA"
echo "init=$INIT_CHECKPOINT"
echo "out_dir=$OUT_DIR"
echo "heldout_cases=$CASES"
echo "============================================================"

python -m qtrm_mm.training.train \
  --config "$CONFIG" \
  --use-donor \
  --data-jsonl "$DATA" \
  --tokenizer-model-id Qwen/Qwen3.5-2B-Base \
  --init-checkpoint "$INIT_CHECKPOINT"

CHECKPOINT="$OUT_DIR/last.pt" \
CONFIG="$CONFIG" \
CASES="$CASES" \
MAX_CASES="$MAX_CASES" \
MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
QTRM_LOGITS_SCALE="$QTRM_LOGITS_SCALE" \
DONOR_LOGITS_SCALE="$DONOR_LOGITS_SCALE" \
OUT="runs/eval/intervention_preference_train24_s080_answer_gate_${MAX_CASES}.jsonl" \
AUDIT_OUT="runs/eval/intervention_preference_train24_s080_answer_gate_${MAX_CASES}_audit.jsonl" \
ROOT_MD="docs/wiki/decisions/intervention-preference-train24-s080-answer-gate-${MAX_CASES}.md" \
ROOT_JSON="docs/wiki/decisions/intervention-preference-train24-s080-answer-gate-${MAX_CASES}-summary.json" \
bash scripts/166_run_canonical_ssot_answer_gate.sh
