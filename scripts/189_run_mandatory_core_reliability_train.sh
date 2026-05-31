#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi
export PYTHONPATH=${PYTHONPATH:-$PWD/src}
export HF_HOME=${HF_HOME:-~/.cache/huggingface}

CONFIG="${CONFIG:-configs/qwen35_2b_4090_mandatory_core_reliability_s120.yaml}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-runs/qwen35_2b_4090_mandatory_core_intervention_preference_s080/last.pt}"
OUT_DIR="${OUT_DIR:-runs/qwen35_2b_4090_mandatory_core_reliability_s120}"
TRAIN_CASES="${TRAIN_CASES:-data/filtered/memory_reasoning_synth_train_cases.jsonl}"
TRAIN_MAX_CASES="${TRAIN_MAX_CASES:-144}"
TRAIN_EVAL_OUT="${TRAIN_EVAL_OUT:-runs/eval/mandatory_core_reliability_source_train${TRAIN_MAX_CASES}.jsonl}"
TRAIN_AUDIT_OUT="${TRAIN_AUDIT_OUT:-runs/eval/mandatory_core_reliability_source_train${TRAIN_MAX_CASES}_audit.jsonl}"
DATA="${DATA:-data/filtered/memory_reasoning_reliability_hardneg_train${TRAIN_MAX_CASES}.jsonl}"
HELDOUT_CASES="${HELDOUT_CASES:-data/eval/memory_reasoning_heldout_expanded_72.jsonl}"
MAX_CASES="${MAX_CASES:-72}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-24}"
QTRM_LOGITS_SCALE="${QTRM_LOGITS_SCALE:-0.75}"
DONOR_LOGITS_SCALE="${DONOR_LOGITS_SCALE:-1.0}"

if [[ ! -f "$INIT_CHECKPOINT" ]]; then
  echo "Missing init checkpoint: $INIT_CHECKPOINT" >&2
  exit 1
fi
if [[ ! -f "$TRAIN_CASES" ]]; then
  echo "Missing train cases: $TRAIN_CASES" >&2
  exit 1
fi

echo "============================================================"
echo "Mandatory core reliability hard-negative training"
echo "config=$CONFIG"
echo "init=$INIT_CHECKPOINT"
echo "train_cases=$TRAIN_CASES max=$TRAIN_MAX_CASES"
echo "heldout_cases=$HELDOUT_CASES max=$MAX_CASES"
echo "============================================================"

CHECKPOINT="$INIT_CHECKPOINT" \
CONFIG="$CONFIG" \
CASES="$TRAIN_CASES" \
MAX_CASES="$TRAIN_MAX_CASES" \
MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
QTRM_LOGITS_SCALE="$QTRM_LOGITS_SCALE" \
DONOR_LOGITS_SCALE="$DONOR_LOGITS_SCALE" \
OUT="$TRAIN_EVAL_OUT" \
AUDIT_OUT="$TRAIN_AUDIT_OUT" \
ROOT_MD="docs/wiki/decisions/mandatory-core-reliability-source-train${TRAIN_MAX_CASES}.md" \
ROOT_JSON="docs/wiki/decisions/mandatory-core-reliability-source-train${TRAIN_MAX_CASES}-summary.json" \
bash scripts/183_run_mandatory_identity_core_candidate_gate.sh

python scripts/188_build_reliability_hard_negative_preferences.py \
  --cases-jsonl "$TRAIN_CASES" \
  --eval-jsonl "$TRAIN_EVAL_OUT" \
  --out-jsonl "$DATA"

if [[ ! -s "$DATA" ]]; then
  echo "Reliability hard-negative data is empty: $DATA" >&2
  exit 1
fi

echo "training data=$DATA"
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
MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
QTRM_LOGITS_SCALE="$QTRM_LOGITS_SCALE" \
DONOR_LOGITS_SCALE="$DONOR_LOGITS_SCALE" \
OUT="runs/eval/mandatory_core_reliability_s120_gate_${MAX_CASES}.jsonl" \
AUDIT_OUT="runs/eval/mandatory_core_reliability_s120_gate_${MAX_CASES}_audit.jsonl" \
ROOT_MD="docs/wiki/decisions/mandatory-core-reliability-s120-gate-${MAX_CASES}.md" \
ROOT_JSON="docs/wiki/decisions/mandatory-core-reliability-s120-gate-${MAX_CASES}-summary.json" \
bash scripts/183_run_mandatory_identity_core_candidate_gate.sh
