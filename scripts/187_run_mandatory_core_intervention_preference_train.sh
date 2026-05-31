#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi
export PYTHONPATH=${PYTHONPATH:-$PWD/src}
export HF_HOME=${HF_HOME:-~/.cache/huggingface}

RAW_DATA="${RAW_DATA:-data/filtered/memory_reasoning_intervention_preferences_train24.jsonl}"
DATA="${DATA:-data/filtered/memory_reasoning_intervention_preferences_clean_train24.jsonl}"
CONFIG="${CONFIG:-configs/qwen35_2b_4090_mandatory_core_intervention_preference_s080.yaml}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-runs/qwen35_2b_4090_mandatory_core_answer_bottleneck_causal_s120/last.pt}"
OUT_DIR="${OUT_DIR:-runs/qwen35_2b_4090_mandatory_core_intervention_preference_s080}"
CASES="${CASES:-data/eval/memory_reasoning_heldout_expanded_72.jsonl}"
MAX_CASES="${MAX_CASES:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-24}"
QTRM_LOGITS_SCALE="${QTRM_LOGITS_SCALE:-0.75}"
DONOR_LOGITS_SCALE="${DONOR_LOGITS_SCALE:-1.0}"

if [[ ! -f "$RAW_DATA" ]]; then
  echo "Missing raw intervention preference data: $RAW_DATA" >&2
  exit 1
fi

python scripts/186_build_clean_intervention_preferences.py \
  --input-jsonl "$RAW_DATA" \
  --output-jsonl "$DATA"

if [[ ! -f "$INIT_CHECKPOINT" ]]; then
  echo "Missing init checkpoint: $INIT_CHECKPOINT" >&2
  exit 1
fi

echo "============================================================"
echo "Mandatory core intervention-preference training"
echo "config=$CONFIG"
echo "data=$DATA"
echo "init=$INIT_CHECKPOINT"
echo "out_dir=$OUT_DIR"
echo "heldout_cases=$CASES"
echo "============================================================"

python -m wgram_lm.training.train \
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
OUT="runs/eval/mandatory_core_intervention_preference_s080_gate_${MAX_CASES}.jsonl" \
AUDIT_OUT="runs/eval/mandatory_core_intervention_preference_s080_gate_${MAX_CASES}_audit.jsonl" \
ROOT_MD="docs/wiki/decisions/mandatory-core-intervention-preference-s080-gate-${MAX_CASES}.md" \
ROOT_JSON="docs/wiki/decisions/mandatory-core-intervention-preference-s080-gate-${MAX_CASES}-summary.json" \
bash scripts/183_run_mandatory_identity_core_candidate_gate.sh
