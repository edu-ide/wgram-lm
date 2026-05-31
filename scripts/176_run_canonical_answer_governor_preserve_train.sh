#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi
export PYTHONPATH=${PYTHONPATH:-$PWD/src}
export HF_HOME=${HF_HOME:-~/.cache/huggingface}

CONFIG="${CONFIG:-configs/qwen35_2b_4090_canonical_answer_governor_preserve_s120.yaml}"
DATA="${DATA:-data/filtered/memory_reasoning_canonical_plain_answer.jsonl}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-runs/qwen35_2b_4090_canonical_answer_governor_s120/last.pt}"
OUT_DIR="${OUT_DIR:-runs/qwen35_2b_4090_canonical_answer_governor_preserve_s120}"
CASES="${CASES:-data/eval/memory_reasoning_heldout_expanded_72.jsonl}"
MAX_CASES="${MAX_CASES:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-24}"
QTRM_LOGITS_SCALE="${QTRM_LOGITS_SCALE:-0.75}"
DONOR_LOGITS_SCALE="${DONOR_LOGITS_SCALE:-1.0}"

if [[ ! -f "$DATA" ]]; then
  python scripts/173_build_canonical_plain_answer_data.py \
    --input-jsonl data/filtered/memory_reasoning_synth_span_reader_truthcal.jsonl \
    --output-jsonl "$DATA" \
    --evidence-mode all \
    --top-k 3
fi

if [[ ! -f "$INIT_CHECKPOINT" ]]; then
  echo "Missing init checkpoint: $INIT_CHECKPOINT" >&2
  exit 1
fi

echo "============================================================"
echo "Canonical answer governor + donor-correct preservation training"
echo "config=$CONFIG"
echo "data=$DATA"
echo "init=$INIT_CHECKPOINT"
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
CASES="$CASES" \
MAX_CASES="$MAX_CASES" \
MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
QTRM_LOGITS_SCALE="$QTRM_LOGITS_SCALE" \
DONOR_LOGITS_SCALE="$DONOR_LOGITS_SCALE" \
OUT="runs/eval/canonical_answer_governor_preserve_s120_answer_gate_${MAX_CASES}.jsonl" \
AUDIT_OUT="runs/eval/canonical_answer_governor_preserve_s120_answer_gate_${MAX_CASES}_audit.jsonl" \
ROOT_MD="docs/wiki/decisions/canonical-answer-governor-preserve-s120-answer-gate-${MAX_CASES}.md" \
ROOT_JSON="docs/wiki/decisions/canonical-answer-governor-preserve-s120-answer-gate-${MAX_CASES}-summary.json" \
bash scripts/166_run_canonical_ssot_answer_gate.sh
