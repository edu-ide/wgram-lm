#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export PYTHONPATH=${PYTHONPATH:-$PWD/src}
export HF_HOME=${HF_HOME:-~/.cache/huggingface}

CONFIG="${CONFIG:-configs/qwen35_2b_4090_canonical_ssot_greedy_causal_s050.yaml}"
TRAIN_CASES="${TRAIN_CASES:-data/filtered/memory_reasoning_synth_train_cases.jsonl}"
TRAIN_JSONL="${TRAIN_JSONL:-data/filtered/memory_reasoning_synth_traces.jsonl}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-runs/qwen35_2b_4090_donor_residual_s010_1000/last.pt}"
EVAL_MAX_CASES="${EVAL_MAX_CASES:-8}"

if [[ ! -f "$TRAIN_JSONL" ]]; then
  python scripts/99_build_memory_trace_data.py \
    --cases "$TRAIN_CASES" \
    --out "$TRAIN_JSONL" \
    --variant target \
    --variant all \
    --variant lexical \
    --top-k 5 \
    --max-evidence-chars 4000
fi

echo "============================================================"
echo "Canonical SSOT Greedy Causal Training"
echo "============================================================"
echo "Config: $CONFIG"
echo "Trace data: $TRAIN_JSONL"
echo "Init checkpoint: $INIT_CHECKPOINT"
echo "============================================================"

python -m qtrm_mm.training.train \
  --config "$CONFIG" \
  --use-donor \
  --data-jsonl "$TRAIN_JSONL" \
  --init-checkpoint "$INIT_CHECKPOINT" \
  --diag-every 0

CHECKPOINT="$(python - "$CONFIG" <<'PY'
import sys
from qtrm_mm.config import load_config

cfg = load_config(sys.argv[1])
print(f"{cfg.train.out_dir}/last.pt")
PY
)"

EVAL_OUT="${OUT:-runs/eval/canonical_ssot_answer_gate_after_causal_train_${EVAL_MAX_CASES}.jsonl}"
EVAL_AUDIT_OUT="${AUDIT_OUT:-runs/eval/canonical_ssot_answer_gate_after_causal_train_${EVAL_MAX_CASES}_audit.jsonl}"
EVAL_ROOT_MD="${ROOT_MD:-docs/wiki/decisions/canonical-ssot-answer-gate-after-causal-train-${EVAL_MAX_CASES}.md}"
EVAL_ROOT_JSON="${ROOT_JSON:-docs/wiki/decisions/canonical-ssot-answer-gate-after-causal-train-${EVAL_MAX_CASES}-summary.json}"

CONFIG="$CONFIG" \
CHECKPOINT="$CHECKPOINT" \
MAX_CASES="$EVAL_MAX_CASES" \
OUT="$EVAL_OUT" \
AUDIT_OUT="$EVAL_AUDIT_OUT" \
ROOT_MD="$EVAL_ROOT_MD" \
ROOT_JSON="$EVAL_ROOT_JSON" \
bash scripts/166_run_canonical_ssot_answer_gate.sh
