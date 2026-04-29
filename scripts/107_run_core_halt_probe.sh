#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

export PYTHONPATH=${PYTHONPATH:-$PWD/src}
export HF_HOME=${HF_HOME:-~/.cache/huggingface}

CONFIG=${CONFIG:-configs/qwen35_2b_4090_core_halt_probe.yaml}
DATA_JSONL=${DATA_JSONL:-data/filtered/qtrm_clean_pilot.jsonl}
OUT_DIR=${OUT_DIR:-runs/qwen35_2b_4090_core_halt_probe}
CHECKPOINT=${CHECKPOINT:-$OUT_DIR/last.pt}
DEFAULT_INIT_CHECKPOINT=${DEFAULT_INIT_CHECKPOINT:-runs/qwen35_2b_4090_memory_synth_generalization_s050/last.pt}
DIAG_EVERY=${DIAG_EVERY:-50}
DIAG_MAX_NEW_TOKENS=${DIAG_MAX_NEW_TOKENS:-8}
EVAL_MAX_NEW_TOKENS=${EVAL_MAX_NEW_TOKENS:-16}
EVAL_MAX_SAMPLES=${EVAL_MAX_SAMPLES:-16}
SAVE_EVERY=${SAVE_EVERY:-100}

if [[ ! -f "$DATA_JSONL" ]]; then
  echo "Missing clean pilot data: $DATA_JSONL" >&2
  echo "Run scripts/94_build_clean_pilot_data.sh first." >&2
  exit 1
fi

INIT_CHECKPOINT=${INIT_CHECKPOINT:-}
if [[ -z "$INIT_CHECKPOINT" && -f "$DEFAULT_INIT_CHECKPOINT" ]]; then
  INIT_CHECKPOINT=$DEFAULT_INIT_CHECKPOINT
fi

INIT_ARGS=()
if [[ -n "$INIT_CHECKPOINT" ]]; then
  INIT_ARGS=(--init-checkpoint "$INIT_CHECKPOINT")
fi

echo "============================================================"
echo "Core-halt probe"
echo "Config: $CONFIG"
echo "Data: $DATA_JSONL"
echo "Output: $OUT_DIR"
echo "Init checkpoint: ${INIT_CHECKPOINT:-none}"
echo "============================================================"

MULTIMODAL=0 DATA_JSONL="$DATA_JSONL" bash scripts/08_train_donor_adapter.sh "$CONFIG" \
  --diag-every "$DIAG_EVERY" \
  --diag-max-new-tokens "$DIAG_MAX_NEW_TOKENS" \
  --save-every "$SAVE_EVERY" \
  --diag-prompt "양자 컴퓨팅이란 무엇인가요?" \
  --diag-prompt "Quantum entanglement means" \
  --diag-prompt "Solve step by step: if x + 3 = 7, what is x?" \
  "${INIT_ARGS[@]}"

mkdir -p "$OUT_DIR"

python scripts/92_eval_qtrm_logits.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --data-jsonl "$DATA_JSONL" \
  --max-samples "$EVAL_MAX_SAMPLES" \
  --max-new-tokens "$EVAL_MAX_NEW_TOKENS" \
  --enable-core-halt \
  --json \
  > "$OUT_DIR/post_eval_core_halt.jsonl"

python scripts/92_eval_qtrm_logits.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --prompt "양자 컴퓨팅이란 무엇인가요?" \
  --prompt "Quantum entanglement means" \
  --prompt "Solve step by step: if x + 3 = 7, what is x?" \
  --max-new-tokens "$EVAL_MAX_NEW_TOKENS" \
  --enable-core-halt \
  > "$OUT_DIR/post_eval_core_halt_prompts.txt"

echo "wrote $CHECKPOINT"
echo "wrote $OUT_DIR/post_eval_core_halt.jsonl"
echo "wrote $OUT_DIR/post_eval_core_halt_prompts.txt"
