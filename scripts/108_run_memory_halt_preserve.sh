#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

export PYTHONPATH=${PYTHONPATH:-$PWD/src}
export HF_HOME=${HF_HOME:-~/.cache/huggingface}

CONFIG=${CONFIG:-configs/qwen35_2b_4090_memory_halt_preserve_s050.yaml}
DATA_JSONL=${DATA_JSONL:-data/filtered/memory_reasoning_synth_traces.jsonl}
OUT_DIR=${OUT_DIR:-runs/qwen35_2b_4090_memory_halt_preserve_s050}
CHECKPOINT=${CHECKPOINT:-$OUT_DIR/last.pt}
INIT_CHECKPOINT=${INIT_CHECKPOINT:-runs/qwen35_2b_4090_memory_synth_generalization_s050/last.pt}
SAVE_EVERY=${SAVE_EVERY:-0}

HARD_CASES=${HARD_CASES:-data/eval/memory_reasoning_probe.jsonl}
HELDOUT_CASES=${HELDOUT_CASES:-data/eval/memory_reasoning_heldout_probe.jsonl}
HARD_INDEX=${HARD_INDEX:-runs/eval/memory_reasoning_harrier270m_index}
HELDOUT_INDEX=${HELDOUT_INDEX:-runs/eval/memory_reasoning_heldout_harrier270m_index}
MAX_LENGTH=${MAX_LENGTH:-384}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-32}
RETRIEVE_TOP_N=${RETRIEVE_TOP_N:-20}
RERANK_TOP_K=${RERANK_TOP_K:-5}
MEMORY_LINK_EXPANSION=${MEMORY_LINK_EXPANSION:-2}
RERANK_BACKEND=${RERANK_BACKEND:-cross_encoder}
RERANKER_MODEL_ID=${RERANKER_MODEL_ID:-Qwen/Qwen3-Reranker-0.6B}
QTRM_LOGITS_SCALE=${QTRM_LOGITS_SCALE:-0.5}

if [[ ! -f "$DATA_JSONL" ]]; then
  echo "Missing MemoryOS synthetic traces: $DATA_JSONL" >&2
  echo "Run scripts/100_build_synthetic_memory_cases.py and scripts/99_build_memory_trace_data.py first." >&2
  exit 1
fi

if [[ ! -f "$INIT_CHECKPOINT" ]]; then
  echo "Missing init checkpoint: $INIT_CHECKPOINT" >&2
  echo "Run the MemoryOS synthetic generalization checkpoint first." >&2
  exit 1
fi

ensure_index() {
  local cases=$1
  local index=$2
  if [[ -f "$index/records.jsonl" ]]; then
    return
  fi
  python scripts/96_build_memory_retrieval_probe_index.py \
    --cases "$cases" \
    --out-dir "$index"
}

eval_gate() {
  local cases=$1
  local index=$2
  local mode=$3
  local out=$4
  python scripts/95_eval_memory_retrieval.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --cases "$cases" \
    --mode qtrm_residual_with_evidence \
    --evidence-mode memoryos \
    --memory-index "$index" \
    --retrieve-top-n "$RETRIEVE_TOP_N" \
    --retrieval-top-k "$RERANK_TOP_K" \
    --memory-link-expansion "$MEMORY_LINK_EXPANSION" \
    --rerank-backend "$RERANK_BACKEND" \
    --reranker-model-id "$RERANKER_MODEL_ID" \
    --max-length "$MAX_LENGTH" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --qtrm-logits-scale "$QTRM_LOGITS_SCALE" \
    --core-halt-mode "$mode" \
    --no-logit-shift \
    --jsonl-out "$out"
}

echo "============================================================"
echo "MemoryOS-preserving core halt probe"
echo "Config: $CONFIG"
echo "Data: $DATA_JSONL"
echo "Output: $OUT_DIR"
echo "Init checkpoint: $INIT_CHECKPOINT"
echo "Trainable policy: core_halt_only"
echo "============================================================"

MULTIMODAL=0 DATA_JSONL="$DATA_JSONL" bash scripts/08_train_donor_adapter.sh "$CONFIG" \
  --init-checkpoint "$INIT_CHECKPOINT" \
  --save-every "$SAVE_EVERY"

mkdir -p "$OUT_DIR" runs/eval
ensure_index "$HARD_CASES" "$HARD_INDEX"
ensure_index "$HELDOUT_CASES" "$HELDOUT_INDEX"

eval_gate "$HARD_CASES" "$HARD_INDEX" disabled \
  runs/eval/memory_reasoning_halt_preserve_full_depth_32tok.jsonl
eval_gate "$HARD_CASES" "$HARD_INDEX" enabled \
  runs/eval/memory_reasoning_halt_preserve_enabled_32tok.jsonl
eval_gate "$HELDOUT_CASES" "$HELDOUT_INDEX" disabled \
  runs/eval/memory_reasoning_heldout_halt_preserve_full_depth_32tok.jsonl
eval_gate "$HELDOUT_CASES" "$HELDOUT_INDEX" enabled \
  runs/eval/memory_reasoning_heldout_halt_preserve_enabled_32tok.jsonl

echo "wrote $CHECKPOINT"
echo "wrote runs/eval/memory_reasoning_halt_preserve_full_depth_32tok.jsonl"
echo "wrote runs/eval/memory_reasoning_halt_preserve_enabled_32tok.jsonl"
echo "wrote runs/eval/memory_reasoning_heldout_halt_preserve_full_depth_32tok.jsonl"
echo "wrote runs/eval/memory_reasoning_heldout_halt_preserve_enabled_32tok.jsonl"
