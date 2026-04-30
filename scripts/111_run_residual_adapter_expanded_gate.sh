#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

export PYTHONPATH=${PYTHONPATH:-$PWD/src}
export HF_HOME=${HF_HOME:-~/.cache/huggingface}

CONFIG="${CONFIG:-configs/qwen35_2b_4090_memory_synth_generalization_s050.yaml}"
CHECKPOINT="${CHECKPOINT:-runs/qwen35_2b_4090_memory_synth_generalization_s050/last.pt}"
CASES="${CASES:-data/eval/memory_reasoning_heldout_expanded_72.jsonl}"
INDEX="${INDEX:-runs/eval/memory_reasoning_heldout_expanded_harrier270m_index}"
OUT="${OUT:-runs/eval/memory_reasoning_heldout_expanded_qwen3_rerank_32tok_synth_generalization_s050.jsonl}"
MAX_LENGTH="${MAX_LENGTH:-384}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
RETRIEVE_TOP_N="${RETRIEVE_TOP_N:-20}"
RETRIEVAL_TOP_K="${RETRIEVAL_TOP_K:-5}"
MEMORY_LINK_EXPANSION="${MEMORY_LINK_EXPANSION:-2}"
RERANK_BACKEND="${RERANK_BACKEND:-cross_encoder}"
RERANKER_MODEL_ID="${RERANKER_MODEL_ID:-Qwen/Qwen3-Reranker-0.6B}"
QTRM_LOGITS_SCALE="${QTRM_LOGITS_SCALE:-0.5}"

echo "============================================================"
echo "Residual adapter expanded held-out gate"
echo "config=${CONFIG}"
echo "checkpoint=${CHECKPOINT}"
echo "cases=${CASES}"
echo "index=${INDEX}"
echo "out=${OUT}"
echo "============================================================"

if [[ ! -f "$CASES" ]]; then
  python scripts/110_build_expanded_memory_reasoning_heldout.py --out "$CASES"
fi

if [[ ! -f "$INDEX/records.jsonl" ]]; then
  python scripts/96_build_memory_retrieval_probe_index.py \
    --cases "$CASES" \
    --out-dir "$INDEX"
fi

python scripts/95_eval_memory_retrieval.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --cases "$CASES" \
  --mode donor_only_with_evidence \
  --mode qtrm_residual_with_evidence \
  --evidence-mode memoryos \
  --memory-index "$INDEX" \
  --retrieve-top-n "$RETRIEVE_TOP_N" \
  --retrieval-top-k "$RETRIEVAL_TOP_K" \
  --memory-link-expansion "$MEMORY_LINK_EXPANSION" \
  --rerank-backend "$RERANK_BACKEND" \
  --reranker-model-id "$RERANKER_MODEL_ID" \
  --max-length "$MAX_LENGTH" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --qtrm-logits-scale "$QTRM_LOGITS_SCALE" \
  --no-logit-shift \
  --jsonl-out "$OUT"
