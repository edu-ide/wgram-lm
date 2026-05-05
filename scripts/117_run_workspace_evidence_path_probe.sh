#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

export PYTHONPATH=${PYTHONPATH:-$PWD/src}
export HF_HOME=${HF_HOME:-~/.cache/huggingface}

CONFIG="${CONFIG:-configs/qwen35_2b_4090_memory_gated_workspace_s050.yaml}"
CHECKPOINT="${CHECKPOINT:-runs/qwen35_2b_4090_memory_gated_workspace_s050/last.pt}"
CASES="${CASES:-data/eval/memory_reasoning_heldout_expanded_72.jsonl}"
INDEX="${INDEX:-runs/eval/memory_reasoning_heldout_expanded_harrier270m_index}"
OUT="${OUT:-runs/eval/memory_reasoning_heldout_expanded_workspace_evidence_path_32tok_gated_workspace_s050.jsonl}"
AUDIT_OUT="${AUDIT_OUT:-runs/eval/memory_reasoning_heldout_expanded_workspace_evidence_path_32tok_gated_workspace_s050_audit.jsonl}"
PROOF_MD="${PROOF_MD:-docs/wiki/decisions/workspace-evidence-path-ablation.md}"
PROOF_JSON="${PROOF_JSON:-docs/wiki/decisions/workspace-evidence-path-ablation-summary.json}"
MAX_LENGTH="${MAX_LENGTH:-384}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
SUPPRESS_VISIBLE_REASONING="${SUPPRESS_VISIBLE_REASONING:-0}"
NO_REPEAT_NGRAM_SIZE="${NO_REPEAT_NGRAM_SIZE:-0}"
SHORT_ANSWER_GOVERNOR="${SHORT_ANSWER_GOVERNOR:-0}"
RETRIEVE_TOP_N="${RETRIEVE_TOP_N:-20}"
RETRIEVAL_TOP_K="${RETRIEVAL_TOP_K:-5}"
MEMORY_LINK_EXPANSION="${MEMORY_LINK_EXPANSION:-2}"
RERANK_BACKEND="${RERANK_BACKEND:-cross_encoder}"
RERANKER_MODEL_ID="${RERANKER_MODEL_ID:-Qwen/Qwen3-Reranker-0.6B}"
QTRM_LOGITS_SCALE="${QTRM_LOGITS_SCALE:-0.5}"
MAX_CASES="${MAX_CASES:-}"
MAX_CASES_ARGS=()
if [[ -n "$MAX_CASES" ]]; then
  MAX_CASES_ARGS=(--max-cases "$MAX_CASES")
fi
DECODE_GUARD_ARGS=()
if [[ "$SUPPRESS_VISIBLE_REASONING" == "1" ]]; then
  DECODE_GUARD_ARGS+=(--suppress-visible-reasoning-tokens)
fi
if [[ "$NO_REPEAT_NGRAM_SIZE" != "0" ]]; then
  DECODE_GUARD_ARGS+=(--no-repeat-ngram-size "$NO_REPEAT_NGRAM_SIZE")
fi
if [[ "$SHORT_ANSWER_GOVERNOR" == "1" ]]; then
  DECODE_GUARD_ARGS+=(--short-answer-governor)
fi

echo "============================================================"
echo "Workspace evidence-path causality probe"
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
  --mode qtrm_workspace_off_with_evidence \
  --mode qtrm_core_off_with_evidence \
  --mode qtrm_core_context_off_with_evidence \
  --mode qtrm_coda_off_with_evidence \
  --mode qtrm_residual_head_off_with_evidence \
  --mode qtrm_workspace_gate_off_with_evidence \
  --mode qtrm_workspace_memory_off_with_evidence \
  --mode qtrm_evidence_bottleneck_off_with_evidence \
  --evidence-mode memoryos \
  --evidence-injection workspace \
  --memory-index "$INDEX" \
  --retrieve-top-n "$RETRIEVE_TOP_N" \
  --retrieval-top-k "$RETRIEVAL_TOP_K" \
  --memory-link-expansion "$MEMORY_LINK_EXPANSION" \
  --rerank-backend "$RERANK_BACKEND" \
  --reranker-model-id "$RERANKER_MODEL_ID" \
  --max-length "$MAX_LENGTH" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  "${DECODE_GUARD_ARGS[@]}" \
  --qtrm-logits-scale "$QTRM_LOGITS_SCALE" \
  --no-logit-shift \
  --jsonl-out "$OUT" \
  --audit-jsonl-out "$AUDIT_OUT" \
  "${MAX_CASES_ARGS[@]}"

python scripts/113_build_expanded_ablation_proof.py \
  --eval "workspace evidence path=$OUT" \
  --markdown-out "$PROOF_MD" \
  --json-out "$PROOF_JSON"

echo "wrote $OUT"
echo "wrote $AUDIT_OUT"
echo "wrote $PROOF_MD"
echo "wrote $PROOF_JSON"
