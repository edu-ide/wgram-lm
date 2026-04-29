#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/qwen35_2b_4090_memory_synth_generalization_s050.yaml}"
CHECKPOINT="${CHECKPOINT:-runs/qwen35_2b_4090_memory_synth_generalization_s050/last.pt}"
CASES="${CASES:-data/eval/memory_reasoning_heldout_probe.jsonl}"
OUT="${OUT:-runs/eval/qtrm_ablation_proof_heldout_target_s050.jsonl}"
MAX_LENGTH="${MAX_LENGTH:-256}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-24}"
EVIDENCE_MODE="${EVIDENCE_MODE:-target}"
RETRIEVAL_TOP_K="${RETRIEVAL_TOP_K:-5}"

echo "============================================================"
echo "QTRM ablation proof"
echo "config=${CONFIG}"
echo "checkpoint=${CHECKPOINT}"
echo "cases=${CASES}"
echo "out=${OUT}"
echo "evidence_mode=${EVIDENCE_MODE}"
echo "============================================================"

PYTHONPATH="${PYTHONPATH:-src}" .venv/bin/python scripts/95_eval_memory_retrieval.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --cases "${CASES}" \
  --evidence-mode "${EVIDENCE_MODE}" \
  --retrieval-top-k "${RETRIEVAL_TOP_K}" \
  --max-length "${MAX_LENGTH}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --jsonl-out "${OUT}" \
  --mode donor_only_with_evidence \
  --mode qtrm_residual_with_evidence \
  --mode qtrm_workspace_off_with_evidence \
  --mode qtrm_core_off_with_evidence \
  --mode donor_only_no_evidence \
  --mode qtrm_residual_no_evidence \
  --no-logit-shift
