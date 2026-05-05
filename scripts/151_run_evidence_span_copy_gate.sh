#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

export PYTHONPATH=${PYTHONPATH:-$PWD/src}
export HF_HOME=${HF_HOME:-~/.cache/huggingface}

CONFIG="${CONFIG:-configs/qwen35_2b_4090_evidence_span_reader_wsno_hardneg_s050.yaml}"
CHECKPOINT="${CHECKPOINT:-runs/qwen35_2b_4090_evidence_span_reader_wsno_hardneg_s050/last.pt}"
MAX_CASES="${MAX_CASES:-4}"
NORMAL_CASES="${NORMAL_CASES:-data/eval/memory_reasoning_heldout_expanded_72.jsonl}"
SWAP_CASES="${SWAP_CASES:-data/eval/memory_reasoning_heldout_workspace_swap_${MAX_CASES}.jsonl}"
NORMAL_OUT="${NORMAL_OUT:-runs/eval/memory_reasoning_heldout_evidence_span_copy_hardneg_${MAX_CASES}.jsonl}"
NORMAL_AUDIT_OUT="${NORMAL_AUDIT_OUT:-runs/eval/memory_reasoning_heldout_evidence_span_copy_hardneg_${MAX_CASES}_audit.jsonl}"
NORMAL_ROOT_MD="${NORMAL_ROOT_MD:-docs/wiki/decisions/evidence-span-copy-hardneg-normal-root-gate.md}"
NORMAL_ROOT_JSON="${NORMAL_ROOT_JSON:-docs/wiki/decisions/evidence-span-copy-hardneg-normal-root-gate-summary.json}"
SWAP_OUT="${SWAP_OUT:-runs/eval/memory_reasoning_heldout_workspace_swap_${MAX_CASES}_evidence_span_copy_hardneg.jsonl}"
SWAP_AUDIT_OUT="${SWAP_AUDIT_OUT:-runs/eval/memory_reasoning_heldout_workspace_swap_${MAX_CASES}_evidence_span_copy_hardneg_audit.jsonl}"
SWAP_ROOT_MD="${SWAP_ROOT_MD:-docs/wiki/decisions/evidence-span-copy-hardneg-swap-root-gate.md}"
SWAP_ROOT_JSON="${SWAP_ROOT_JSON:-docs/wiki/decisions/evidence-span-copy-hardneg-swap-root-gate-summary.json}"
EVIDENCE_SPAN_MAX_TOKENS="${EVIDENCE_SPAN_MAX_TOKENS:-12}"
EVIDENCE_SPAN_NO_ANSWER_THRESHOLD="${EVIDENCE_SPAN_NO_ANSWER_THRESHOLD:-0.5}"

echo "============================================================"
echo "Evidence span-copy answer-channel gate"
echo "config=${CONFIG}"
echo "checkpoint=${CHECKPOINT}"
echo "max_cases=${MAX_CASES}"
echo "============================================================"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing checkpoint: $CHECKPOINT" >&2
  exit 1
fi

if [[ ! -f "$NORMAL_CASES" ]]; then
  python scripts/110_build_expanded_memory_reasoning_heldout.py --out "$NORMAL_CASES"
fi

COMMON_ARGS=(
  --config "$CONFIG"
  --checkpoint "$CHECKPOINT"
  --mode qtrm_residual_with_evidence
  --mode qtrm_workspace_off_with_evidence
  --mode qtrm_core_off_with_evidence
  --mode qtrm_workspace_memory_off_with_evidence
  --mode qtrm_core_context_off_with_evidence
  --mode qtrm_evidence_bottleneck_off_with_evidence
  --mode qtrm_evidence_span_reader_off_with_evidence
  --evidence-mode all
  --evidence-injection workspace
  --answer-channel evidence_span_copy
  --evidence-span-max-tokens "$EVIDENCE_SPAN_MAX_TOKENS"
  --evidence-span-no-answer-threshold "$EVIDENCE_SPAN_NO_ANSWER_THRESHOLD"
  --max-length 384
  --max-new-tokens 1
  --qtrm-logits-scale 0.5
  --donor-logits-scale 1.0
  --no-logit-shift
)

python scripts/95_eval_memory_retrieval.py \
  "${COMMON_ARGS[@]}" \
  --cases "$NORMAL_CASES" \
  --max-cases "$MAX_CASES" \
  --jsonl-out "$NORMAL_OUT" \
  --audit-jsonl-out "$NORMAL_AUDIT_OUT"

python scripts/148_build_root_architecture_gate.py \
  --eval-jsonl "$NORMAL_OUT" \
  --markdown-out "$NORMAL_ROOT_MD" \
  --json-out "$NORMAL_ROOT_JSON"

python scripts/build_workspace_counterfactual_eval_cases.py \
  --cases "$NORMAL_CASES" \
  --out "$SWAP_CASES" \
  --max-cases "$MAX_CASES"

python scripts/95_eval_memory_retrieval.py \
  "${COMMON_ARGS[@]}" \
  --cases "$SWAP_CASES" \
  --jsonl-out "$SWAP_OUT" \
  --audit-jsonl-out "$SWAP_AUDIT_OUT"

python scripts/148_build_root_architecture_gate.py \
  --eval-jsonl "$SWAP_OUT" \
  --markdown-out "$SWAP_ROOT_MD" \
  --json-out "$SWAP_ROOT_JSON"

echo "wrote $NORMAL_OUT"
echo "wrote $NORMAL_ROOT_MD"
echo "wrote $SWAP_OUT"
echo "wrote $SWAP_ROOT_MD"
