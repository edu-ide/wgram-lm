#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi
export PYTHONPATH=${PYTHONPATH:-$PWD/src}
export HF_HOME=${HF_HOME:-~/.cache/huggingface}

CONFIG="${CONFIG:-configs/qwen35_2b_4090_donor_residual_s010_1000.yaml}"
CHECKPOINT="${CHECKPOINT:-runs/qwen35_2b_4090_donor_residual_s010_1000/last.pt}"
CASES="${CASES:-data/eval/memory_reasoning_heldout_expanded_72.jsonl}"
MAX_CASES="${MAX_CASES:-16}"
OUT="${OUT:-runs/eval/canonical_ssot_answer_gate_${MAX_CASES}.jsonl}"
AUDIT_OUT="${AUDIT_OUT:-runs/eval/canonical_ssot_answer_gate_${MAX_CASES}_audit.jsonl}"
ROOT_MD="${ROOT_MD:-docs/wiki/decisions/canonical-ssot-answer-gate-${MAX_CASES}.md}"
ROOT_JSON="${ROOT_JSON:-docs/wiki/decisions/canonical-ssot-answer-gate-${MAX_CASES}-summary.json}"
MAX_LENGTH="${MAX_LENGTH:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-48}"
QTRM_LOGITS_SCALE="${QTRM_LOGITS_SCALE:-}"
DONOR_LOGITS_SCALE="${DONOR_LOGITS_SCALE:-1.0}"
NO_REPEAT_NGRAM_SIZE="${NO_REPEAT_NGRAM_SIZE:-2}"
STRICT_PROMOTION_GATE="${STRICT_PROMOTION_GATE:-1}"
QTRM_SCALE_ARGS=()
if [[ -n "$QTRM_LOGITS_SCALE" ]]; then
  QTRM_SCALE_ARGS+=(--qtrm-logits-scale "$QTRM_LOGITS_SCALE")
fi
ROOT_GATE_ARGS=()
if [[ "$STRICT_PROMOTION_GATE" != "0" ]]; then
  ROOT_GATE_ARGS+=(--strict-promotion-gate)
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing checkpoint: $CHECKPOINT" >&2
  exit 1
fi

if [[ ! -f "$CASES" ]]; then
  python scripts/110_build_expanded_memory_reasoning_heldout.py --out "$CASES"
fi

echo "============================================================"
echo "Canonical SSOT autoregressive answer gate"
echo "config=$CONFIG"
echo "checkpoint=$CHECKPOINT"
echo "cases=$CASES"
echo "max_cases=$MAX_CASES"
echo "answer_channel=greedy"
echo "evidence_injection=ssot"
echo "============================================================"

python scripts/95_eval_memory_retrieval.py \
  --require-canonical-ssot \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --cases "$CASES" \
  --max-cases "$MAX_CASES" \
  --mode donor_only_with_evidence \
  --mode qtrm_residual_with_evidence \
  --mode qtrm_workspace_off_with_evidence \
  --mode qtrm_core_off_with_evidence \
  --mode qtrm_workspace_memory_off_with_evidence \
  --mode qtrm_core_context_off_with_evidence \
  --mode qtrm_core_to_text_off_with_evidence \
  --mode qtrm_evidence_bottleneck_off_with_evidence \
  --mode qtrm_evidence_span_reader_off_with_evidence \
  --mode qtrm_answer_residual_governor_off_with_evidence \
  --mode donor_only_no_evidence \
  --mode qtrm_residual_no_evidence \
  --evidence-mode all \
  --evidence-injection ssot \
  --answer-channel greedy \
  --max-length "$MAX_LENGTH" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  "${QTRM_SCALE_ARGS[@]}" \
  --donor-logits-scale "$DONOR_LOGITS_SCALE" \
  --suppress-visible-reasoning-tokens \
  --no-repeat-ngram-size "$NO_REPEAT_NGRAM_SIZE" \
  --no-logit-shift \
  --jsonl-out "$OUT" \
  --audit-jsonl-out "$AUDIT_OUT" \
  --history-jsonl-out none

python scripts/148_build_root_architecture_gate.py \
  --eval-jsonl "$OUT" \
  --markdown-out "$ROOT_MD" \
  --json-out "$ROOT_JSON" \
  "${ROOT_GATE_ARGS[@]}"

echo "wrote $OUT"
echo "wrote $AUDIT_OUT"
echo "wrote $ROOT_MD"
echo "wrote $ROOT_JSON"
