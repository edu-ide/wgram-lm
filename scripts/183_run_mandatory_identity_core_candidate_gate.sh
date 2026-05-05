#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi
export PYTHONPATH=${PYTHONPATH:-$PWD/src}
export HF_HOME=${HF_HOME:-~/.cache/huggingface}

CONFIG="${CONFIG:-configs/qwen35_2b_4090_mandatory_identity_core_candidate.yaml}"
CHECKPOINT="${CHECKPOINT:-runs/qwen35_2b_4090_intervention_preference_train24_s080/last.pt}"
CASES="${CASES:-data/eval/memory_reasoning_heldout_expanded_72.jsonl}"
MAX_CASES="${MAX_CASES:-8}"
MAX_LENGTH="${MAX_LENGTH:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-24}"
QTRM_LOGITS_SCALE="${QTRM_LOGITS_SCALE:-0.75}"
DONOR_LOGITS_SCALE="${DONOR_LOGITS_SCALE:-1.0}"
NO_REPEAT_NGRAM_SIZE="${NO_REPEAT_NGRAM_SIZE:-2}"
OUT="${OUT:-runs/eval/mandatory_identity_core_candidate_gate_${MAX_CASES}.jsonl}"
AUDIT_OUT="${AUDIT_OUT:-runs/eval/mandatory_identity_core_candidate_gate_${MAX_CASES}_audit.jsonl}"
ROOT_MD="${ROOT_MD:-docs/wiki/decisions/mandatory-identity-core-candidate-gate-${MAX_CASES}.md}"
ROOT_JSON="${ROOT_JSON:-docs/wiki/decisions/mandatory-identity-core-candidate-gate-${MAX_CASES}-summary.json}"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing checkpoint: $CHECKPOINT" >&2
  exit 1
fi

python scripts/95_eval_memory_retrieval.py \
  --require-canonical-ssot \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --cases "$CASES" \
  --max-cases "$MAX_CASES" \
  --mode donor_only_with_evidence \
  --mode qtrm_residual_with_evidence \
  --mode qtrm_core_off_with_evidence \
  --mode qtrm_workspace_off_with_evidence \
  --evidence-mode all \
  --evidence-injection ssot \
  --answer-channel greedy \
  --max-length "$MAX_LENGTH" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --qtrm-logits-scale "$QTRM_LOGITS_SCALE" \
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
  --strict-promotion-gate \
  --baseline-mode qtrm_residual_with_evidence \
  --critical-mode qtrm_core_off_with_evidence \
  --critical-mode qtrm_workspace_off_with_evidence \
  --comparison-mode donor_only_with_evidence

echo "wrote $OUT"
echo "wrote $AUDIT_OUT"
echo "wrote $ROOT_MD"
echo "wrote $ROOT_JSON"
