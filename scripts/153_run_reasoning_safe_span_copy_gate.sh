#!/usr/bin/env bash
set -euo pipefail

cd ~/qtrm-workspace/wgram-lm
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi
export PYTHONPATH=${PYTHONPATH:-$PWD/src}
export HF_HOME=${HF_HOME:-~/.cache/huggingface}

CONFIG="${CONFIG:-configs/qwen35_2b_4090_evidence_span_reader_trainhardnegx2_s500.yaml}"
CHECKPOINT="${CHECKPOINT:-runs/qwen35_2b_4090_evidence_span_reader_trainhardnegx2_s500/last.pt}"
CASES="${CASES:-data/eval/memory_reasoning_heldout_expanded_72.jsonl}"
MAX_CASES="${MAX_CASES:-16}"
OUT="${OUT:-runs/eval/reasoning_safe_span_copy_confidence_gate_${MAX_CASES}.jsonl}"
AUDIT_OUT="${AUDIT_OUT:-runs/eval/reasoning_safe_span_copy_confidence_gate_${MAX_CASES}_audit.jsonl}"
ROOT_MD="${ROOT_MD:-docs/wiki/decisions/reasoning-safe-span-copy-confidence-gate.md}"
ROOT_JSON="${ROOT_JSON:-docs/wiki/decisions/reasoning-safe-span-copy-confidence-gate-summary.json}"
EVIDENCE_SPAN_MAX_TOKENS="${EVIDENCE_SPAN_MAX_TOKENS:-12}"
EVIDENCE_SPAN_NO_ANSWER_THRESHOLD="${EVIDENCE_SPAN_NO_ANSWER_THRESHOLD:-0.1}"
EVIDENCE_SPAN_MIN_SCORE="${EVIDENCE_SPAN_MIN_SCORE:-12}"
TRUTH_GATE="${TRUTH_GATE:-0}"
TRUTH_SUPPORT_THRESHOLD="${TRUTH_SUPPORT_THRESHOLD:-0.5}"
TRUTH_CAUSAL_THRESHOLD="${TRUTH_CAUSAL_THRESHOLD:-0.5}"
TRUTH_REFUTE_THRESHOLD="${TRUTH_REFUTE_THRESHOLD:-0.5}"
TRUTH_MISSING_THRESHOLD="${TRUTH_MISSING_THRESHOLD:-0.5}"
EVIDENCE_INJECTION="${EVIDENCE_INJECTION:-ssot}"

truth_args=()
if [[ "$TRUTH_GATE" == "1" || "$TRUTH_GATE" == "true" ]]; then
  truth_args+=(
    --truth-gate
    --truth-support-threshold "$TRUTH_SUPPORT_THRESHOLD"
    --truth-causal-threshold "$TRUTH_CAUSAL_THRESHOLD"
    --truth-refute-threshold "$TRUTH_REFUTE_THRESHOLD"
    --truth-missing-threshold "$TRUTH_MISSING_THRESHOLD"
  )
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing checkpoint: $CHECKPOINT" >&2
  exit 1
fi

if [[ ! -f "$CASES" ]]; then
  python scripts/110_build_expanded_memory_reasoning_heldout.py --out "$CASES"
fi

echo "============================================================"
echo "Reasoning-safe span-copy confidence gate (PROBE-ONLY)"
echo "PROBE-ONLY: span-copy is not the canonical autoregressive answer gate."
echo "config=$CONFIG"
echo "checkpoint=$CHECKPOINT"
echo "cases=$CASES"
echo "max_cases=$MAX_CASES"
echo "no_answer_threshold=$EVIDENCE_SPAN_NO_ANSWER_THRESHOLD"
echo "min_span_score=$EVIDENCE_SPAN_MIN_SCORE"
echo "truth_gate=$TRUTH_GATE"
echo "evidence_injection=$EVIDENCE_INJECTION"
echo "============================================================"

python scripts/95_eval_memory_retrieval.py \
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
  --mode qtrm_evidence_bottleneck_off_with_evidence \
  --mode qtrm_evidence_span_reader_off_with_evidence \
  --evidence-mode all \
  --evidence-injection "$EVIDENCE_INJECTION" \
  --answer-channel evidence_span_copy \
  --evidence-span-max-tokens "$EVIDENCE_SPAN_MAX_TOKENS" \
  --evidence-span-no-answer-threshold "$EVIDENCE_SPAN_NO_ANSWER_THRESHOLD" \
  --evidence-span-min-score "$EVIDENCE_SPAN_MIN_SCORE" \
  "${truth_args[@]}" \
  --max-length 384 \
  --max-new-tokens 1 \
  --qtrm-logits-scale 0.5 \
  --donor-logits-scale 1.0 \
  --no-logit-shift \
  --jsonl-out "$OUT" \
  --audit-jsonl-out "$AUDIT_OUT" \
  --history-jsonl-out none

python scripts/148_build_root_architecture_gate.py \
  --eval-jsonl "$OUT" \
  --markdown-out "$ROOT_MD" \
  --json-out "$ROOT_JSON"

echo "wrote $OUT"
echo "wrote $AUDIT_OUT"
echo "wrote $ROOT_MD"
echo "wrote $ROOT_JSON"
