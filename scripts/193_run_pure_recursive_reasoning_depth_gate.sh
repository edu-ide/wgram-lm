#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi
export PYTHONPATH=${PYTHONPATH:-$PWD/src}
export HF_HOME=${HF_HOME:-~/.cache/huggingface}

CONFIG="${CONFIG:-configs/qwen35_2b_4090_mandatory_core_intervention_preference_s080.yaml}"
CHECKPOINT="${CHECKPOINT:-runs/qwen35_2b_4090_mandatory_core_intervention_preference_s080/last.pt}"
CASES="${CASES:-data/eval/pure_recursive_reasoning_heldout_72.jsonl}"
OUT="${OUT:-runs/eval/pure_recursive_reasoning_depth_sweep.jsonl}"
ROOT_MD="${ROOT_MD:-docs/wiki/decisions/pure-recursive-reasoning-depth-gate.md}"
ROOT_JSON="${ROOT_JSON:-docs/wiki/decisions/pure-recursive-reasoning-depth-gate-summary.json}"
MAX_CASES="${MAX_CASES:-72}"
MAX_LENGTH="${MAX_LENGTH:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-12}"
NO_REPEAT_NGRAM_SIZE="${NO_REPEAT_NGRAM_SIZE:-2}"
SCORING="${SCORING:-causal_forced_choice}"
CHOICE_SCORE_NORMALIZATION="${CHOICE_SCORE_NORMALIZATION:-mean}"
INCLUDE_TRANSITION_STATE_OFF="${INCLUDE_TRANSITION_STATE_OFF:-0}"
CASES_PER_FAMILY="${CASES_PER_FAMILY:-18}"
START_INDEX="${START_INDEX:-0}"
INCLUDE_FAMILIES="${INCLUDE_FAMILIES:-}"

QTRM_SCALE_ARGS=()
if [[ -n "${QTRM_LOGITS_SCALE:-}" ]]; then
  QTRM_SCALE_ARGS+=(--qtrm-logits-scale "$QTRM_LOGITS_SCALE")
fi
if [[ -n "${DONOR_LOGITS_SCALE:-}" ]]; then
  QTRM_SCALE_ARGS+=(--donor-logits-scale "$DONOR_LOGITS_SCALE")
fi

MODE_ARGS=()
if [[ "$INCLUDE_TRANSITION_STATE_OFF" == "1" || "$INCLUDE_TRANSITION_STATE_OFF" == "true" ]]; then
  MODE_ARGS+=(--mode donor_only_no_evidence)
  MODE_ARGS+=(--mode qtrm_core_off_no_evidence)
  MODE_ARGS+=(--mode qtrm_core_steps_1_no_evidence)
  MODE_ARGS+=(--mode qtrm_core_steps_2_no_evidence)
  MODE_ARGS+=(--mode qtrm_core_steps_4_no_evidence)
  MODE_ARGS+=(--mode qtrm_core_steps_8_no_evidence)
  MODE_ARGS+=(--mode qtrm_core_steps_8_transition_state_off_no_evidence)
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing checkpoint: $CHECKPOINT" >&2
  exit 1
fi

CASE_ARGS=()
if [[ -n "$INCLUDE_FAMILIES" ]]; then
  CASE_ARGS+=(--include-family "$INCLUDE_FAMILIES")
fi

python scripts/190_build_pure_recursive_reasoning_cases.py \
  --out "$CASES" \
  --cases-per-family "$CASES_PER_FAMILY" \
  --start-index "$START_INDEX" \
  "${CASE_ARGS[@]}"

echo "============================================================"
echo "Pure recursive reasoning depth gate"
echo "config=$CONFIG"
echo "checkpoint=$CHECKPOINT"
echo "cases=$CASES"
echo "out=$OUT"
echo "max_cases=$MAX_CASES"
echo "scoring=$SCORING"
echo "choice_score_normalization=$CHOICE_SCORE_NORMALIZATION"
echo "include_transition_state_off=$INCLUDE_TRANSITION_STATE_OFF"
echo "include_families=${INCLUDE_FAMILIES:-all}"
echo "============================================================"

python scripts/192_eval_raw_intelligence.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --cases "$CASES" \
  --max-cases "$MAX_CASES" \
  --max-length "$MAX_LENGTH" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --scoring "$SCORING" \
  --choice-score-normalization "$CHOICE_SCORE_NORMALIZATION" \
  --no-repeat-ngram-size "$NO_REPEAT_NGRAM_SIZE" \
  --suppress-visible-reasoning-tokens \
  --out "$OUT" \
  "${MODE_ARGS[@]}" \
  "${QTRM_SCALE_ARGS[@]}"

python scripts/191_build_raw_intelligence_gate.py \
  --gate-type pure_recursive_reasoning \
  --eval-jsonl "$OUT" \
  --markdown-out "$ROOT_MD" \
  --json-out "$ROOT_JSON"

echo "wrote $OUT"
echo "wrote $ROOT_MD"
echo "wrote $ROOT_JSON"
