#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

export PYTHONPATH=${PYTHONPATH:-$PWD/src}
export HF_HOME=${HF_HOME:-~/.cache/huggingface}

CONFIG="${CONFIG:-configs/qwen35_2b_4090_workspace_answer_bottleneck_causal_s050.yaml}"
SOURCE_JSONL="${SOURCE_JSONL:-data/filtered/memory_self_improvement_preferences_analysis.jsonl}"
DATA_JSONL="${DATA_JSONL:-data/filtered/memory_self_improvement_preferences_workspace_counterfactual.jsonl}"
OUT_DIR="${OUT_DIR:-runs/qwen35_2b_4090_workspace_answer_bottleneck_causal_s050}"
CHECKPOINT="${CHECKPOINT:-$OUT_DIR/last.pt}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-runs/qwen35_2b_4090_workspace_answer_bottleneck_s050/last.pt}"
DIAG_EVERY="${DIAG_EVERY:-0}"
DIAG_MAX_NEW_TOKENS="${DIAG_MAX_NEW_TOKENS:-16}"
SAVE_EVERY="${SAVE_EVERY:-100}"
MAX_CASES="${MAX_CASES:-4}"
SUPPRESS_VISIBLE_REASONING="${SUPPRESS_VISIBLE_REASONING:-1}"
NO_REPEAT_NGRAM_SIZE="${NO_REPEAT_NGRAM_SIZE:-2}"
SHORT_ANSWER_GOVERNOR="${SHORT_ANSWER_GOVERNOR:-1}"
PREFERENCE_EVAL_OUT="${PREFERENCE_EVAL_OUT:-runs/eval/workspace_answer_bottleneck_causal_pair_eval_s050.jsonl}"

NORMAL_OUT="${NORMAL_OUT:-runs/eval/memory_reasoning_heldout_expanded_workspace_answer_bottleneck_causal_32tok_s050.jsonl}"
NORMAL_AUDIT_OUT="${NORMAL_AUDIT_OUT:-runs/eval/memory_reasoning_heldout_expanded_workspace_answer_bottleneck_causal_32tok_s050_audit.jsonl}"
NORMAL_PROOF_MD="${NORMAL_PROOF_MD:-docs/wiki/decisions/workspace-answer-bottleneck-causal-ablation.md}"
NORMAL_PROOF_JSON="${NORMAL_PROOF_JSON:-docs/wiki/decisions/workspace-answer-bottleneck-causal-ablation-summary.json}"
NORMAL_ROOT_MD="${NORMAL_ROOT_MD:-docs/wiki/decisions/workspace-answer-bottleneck-causal-root-gate.md}"
NORMAL_ROOT_JSON="${NORMAL_ROOT_JSON:-docs/wiki/decisions/workspace-answer-bottleneck-causal-root-gate-summary.json}"

SWAP_CASES="${SWAP_CASES:-data/eval/memory_reasoning_heldout_workspace_swap_${MAX_CASES}.jsonl}"
SWAP_OUT="${SWAP_OUT:-runs/eval/memory_reasoning_heldout_workspace_swap_${MAX_CASES}_workspace_answer_bottleneck_causal_s050.jsonl}"
SWAP_AUDIT_OUT="${SWAP_AUDIT_OUT:-runs/eval/memory_reasoning_heldout_workspace_swap_${MAX_CASES}_workspace_answer_bottleneck_causal_s050_audit.jsonl}"
SWAP_ROOT_MD="${SWAP_ROOT_MD:-docs/wiki/decisions/workspace-answer-bottleneck-causal-swap-root-gate.md}"
SWAP_ROOT_JSON="${SWAP_ROOT_JSON:-docs/wiki/decisions/workspace-answer-bottleneck-causal-swap-root-gate-summary.json}"

echo "============================================================"
echo "Workspace answer-bottleneck causal training"
echo "config=${CONFIG}"
echo "source=${SOURCE_JSONL}"
echo "data=${DATA_JSONL}"
echo "init=${INIT_CHECKPOINT}"
echo "out_dir=${OUT_DIR}"
echo "max_cases=${MAX_CASES}"
echo "diag_every=${DIAG_EVERY} (plain prompt diagnostics do not inject hidden workspace evidence)"
echo "============================================================"

if [[ ! -f "$SOURCE_JSONL" ]]; then
  echo "Missing source preference data: $SOURCE_JSONL" >&2
  exit 1
fi
if [[ ! -f "$INIT_CHECKPOINT" ]]; then
  echo "Missing init checkpoint: $INIT_CHECKPOINT" >&2
  exit 1
fi

python scripts/125_build_workspace_counterfactual_preferences.py \
  --input-jsonl "$SOURCE_JSONL" \
  --output-jsonl "$DATA_JSONL"

MULTIMODAL=0 DATA_JSONL="$DATA_JSONL" bash scripts/08_train_donor_adapter.sh "$CONFIG" \
  --init-checkpoint "$INIT_CHECKPOINT" \
  --diag-every "$DIAG_EVERY" \
  --diag-max-new-tokens "$DIAG_MAX_NEW_TOKENS" \
  --save-every "$SAVE_EVERY" \
  --diag-prompt "Answer using only the hidden MemoryOS workspace evidence. Return only the short answer." \
  --diag-prompt "Change the answer when the hidden evidence changes."

python scripts/121_eval_preference_pairs.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --data-jsonl "$DATA_JSONL" \
  --jsonl-out "$PREFERENCE_EVAL_OUT"

CONFIG="$CONFIG" CHECKPOINT="$CHECKPOINT" MAX_CASES="$MAX_CASES" \
  SUPPRESS_VISIBLE_REASONING="$SUPPRESS_VISIBLE_REASONING" \
  NO_REPEAT_NGRAM_SIZE="$NO_REPEAT_NGRAM_SIZE" \
  SHORT_ANSWER_GOVERNOR="$SHORT_ANSWER_GOVERNOR" \
  OUT="$NORMAL_OUT" \
  AUDIT_OUT="$NORMAL_AUDIT_OUT" \
  PROOF_MD="$NORMAL_PROOF_MD" \
  PROOF_JSON="$NORMAL_PROOF_JSON" \
  bash scripts/117_run_workspace_evidence_path_probe.sh

python scripts/148_build_root_architecture_gate.py \
  --eval-jsonl "$NORMAL_OUT" \
  --markdown-out "$NORMAL_ROOT_MD" \
  --json-out "$NORMAL_ROOT_JSON"

python scripts/build_workspace_counterfactual_eval_cases.py \
  --cases data/eval/memory_reasoning_heldout_expanded_72.jsonl \
  --out "$SWAP_CASES" \
  --max-cases "$MAX_CASES"

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

python scripts/95_eval_memory_retrieval.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --cases "$SWAP_CASES" \
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
  --evidence-mode all \
  --evidence-injection workspace \
  --max-length 384 \
  --max-new-tokens 32 \
  --qtrm-logits-scale 0.5 \
  --donor-logits-scale 1.0 \
  --no-logit-shift \
  "${DECODE_GUARD_ARGS[@]}" \
  --jsonl-out "$SWAP_OUT" \
  --audit-jsonl-out "$SWAP_AUDIT_OUT"

python scripts/148_build_root_architecture_gate.py \
  --eval-jsonl "$SWAP_OUT" \
  --markdown-out "$SWAP_ROOT_MD" \
  --json-out "$SWAP_ROOT_JSON"

echo "wrote $CHECKPOINT"
echo "wrote $NORMAL_ROOT_MD"
echo "wrote $SWAP_ROOT_MD"
