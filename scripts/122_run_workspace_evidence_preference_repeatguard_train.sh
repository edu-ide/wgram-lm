#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

export PYTHONPATH=${PYTHONPATH:-$PWD/src}
export HF_HOME=${HF_HOME:-~/.cache/huggingface}

CONFIG="${CONFIG:-configs/qwen35_2b_4090_workspace_evidence_preference_repeatguard_s050.yaml}"
DATA_JSONL="${DATA_JSONL:-data/filtered/memory_self_improvement_preferences_analysis.jsonl}"
OUT_DIR="${OUT_DIR:-runs/qwen35_2b_4090_workspace_evidence_preference_repeatguard_s050}"
CHECKPOINT="${CHECKPOINT:-$OUT_DIR/last.pt}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-runs/qwen35_2b_4090_memory_gated_workspace_s050/last.pt}"
DIAG_EVERY="${DIAG_EVERY:-100}"
DIAG_MAX_NEW_TOKENS="${DIAG_MAX_NEW_TOKENS:-16}"
SAVE_EVERY="${SAVE_EVERY:-0}"
PREFERENCE_EVAL_OUT="${PREFERENCE_EVAL_OUT:-runs/eval/workspace_evidence_preference_repeatguard_pair_eval_s050.jsonl}"

echo "============================================================"
echo "Workspace evidence-path preference + repeatguard training"
echo "config=${CONFIG}"
echo "data=${DATA_JSONL}"
echo "init=${INIT_CHECKPOINT}"
echo "out_dir=${OUT_DIR}"
echo "============================================================"

if [[ ! -f "$DATA_JSONL" ]]; then
  echo "Missing preference data: $DATA_JSONL" >&2
  exit 1
fi
if [[ ! -f "$INIT_CHECKPOINT" ]]; then
  echo "Missing init checkpoint: $INIT_CHECKPOINT" >&2
  exit 1
fi

MULTIMODAL=0 DATA_JSONL="$DATA_JSONL" bash scripts/08_train_donor_adapter.sh "$CONFIG" \
  --init-checkpoint "$INIT_CHECKPOINT" \
  --diag-every "$DIAG_EVERY" \
  --diag-max-new-tokens "$DIAG_MAX_NEW_TOKENS" \
  --save-every "$SAVE_EVERY" \
  --diag-prompt "Answer using only the hidden MemoryOS workspace evidence. Return only the short answer." \
  --diag-prompt "Prefer the verified answer over tempting but contradicted evidence."

python scripts/121_eval_preference_pairs.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --data-jsonl "$DATA_JSONL" \
  --jsonl-out "$PREFERENCE_EVAL_OUT"

CONFIG="$CONFIG" CHECKPOINT="$CHECKPOINT" \
  OUT="${OUT:-runs/eval/memory_reasoning_heldout_expanded_workspace_evidence_preference_repeatguard_32tok_trained_s050.jsonl}" \
  AUDIT_OUT="${AUDIT_OUT:-runs/eval/memory_reasoning_heldout_expanded_workspace_evidence_preference_repeatguard_32tok_trained_s050_audit.jsonl}" \
  PROOF_MD="${PROOF_MD:-docs/wiki/decisions/workspace-evidence-preference-repeatguard-trained-ablation.md}" \
  PROOF_JSON="${PROOF_JSON:-docs/wiki/decisions/workspace-evidence-preference-repeatguard-trained-ablation-summary.json}" \
  bash scripts/117_run_workspace_evidence_path_probe.sh

echo "wrote $CHECKPOINT"
