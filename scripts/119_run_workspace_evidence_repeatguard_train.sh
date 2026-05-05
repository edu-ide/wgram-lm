#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

export PYTHONPATH=${PYTHONPATH:-$PWD/src}
export HF_HOME=${HF_HOME:-~/.cache/huggingface}

CONFIG="${CONFIG:-configs/qwen35_2b_4090_workspace_evidence_repeatguard_s050.yaml}"
DATA_JSONL="${DATA_JSONL:-data/filtered/memory_reasoning_synth_traces.jsonl}"
OUT_DIR="${OUT_DIR:-runs/qwen35_2b_4090_workspace_evidence_repeatguard_s050}"
CHECKPOINT="${CHECKPOINT:-$OUT_DIR/last.pt}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-runs/qwen35_2b_4090_memory_gated_workspace_s050/last.pt}"
DIAG_EVERY="${DIAG_EVERY:-100}"
DIAG_MAX_NEW_TOKENS="${DIAG_MAX_NEW_TOKENS:-16}"
SAVE_EVERY="${SAVE_EVERY:-0}"

echo "============================================================"
echo "Workspace evidence-path repeatguard training"
echo "config=${CONFIG}"
echo "data=${DATA_JSONL}"
echo "init=${INIT_CHECKPOINT}"
echo "out_dir=${OUT_DIR}"
echo "============================================================"

if [[ ! -f "$DATA_JSONL" ]]; then
  echo "Missing training data: $DATA_JSONL" >&2
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
  --diag-prompt "Answer using only the hidden MemoryOS workspace evidence. Question: What is the access code?" \
  --diag-prompt "Repeat only if the evidence itself requires it. Question: What is the final answer?"

CONFIG="$CONFIG" CHECKPOINT="$CHECKPOINT" \
  OUT="${OUT:-runs/eval/memory_reasoning_heldout_expanded_workspace_evidence_repeatguard_32tok_trained_s050.jsonl}" \
  AUDIT_OUT="${AUDIT_OUT:-runs/eval/memory_reasoning_heldout_expanded_workspace_evidence_repeatguard_32tok_trained_s050_audit.jsonl}" \
  PROOF_MD="${PROOF_MD:-docs/wiki/decisions/workspace-evidence-repeatguard-trained-ablation.md}" \
  PROOF_JSON="${PROOF_JSON:-docs/wiki/decisions/workspace-evidence-repeatguard-trained-ablation-summary.json}" \
  bash scripts/117_run_workspace_evidence_path_probe.sh

echo "wrote $CHECKPOINT"
