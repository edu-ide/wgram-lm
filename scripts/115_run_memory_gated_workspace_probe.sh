#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

export PYTHONPATH=${PYTHONPATH:-$PWD/src}
export HF_HOME=${HF_HOME:-~/.cache/huggingface}

CONFIG="${CONFIG:-configs/qwen35_2b_4090_memory_gated_workspace_s050.yaml}"
DATA_JSONL="${DATA_JSONL:-data/filtered/memory_reasoning_synth_traces.jsonl}"
OUT_DIR="${OUT_DIR:-runs/qwen35_2b_4090_memory_gated_workspace_s050}"
CHECKPOINT="${CHECKPOINT:-$OUT_DIR/last.pt}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-runs/qwen35_2b_4090_memory_synth_generalization_s050/last.pt}"
DIAG_EVERY="${DIAG_EVERY:-100}"
DIAG_MAX_NEW_TOKENS="${DIAG_MAX_NEW_TOKENS:-16}"
SAVE_EVERY="${SAVE_EVERY:-0}"

CASES="${CASES:-data/eval/memory_reasoning_heldout_expanded_72.jsonl}"
INDEX="${INDEX:-runs/eval/memory_reasoning_heldout_expanded_harrier270m_index}"
BASE_OUT="${BASE_OUT:-runs/eval/memory_reasoning_heldout_expanded_qwen3_rerank_32tok_gated_workspace_s050.jsonl}"
WORKSPACE_OUT="${WORKSPACE_OUT:-runs/eval/memory_reasoning_heldout_expanded_workspace_core_ablation_32tok_gated_workspace_s050.jsonl}"
STRICT_OUT="${STRICT_OUT:-runs/eval/memory_reasoning_heldout_expanded_strict_causality_ablation_32tok_gated_workspace_s050.jsonl}"
PROOF_MD="${PROOF_MD:-docs/wiki/decisions/gated-workspace-ablation.md}"
PROOF_JSON="${PROOF_JSON:-docs/wiki/decisions/gated-workspace-ablation-summary.json}"

echo "============================================================"
echo "Memory gated workspace probe"
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
  --diag-prompt "If the evidence is missing, answer UNKNOWN." \
  --diag-prompt "양자 컴퓨팅이란 무엇인가요?"

CONFIG="$CONFIG" CHECKPOINT="$CHECKPOINT" OUT="$BASE_OUT" \
  bash scripts/111_run_residual_adapter_expanded_gate.sh

CONFIG="$CONFIG" CHECKPOINT="$CHECKPOINT" OUT="$WORKSPACE_OUT" \
  bash scripts/112_run_expanded_workspace_core_ablation.sh

# Strict runner includes qtrm_workspace_gate_off_with_evidence.
CONFIG="$CONFIG" CHECKPOINT="$CHECKPOINT" OUT="$STRICT_OUT" \
  bash scripts/114_run_expanded_strict_causality_ablation.sh

python scripts/113_build_expanded_ablation_proof.py \
  --eval "gated residual gate=$BASE_OUT" \
  --eval "gated workspace/core ablation=$WORKSPACE_OUT" \
  --eval "gated strict causality ablation=$STRICT_OUT" \
  --markdown-out "$PROOF_MD" \
  --json-out "$PROOF_JSON"

echo "wrote $CHECKPOINT"
echo "wrote $BASE_OUT"
echo "wrote $WORKSPACE_OUT"
echo "wrote $STRICT_OUT"
echo "wrote $PROOF_MD"
echo "wrote $PROOF_JSON"
