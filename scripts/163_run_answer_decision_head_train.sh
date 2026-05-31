#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/qwen35_2b_4090_answer_decision_head_truthcal_s200.yaml}"
TRAIN_CASES="${TRAIN_CASES:-data/filtered/memory_reasoning_synth_train_cases.jsonl}"
TRAIN_RECORDS="${TRAIN_RECORDS:-docs/wiki/decisions/evidence-span-truthcal-train144-answer-channel-records.jsonl}"
TRAIN_MIX="${TRAIN_MIX:-data/filtered/memory_reasoning_answer_decision_truthcal_train144.jsonl}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-runs/qwen35_2b_4090_evidence_span_reader_truthcal_s300/last.pt}"
HF_HOME="${HF_HOME:-/mnt/nvme1n1p2/hf-cache-qtrm}"
PYTHONPATH="${PYTHONPATH:-src}"

export HF_HOME PYTHONPATH

echo "=== Building answer-decision training mix ==="
uv run python scripts/162_build_answer_decision_training_mix.py \
  --cases-jsonl "$TRAIN_CASES" \
  --records-jsonl "$TRAIN_RECORDS" \
  --out-jsonl "$TRAIN_MIX" \
  --evidence-mode all \
  --retrieval-top-k 4 \
  --record-mode qtrm_residual_with_evidence

echo "=== Training QTRM in-model answer-decision head ==="
uv run python -m wgram_lm.training.train \
  --config "$CONFIG" \
  --use-donor \
  --data-jsonl "$TRAIN_MIX" \
  --init-checkpoint "$INIT_CHECKPOINT" \
  --tokenizer-model-id Qwen/Qwen3.5-2B-Base

echo
echo "Evaluate with:"
echo "HF_HOME=$HF_HOME PYTHONPATH=$PYTHONPATH uv run python scripts/95_eval_memory_retrieval.py \\"
echo "  --config $CONFIG \\"
echo "  --checkpoint runs/qwen35_2b_4090_answer_decision_head_truthcal_s200/last.pt \\"
echo "  --cases data/eval/memory_reasoning_heldout_expanded_72.jsonl \\"
echo "  --mode qtrm_residual_with_evidence --mode qtrm_answer_decision_off_with_evidence \\"
echo "  --answer-channel evidence_span_copy --truth-gate --model-answer-decision \\"
echo "  --evidence-mode all --evidence-injection workspace --retrieval-top-k 4 \\"
echo "  --evidence-span-max-tokens 16 --evidence-span-no-answer-threshold 0.5 \\"
echo "  --short-answer-governor --suppress-visible-reasoning-tokens"
