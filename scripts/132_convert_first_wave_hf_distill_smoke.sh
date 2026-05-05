#!/usr/bin/env bash
set -euo pipefail

MAX_ROWS="${MAX_ROWS:-100}"
OUT_DIR="${OUT_DIR:-data/filtered/hf_distill_smoke}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$OUT_DIR"

echo "Converting first-wave HF distillation datasets"
echo "max_rows=$MAX_ROWS"
echo "out_dir=$OUT_DIR"

"$PYTHON_BIN" scripts/131_convert_hf_distill_dataset.py \
  --adapter yana_reasoning_dpo \
  --hf-id Yana/ft-llm-2026-reasoning-dpo \
  --split train \
  --out "$OUT_DIR/yana_reasoning_dpo_s${MAX_ROWS}.jsonl" \
  --max-rows "$MAX_ROWS"

"$PYTHON_BIN" scripts/131_convert_hf_distill_dataset.py \
  --adapter noesis_text_sft \
  --hf-id AMAImedia/NOESIS-50K-reasoning-router-code-math-psych-opus47-deepseek4-qwen36-gemini31-r1-gpt54 \
  --split train \
  --out "$OUT_DIR/noesis_50k_reasoning_sft_s${MAX_ROWS}.jsonl" \
  --max-rows "$MAX_ROWS"

"$PYTHON_BIN" scripts/131_convert_hf_distill_dataset.py \
  --adapter ragognize \
  --hf-id F4biian/RAGognize \
  --split train \
  --out "$OUT_DIR/ragognize_evidence_s${MAX_ROWS}.jsonl" \
  --max-rows "$MAX_ROWS"

"$PYTHON_BIN" scripts/131_convert_hf_distill_dataset.py \
  --adapter halluclaim_76k \
  --hf-id lrsbrgrn/HalluClaim-76k \
  --split train \
  --out "$OUT_DIR/halluclaim_76k_s${MAX_ROWS}.jsonl" \
  --max-rows "$MAX_ROWS"

wc -l "$OUT_DIR"/*.jsonl
