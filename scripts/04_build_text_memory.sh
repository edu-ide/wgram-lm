#!/usr/bin/env bash
set -euo pipefail
INPUT_DIR=${1:-data/docs}
OUT_DIR=${2:-memory/text}
MODEL_ID=${EMBED_MODEL:-microsoft/harrier-oss-v1-270m}
BACKEND=${MEMORY_BACKEND:-faiss_flat}
HNSW_M=${HNSW_M:-32}
HNSW_EF_CONSTRUCTION=${HNSW_EF_CONSTRUCTION:-200}
export PYTHONPATH=$PWD/src
mkdir -p "$INPUT_DIR"
if [ ! -f "$INPUT_DIR/example.md" ]; then
  cat > "$INPUT_DIR/example.md" <<'DOC'
# QTRM MemoryOS
QTRM uses recursive latent states, MemoryOS, LLM Wiki, embeddings, and verifier-grounded retrieval.
DOC
fi
python -m wgram_lm.memoryos.wiki_compile "$INPUT_DIR" memory/wiki
python -m wgram_lm.memoryos.text_index "$INPUT_DIR" "$OUT_DIR" \
  --model-id "$MODEL_ID" \
  --backend "$BACKEND" \
  --hnsw-m "$HNSW_M" \
  --hnsw-ef-construction "$HNSW_EF_CONSTRUCTION"
