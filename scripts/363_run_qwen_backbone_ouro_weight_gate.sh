#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="${PYTHONPATH:-src}"
export HF_HOME="${HF_HOME:-/mnt/sdc1/huggingface_cache/qtrm-ouro}"

QWEN_MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen3.5-2B-Base}"
OURO_MODEL_ID="${OURO_MODEL_ID:-/mnt/sdc1/models/ByteDance-Ouro-2.6B-Thinking}"
OURO_LAYER="${OURO_LAYER:-24}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-64}"
STEPS="${STEPS:-80}"
BATCH_SIZE="${BATCH_SIZE:-2}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

if [[ ! -f "$OURO_MODEL_ID/model.safetensors" ]]; then
  echo "Missing Ouro weights: $OURO_MODEL_ID/model.safetensors" >&2
  echo "Download first, or set OURO_MODEL_ID to a complete local snapshot." >&2
  exit 2
fi

SMOKE_DIR="${SMOKE_DIR:-local_eval/qwen_backbone_qtrm_ouro_weight_wrapped_smoke_${RUN_STAMP}}"
TRAIN_DIR="${TRAIN_DIR:-local_eval/qwen_backbone_qtrm_ouro_weight_wrapped_train_gate_s${STEPS}_${RUN_STAMP}}"

echo "=== Actual Ouro-weight QTRM smoke ==="
echo "Qwen:  $QWEN_MODEL_ID"
echo "Ouro:  $OURO_MODEL_ID"
echo "Layer: $OURO_LAYER"
echo "Out:   $SMOKE_DIR"

.venv/bin/python scripts/361_qwen_backbone_qtrm_smoke.py \
  --model-id "$QWEN_MODEL_ID" \
  --out-dir "$SMOKE_DIR" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --core-impl ouro_weight_wrapped \
  --ouro-model-id "$OURO_MODEL_ID" \
  --ouro-core-layer-indices "$OURO_LAYER" \
  --core-adapter-dim 0 \
  --h-cycles 1 \
  --l-cycles 1 \
  --outer-steps 1 \
  --core-gate-on 0.0625

echo "=== Actual Ouro-weight QTRM short train gate ==="
echo "Out: $TRAIN_DIR"

.venv/bin/python scripts/362_train_qwen_backbone_qtrm_core_gate.py \
  --model-id "$QWEN_MODEL_ID" \
  --out-dir "$TRAIN_DIR" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --core-impl ouro_weight_wrapped \
  --ouro-model-id "$OURO_MODEL_ID" \
  --ouro-core-layer-indices "$OURO_LAYER" \
  --core-adapter-dim 64 \
  --steps "$STEPS" \
  --batch-size "$BATCH_SIZE" \
  --train-cases 256 \
  --eval-cases 96 \
  --log-every 20
