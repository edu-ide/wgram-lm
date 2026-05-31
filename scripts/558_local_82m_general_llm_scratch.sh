#!/usr/bin/env bash
set -euo pipefail

# Minimal, proven-working launch for local 82M-scale Dynamic-BLT + one_body general LLM pretrain on RTX 4090.
# Goal: Train a compact "reasoning engine" (math / code / logic) using high-quality UltraData-style sampled data.
# Architecture highlights: dynamic hnet_dechunk boundary + one_body recurrent core + light attractor.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

SAMPLED_DATA="/mnt/nvme0n1p2/tmp/hrm_text_dataio_sample_82m_20260530/sampled"
RUN_NAME="20260530_82M_GENERAL_LLM_SCRATCH"
OUT_DIR="${ROOT}/local_eval/${RUN_NAME}"

PTXAS="/usr/local/cuda-12.8/bin/ptxas"
export REQUIRED_TRITON_PTXAS_PATH="${PTXAS}"
export TRITON_PTXAS_PATH="${PTXAS}"
export PATH="$(dirname "${PTXAS}"):${PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:64

mkdir -p "${OUT_DIR}/tensorboard"

echo "================================================================"
echo "  LOCAL 82M GENERAL REASONING LLM — SCRATCH PRETRAIN (Phase 1)"
echo "================================================================"
echo "Run           : ${RUN_NAME}"
echo "Data          : ${SAMPLED_DATA} (high-quality math/code/SFT mix)"
echo "Target        : 2000 steps | batch=3, seq=256 (~1.5M tokens / 1000 steps)"
echo "Arch          : Dynamic-BLT (hnet_dechunk) + one_body recurrent + attractor"
echo "d_model       : 384 (n_heads=6, d_ff=1024, hybrid_layers=4)"
echo "Output        : ${OUT_DIR}"
echo "================================================================"

pkill -f "557_train_blt_d_prefixlm_dataio.*${RUN_NAME}" || true
sleep 1

# Proven minimal flag set that successfully completed step 1 + eval with full dynamic features
nohup "${PYTHON}" -u "${ROOT}/scripts/557_train_blt_d_prefixlm_dataio.py" \
  --sampled-data "${SAMPLED_DATA}" \
  --out-dir "${OUT_DIR}" \
  --steps 2000 \
  --checkpoint-every 200 \
  --batch-size 3 \
  --seq-len 256 \
  --eval-sampled-data "${SAMPLED_DATA}" \
  --eval-every 200 \
  --eval-epoch 0 \
  --eval-max-rows 64 \
  --eval-batch-size 1 \
  --eval-max-batches 0 \
  --lr 2.2e-4 \
  --lr-warmup-steps 80 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --weight-decay 0.1 \
  --grad-clip 1.0 \
  --amp-dtype bf16 \
  --matmul-precision high \
  --log-every 20 \
  --tensorboard-dir "${OUT_DIR}/tensorboard" \
  --patch-size 4 \
  --patch-boundary-mode hnet_dechunk \
  --dynamic-min-patch-size 2 \
  --dynamic-soft-patch-size 0 \
  --hbf-boundary-threshold 0.35 \
  --boundary-prior-weight 0.02 \
  --boundary-target-ratio 0.25 \
  --decoder-latent-mode one_body \
  --diffusion-weight 0.0 \
  --diffusion-mask-prob 0.0 \
  --d-model 384 \
  --n-heads 6 \
  --n-kv-heads 2 \
  --d-ff 1024 \
  --dropout 0.0 \
  --backbone trm_qwen35_3to1 \
  --think-structure trm_dual_z \
  --train-think-steps 2 \
  --answer-attractor-depths 1 \
  --answer-attractor-ce-weight 0.0 \
  --hybrid-layers 4 \
  --attn-every 4 \
  --delta-backend official_gated_delta2 \
  --strict-backends \
  --attention-backend sdpa \
  --seed 10602 \
  --no-save-optimizer-checkpoint \
  --no-online-opus-enabled \
  --allow-missing-past-success-preflight \
  --acknowledge-past-success-restoration-gap \
  >> "${OUT_DIR}/train.log" 2>&1 &

TRAIN_PID=$!
echo "${TRAIN_PID}" > "${OUT_DIR}/trainer.pid"

sleep 6

if ps -p "${TRAIN_PID}" > /dev/null 2>&1; then
  echo "✅ Launched (PID ${TRAIN_PID})"
  echo "   Monitor: tail -f ${OUT_DIR}/train.log"
  echo "   TB     : http://localhost:6006"
  echo ""
  echo "=== First 25 log lines ==="
  head -n 25 "${OUT_DIR}/train.log"
else
  echo "❌ Died immediately:"
  cat "${OUT_DIR}/train.log" | head -80
  exit 1
fi
