#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

# Use the pre-packed high-quality math/code/SFT mix data (UltraData rehearsal style)
SAMPLED_DATA="/mnt/nvme0n1p2/tmp/hrm_text_dataio_sample_82m_20260530/sampled"
RUN_NAME="20260530_LOCAL_82M_GENERAL_REASONING_LLM"
OUT_DIR="${ROOT}/local_eval/${RUN_NAME}"

# Local CUDA 12.8 ptxas (required for Triton on this box)
PTXAS="/usr/local/cuda-12.8/bin/ptxas"
export REQUIRED_TRITON_PTXAS_PATH="${PTXAS}"
export TRITON_PTXAS_PATH="${PTXAS}"
export PATH="$(dirname "${PTXAS}"):${PATH}"

# Memory allocator hints for 24GB 4090 + heavy Triton GDN2 kernels + attractor
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.75,max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1

mkdir -p "${OUT_DIR}/tensorboard"

echo "================================================================"
echo "  LOCAL 82M GENERAL REASONING LLM (Dynamic-BLT + Attractor)"
echo "================================================================"
echo "Run name     : ${RUN_NAME}"
echo "Output       : ${OUT_DIR}"
echo "Dataset      : ${SAMPLED_DATA}"
echo "Target steps : 2000 (batch=4, seq=256 → ~2.0M tokens seen per 1000 steps; safe for 4090 GDN2)"
echo "Model        : d_model=384 / hybrid=4 / one_body + hnet_dechunk (attractor light for 4090 fit)"
echo "GPU          : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'RTX 4090')"
echo "================================================================"

# Kill any stale trainer on this exact run (defensive)
pkill -f "557_train_blt_d_prefixlm_dataio.*${RUN_NAME}" || true
sleep 1

# Launch with proper nohup + disown so it survives terminal close
nohup "${PYTHON}" -u "${ROOT}/scripts/557_train_blt_d_prefixlm_dataio.py" \
  --sampled-data "${SAMPLED_DATA}" \
  --out-dir "${OUT_DIR}" \
  --steps 2000 \
  --checkpoint-every 200 \
  --batch-size 4 \
  --seq-len 256 \
  --eval-sampled-data "${SAMPLED_DATA}" \
  --eval-every 200 \
  --eval-epoch 0 \
  --eval-max-rows 128 \
  --eval-batch-size 2 \
  --eval-max-batches 0 \
  --lr 2.2e-4 \
  --lr-warmup-steps 100 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --weight-decay 0.1 \
  --grad-clip 1.0 \
  --amp-dtype bf16 \
  --matmul-precision high \
  --log-every 25 \
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
  --answer-attractor-depths 1 2 \
  --answer-attractor-ce-weight 0.0 \
  --hybrid-layers 4 \
  --attn-every 4 \
  --delta-backend official_gated_delta2 \
  --strict-backends \
  --attention-backend sdpa \
  --activation-checkpointing \
  --seed 10602 \
  --no-save-optimizer-checkpoint \
  --no-online-opus-enabled \
  --allow-missing-past-success-preflight \
  --acknowledge-past-success-restoration-gap \
  >> "${OUT_DIR}/train.log" 2>&1 &

TRAIN_PID=$!
echo "${TRAIN_PID}" > "${OUT_DIR}/trainer.pid"

# Give it a moment to start / hit first compile or data load error
sleep 4

if ps -p "${TRAIN_PID}" > /dev/null 2>&1; then
  echo "✅ Training launched successfully (PID ${TRAIN_PID})"
  echo "   Log      : tail -f ${OUT_DIR}/train.log"
  echo "   TensorBoard: http://localhost:6006  (pointed at local_eval/)"
  echo ""
  echo "First 40 lines of log:"
  head -n 40 "${OUT_DIR}/train.log"
else
  echo "❌ Training process died immediately. Dumping head of log:"
  cat "${OUT_DIR}/train.log" | head -100
  exit 1
fi
