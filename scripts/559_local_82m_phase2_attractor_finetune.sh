#!/usr/bin/env bash
set -euo pipefail

# Phase 2: Attractor "Thinking" Finetune on the Phase 1 foundation checkpoint.
# Goal: Explicitly train multi-step reasoning / fixed-point convergence (Attractor) on top of
# the already-trained Dynamic-BLT + one-body recurrent core.
#
# This is the step where we activate the "더 깊게 생각하면 더 잘 맞춘다" signal (RI-1 style).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

# === Phase 1 best checkpoint (foundation) ===
PHASE1_CKPT="local_eval/20260530_82M_GENERAL_LLM_SCRATCH/best_eval_model.pt"

# === Data (same high-quality mix as Phase 1) ===
SAMPLED_DATA="/mnt/nvme0n1p2/tmp/hrm_text_dataio_sample_82m_20260530/sampled"

# === Phase 2 run identity ===
RUN_NAME="20260530_82M_ATTRACTOR_PHASE2_THINKING"
OUT_DIR="${ROOT}/local_eval/${RUN_NAME}"

# CUDA / Triton (same as before)
PTXAS="/usr/local/cuda-12.8/bin/ptxas"
export REQUIRED_TRITON_PTXAS_PATH="${PTXAS}"
export TRITON_PTXAS_PATH="${PTXAS}"
export PATH="$(dirname "${PTXAS}"):${PATH}"

# Memory friendly allocator (attractor adds extra forwards)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:64

mkdir -p "${OUT_DIR}/tensorboard"

echo "================================================================"
echo "  PHASE 2: ATTRACTOR THINKING FINETUNE (82M Dynamic-BLT)"
echo "================================================================"
echo "Run name      : ${RUN_NAME}"
echo "Resume from   : ${PHASE1_CKPT}"
echo "Output        : ${OUT_DIR}"
echo "Data          : ${SAMPLED_DATA}"
echo "Target steps  : 400 (short, focused attractor training)"
echo "Key changes   : train-think-steps=4, attractor depths 1/2/4, ce-weight 0.05"
echo "================================================================"

# Kill any previous attempt on the exact same run name
pkill -f "557_train_blt_d_prefixlm_dataio.*${RUN_NAME}" || true
sleep 1

# Launch Phase 2
# - Resume the Phase 1 best checkpoint (non-strict so any new attractor heads can init cleanly)
# - Attractor regularization now turned ON with meaningful weight and multiple depths
# - Keep the same core architecture and data contract as Phase 1
nohup "${PYTHON}" -u "${ROOT}/scripts/557_train_blt_d_prefixlm_dataio.py" \
  --resume "${PHASE1_CKPT}" \
  --no-resume-strict \
  --acknowledge-past-success-restoration-gap \
  --sampled-data "${SAMPLED_DATA}" \
  --out-dir "${OUT_DIR}" \
  --steps 400 \
  --checkpoint-every 100 \
  --batch-size 3 \
  --seq-len 256 \
  --eval-sampled-data "${SAMPLED_DATA}" \
  --eval-every 100 \
  --eval-epoch 0 \
  --eval-max-rows 64 \
  --eval-batch-size 1 \
  --eval-max-batches 0 \
  --lr 1.6e-4 \
  --lr-warmup-steps 40 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --weight-decay 0.1 \
  --grad-clip 1.0 \
  --amp-dtype bf16 \
  --matmul-precision high \
  --log-every 15 \
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
  --train-think-steps 4 \
  --answer-attractor-depths 1 2 \
  --answer-attractor-ce-weight 0.05 \
  --answer-attractor-monotonic-weight 0.02 \
  --answer-attractor-residual-wrong-weight 0.01 \
  --hybrid-layers 4 \
  --attn-every 4 \
  --delta-backend official_gated_delta2 \
  --strict-backends \
  --attention-backend sdpa \
  --seed 10603 \
  --no-save-optimizer-checkpoint \
  --no-online-opus-enabled \
  --allow-missing-past-success-preflight \
  >> "${OUT_DIR}/train.log" 2>&1 &

TRAIN_PID=$!
echo "${TRAIN_PID}" > "${OUT_DIR}/trainer.pid"

sleep 8

if ps -p "${TRAIN_PID}" > /dev/null 2>&1; then
  echo "✅ Phase 2 Attractor Finetune launched successfully (PID ${TRAIN_PID})"
  echo ""
  echo "   Log      : tail -f ${OUT_DIR}/train.log"
  echo "   TensorBoard : http://localhost:6006"
  echo "   Expected duration : ~60-90 minutes for 400 steps"
  echo ""
  echo "=== First 30 lines of log (should show resume + attractor activation) ==="
  head -n 30 "${OUT_DIR}/train.log"
else
  echo "❌ Phase 2 process died immediately. Dumping head of log:"
  cat "${OUT_DIR}/train.log" | head -100
  exit 1
fi
