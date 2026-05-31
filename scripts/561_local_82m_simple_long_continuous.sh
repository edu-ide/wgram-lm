#!/usr/bin/env bash
# Simple long continuous run (no chunking).
# One single python process with a very high step count.
# Pros: Clean continuous log, no repeated "resume_loaded" spam.
# Cons: If it crashes, you have to manually resume. No automatic old checkpoint pruning.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

# === CONFIG - Change these as needed ===
START_CKPT="local_eval/20260530_82M_ATTRACTOR_CONTINUOUS_SAFE_LONG/last_model.pt"  # Latest from the previous safe run
SAMPLED_DATA="/mnt/nvme0n1p2/tmp/hrm_text_dataio_sample_82m_20260530/sampled"

RUN_NAME="20260530_82M_SIMPLE_LONG_CONTINUOUS"  # Clean log + pruner will keep only last ~3 checkpoints
OUT_DIR="${ROOT}/local_eval/${RUN_NAME}"

TOTAL_STEPS=15000          # Just set this high. It will run until it finishes or you kill it.
CHECKPOINT_EVERY=200

# Attractor settings (same conservative safe values as the other long run)
ATTRACTOR_CE_WEIGHT=0.04
ATTRACTOR_DEPTHS="1 2"

LR=1.0e-4
# =====================================

PTXAS="/usr/local/cuda-12.8/bin/ptxas"
export REQUIRED_TRITON_PTXAS_PATH="${PTXAS}"
export TRITON_PTXAS_PATH="${PTXAS}"
export PATH="$(dirname "${PTXAS}"):${PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:64

mkdir -p "${OUT_DIR}/tensorboard"

echo "================================================================"
echo "  SIMPLE LONG CONTINUOUS RUN (clean log version)"
echo "================================================================"
echo "Run name       : ${RUN_NAME}"
echo "Output         : ${OUT_DIR}"
echo "Resume from    : ${START_CKPT}"
echo "Total steps    : ${TOTAL_STEPS}"
echo "Attractor      : depths=${ATTRACTOR_DEPTHS}, weight=${ATTRACTOR_CE_WEIGHT}"
echo "================================================================"

# Kill any previous process using the same run dir (defensive)
pkill -f "${RUN_NAME}" || true
sleep 2

nohup "${PYTHON}" -u "${ROOT}/scripts/557_train_blt_d_prefixlm_dataio.py" \
  --resume "${START_CKPT}" \
  --no-resume-strict \
  --acknowledge-past-success-restoration-gap \
  --sampled-data "${SAMPLED_DATA}" \
  --out-dir "${OUT_DIR}" \
  --steps "${TOTAL_STEPS}" \
  --checkpoint-every "${CHECKPOINT_EVERY}" \
  --batch-size 3 \
  --seq-len 256 \
  --eval-sampled-data "${SAMPLED_DATA}" \
  --eval-every 400 \
  --eval-epoch 0 \
  --eval-max-rows 64 \
  --eval-batch-size 1 \
  --lr "${LR}" \
  --lr-warmup-steps 30 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --weight-decay 0.1 \
  --grad-clip 1.0 \
  --amp-dtype bf16 \
  --matmul-precision high \
  --log-every 30 \
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
  --answer-attractor-depths ${ATTRACTOR_DEPTHS} \
  --answer-attractor-ce-weight "${ATTRACTOR_CE_WEIGHT}" \
  --answer-attractor-monotonic-weight 0.015 \
  --hybrid-layers 4 \
  --attn-every 4 \
  --delta-backend official_gated_delta2 \
  --strict-backends \
  --attention-backend sdpa \
  --seed 10605 \
  --no-save-optimizer-checkpoint \
  --no-online-opus-enabled \
  --allow-missing-past-success-preflight \
  >> "${OUT_DIR}/train.log" 2>&1 &

TRAIN_PID=$!
echo "${TRAIN_PID}" > "${OUT_DIR}/trainer.pid"

echo ""
echo "✅ Simple long continuous training launched (PID ${TRAIN_PID})"
echo "   Log: tail -f ${OUT_DIR}/train.log"
echo "   This will run as ONE single python process → clean continuous log."
echo ""
echo "To stop: pkill -f ${RUN_NAME}   or   kill ${TRAIN_PID}"
echo "To resume later: edit this script and point --resume to ${OUT_DIR}/last_model.pt"
