#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

SAMPLED_DATA="/mnt/nvme0n1p2/tmp/hrm_text_dataio_sample_82m_20260530/sampled"
OUT_DIR="${ROOT}/local_eval/20260530_TINY_NANO_TEST"

# Point to local ptxas (using CUDA 12.8 on local 4090)
PTXAS="/usr/local/cuda-12.8/bin/ptxas"
export REQUIRED_TRITON_PTXAS_PATH="${PTXAS}"
export TRITON_PTXAS_PATH="${PTXAS}"
export PATH="$(dirname "${PTXAS}"):${PATH}"

mkdir -p "${OUT_DIR}"

echo "Starting local Nano-size (345K) BLT pretrain smoke test on dataset: ${SAMPLED_DATA}"

"${PYTHON}" "${ROOT}/scripts/557_train_blt_d_prefixlm_dataio.py" \
  --sampled-data "${SAMPLED_DATA}" \
  --out-dir "${OUT_DIR}" \
  --steps 10 \
  --checkpoint-every 10 \
  --batch-size 4 \
  --seq-len 256 \
  --eval-sampled-data "${SAMPLED_DATA}" \
  --eval-every 10 \
  --eval-epoch 0 \
  --eval-max-rows 16 \
  --eval-batch-size 2 \
  --eval-max-batches 0 \
  --lr 5e-4 \
  --lr-warmup-steps 2 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --weight-decay 0.1 \
  --grad-clip 1.0 \
  --amp-dtype bf16 \
  --matmul-precision high \
  --log-every 2 \
  --tensorboard-dir "${OUT_DIR}/tensorboard" \
  --patch-size 4 \
  --patch-boundary-mode hnet_dechunk \
  --dynamic-min-patch-size 2 \
  --dynamic-soft-patch-size 0 \
  --hbf-boundary-threshold 0.35 \
  --boundary-prior-weight 0.0 \
  --boundary-target-ratio 0.5 \
  --decoder-latent-mode one_body \
  --diffusion-weight 0.0 \
  --diffusion-mask-prob 0.0 \
  --d-model 32 \
  --n-heads 1 \
  --n-kv-heads 1 \
  --d-ff 64 \
  --dropout 0.0 \
  --backbone trm_qwen35_3to1 \
  --think-structure trm_dual_z \
  --train-think-steps 1 \
  --hybrid-layers 1 \
  --attn-every 4 \
  --delta-backend official_gated_delta2 \
  --strict-backends \
  --attention-backend sdpa \
  --seed 9999 \
  --no-save-optimizer-checkpoint \
  --no-online-opus-enabled \
  --allow-missing-past-success-preflight \
  --acknowledge-past-success-restoration-gap

echo "Local Nano smoke test execution triggered."
