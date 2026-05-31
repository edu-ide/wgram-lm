#!/usr/bin/env bash
# Long-running continuous Attractor training for 82M Dynamic-BLT
# Designed to be left running while the user is away.
# It trains in chunks, always resumes from the latest checkpoint,
# and automatically prunes old checkpoints to protect disk space.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

# === CONFIGURATION - TUNE THESE BEFORE LEAVING ===
START_CKPT="local_eval/20260530_82M_GENERAL_LLM_SCRATCH/best_eval_model.pt"
SAMPLED_DATA="/mnt/nvme0n1p2/tmp/hrm_text_dataio_sample_82m_20260530/sampled"

RUN_NAME="20260530_82M_ATTRACTOR_CONTINUOUS_SAFE_LONG"
OUT_DIR="${ROOT}/local_eval/${RUN_NAME}"

TOTAL_TARGET_STEPS=12000          # Very high - run until you come back or kill it
CHUNK_STEPS=300                   # Train this many steps, then save + resume
CHECKPOINT_EVERY=150              # Inside each chunk
PRUNE_KEEP_LAST=6                 # Keep only the last N full checkpoints (saves disk)

LR=1.1e-4
LR_WARMUP=50
ATTRACTOR_CE_WEIGHT=0.04
ATTRACTOR_DEPTHS="1 2"

# ================================================

PTXAS="/usr/local/cuda-12.8/bin/ptxas"
export REQUIRED_TRITON_PTXAS_PATH="${PTXAS}"
export TRITON_PTXAS_PATH="${PTXAS}"
export PATH="$(dirname "${PTXAS}"):${PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:64

mkdir -p "${OUT_DIR}/tensorboard"
mkdir -p "${OUT_DIR}/checkpoints"

echo "================================================================"
echo "  82M ATTRACTOR - CONTINUOUS LONG RUN (leave & return)"
echo "================================================================"
echo "Run name           : ${RUN_NAME}"
echo "Output dir         : ${OUT_DIR}"
echo "Starting from      : ${START_CKPT}"
echo "Target total steps : ${TOTAL_TARGET_STEPS}"
echo "Chunk size         : ${CHUNK_STEPS} steps"
echo "Attractor          : depths=${ATTRACTOR_DEPTHS}, ce_weight=${ATTRACTOR_CE_WEIGHT}"
echo "================================================================"

# Function to get current step from latest checkpoint
get_current_step() {
    local latest=""
    if [[ -f "${OUT_DIR}/last_model.pt" ]]; then
        latest="${OUT_DIR}/last_model.pt"
    elif [[ -f "${OUT_DIR}/best_eval_model.pt" ]]; then
        latest="${OUT_DIR}/best_eval_model.pt"
    else
        # First run - will use START_CKPT
        echo "0"
        return
    fi

    # Try to read step from checkpoint metadata if possible, fallback to 0
    python3 -c "
import torch, sys
try:
    ckpt = torch.load('${latest}', map_location='cpu', weights_only=False)
    step = ckpt.get('step', 0)
    print(step)
except:
    print(0)
" 2>/dev/null || echo "0"
}

# Function to prune old checkpoints
prune_old_checkpoints() {
    echo "[$(date '+%Y-%m-%d %H:%M')] Pruning old checkpoints, keeping last ${PRUNE_KEEP_LAST}..."
    # Move older ones to a subdir or delete (safer to move first)
    find "${OUT_DIR}" -maxdepth 1 -name 'step_*_model.pt' -o -name 'last_model.pt' -o -name 'best_eval_model.pt' 2>/dev/null | \
        sort -t_ -k2 -n | head -n -${PRUNE_KEEP_LAST} | while read f; do
        if [[ -f "$f" ]]; then
            mv "$f" "${OUT_DIR}/checkpoints/" 2>/dev/null || rm -f "$f"
        fi
    done
    # Also prune old tensorboard events if they get too big (optional)
}

CURRENT_STEP=$(get_current_step)
echo "Current step detected: ${CURRENT_STEP}"

if [[ ${CURRENT_STEP} -ge ${TOTAL_TARGET_STEPS} ]]; then
    echo "Already reached or exceeded target steps. Exiting."
    exit 0
fi

REMAINING=$(( TOTAL_TARGET_STEPS - CURRENT_STEP ))
echo "Remaining steps to target: ${REMAINING}"

# Main training loop
LOOP_COUNT=0
while [[ ${CURRENT_STEP} -lt ${TOTAL_TARGET_STEPS} ]]; do
    LOOP_COUNT=$((LOOP_COUNT + 1))
    echo ""
    echo "=== Loop ${LOOP_COUNT} | Current step: ${CURRENT_STEP} | Target: ${TOTAL_TARGET_STEPS} ==="

    REMAINING=$(( TOTAL_TARGET_STEPS - CURRENT_STEP ))
    THIS_CHUNK=$CHUNK_STEPS
    if [[ ${REMAINING} -lt ${THIS_CHUNK} ]]; then
        THIS_CHUNK=${REMAINING}
    fi

    RESUME_PATH=""
    if [[ -f "${OUT_DIR}/last_model.pt" ]]; then
        RESUME_PATH="${OUT_DIR}/last_model.pt"
    elif [[ -f "${OUT_DIR}/best_eval_model.pt" ]]; then
        RESUME_PATH="${OUT_DIR}/best_eval_model.pt"
    else
        RESUME_PATH="${START_CKPT}"
    fi

    echo "Resuming from: ${RESUME_PATH}"
    echo "Training ${THIS_CHUNK} steps this chunk..."

    "${PYTHON}" -u "${ROOT}/scripts/557_train_blt_d_prefixlm_dataio.py" \
        --resume "${RESUME_PATH}" \
        --no-resume-strict \
        --acknowledge-past-success-restoration-gap \
        --sampled-data "${SAMPLED_DATA}" \
        --out-dir "${OUT_DIR}" \
        --steps "${THIS_CHUNK}" \
        --checkpoint-every "${CHECKPOINT_EVERY}" \
        --batch-size 3 \
        --seq-len 256 \
        --eval-sampled-data "${SAMPLED_DATA}" \
        --eval-every 300 \
        --eval-epoch 0 \
        --eval-max-rows 64 \
        --eval-batch-size 1 \
        --lr "${LR}" \
        --lr-warmup-steps "${LR_WARMUP}" \
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
        --train-think-steps 4 \
        --answer-attractor-depths ${ATTRACTOR_DEPTHS} \
        --answer-attractor-ce-weight "${ATTRACTOR_CE_WEIGHT}" \
        --answer-attractor-monotonic-weight 0.015 \
        --hybrid-layers 4 \
        --attn-every 4 \
        --delta-backend official_gated_delta2 \
        --strict-backends \
        --attention-backend sdpa \
        --seed 10604 \
        --no-save-optimizer-checkpoint \
        --no-online-opus-enabled \
        --allow-missing-past-success-preflight \
        >> "${OUT_DIR}/train.log" 2>&1

    EXIT_CODE=$?
    echo "Chunk finished with exit code ${EXIT_CODE}"

    # Update current step
    CURRENT_STEP=$(get_current_step)
    echo "New current step: ${CURRENT_STEP}"

    # Prune old checkpoints to protect disk
    prune_old_checkpoints

    if [[ ${EXIT_CODE} -ne 0 ]]; then
        echo "Non-zero exit code detected. Sleeping 60s before next attempt..."
        sleep 60
    else
        # Small cool-down between chunks (optional)
        sleep 15
    fi

    # Safety: if we're very close to target, stop
    if [[ ${CURRENT_STEP} -ge ${TOTAL_TARGET_STEPS} ]]; then
        echo "Reached target steps (${TOTAL_TARGET_STEPS}). Stopping continuous loop."
        break
    fi

    # Optional hard stop condition (uncomment if you want a hard wall)
    # if [[ ${LOOP_COUNT} -gt 100 ]]; then break; fi
done

echo ""
echo "=== Continuous long run finished or stopped ==="
echo "Final step: ${CURRENT_STEP}"
echo "Log: ${OUT_DIR}/train.log"
echo "Latest best: ${OUT_DIR}/best_eval_model.pt (if saved)"
echo ""
echo "To resume manually later: edit this script or run the trainer directly with --resume ${OUT_DIR}/last_model.pt"
