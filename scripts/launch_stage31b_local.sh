#!/usr/bin/env bash
set -euo pipefail

WAIT_PID="${1:-}"
ROOT="/home/tripleyoung/qtrm-workspace/qtrm_multimodal_memoryos"

if [ -n "${WAIT_PID}" ]; then
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    sleep 30
  done
fi

cd "${ROOT}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN="${TS}_PERSIST_STAGE31B_LSCR_SDST_IdentityBiased_seed121"
OUT="/mnt/sdc1/tripleyoung/qtrm_eval/${RUN}"
LOG="/tmp/${RUN}.log"
mkdir -p "${OUT}"

echo "Starting Stage 31B: ${RUN}"
echo "Logging to: ${LOG}"

env PYTHONUNBUFFERED=1 QTRM_AIM_REPO=/mnt/sdc1/tripleyoung/qtrm_aim PYTHONPATH=src \
  .venv/bin/python scripts/511_train_qwen_state_transition_hrmtext.py \
  --out-dir "${OUT}" \
  --run-name "${RUN}" \
  --aim-run-name "${RUN}" \
  --aim-experiment qwen35_hrmtext_stage31_lscr_sdst \
  --aim-description "Stage31B Identity-Biased Recurrence with LayerScale 1e-5 and SDST decay 0.95" \
  --resume /mnt/sdc1/tripleyoung/qtrm_eval/20260521_113012_PERSIST_STAGE24A_SilentIdentityCore_from20A_seed102/last.pt \
  --qwen-model-id Qwen/Qwen3.5-0.8B-Base \
  --synthetic-schema generalized \
  --train-depths 4 6 8 \
  --synthetic-sampling-strategy stratified \
  --synthetic-family-mix chain2_checksum1 \
  --reasoning-condition-prefix synth \
  --reasoning-count 512 \
  --reasoning-weight 2.0 \
  --healing-data-path /mnt/sdc1/tripleyoung/qtrm_data/20260521_102930_HRMText_verified_dolly_mix_v1 \
  --healing-count 3000 \
  --healing-weight 0.10 \
  --healing-target-tokens 8 \
  --healing-rows-per-file-cap 2000 \
  --source-eval-data-path /mnt/sdc1/tripleyoung/qtrm_data/20260521_105939_HRMText_verified_eval_offset500_v1 \
  --source-eval-include-glob data/verified_reasoning.jsonl \
  --source-eval-count 512 \
  --source-eval-target-tokens 8 \
  --source-eval-batch-size 16 \
  --workspace-pooling attention \
  --operation-arg-conditioning \
  --core-impl state_transition \
  --core-update mini_gated_delta \
  --state-update-schedule nested \
  --recurrent-readout-pooling final \
  --freeze-qwen \
  --continuous-time \
  --zero-step-embeddings \
  --freeze-step-embeddings \
  --transition-scale-init 0.05 \
  --override-transition-scale 0.05 \
  --override-injection-gate-logit 3.0 \
  --layerscale \
  --layerscale-init 1e-5 \
  --gate-type sigmoid \
  --gate-bias-init -2.0 \
  --state-supervision-weight 1.0 \
  --state-supervision-decay-rate 0.95 \
  --state-supervision-min-weight 0.05 \
  --aux-step-answer-weight 0.1 \
  --depth-consistency-weight 0.05 \
  --depth-consistency-temperature 1.0 \
  --consistency-min-steps 4 \
  --latent-shortcut-consistency-weight 0.2 \
  --latent-shortcut-consistency-min-step 1 \
  --lattice-candidate-weight 0.0 \
  --n-steps 12 \
  --depth-sample-min 4 \
  --epochs 8 \
  --batch-size 8 \
  --lr 3e-5 \
  --grad-clip 1.0 \
  --max-length 160 \
  --seed 121 \
  --dataloader-seed 121 \
  --eval-every 1 \
  --eval-count 256 \
  --eval-batch-size 4 \
  --eval-depths 4 6 8 10 12 \
  --eval-seed 10042 \
  --checkpoint-every 1 \
  --save-best-checkpoint \
  --save-best-generalization-checkpoint \
  --save-trainable-only \
  --log-every 20 > "${LOG}" 2>&1
