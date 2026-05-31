#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data4tb/wgram-lm"
cd "${ROOT}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN="${TS}_DGX_STAGE31B_LSCR_SDST_miniGatedDelta_seed121"
OUT="/mnt/data4tb/qtrm_eval/${RUN}"
LOG="/tmp/${RUN}.log"
mkdir -p "${OUT}"

env PYTHONUNBUFFERED=1 QTRM_AIM_REPO=/mnt/data4tb/qtrm_aim PYTHONPATH=src \
  /mnt/data4tb/venv_sglang_pr23000/bin/python scripts/511_train_qwen_state_transition_hrmtext.py \
  --out-dir "${OUT}" \
  --run-name "${RUN}" \
  --aim-run-name "${RUN}" \
  --aim-experiment qwen35_hrmtext_stage31_lscr_sdst \
  --aim-description "Stage31B DGX LSCR plus SDST with mini_gated_delta recurrent core" \
  --resume /mnt/data4tb/qtrm_eval/20260521_113013_DGX_STAGE24B_SilentIdentityCore_from20B_seed103/last.pt \
  --qwen-model-id Qwen/Qwen3.5-0.8B-Base \
  --synthetic-schema generalized \
  --train-depths 4 6 8 \
  --synthetic-sampling-strategy stratified \
  --synthetic-family-mix chain2_checksum1 \
  --reasoning-condition-prefix synth \
  --reasoning-count 512 \
  --reasoning-weight 2.0 \
  --healing-data-path /mnt/data4tb/qtrm_data/20260521_102930_HRMText_verified_dolly_mix_v1 \
  --healing-count 3000 \
  --healing-weight 0.10 \
  --healing-target-tokens 8 \
  --healing-rows-per-file-cap 2000 \
  --source-eval-data-path /mnt/data4tb/qtrm_data/20260521_105939_HRMText_verified_eval_offset500_v1 \
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
  --layerscale-init 0.1 \
  --state-supervision-weight 1.0 \
  --state-supervision-decay-rate 0.9 \
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
