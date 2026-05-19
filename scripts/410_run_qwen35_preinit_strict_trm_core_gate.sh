#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/mnt/data4tb/hf-cache-qtrm}"
export PYTHONPATH="${PYTHONPATH:-src}"

PYTHON="${PYTHON:-.venv/bin/python}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-2B-Base}"
OUT_DIR="${OUT_DIR:-local_eval/qwen35_preinit_strict_trm_core_gate_s${STEPS:-120}_$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
EXTRA_ARGS=()
if [[ "${EVAL_FORCE_TRAJECTORY_CARRY_OFF:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--eval-force-trajectory-carry-off)
fi

"${PYTHON}" scripts/362_train_qwen_backbone_qtrm_core_gate.py \
  --model-id "${MODEL_ID}" \
  --out-dir "${OUT_DIR}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --max-seq-len "${MAX_SEQ_LEN:-128}" \
  --core-impl qwen_shared_layer_wrapped \
  --qwen-core-layer-indices "${QWEN_CORE_LAYER_INDICES:-3}" \
  --mandatory-core \
  --core-adapter-dim "${CORE_ADAPTER_DIM:-128}" \
  --core-delta-adapter-mode "${CORE_DELTA_ADAPTER_MODE:-add}" \
  --core-insertion-mode "${CORE_INSERTION_MODE:-final_residual}" \
  --core-insert-after-layer "${CORE_INSERT_AFTER_LAYER:--1}" \
  --core-residual-gate-mode "${CORE_RESIDUAL_GATE_MODE:-constant}" \
  --core-residual-gate-dim "${CORE_RESIDUAL_GATE_DIM:-128}" \
  --core-residual-gate-init "${CORE_RESIDUAL_GATE_INIT:--2.0}" \
  --core-trajectory-carry-mode "${CORE_TRAJECTORY_CARRY_MODE:-none}" \
  --core-trajectory-carry-gate-init "${CORE_TRAJECTORY_CARRY_GATE_INIT:-0.0}" \
  --residual-scale "${RESIDUAL_SCALE:-0.05}" \
  --h-cycles "${H_CYCLES:-1}" \
  --l-cycles "${L_CYCLES:-3}" \
  --outer-steps "${OUTER_STEPS:-1}" \
  --core-step-conditioning-enabled \
  --core-step-conditioning-max-steps "${CORE_STEP_CONDITIONING_MAX_STEPS:-64}" \
  --core-step-conditioning-scale "${CORE_STEP_CONDITIONING_SCALE:-1.0}" \
  --steps "${STEPS:-120}" \
  --batch-size "${BATCH_SIZE:-2}" \
  --train-cases "${TRAIN_CASES:-1024}" \
  --train-seed-offsets "${TRAIN_SEED_OFFSETS:-0}" \
  --eval-cases "${EVAL_CASES:-512}" \
  --eval-seed-offsets "${EVAL_SEED_OFFSETS:-10000}" \
  --case-mode "${CASE_MODE:-hard_v1}" \
  --acceptance-metric "${ACCEPTANCE_METRIC:-label_choice}" \
  --lr "${LR:-2.0e-4}" \
  --qwen-lr "${QWEN_LR:-5.0e-5}" \
  --unfreeze-qwen-layer-indices "${UNFREEZE_QWEN_LAYER_INDICES:-}" \
  --weight-decay "${WEIGHT_DECAY:-0.0}" \
  --qwen-weight-decay "${QWEN_WEIGHT_DECAY:-0.0}" \
  --grad-clip "${GRAD_CLIP:-1.0}" \
  --kl-weight "${KL_WEIGHT:-0.05}" \
  --language-kl-weight "${LANGUAGE_KL_WEIGHT:-0.05}" \
  --language-kl-batch-size "${LANGUAGE_KL_BATCH_SIZE:-2}" \
  --language-probe-set "${LANGUAGE_PROBE_SET:-basic}" \
  --language-healing-weight "${LANGUAGE_HEALING_WEIGHT:-0.0}" \
  --language-healing-kl-weight "${LANGUAGE_HEALING_KL_WEIGHT:-0.0}" \
  --language-healing-batch-size "${LANGUAGE_HEALING_BATCH_SIZE:-2}" \
  --selection-language-weight "${SELECTION_LANGUAGE_WEIGHT:-0.0}" \
  --selection-min-language-top1 "${SELECTION_MIN_LANGUAGE_TOP1:-0.0}" \
  --core-advantage-weight "${CORE_ADVANTAGE_WEIGHT:-0.0}" \
  --core-advantage-margin "${CORE_ADVANTAGE_MARGIN:-0.0}" \
  --core-advantage-mode "${CORE_ADVANTAGE_MODE:-target_logp}" \
  --family-loss-weights "${FAMILY_LOSS_WEIGHTS:-}" \
  --checksum-counterfactual-weight "${CHECKSUM_COUNTERFACTUAL_WEIGHT:-0.0}" \
  --checksum-counterfactual-variants "${CHECKSUM_COUNTERFACTUAL_VARIANTS:-1}" \
  --checksum-base-error-advantage-weight "${CHECKSUM_BASE_ERROR_ADVANTAGE_WEIGHT:-0.0}" \
  --checksum-base-error-margin "${CHECKSUM_BASE_ERROR_MARGIN:-0.05}" \
  --checksum-base-error-base-margin-threshold "${CHECKSUM_BASE_ERROR_BASE_MARGIN_THRESHOLD:-0.0}" \
  --checksum-latent-answer-weight "${CHECKSUM_LATENT_ANSWER_WEIGHT:-0.0}" \
  --checksum-latent-answer-source "${CHECKSUM_LATENT_ANSWER_SOURCE:-z_h}" \
  --checksum-latent-answer-lr "${CHECKSUM_LATENT_ANSWER_LR:-1.0e-3}" \
  --checksum-latent-answer-weight-decay "${CHECKSUM_LATENT_ANSWER_WEIGHT_DECAY:-0.01}" \
  --checksum-trajectory-weight "${CHECKSUM_TRAJECTORY_WEIGHT:-0.0}" \
  --eval-every-steps "${EVAL_EVERY_STEPS:-40}" \
  --restore-best-checkpoint \
  --min-reasoning-gain "${MIN_REASONING_GAIN:-0.02}" \
  --min-language-top1-agreement "${MIN_LANGUAGE_TOP1_AGREEMENT:-0.75}" \
  --min-family-gain "${MIN_FAMILY_GAIN:-0.0}" \
  --min-family-core-accuracy "${MIN_FAMILY_CORE_ACCURACY:-0.08}" \
  --init-checkpoint "${INIT_CHECKPOINT:-}" \
  --seed "${SEED:-20260519}" \
  --log-every "${LOG_EVERY:-20}" \
  "${EXTRA_ARGS[@]}"
