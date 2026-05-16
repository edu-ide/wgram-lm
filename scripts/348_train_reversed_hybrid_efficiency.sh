#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-777}"
EVAL_SEED="${EVAL_SEED:-9777}"
STAGE="${STAGE:-len4}"
OUT_ROOT="${OUT_ROOT:-local_eval/qtrm_native_reversed_efficiency_$(date +%Y%m%d_%H%M%S)}"
RESUME_FROM="${RESUME_FROM:-}"
TASK_FAMILIES="${TASK_FAMILIES:-checksum,modchain,revchain}"
EVAL_TASK_FAMILIES="${EVAL_TASK_FAMILIES:-checksum,modchain,revchain}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONPATH="references/official/flash-linear-attention:local_deps/mamba3_runtime:src${PYTHONPATH:+:${PYTHONPATH}}"

case "${STAGE}" in
  smoke)
    STEPS="${STEPS:-8}"
    TRAIN_CASES="${TRAIN_CASES:-96}"
    EVAL_CASES="${EVAL_CASES:-48}"
    PROGRAM_LEN="${PROGRAM_LEN:-4}"
    ACTIVE_MIN="${ACTIVE_MIN:-2}"
    ACTIVE_MAX="${ACTIVE_MAX:-4}"
    BATCH_SIZE="${BATCH_SIZE:-8}"
    D_MODEL="${D_MODEL:-64}"
    D_FF="${D_FF:-128}"
    LR="${LR:-1e-4}"
    LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-2}"
    LR_MIN_RATIO="${LR_MIN_RATIO:-0.2}"
    ;;
  len4)
    STEPS="${STEPS:-1800}"
    TRAIN_CASES="${TRAIN_CASES:-8192}"
    EVAL_CASES="${EVAL_CASES:-384}"
    PROGRAM_LEN="${PROGRAM_LEN:-4}"
    ACTIVE_MIN="${ACTIVE_MIN:-2}"
    ACTIVE_MAX="${ACTIVE_MAX:-4}"
    BATCH_SIZE="${BATCH_SIZE:-24}"
    D_MODEL="${D_MODEL:-128}"
    D_FF="${D_FF:-256}"
    LR="${LR:-8e-5}"
    LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-180}"
    LR_MIN_RATIO="${LR_MIN_RATIO:-0.2}"
    ;;
  len6)
    STEPS="${STEPS:-2200}"
    TRAIN_CASES="${TRAIN_CASES:-12288}"
    EVAL_CASES="${EVAL_CASES:-512}"
    PROGRAM_LEN="${PROGRAM_LEN:-6}"
    ACTIVE_MIN="${ACTIVE_MIN:-3}"
    ACTIVE_MAX="${ACTIVE_MAX:-6}"
    BATCH_SIZE="${BATCH_SIZE:-20}"
    D_MODEL="${D_MODEL:-128}"
    D_FF="${D_FF:-256}"
    LR="${LR:-6e-5}"
    LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-220}"
    LR_MIN_RATIO="${LR_MIN_RATIO:-0.2}"
    ;;
  len8)
    STEPS="${STEPS:-2800}"
    TRAIN_CASES="${TRAIN_CASES:-16384}"
    EVAL_CASES="${EVAL_CASES:-768}"
    PROGRAM_LEN="${PROGRAM_LEN:-8}"
    ACTIVE_MIN="${ACTIVE_MIN:-3}"
    ACTIVE_MAX="${ACTIVE_MAX:-8}"
    BATCH_SIZE="${BATCH_SIZE:-16}"
    D_MODEL="${D_MODEL:-128}"
    D_FF="${D_FF:-256}"
    LR="${LR:-5e-5}"
    LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-280}"
    LR_MIN_RATIO="${LR_MIN_RATIO:-0.2}"
    ;;
  *)
    echo "Unsupported STAGE=${STAGE}; expected smoke, len4, len6, or len8" >&2
    exit 2
    ;;
esac

TRAIN_ACTIVE_MIN="${TRAIN_ACTIVE_MIN:-${ACTIVE_MIN}}"
TRAIN_ACTIVE_MAX="${TRAIN_ACTIVE_MAX:-${ACTIVE_MAX}}"
EVAL_ACTIVE_MIN="${EVAL_ACTIVE_MIN:-${ACTIVE_MIN}}"
EVAL_ACTIVE_MAX="${EVAL_ACTIVE_MAX:-${ACTIVE_MAX}}"
REPLAY_ACTIVE_MIN="${REPLAY_ACTIVE_MIN:-${TRAIN_ACTIVE_MIN}}"
REPLAY_ACTIVE_MAX="${REPLAY_ACTIVE_MAX:-${TRAIN_ACTIVE_MAX}}"

mkdir -p "${OUT_ROOT}"
OUT_DIR="${OUT_ROOT}/${STAGE}_seed${SEED}"

command=(
  "${PYTHON_BIN}" scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py
  --out-dir "${OUT_DIR}"
  --target-level "reversed_hybrid_3to1 training-efficiency ${STAGE}"
  --steps "${STEPS}"
  --train-cases "${TRAIN_CASES}"
  --eval-cases "${EVAL_CASES}"
  --task-families "${TASK_FAMILIES}"
  --eval-task-families "${EVAL_TASK_FAMILIES}"
  --program-len "${PROGRAM_LEN}"
  --modulus 32
  --d-model "${D_MODEL}"
  --n-heads 4
  --n-kv-heads 2
  --d-ff "${D_FF}"
  --backbone trm_official
  --encode-backbone mha_etd
  --think-backbone trm_official
  --decode-backbone mha_etd
  --think-structure "${THINK_STRUCTURE:-trm_dual_z_reversed_hybrid_3to1}"
  --delta-backend fla_gated_delta
  --delta-head-dim 32
  --delta-num-v-heads 4
  --delta-expand-v 1.0
  --strict-backends
  --trm-l-cycles 6
  --halt-pooling dedicated
  --batch-size "${BATCH_SIZE}"
  --lr "${LR}"
  --lr-schedule "${LR_SCHEDULE:-linear_warmup_cosine}"
  --lr-warmup-steps "${LR_WARMUP_STEPS}"
  --lr-min-ratio "${LR_MIN_RATIO}"
  --weight-decay 0.01
  --grad-clip 1.0
  --answer-loss-type cross_entropy
  --family-dro-loss-weight "${FAMILY_DRO_LOSS_WEIGHT:-0.0}"
  --family-dro-temperature "${FAMILY_DRO_TEMPERATURE:-0.0}"
  --train-think-steps 3
  --eval-think-steps 3
  --depth-intermediate-loss-weight "${DEPTH_INTERMEDIATE_LOSS_WEIGHT:-0.20}"
  --depth-intermediate-min-depth 1
  --depth-intermediate-weight-power "${DEPTH_INTERMEDIATE_WEIGHT_POWER:-1.0}"
  --depth-intermediate-family-dro-temperature "${DEPTH_INTERMEDIATE_FAMILY_DRO_TEMPERATURE:-0.0}"
  --prefix-depth-anchor-loss-weight "${PREFIX_DEPTH_ANCHOR_LOSS_WEIGHT:-0.0}"
  --prefix-depth-anchor-min-depth "${PREFIX_DEPTH_ANCHOR_MIN_DEPTH:-1}"
  --prefix-depth-anchor-weight-power "${PREFIX_DEPTH_ANCHOR_WEIGHT_POWER:-1.0}"
  --answer-space-ranking-loss-weight "${ANSWER_SPACE_RANKING_LOSS_WEIGHT:-0.02}"
  --answer-space-ranking-max-cases "${ANSWER_SPACE_RANKING_MAX_CASES:-16}"
  --answer-space-ranking-every "${ANSWER_SPACE_RANKING_EVERY:-2}"
  --order-router-aux-loss-weight "${ORDER_ROUTER_AUX_LOSS_WEIGHT:-0.0}"
  --order-router-aux-target-mode "${ORDER_ROUTER_AUX_TARGET_MODE:-family_order}"
  --order-router-lr-multiplier "${ORDER_ROUTER_LR_MULTIPLIER:-1.0}"
  --forced-route-answer-loss-weight "${FORCED_ROUTE_ANSWER_LOSS_WEIGHT:-0.0}"
  --forced-route-answer-route "${FORCED_ROUTE_ANSWER_ROUTE:-1}"
  --forced-route-answer-families "${FORCED_ROUTE_ANSWER_FAMILIES:-revchain}"
  --forced-route-answer-max-cases "${FORCED_ROUTE_ANSWER_MAX_CASES:-0}"
  --forced-route-answer-every "${FORCED_ROUTE_ANSWER_EVERY:-1}"
  --forced-route-depth-loss-weight "${FORCED_ROUTE_DEPTH_LOSS_WEIGHT:-0.0}"
  --forced-route-depth-route "${FORCED_ROUTE_DEPTH_ROUTE:-1}"
  --forced-route-depth-families "${FORCED_ROUTE_DEPTH_FAMILIES:-revchain}"
  --forced-route-depth-max-cases "${FORCED_ROUTE_DEPTH_MAX_CASES:-0}"
  --forced-route-depth-every "${FORCED_ROUTE_DEPTH_EVERY:-1}"
  --forced-route-depth-min-depth "${FORCED_ROUTE_DEPTH_MIN_DEPTH:-1}"
  --forced-route-depth-weight-power "${FORCED_ROUTE_DEPTH_WEIGHT_POWER:-1.0}"
  --eval-active-len-cycle
  --active-len-cycle-min "${EVAL_ACTIVE_MIN}"
  --active-len-cycle-max "${EVAL_ACTIVE_MAX}"
  --train-active-len-cycle-min "${TRAIN_ACTIVE_MIN}"
  --train-active-len-cycle-max "${TRAIN_ACTIVE_MAX}"
  --active-len-replay-loss-weight "${ACTIVE_LEN_REPLAY_LOSS_WEIGHT:-0.02}"
  --active-len-replay-min "${REPLAY_ACTIVE_MIN}"
  --active-len-replay-max "${REPLAY_ACTIVE_MAX}"
  --active-len-replay-max-cases "${ACTIVE_LEN_REPLAY_MAX_CASES:-16}"
  --active-len-replay-every "${ACTIVE_LEN_REPLAY_EVERY:-2}"
  --retention-reference-checkpoint "${RETENTION_REFERENCE_CHECKPOINT:-}"
  --retention-kl-loss-weight "${RETENTION_KL_LOSS_WEIGHT:-0.0}"
  --retention-active-len-min "${RETENTION_ACTIVE_LEN_MIN:-${ACTIVE_MIN}}"
  --retention-active-len-max "${RETENTION_ACTIVE_LEN_MAX:-${ACTIVE_MAX}}"
  --retention-max-cases "${RETENTION_MAX_CASES:-16}"
  --retention-every "${RETENTION_EVERY:-4}"
  --retention-temperature "${RETENTION_TEMPERATURE:-1.0}"
  --operation-counterfactual-loss-weight "${OPERATION_COUNTERFACTUAL_LOSS_WEIGHT:-0.0}"
  --operation-counterfactual-warmup-steps "${OPERATION_COUNTERFACTUAL_WARMUP_STEPS:-1200}"
  --operation-counterfactual-margin "${OPERATION_COUNTERFACTUAL_MARGIN:-0.5}"
  --operation-counterfactual-max-cases "${OPERATION_COUNTERFACTUAL_MAX_CASES:-16}"
  --operation-counterfactual-every "${OPERATION_COUNTERFACTUAL_EVERY:-4}"
  --core-step-codec-loss-weight "${CORE_STEP_CODEC_LOSS_WEIGHT:-0.0}"
  --core-step-codec-state-source "${CORE_STEP_CODEC_STATE_SOURCE:-both}"
  --core-step-codec-pooling "${CORE_STEP_CODEC_POOLING:-last}"
  --seed "${SEED}"
  --eval-seed "${EVAL_SEED}"
  --device "${DEVICE}"
  --log-every "${LOG_EVERY:-200}"
  --max-examples 3
  --eval-during-training-every "${EVAL_DURING_TRAINING_EVERY:-600}"
  --eval-during-training-cases "${EVAL_DURING_TRAINING_CASES:-192}"
  --periodic-eval-score-mode "${PERIODIC_EVAL_SCORE_MODE:-strict}"
  --core-answer-probe-state-source "${CORE_ANSWER_PROBE_STATE_SOURCE:-h}"
  --core-answer-probe-pooling "${CORE_ANSWER_PROBE_POOLING:-last}"
  --core-answer-probe-train-cases "${CORE_ANSWER_PROBE_TRAIN_CASES:-1024}"
  --core-answer-probe-eval-cases "${CORE_ANSWER_PROBE_EVAL_CASES:-0}"
  --core-answer-probe-steps "${CORE_ANSWER_PROBE_STEPS:-300}"
  --core-answer-probe-batch-size "${CORE_ANSWER_PROBE_BATCH_SIZE:-128}"
  --core-answer-probe-lr "${CORE_ANSWER_PROBE_LR:-0.01}"
  --core-answer-probe-weight-decay "${CORE_ANSWER_PROBE_WEIGHT_DECAY:-0.0}"
  --restore-best-eval-checkpoint
  --accept-min-exact "${ACCEPT_MIN_EXACT:-0.0}"
  --accept-min-depth-gain "${ACCEPT_MIN_DEPTH_GAIN:--1.0}"
  --accept-min-ablation-drop "${ACCEPT_MIN_ABLATION_DROP:--1.0}"
  --accept-min-family-exact "${ACCEPT_MIN_FAMILY_EXACT:-0.0}"
  --accepted-decision "${ACCEPTED_DECISION:-diagnostic_reversed_efficiency}"
)

if [[ "${ACTIVE_LEN_BATCH_CYCLE:-1}" == "1" ]]; then
  command+=(--active-len-batch-cycle)
fi

if [[ "${DEPTH_INTERMEDIATE_FAMILY_DRO:-0}" == "1" ]]; then
  command+=(--depth-intermediate-family-dro)
fi

if [[ "${ACTIVE_LEN_CURRICULUM:-0}" == "1" ]]; then
  command+=(
    --active-len-curriculum
    --active-len-curriculum-min "${ACTIVE_LEN_CURRICULUM_MIN:-${TRAIN_ACTIVE_MIN}}"
    --active-len-curriculum-warmup-frac "${ACTIVE_LEN_CURRICULUM_WARMUP_FRAC:-0.5}"
  )
fi

if [[ "${EVAL_ANSWER_SPACE_ARGMAX:-1}" == "1" ]]; then
  command+=(--eval-answer-space-argmax)
fi

if [[ "${EVAL_CORE_ANSWER_PROBE:-0}" == "1" ]]; then
  command+=(--eval-core-answer-probe)
fi

if [[ "${EVAL_CORE_STEP_PROBE:-1}" == "1" ]]; then
  command+=(--eval-core-step-probe)
fi

if [[ "${EVAL_ORDER_ROUTER_PROBE:-0}" == "1" ]]; then
  command+=(--eval-order-router-probe)
fi

if [[ "${EVAL_ORDER_ROUTER_ROUTE_ABLATION:-0}" == "1" ]]; then
  command+=(--eval-order-router-route-ablation)
fi

if [[ "${EVAL_INITIAL_CHECKPOINT:-0}" == "1" ]]; then
  command+=(--eval-initial-checkpoint)
fi

if [[ -n "${RESUME_FROM}" ]]; then
  command+=(--resume-from "${RESUME_FROM}" --resume-allow-missing)
fi

if [[ "${EVAL_OPERATION_BREAKDOWN:-0}" == "1" ]]; then
  command+=(--eval-operation-breakdown)
fi

if [[ "${EVAL_DEPTH_SWEEP:-0}" == "1" ]]; then
  command+=(--eval-depth-sweep)
fi

if [[ "${EVAL_STATE_TRACE:-0}" == "1" ]]; then
  command+=(--eval-state-trace)
fi

printf 'Running reversed_hybrid_3to1 efficiency stage=%s out=%s\n' "${STAGE}" "${OUT_DIR}"
"${command[@]}" 2>&1 | tee "${OUT_DIR}.log"
