#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
PYTHONPATH_VALUE="${PYTHONPATH:-local_deps/mamba3_runtime:src}"
SEED="${SEED:-337}"
EVAL_SEED="${EVAL_SEED:-9337}"
PROFILE="${PROFILE:-standard}"
OUT_ROOT="${OUT_ROOT:-local_eval/qtrm_native_hardlen_recipe_seed${SEED}}"
RESUME_ALLOW_MISSING="${RESUME_ALLOW_MISSING:-0}"
PROMPT_STATE_ANCHOR="${PROMPT_STATE_ANCHOR:-0}"
PROMPT_STATE_ANCHOR_POSITION="${PROMPT_STATE_ANCHOR_POSITION:-before_answer}"
TASK_FAMILIES="${TASK_FAMILIES:-modchain}"
EVAL_TASK_FAMILIES="${EVAL_TASK_FAMILIES:-}"
THINK_STRUCTURE="${THINK_STRUCTURE:-trm_dual_z_interactive}"
TRAIN_HARD_OP_IDS="${TRAIN_HARD_OP_IDS:-}"
TRAIN_HARD_OP_PROBABILITY="${TRAIN_HARD_OP_PROBABILITY:-0}"
TRAIN_HARD_OP_POSITIONS="${TRAIN_HARD_OP_POSITIONS:-}"
TOKENIZER_MODE="${TOKENIZER_MODE:-char}"
NUMBER_TOKENIZER_MAX_VALUE="${NUMBER_TOKENIZER_MAX_VALUE:-99}"
EVAL_DEPTH_SWEEP="${EVAL_DEPTH_SWEEP:-0}"
EVAL_OPERATION_BREAKDOWN="${EVAL_OPERATION_BREAKDOWN:-0}"
EVAL_CORE_ANSWER_PROBE="${EVAL_CORE_ANSWER_PROBE:-0}"
EVAL_CORE_STEP_PROBE="${EVAL_CORE_STEP_PROBE:-0}"
CORE_ANSWER_PROBE_STATE_SOURCE="${CORE_ANSWER_PROBE_STATE_SOURCE:-h}"
CORE_ANSWER_PROBE_POOLING="${CORE_ANSWER_PROBE_POOLING:-last}"
CORE_ANSWER_PROBE_TRAIN_CASES="${CORE_ANSWER_PROBE_TRAIN_CASES:-1024}"
CORE_ANSWER_PROBE_EVAL_CASES="${CORE_ANSWER_PROBE_EVAL_CASES:-0}"
CORE_ANSWER_PROBE_STEPS="${CORE_ANSWER_PROBE_STEPS:-300}"
CORE_ANSWER_PROBE_BATCH_SIZE="${CORE_ANSWER_PROBE_BATCH_SIZE:-128}"
CORE_ANSWER_PROBE_LR="${CORE_ANSWER_PROBE_LR:-0.01}"
CORE_ANSWER_PROBE_WEIGHT_DECAY="${CORE_ANSWER_PROBE_WEIGHT_DECAY:-0}"
EVAL_BEAM_WIDTH="${EVAL_BEAM_WIDTH:-1}"
EVAL_ANSWER_SPACE_ARGMAX="${EVAL_ANSWER_SPACE_ARGMAX:-0}"
EVAL_ANSWER_SPACE_ARGMAX_BATCH_SIZE="${EVAL_ANSWER_SPACE_ARGMAX_BATCH_SIZE:-512}"
EVAL_DURING_TRAINING_EVERY="${EVAL_DURING_TRAINING_EVERY:-0}"
EVAL_DURING_TRAINING_CASES="${EVAL_DURING_TRAINING_CASES:-64}"
PERIODIC_EVAL_SCORE_MODE="${PERIODIC_EVAL_SCORE_MODE:-strict}"
RESTORE_BEST_EVAL_CHECKPOINT="${RESTORE_BEST_EVAL_CHECKPOINT:-0}"
PREFIX_DEPTH_ANCHOR_LOSS_WEIGHT="${PREFIX_DEPTH_ANCHOR_LOSS_WEIGHT:-0}"
PREFIX_DEPTH_ANCHOR_MIN_DEPTH="${PREFIX_DEPTH_ANCHOR_MIN_DEPTH:-1}"
PREFIX_DEPTH_ANCHOR_WEIGHT_POWER="${PREFIX_DEPTH_ANCHOR_WEIGHT_POWER:-0}"
RESIDUE_AUX_LOSS_WEIGHT="${RESIDUE_AUX_LOSS_WEIGHT:-0}"
RESIDUE_AUX_MODULI="${RESIDUE_AUX_MODULI:-2,4,8}"
SEQUENCE_PREFERENCE_LOSS_WEIGHT="${SEQUENCE_PREFERENCE_LOSS_WEIGHT:-0}"
SEQUENCE_PREFERENCE_DELTAS="${SEQUENCE_PREFERENCE_DELTAS:-2,4,8,16}"
SEQUENCE_PREFERENCE_MARGIN="${SEQUENCE_PREFERENCE_MARGIN:-1.0}"
ANSWER_SPACE_RANKING_LOSS_WEIGHT="${ANSWER_SPACE_RANKING_LOSS_WEIGHT:-0}"
ANSWER_SPACE_RANKING_MAX_CASES="${ANSWER_SPACE_RANKING_MAX_CASES:-0}"
ANSWER_SPACE_RANKING_EVERY="${ANSWER_SPACE_RANKING_EVERY:-1}"
ANSWER_SPACE_RANKING_TEMPERATURE="${ANSWER_SPACE_RANKING_TEMPERATURE:-1.0}"
PREFIX_STATE_ALIGNMENT_LOSS_WEIGHT="${PREFIX_STATE_ALIGNMENT_LOSS_WEIGHT:-0}"
PREFIX_STATE_ALIGNMENT_MAX_CASES="${PREFIX_STATE_ALIGNMENT_MAX_CASES:-0}"
PREFIX_STATE_ALIGNMENT_EVERY="${PREFIX_STATE_ALIGNMENT_EVERY:-1}"
PREFIX_STATE_CONTRASTIVE_LOSS_WEIGHT="${PREFIX_STATE_CONTRASTIVE_LOSS_WEIGHT:-0}"
PREFIX_STATE_CONTRASTIVE_MAX_CASES="${PREFIX_STATE_CONTRASTIVE_MAX_CASES:-0}"
PREFIX_STATE_CONTRASTIVE_EVERY="${PREFIX_STATE_CONTRASTIVE_EVERY:-1}"
PREFIX_STATE_CONTRASTIVE_TEMPERATURE="${PREFIX_STATE_CONTRASTIVE_TEMPERATURE:-0.1}"
PREFIX_STATE_CONTRASTIVE_STATE_SOURCE="${PREFIX_STATE_CONTRASTIVE_STATE_SOURCE:-both}"
PREFIX_STATE_CONTRASTIVE_POOLING="${PREFIX_STATE_CONTRASTIVE_POOLING:-last}"
RETENTION_REFERENCE_CHECKPOINT="${RETENTION_REFERENCE_CHECKPOINT:-}"
RETENTION_KL_LOSS_WEIGHT="${RETENTION_KL_LOSS_WEIGHT:-0}"
RETENTION_ACTIVE_LEN_MIN="${RETENTION_ACTIVE_LEN_MIN:-1}"
RETENTION_ACTIVE_LEN_MAX="${RETENTION_ACTIVE_LEN_MAX:--1}"
RETENTION_MAX_CASES="${RETENTION_MAX_CASES:-0}"
RETENTION_EVERY="${RETENTION_EVERY:-1}"
RETENTION_TEMPERATURE="${RETENTION_TEMPERATURE:-1.0}"
ACTIVE_LEN_REPLAY_LOSS_WEIGHT="${ACTIVE_LEN_REPLAY_LOSS_WEIGHT:-0}"
ACTIVE_LEN_REPLAY_MIN="${ACTIVE_LEN_REPLAY_MIN:-1}"
ACTIVE_LEN_REPLAY_MAX="${ACTIVE_LEN_REPLAY_MAX:--1}"
ACTIVE_LEN_REPLAY_MAX_CASES="${ACTIVE_LEN_REPLAY_MAX_CASES:-0}"
ACTIVE_LEN_REPLAY_EVERY="${ACTIVE_LEN_REPLAY_EVERY:-1}"
ONLINE_GREEDY_PREFERENCE_LOSS_WEIGHT="${ONLINE_GREEDY_PREFERENCE_LOSS_WEIGHT:-0}"
ONLINE_GREEDY_PREFERENCE_MARGIN="${ONLINE_GREEDY_PREFERENCE_MARGIN:-1.0}"
ONLINE_GREEDY_PREFERENCE_MAX_CASES="${ONLINE_GREEDY_PREFERENCE_MAX_CASES:-0}"
ONLINE_GREEDY_PREFERENCE_EVERY="${ONLINE_GREEDY_PREFERENCE_EVERY:-1}"
CORE_STEP_CODEC_LOSS_WEIGHT="${CORE_STEP_CODEC_LOSS_WEIGHT:-0}"
CORE_STEP_CODEC_STATE_SOURCE="${CORE_STEP_CODEC_STATE_SOURCE:-both}"
CORE_STEP_CODEC_POOLING="${CORE_STEP_CODEC_POOLING:-last}"
PROGRAM_LEN="${PROGRAM_LEN:-4}"
THINK_STEPS="${THINK_STEPS:-${PROGRAM_LEN}}"
ACTIVE_LEN_MIN="${ACTIVE_LEN_MIN:-2}"
ACTIVE_LEN_MAX="${ACTIVE_LEN_MAX:-${PROGRAM_LEN}}"
TRAIN_ACTIVE_LEN_MIN="${TRAIN_ACTIVE_LEN_MIN:--1}"
TRAIN_ACTIVE_LEN_MAX="${TRAIN_ACTIVE_LEN_MAX:--1}"
ACCEPT_MAX_MEAN_HALT_STEPS="${ACCEPT_MAX_MEAN_HALT_STEPS:-3.2}"
D_MODEL="${D_MODEL:-64}"
D_FF="${D_FF:-128}"
N_HEADS="${N_HEADS:-4}"
N_KV_HEADS="${N_KV_HEADS:-2}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TRM_L_CYCLES="${TRM_L_CYCLES:-1}"
TRAIN_ACTIVE_LEN_MODE="${TRAIN_ACTIVE_LEN_MODE:-batch_cycle}"
STAGE1_ACTIVE_LEN_MODE="${STAGE1_ACTIVE_LEN_MODE:-${TRAIN_ACTIVE_LEN_MODE}}"
STAGE2_ACTIVE_LEN_MODE="${STAGE2_ACTIVE_LEN_MODE:-${TRAIN_ACTIVE_LEN_MODE}}"
STAGE3_ACTIVE_LEN_MODE="${STAGE3_ACTIVE_LEN_MODE:-${TRAIN_ACTIVE_LEN_MODE}}"
STAGE4_ACTIVE_LEN_MODE="${STAGE4_ACTIVE_LEN_MODE:-${STAGE3_ACTIVE_LEN_MODE}}"
CURRICULUM_MIN="${CURRICULUM_MIN:-1}"
CURRICULUM_WARMUP_FRAC="${CURRICULUM_WARMUP_FRAC:-0.75}"

case "${PROFILE}" in
  smoke)
    STAGE1_STEPS="${STAGE1_STEPS:-8}"
    STAGE2_STEPS="${STAGE2_STEPS:-8}"
    STAGE3_STEPS="${STAGE3_STEPS:-8}"
    REFINE_STEPS="${REFINE_STEPS:-0}"
    TRAIN_CASES="${TRAIN_CASES:-64}"
    EVAL_CASES="${EVAL_CASES:-24}"
    LOG_EVERY="${LOG_EVERY:-4}"
    ;;
  standard)
    STAGE1_STEPS="${STAGE1_STEPS:-8000}"
    STAGE2_STEPS="${STAGE2_STEPS:-8000}"
    STAGE3_STEPS="${STAGE3_STEPS:-4000}"
    REFINE_STEPS="${REFINE_STEPS:-2000}"
    TRAIN_CASES="${TRAIN_CASES:-4096}"
    EVAL_CASES="${EVAL_CASES:-192}"
    LOG_EVERY="${LOG_EVERY:-1000}"
    ;;
  *)
    echo "Unsupported PROFILE=${PROFILE}; expected smoke or standard" >&2
    exit 2
    ;;
esac

mkdir -p "${OUT_ROOT}"

common_args=(
  scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py
  --target-level "L5N hard active-len recursive gate"
  --train-cases "${TRAIN_CASES}"
  --eval-cases "${EVAL_CASES}"
  --task-families "${TASK_FAMILIES}"
  --eval-task-families "${EVAL_TASK_FAMILIES}"
  --train-hard-op-ids "${TRAIN_HARD_OP_IDS}"
  --train-hard-op-probability "${TRAIN_HARD_OP_PROBABILITY}"
  --train-hard-op-positions "${TRAIN_HARD_OP_POSITIONS}"
  --tokenizer-mode "${TOKENIZER_MODE}"
  --number-tokenizer-max-value "${NUMBER_TOKENIZER_MAX_VALUE}"
  --program-len "${PROGRAM_LEN}"
  --modulus 32
  --d-model "${D_MODEL}"
  --n-heads "${N_HEADS}"
  --n-kv-heads "${N_KV_HEADS}"
  --d-ff "${D_FF}"
  --backbone trm_official
  --encode-backbone mha_etd
  --think-backbone trm_official
  --decode-backbone mha_etd
  --think-structure "${THINK_STRUCTURE}"
  --trm-l-cycles "${TRM_L_CYCLES}"
  --halt-pooling dedicated
  --batch-size "${BATCH_SIZE}"
  --weight-decay 0.01
  --grad-clip 1.0
  --train-think-steps "${THINK_STEPS}"
  --eval-think-steps "${THINK_STEPS}"
  --adaptive-halt-eval
  --halt-min-steps 1
  --adaptive-halt-loss-weight 5.0
  --adaptive-halt-target-mode active_len
  --adaptive-halt-active-len-target first_step
  --depth-intermediate-loss-weight 0.25
  --prefix-depth-anchor-loss-weight "${PREFIX_DEPTH_ANCHOR_LOSS_WEIGHT}"
  --prefix-depth-anchor-min-depth "${PREFIX_DEPTH_ANCHOR_MIN_DEPTH}"
  --prefix-depth-anchor-weight-power "${PREFIX_DEPTH_ANCHOR_WEIGHT_POWER}"
  --residue-aux-loss-weight "${RESIDUE_AUX_LOSS_WEIGHT}"
  --residue-aux-moduli "${RESIDUE_AUX_MODULI}"
  --sequence-preference-loss-weight "${SEQUENCE_PREFERENCE_LOSS_WEIGHT}"
  --sequence-preference-deltas "${SEQUENCE_PREFERENCE_DELTAS}"
  --sequence-preference-margin "${SEQUENCE_PREFERENCE_MARGIN}"
  --answer-space-ranking-loss-weight "${ANSWER_SPACE_RANKING_LOSS_WEIGHT}"
  --answer-space-ranking-max-cases "${ANSWER_SPACE_RANKING_MAX_CASES}"
  --answer-space-ranking-every "${ANSWER_SPACE_RANKING_EVERY}"
  --answer-space-ranking-temperature "${ANSWER_SPACE_RANKING_TEMPERATURE}"
  --prefix-state-alignment-loss-weight "${PREFIX_STATE_ALIGNMENT_LOSS_WEIGHT}"
  --prefix-state-alignment-max-cases "${PREFIX_STATE_ALIGNMENT_MAX_CASES}"
  --prefix-state-alignment-every "${PREFIX_STATE_ALIGNMENT_EVERY}"
  --prefix-state-contrastive-loss-weight "${PREFIX_STATE_CONTRASTIVE_LOSS_WEIGHT}"
  --prefix-state-contrastive-max-cases "${PREFIX_STATE_CONTRASTIVE_MAX_CASES}"
  --prefix-state-contrastive-every "${PREFIX_STATE_CONTRASTIVE_EVERY}"
  --prefix-state-contrastive-temperature "${PREFIX_STATE_CONTRASTIVE_TEMPERATURE}"
  --prefix-state-contrastive-state-source "${PREFIX_STATE_CONTRASTIVE_STATE_SOURCE}"
  --prefix-state-contrastive-pooling "${PREFIX_STATE_CONTRASTIVE_POOLING}"
  --retention-reference-checkpoint "${RETENTION_REFERENCE_CHECKPOINT}"
  --retention-kl-loss-weight "${RETENTION_KL_LOSS_WEIGHT}"
  --retention-active-len-min "${RETENTION_ACTIVE_LEN_MIN}"
  --retention-active-len-max "${RETENTION_ACTIVE_LEN_MAX}"
  --retention-max-cases "${RETENTION_MAX_CASES}"
  --retention-every "${RETENTION_EVERY}"
  --retention-temperature "${RETENTION_TEMPERATURE}"
  --active-len-replay-loss-weight "${ACTIVE_LEN_REPLAY_LOSS_WEIGHT}"
  --active-len-replay-min "${ACTIVE_LEN_REPLAY_MIN}"
  --active-len-replay-max "${ACTIVE_LEN_REPLAY_MAX}"
  --active-len-replay-max-cases "${ACTIVE_LEN_REPLAY_MAX_CASES}"
  --active-len-replay-every "${ACTIVE_LEN_REPLAY_EVERY}"
  --online-greedy-preference-loss-weight "${ONLINE_GREEDY_PREFERENCE_LOSS_WEIGHT}"
  --online-greedy-preference-margin "${ONLINE_GREEDY_PREFERENCE_MARGIN}"
  --online-greedy-preference-max-cases "${ONLINE_GREEDY_PREFERENCE_MAX_CASES}"
  --online-greedy-preference-every "${ONLINE_GREEDY_PREFERENCE_EVERY}"
  --core-step-codec-loss-weight "${CORE_STEP_CODEC_LOSS_WEIGHT}"
  --core-step-codec-state-source "${CORE_STEP_CODEC_STATE_SOURCE}"
  --core-step-codec-pooling "${CORE_STEP_CODEC_POOLING}"
  --eval-active-len-cycle
  --active-len-cycle-min "${ACTIVE_LEN_MIN}"
  --active-len-cycle-max "${ACTIVE_LEN_MAX}"
  --train-active-len-cycle-min "${TRAIN_ACTIVE_LEN_MIN}"
  --train-active-len-cycle-max "${TRAIN_ACTIVE_LEN_MAX}"
  --eval-seed "${EVAL_SEED}"
  --log-every "${LOG_EVERY}"
  --eval-during-training-every "${EVAL_DURING_TRAINING_EVERY}"
  --eval-during-training-cases "${EVAL_DURING_TRAINING_CASES}"
  --periodic-eval-score-mode "${PERIODIC_EVAL_SCORE_MODE}"
  --core-answer-probe-state-source "${CORE_ANSWER_PROBE_STATE_SOURCE}"
  --core-answer-probe-pooling "${CORE_ANSWER_PROBE_POOLING}"
  --core-answer-probe-train-cases "${CORE_ANSWER_PROBE_TRAIN_CASES}"
  --core-answer-probe-eval-cases "${CORE_ANSWER_PROBE_EVAL_CASES}"
  --core-answer-probe-steps "${CORE_ANSWER_PROBE_STEPS}"
  --core-answer-probe-batch-size "${CORE_ANSWER_PROBE_BATCH_SIZE}"
  --core-answer-probe-lr "${CORE_ANSWER_PROBE_LR}"
  --core-answer-probe-weight-decay "${CORE_ANSWER_PROBE_WEIGHT_DECAY}"
  --eval-beam-width "${EVAL_BEAM_WIDTH}"
  --eval-answer-space-argmax-batch-size "${EVAL_ANSWER_SPACE_ARGMAX_BATCH_SIZE}"
)

if [[ "${EVAL_DEPTH_SWEEP}" == "1" ]]; then
  common_args+=(--eval-depth-sweep)
fi

if [[ "${EVAL_OPERATION_BREAKDOWN}" == "1" ]]; then
  common_args+=(--eval-operation-breakdown)
fi

if [[ "${EVAL_CORE_ANSWER_PROBE}" == "1" ]]; then
  common_args+=(--eval-core-answer-probe)
fi

if [[ "${EVAL_CORE_STEP_PROBE}" == "1" ]]; then
  common_args+=(--eval-core-step-probe)
fi

if [[ "${EVAL_ANSWER_SPACE_ARGMAX}" == "1" ]]; then
  common_args+=(--eval-answer-space-argmax)
fi

if [[ "${RESTORE_BEST_EVAL_CHECKPOINT}" == "1" ]]; then
  common_args+=(--restore-best-eval-checkpoint)
fi

if [[ "${RESUME_ALLOW_MISSING}" == "1" ]]; then
  common_args+=(--resume-allow-missing)
fi

if [[ "${PROMPT_STATE_ANCHOR}" == "1" ]]; then
  common_args+=(
    --prompt-state-anchor
    --prompt-state-anchor-position "${PROMPT_STATE_ANCHOR_POSITION}"
  )
fi

lenient_accept_args=(
  --accept-min-exact 0.0
  --accept-min-depth-gain -1.0
  --accept-min-ablation-drop -1.0
  --accept-min-family-exact 0.0
  --accept-max-adaptive-halt-exact-drop 1.0
  --accept-max-mean-halt-steps 4.0
  --accept-min-halted-fraction 0.0
)

strict_accept_args=(
  --accept-min-exact 0.30
  --accept-min-depth-gain 0.10
  --accept-min-ablation-drop 0.03
  --accept-min-family-exact 0.20
  --accept-require-adaptive-halt
  --accept-max-adaptive-halt-exact-drop 0.04
  --accept-max-mean-halt-steps "${ACCEPT_MAX_MEAN_HALT_STEPS}"
  --accept-min-halted-fraction 0.95
)

if [[ "${PROFILE}" == "smoke" ]]; then
  final_accept_args=("${lenient_accept_args[@]}")
else
  final_accept_args=("${strict_accept_args[@]}")
fi

run_stage() {
  local out_dir="$1"
  shift
  PYTHONPATH="${PYTHONPATH_VALUE}" "${PYTHON_BIN}" "${common_args[@]}" "$@" --out-dir "${out_dir}"
}

active_len_mode_args() {
  local mode="$1"
  case "${mode}" in
    batch_cycle)
      printf '%s\n' --active-len-batch-cycle
      ;;
    curriculum)
      printf '%s\n' \
        --active-len-curriculum \
        --active-len-curriculum-min "${CURRICULUM_MIN}" \
        --active-len-curriculum-warmup-frac "${CURRICULUM_WARMUP_FRAC}"
      ;;
    none)
      ;;
    *)
      echo "Unsupported active length mode: ${mode}" >&2
      exit 2
      ;;
  esac
}

stage1="${OUT_ROOT}/stage1_active_len_s${STAGE1_STEPS}"
stage2="${OUT_ROOT}/stage2_resume_s${STAGE2_STEPS}"
stage3="${OUT_ROOT}/stage3_halt_depth_s${STAGE3_STEPS}"
stage4="${OUT_ROOT}/stage4_halt_depth_refine_s${REFINE_STEPS}"

mapfile -t stage1_active_len_args < <(active_len_mode_args "${STAGE1_ACTIVE_LEN_MODE}")
mapfile -t stage2_active_len_args < <(active_len_mode_args "${STAGE2_ACTIVE_LEN_MODE}")
mapfile -t stage3_active_len_args < <(active_len_mode_args "${STAGE3_ACTIVE_LEN_MODE}")
mapfile -t stage4_active_len_args < <(active_len_mode_args "${STAGE4_ACTIVE_LEN_MODE}")

run_stage "${stage1}" \
  --steps "${STAGE1_STEPS}" \
  --lr 3e-4 \
  --seed "${SEED}" \
  "${stage1_active_len_args[@]}" \
  "${lenient_accept_args[@]}"

run_stage "${stage2}" \
  --resume-from "${stage1}/last.pt" \
  --steps "${STAGE2_STEPS}" \
  --lr 2e-4 \
  --seed "$((SEED + 3))" \
  "${stage2_active_len_args[@]}" \
  "${lenient_accept_args[@]}"

set +e
run_stage "${stage3}" \
  --resume-from "${stage2}/last.pt" \
  --steps "${STAGE3_STEPS}" \
  --lr 1e-4 \
  --seed "$((SEED + 5))" \
  --halt-depth-final-loss-weight 1.0 \
  "${stage3_active_len_args[@]}" \
  "${final_accept_args[@]}"
stage3_status="$?"
set -e

final_stage="${stage3}"
if [[ "${stage3_status}" != "0" ]]; then
  if [[ ! -f "${stage3}/report.json" ]]; then
    exit "${stage3_status}"
  fi
  if [[ "${PROFILE}" == "standard" && "${REFINE_STEPS}" -gt 0 ]]; then
    run_stage "${stage4}" \
      --resume-from "${stage3}/last.pt" \
      --steps "${REFINE_STEPS}" \
      --lr 5e-5 \
      --seed "$((SEED + 8))" \
      --halt-depth-final-loss-weight 1.0 \
      "${stage4_active_len_args[@]}" \
      "${strict_accept_args[@]}"
    final_stage="${stage4}"
  else
    exit "${stage3_status}"
  fi
fi

echo "Final report: ${final_stage}/report.json"
