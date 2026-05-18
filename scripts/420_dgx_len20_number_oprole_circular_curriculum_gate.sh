#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-run}"
DGX_HOST="${DGX_HOST:-dgx}"
DGX_REPO="${DGX_REPO:-/mnt/data4tb/qtrm_multimodal_memoryos_gate}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/mnt/data4tb/ws_llm/.venv/bin/python}"
OUT_TAG="${OUT_TAG:-$(date +%Y%m%d_%H%M%S)}"

RESUME_FROM="${RESUME_FROM:-local_eval/qtrm_native_l6_d256_number_oprole_circular_trace_revrepair_s3000_20260515/last.pt}"
PROGRAM_LEN="${PROGRAM_LEN:-20}"
THINK_STEPS="${THINK_STEPS:-20}"
STEPS="${STEPS:-900}"
TRAIN_CASES="${TRAIN_CASES:-16384}"
EVAL_CASES="${EVAL_CASES:-512}"
EVAL_SEED="${EVAL_SEED:-9338}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-3.0e-5}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-100}"
LOG_EVERY="${LOG_EVERY:-100}"
EVAL_EVERY="${EVAL_EVERY:-100}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-local_eval/runner_logs}"
REMOTE_LOG="${REMOTE_LOG:-${REMOTE_LOG_DIR}/len20_number_oprole_circular_${OUT_TAG}.log}"
REMOTE_PID="${REMOTE_PID:-${REMOTE_LOG_DIR}/len20_number_oprole_circular_${OUT_TAG}.pid}"

usage() {
  cat <<USAGE
Usage:
  $0 [plan|status|tail|submit|run|run-local]

Purpose:
  DGX len20 QTRM-native number/op-role circular-value curriculum gate.
  This is the direct follow-up to the reduced transition-control probe and the
  accepted L6 d256 number/op-role circular checkpoint.
USAGE
}

remote() {
  ssh "${DGX_HOST}" "cd '${DGX_REPO}' && $*"
}

case "${ACTION}" in
  -h|--help|help)
    usage
    ;;
  plan)
    cat <<'PLAN'
Len20 number/op-role circular curriculum gate:

1. Resume the accepted L6 d256 number/op-role circular checkpoint.
2. Resize position embeddings from len6 to len20 with repeat_last.
3. Keep the canonical LM path:
   prompt text -> number/op-role tokenizer -> token embeddings ->
   dual/nested recurrent core -> shared circular value LM logits.
4. Use active-length curriculum plus L6 replay/retention.
5. Promote only if len20 seed9338 passes family-floor and destructive
   core/state/op/carrier ablations remove the gain.
PLAN
    ;;
  status)
    remote "pwd; git status --short --branch; git rev-parse --short HEAD; \
      echo; echo '[processes]'; pgrep -af '337_train|420_dgx_len20_number_oprole' || true; \
      echo; echo '[latest outputs]'; ls -dt local_eval/qtrm_native_len20_number_oprole_circular_curriculum_seed* 2>/dev/null | head -5 || true; \
      echo; echo '[latest progress]'; latest=\$(ls -dt local_eval/qtrm_native_len20_number_oprole_circular_curriculum_seed* 2>/dev/null | head -1 || true); \
      if [ -n \"\$latest\" ] && [ -f \"\$latest/latest_progress.json\" ]; then cat \"\$latest/latest_progress.json\"; fi; \
      echo; echo '[runner logs]'; ls -lt '${REMOTE_LOG_DIR}'/len20_number_oprole_circular_*.log 2>/dev/null | head -5 || true"
    ;;
  tail)
    remote "latest_log=\$(ls -t '${REMOTE_LOG_DIR}'/len20_number_oprole_circular_*.log 2>/dev/null | head -1 || true); \
      if [ -z \"\$latest_log\" ]; then echo 'no runner log found'; exit 0; fi; \
      echo \"==> \$latest_log <==\"; tail -100 \"\$latest_log\""
    ;;
  submit)
    remote "mkdir -p '${REMOTE_LOG_DIR}'; \
      git pull --ff-only; \
      nohup env PYTHONUNBUFFERED=1 OUT_TAG='${OUT_TAG}' RESUME_FROM='${RESUME_FROM}' PROGRAM_LEN='${PROGRAM_LEN}' THINK_STEPS='${THINK_STEPS}' \
        STEPS='${STEPS}' TRAIN_CASES='${TRAIN_CASES}' EVAL_CASES='${EVAL_CASES}' EVAL_SEED='${EVAL_SEED}' \
        BATCH_SIZE='${BATCH_SIZE}' LR='${LR}' SAVE_EVERY_STEPS='${SAVE_EVERY_STEPS}' \
        LOG_EVERY='${LOG_EVERY}' EVAL_EVERY='${EVAL_EVERY}' EXTRA_ARGS='${EXTRA_ARGS}' \
        bash scripts/420_dgx_len20_number_oprole_circular_curriculum_gate.sh run-local > '${REMOTE_LOG}' 2>&1 < /dev/null & \
      pid=\$!; echo \"\$pid\" > '${REMOTE_PID}'; \
      echo \"submitted pid=\$pid\"; echo \"log=${REMOTE_LOG}\"; echo \"pid_file=${REMOTE_PID}\""
    ;;
  run)
    remote "env PYTHONUNBUFFERED=1 OUT_TAG='${OUT_TAG}' RESUME_FROM='${RESUME_FROM}' PROGRAM_LEN='${PROGRAM_LEN}' THINK_STEPS='${THINK_STEPS}' \
      STEPS='${STEPS}' TRAIN_CASES='${TRAIN_CASES}' EVAL_CASES='${EVAL_CASES}' EVAL_SEED='${EVAL_SEED}' \
      BATCH_SIZE='${BATCH_SIZE}' LR='${LR}' SAVE_EVERY_STEPS='${SAVE_EVERY_STEPS}' \
      LOG_EVERY='${LOG_EVERY}' EVAL_EVERY='${EVAL_EVERY}' EXTRA_ARGS='${EXTRA_ARGS}' \
      bash scripts/420_dgx_len20_number_oprole_circular_curriculum_gate.sh run-local"
    ;;
  run-local)
    resume_args=()
    if [[ -n "${RESUME_FROM}" && "${RESUME_FROM}" != "none" ]]; then
      resume_args+=(--resume-from "${RESUME_FROM}" --resume-allow-missing)
    fi
    retention_args=()
    if [[ -n "${RESUME_FROM}" && "${RESUME_FROM}" != "none" ]]; then
      retention_args+=(
        --retention-reference-checkpoint resume
        --retention-kl-loss-weight 0.05
        --retention-max-cases 64
        --retention-every 2
      )
    fi

    PYTHONPATH=src "${REMOTE_PYTHON}" scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py \
      --out-dir "local_eval/qtrm_native_len20_number_oprole_circular_curriculum_seed${EVAL_SEED}_${OUT_TAG}" \
      --target-level "QTRM-native len20 number-oprole circular curriculum seed${EVAL_SEED}" \
      "${resume_args[@]}" \
      --steps "${STEPS}" \
      --train-cases "${TRAIN_CASES}" \
      --eval-cases "${EVAL_CASES}" \
      --task-families 'modchain,revchain,modchain,revchain,checksum' \
      --eval-task-families 'modchain,revchain,checksum' \
      --eval-family-order-invariant \
      --include-family-tag \
      --tokenizer-mode number \
      --number-tokenizer-max-value 99 \
      --number-tokenizer-op-role-tokens \
      --value-codec circular \
      --program-len "${PROGRAM_LEN}" \
      --modulus 32 \
      --d-model 256 \
      --n-heads 8 \
      --d-ff 512 \
      --batch-size "${BATCH_SIZE}" \
      --lr "${LR}" \
      --device cuda \
      --train-think-steps "${THINK_STEPS}" \
      --eval-think-steps "${THINK_STEPS}" \
      --backbone mha_etd \
      --think-structure trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier \
      --pos-embed-resize-strategy repeat_last \
      --active-len-curriculum \
      --active-len-curriculum-min 6 \
      --active-len-curriculum-warmup-frac 0.65 \
      --active-len-replay-loss-weight 0.03 \
      --active-len-replay-min 1 \
      --active-len-replay-max 6 \
      --active-len-replay-max-cases 64 \
      --active-len-replay-every 2 \
      "${retention_args[@]}" \
      --family-dro-loss-weight 0.12 \
      --family-dro-temperature 1.0 \
      --depth-intermediate-family-dro \
      --depth-intermediate-family-dro-temperature 1.0 \
      --state-trace-depth-loss-weight 0.35 \
      --state-trace-depth-state-source both \
      --state-trace-depth-min-depth 4 \
      --state-trace-depth-max-depth-samples 6 \
      --state-trace-depth-sample-mode uniform \
      --state-trace-depth-weight-power 1.0 \
      --state-trace-depth-family-dro \
      --state-trace-depth-family-dro-temperature 1.0 \
      --eval-seed "${EVAL_SEED}" \
      --eval-during-training-every "${EVAL_EVERY}" \
      --eval-during-training-cases "${EVAL_CASES}" \
      --periodic-eval-score-mode family_floor \
      --eval-initial-checkpoint \
      --restore-best-eval-checkpoint \
      --save-every-steps "${SAVE_EVERY_STEPS}" \
      --save-best-periodic-checkpoint \
      --eval-state-trace \
      --accept-min-exact 0.10 \
      --accept-min-depth-gain 0.06 \
      --accept-min-ablation-drop 0.06 \
      --accept-min-family-exact 0.06 \
      --accepted-decision "accepted_qtrm_native_len20_number_oprole_circular_seed${EVAL_SEED}" \
      --log-every "${LOG_EVERY}" \
      ${EXTRA_ARGS}
    ;;
  *)
    echo "unknown action: ${ACTION}" >&2
    usage >&2
    exit 2
    ;;
esac
