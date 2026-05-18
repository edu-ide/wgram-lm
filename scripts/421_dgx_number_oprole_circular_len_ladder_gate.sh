#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-run}"
DGX_HOST="${DGX_HOST:-dgx}"
DGX_REPO="${DGX_REPO:-/mnt/data4tb/qtrm_multimodal_memoryos_gate}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/mnt/data4tb/ws_llm/.venv/bin/python}"
OUT_TAG="${OUT_TAG:-$(date +%Y%m%d_%H%M%S)}"

RESUME_FROM="${RESUME_FROM:-local_eval/qtrm_native_l6_d256_number_oprole_circular_trace_revrepair_s3000_20260515/last.pt}"
LENGTHS="${LENGTHS:-8,12,16,20}"
STEPS_PER_STAGE="${STEPS_PER_STAGE:-450}"
TRAIN_CASES="${TRAIN_CASES:-16384}"
EVAL_CASES="${EVAL_CASES:-512}"
EVAL_SEED="${EVAL_SEED:-9338}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-3.0e-5}"
DEVICE="${DEVICE:-cuda}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-150}"
LOG_EVERY="${LOG_EVERY:-150}"
EVAL_EVERY="${EVAL_EVERY:-150}"
ACCEPT_MIN_EXACT="${ACCEPT_MIN_EXACT:-0.10}"
ACCEPT_MIN_DEPTH_GAIN="${ACCEPT_MIN_DEPTH_GAIN:-0.06}"
ACCEPT_MIN_ABLATION_DROP="${ACCEPT_MIN_ABLATION_DROP:-0.06}"
ACCEPT_MIN_FAMILY_EXACT="${ACCEPT_MIN_FAMILY_EXACT:-0.06}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-local_eval/runner_logs}"
REMOTE_LOG="${REMOTE_LOG:-${REMOTE_LOG_DIR}/number_oprole_circular_ladder_${OUT_TAG}.log}"
REMOTE_PID="${REMOTE_PID:-${REMOTE_LOG_DIR}/number_oprole_circular_ladder_${OUT_TAG}.pid}"

usage() {
  cat <<USAGE
Usage:
  $0 [plan|status|tail|submit|run|run-local]

Purpose:
  DGX staged length ladder for the QTRM-native number/op-role circular core.
  It resumes the accepted L6 checkpoint and promotes only checkpoints that pass
  each staged family-floor gate before attempting the next horizon.
USAGE
}

remote() {
  ssh "${DGX_HOST}" "cd '${DGX_REPO}' && $*"
}

run_stage() {
  local stage_len="$1"
  local prev_len="$2"
  local resume_from="$3"
  local out_dir="local_eval/qtrm_native_number_oprole_circular_ladder_len${stage_len}_seed${EVAL_SEED}_${OUT_TAG}"

  local resume_args=()
  local retention_args=()
  if [[ -n "${resume_from}" && "${resume_from}" != "none" ]]; then
    resume_args+=(--resume-from "${resume_from}" --resume-allow-missing)
    retention_args+=(
      --retention-reference-checkpoint resume
      --retention-kl-loss-weight 0.05
      --retention-max-cases 64
      --retention-every 2
    )
  fi

  echo "=== QTRM number/op-role circular ladder stage len=${stage_len} prev_len=${prev_len} ==="
  echo "resume_from=${resume_from}"
  echo "out_dir=${out_dir}"

  PYTHONPATH=src "${REMOTE_PYTHON}" scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py \
    --out-dir "${out_dir}" \
    --target-level "QTRM-native number-oprole circular ladder len${stage_len} seed${EVAL_SEED}" \
    "${resume_args[@]}" \
    --steps "${STEPS_PER_STAGE}" \
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
    --program-len "${stage_len}" \
    --modulus 32 \
    --d-model 256 \
    --n-heads 8 \
    --d-ff 512 \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --device "${DEVICE}" \
    --train-think-steps "${stage_len}" \
    --eval-think-steps "${stage_len}" \
    --backbone mha_etd \
    --think-structure trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier \
    --pos-embed-resize-strategy repeat_last \
    --active-len-curriculum \
    --active-len-curriculum-min "${prev_len}" \
    --active-len-curriculum-warmup-frac 0.70 \
    --active-len-replay-loss-weight 0.05 \
    --active-len-replay-min 1 \
    --active-len-replay-max "${prev_len}" \
    --active-len-replay-max-cases 64 \
    --active-len-replay-every 2 \
    "${retention_args[@]}" \
    --family-dro-loss-weight 0.14 \
    --family-dro-temperature 1.0 \
    --depth-intermediate-family-dro \
    --depth-intermediate-family-dro-temperature 1.0 \
    --state-trace-depth-loss-weight 0.30 \
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
    --accept-min-exact "${ACCEPT_MIN_EXACT}" \
    --accept-min-depth-gain "${ACCEPT_MIN_DEPTH_GAIN}" \
    --accept-min-ablation-drop "${ACCEPT_MIN_ABLATION_DROP}" \
    --accept-min-family-exact "${ACCEPT_MIN_FAMILY_EXACT}" \
    --accepted-decision "accepted_qtrm_native_number_oprole_circular_ladder_len${stage_len}_seed${EVAL_SEED}" \
    --log-every "${LOG_EVERY}" \
    ${EXTRA_ARGS}

  local accepted
  accepted="$("${REMOTE_PYTHON}" - "${out_dir}/report.json" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    report = json.load(f)
print("true" if report.get("accepted") else "false")
PY
)"
  echo "stage len=${stage_len} accepted=${accepted}"
  if [[ "${accepted}" != "true" ]]; then
    echo "stopping ladder at len=${stage_len}; report=${out_dir}/report.json" >&2
    return 10
  fi
  printf '%s\n' "${out_dir}/last.pt" > "${out_dir}/promoted_checkpoint.txt"
}

case "${ACTION}" in
  -h|--help|help)
    usage
    ;;
  plan)
    cat <<'PLAN'
Staged number/op-role circular length ladder:

1. Start from the accepted L6 checkpoint.
2. Train/evaluate len8 and require the same destructive-ablation gate.
3. Only if len8 is accepted, promote its last.pt to len12.
4. Repeat for len16 and len20.
5. Stop immediately on the first rejected length.

This tests horizon scaling without changing the canonical model path:

  prompt tokens -> number/op-role embeddings -> dual/nested recurrent core
  -> circular value LM logits -> greedy answer
PLAN
    ;;
  status)
    remote "pwd; git status --short --branch; git rev-parse --short HEAD; \
      echo; echo '[processes]'; pgrep -af '337_train|421_dgx_number_oprole' || true; \
      echo; echo '[latest ladder outputs]'; ls -dt local_eval/qtrm_native_number_oprole_circular_ladder_len* 2>/dev/null | head -8 || true; \
      echo; echo '[latest progress]'; latest=\$(ls -dt local_eval/qtrm_native_number_oprole_circular_ladder_len* 2>/dev/null | head -1 || true); \
      if [ -n \"\$latest\" ] && [ -f \"\$latest/latest_progress.json\" ]; then cat \"\$latest/latest_progress.json\"; fi; \
      echo; echo '[runner logs]'; ls -lt '${REMOTE_LOG_DIR}'/number_oprole_circular_ladder_*.log 2>/dev/null | head -5 || true"
    ;;
  tail)
    remote "latest_log=\$(ls -t '${REMOTE_LOG_DIR}'/number_oprole_circular_ladder_*.log 2>/dev/null | head -1 || true); \
      if [ -z \"\$latest_log\" ]; then echo 'no runner log found'; exit 0; fi; \
      echo \"==> \$latest_log <==\"; tail -120 \"\$latest_log\""
    ;;
  submit)
    remote "mkdir -p '${REMOTE_LOG_DIR}'; \
      git pull --ff-only; \
      nohup env PYTHONUNBUFFERED=1 OUT_TAG='${OUT_TAG}' RESUME_FROM='${RESUME_FROM}' LENGTHS='${LENGTHS}' \
        STEPS_PER_STAGE='${STEPS_PER_STAGE}' TRAIN_CASES='${TRAIN_CASES}' EVAL_CASES='${EVAL_CASES}' \
        EVAL_SEED='${EVAL_SEED}' BATCH_SIZE='${BATCH_SIZE}' LR='${LR}' DEVICE='${DEVICE}' \
        SAVE_EVERY_STEPS='${SAVE_EVERY_STEPS}' LOG_EVERY='${LOG_EVERY}' EVAL_EVERY='${EVAL_EVERY}' \
        ACCEPT_MIN_EXACT='${ACCEPT_MIN_EXACT}' ACCEPT_MIN_DEPTH_GAIN='${ACCEPT_MIN_DEPTH_GAIN}' \
        ACCEPT_MIN_ABLATION_DROP='${ACCEPT_MIN_ABLATION_DROP}' ACCEPT_MIN_FAMILY_EXACT='${ACCEPT_MIN_FAMILY_EXACT}' \
        EXTRA_ARGS='${EXTRA_ARGS}' \
        bash scripts/421_dgx_number_oprole_circular_len_ladder_gate.sh run-local > '${REMOTE_LOG}' 2>&1 < /dev/null & \
      pid=\$!; echo \"\$pid\" > '${REMOTE_PID}'; \
      echo \"submitted pid=\$pid\"; echo \"log=${REMOTE_LOG}\"; echo \"pid_file=${REMOTE_PID}\""
    ;;
  run)
    remote "env PYTHONUNBUFFERED=1 OUT_TAG='${OUT_TAG}' RESUME_FROM='${RESUME_FROM}' LENGTHS='${LENGTHS}' \
      STEPS_PER_STAGE='${STEPS_PER_STAGE}' TRAIN_CASES='${TRAIN_CASES}' EVAL_CASES='${EVAL_CASES}' \
      EVAL_SEED='${EVAL_SEED}' BATCH_SIZE='${BATCH_SIZE}' LR='${LR}' DEVICE='${DEVICE}' \
      SAVE_EVERY_STEPS='${SAVE_EVERY_STEPS}' LOG_EVERY='${LOG_EVERY}' EVAL_EVERY='${EVAL_EVERY}' \
      ACCEPT_MIN_EXACT='${ACCEPT_MIN_EXACT}' ACCEPT_MIN_DEPTH_GAIN='${ACCEPT_MIN_DEPTH_GAIN}' \
      ACCEPT_MIN_ABLATION_DROP='${ACCEPT_MIN_ABLATION_DROP}' ACCEPT_MIN_FAMILY_EXACT='${ACCEPT_MIN_FAMILY_EXACT}' \
      EXTRA_ARGS='${EXTRA_ARGS}' \
      bash scripts/421_dgx_number_oprole_circular_len_ladder_gate.sh run-local"
    ;;
  run-local)
    IFS=',' read -r -a ladder_lengths <<< "${LENGTHS}"
    current_resume="${RESUME_FROM}"
    prev_len=6
    for stage_len in "${ladder_lengths[@]}"; do
      run_stage "${stage_len}" "${prev_len}" "${current_resume}"
      current_resume="local_eval/qtrm_native_number_oprole_circular_ladder_len${stage_len}_seed${EVAL_SEED}_${OUT_TAG}/last.pt"
      prev_len="${stage_len}"
    done
    echo "ladder completed; final_checkpoint=${current_resume}"
    ;;
  *)
    echo "unknown action: ${ACTION}" >&2
    usage >&2
    exit 2
    ;;
esac
