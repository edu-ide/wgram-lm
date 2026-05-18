#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-run}"
DGX_HOST="${DGX_HOST:-dgx}"
DGX_REPO="${DGX_REPO:-/mnt/data4tb/qtrm_multimodal_memoryos}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/mnt/data4tb/ws_llm/.venv/bin/python}"
OUT_TAG="${OUT_TAG:-$(date +%Y%m%d_%H%M%S)}"

RESUME_FROM="${RESUME_FROM:-local_eval/dgx_single_order_router_len20_familyfloor_select_20260517_222156/last.pt}"
PROGRAM_LEN="${PROGRAM_LEN:-20}"
THINK_STEPS="${THINK_STEPS:-20}"
STEPS="${STEPS:-600}"
TRAIN_CASES="${TRAIN_CASES:-16384}"
EVAL_CASES="${EVAL_CASES:-512}"
EVAL_SEED="${EVAL_SEED:-9338}"
BATCH_SIZE="${BATCH_SIZE:-64}"
D_MODEL="${D_MODEL:-128}"
N_HEADS="${N_HEADS:-8}"
D_FF="${D_FF:-256}"
LR="${LR:-1.0e-4}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-100}"
LOG_EVERY="${LOG_EVERY:-100}"
EVAL_EVERY="${EVAL_EVERY:-100}"
PREFIX_ANCHOR_WEIGHT="${PREFIX_ANCHOR_WEIGHT:-0.2}"
FAMILY_DRO_WEIGHT="${FAMILY_DRO_WEIGHT:-0.35}"
FORCED_ROUTE_ANSWER_WEIGHT="${FORCED_ROUTE_ANSWER_WEIGHT:-0.30}"
FORCED_ROUTE_DEPTH_WEIGHT="${FORCED_ROUTE_DEPTH_WEIGHT:-0.20}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-local_eval/runner_logs}"
REMOTE_LOG="${REMOTE_LOG:-${REMOTE_LOG_DIR}/len20_prefix_anchor_${OUT_TAG}.log}"
REMOTE_PID="${REMOTE_PID:-${REMOTE_LOG_DIR}/len20_prefix_anchor_${OUT_TAG}.pid}"

usage() {
  cat <<USAGE
Usage:
  $0 [plan|status|tail|submit|run]

Purpose:
  DGX len20 route1 prefix-anchor gate for the ordered-chain transition
  bottleneck. This is the follow-up after data-scale continuation failed to
  stabilize eval seed 9338.

Defaults:
  RESUME_FROM=${RESUME_FROM}
  PROGRAM_LEN=${PROGRAM_LEN}
  THINK_STEPS=${THINK_STEPS}
  STEPS=${STEPS}
  TRAIN_CASES=${TRAIN_CASES}
  EVAL_CASES=${EVAL_CASES}
  EVAL_SEED=${EVAL_SEED}
  PREFIX_ANCHOR_WEIGHT=${PREFIX_ANCHOR_WEIGHT}
  SAVE_EVERY_STEPS=${SAVE_EVERY_STEPS}

Background submit:
  $0 submit
  $0 status
  $0 tail
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
Len20 prefix-anchor gate:

1. Resume the accepted len20 route-conditioned checkpoint.
2. Train only route1/router parameters.
3. Add forced-route prefix-depth anchor pressure:
   causal-prefix prompt -> forced route1 -> recurrent depth -> LM logits.
4. Select checkpoints by 512-case family floor on eval seed 9338.
5. Promote only after seed9338 passes and the original seed is rerun.
PLAN
    ;;
  status)
    remote "pwd; git status --short --branch; git rev-parse --short HEAD; \
      echo; echo '[processes]'; pgrep -af '337_train|412_dgx_len20_prefix_anchor' || true; \
      echo; echo '[latest outputs]'; ls -dt local_eval/dgx_single_order_router_len20_prefix_anchor_* 2>/dev/null | head -5 || true; \
      echo; echo '[latest progress]'; latest=\$(ls -dt local_eval/dgx_single_order_router_len20_prefix_anchor_* 2>/dev/null | head -1 || true); \
      if [ -n \"\$latest\" ] && [ -f \"\$latest/latest_progress.json\" ]; then cat \"\$latest/latest_progress.json\"; fi; \
      echo; echo '[runner logs]'; ls -lt '${REMOTE_LOG_DIR}'/len20_prefix_anchor_*.log 2>/dev/null | head -5 || true"
    ;;
  tail)
    remote "latest_log=\$(ls -t '${REMOTE_LOG_DIR}'/len20_prefix_anchor_*.log 2>/dev/null | head -1 || true); \
      if [ -z \"\$latest_log\" ]; then echo 'no runner log found'; exit 0; fi; \
      echo \"==> \$latest_log <==\"; tail -80 \"\$latest_log\""
    ;;
  submit)
    remote "mkdir -p '${REMOTE_LOG_DIR}'; \
      git pull --ff-only; \
      nohup env OUT_TAG='${OUT_TAG}' RESUME_FROM='${RESUME_FROM}' PROGRAM_LEN='${PROGRAM_LEN}' THINK_STEPS='${THINK_STEPS}' \
        STEPS='${STEPS}' TRAIN_CASES='${TRAIN_CASES}' EVAL_CASES='${EVAL_CASES}' EVAL_SEED='${EVAL_SEED}' \
        BATCH_SIZE='${BATCH_SIZE}' D_MODEL='${D_MODEL}' N_HEADS='${N_HEADS}' D_FF='${D_FF}' LR='${LR}' \
        SAVE_EVERY_STEPS='${SAVE_EVERY_STEPS}' LOG_EVERY='${LOG_EVERY}' EVAL_EVERY='${EVAL_EVERY}' \
        PREFIX_ANCHOR_WEIGHT='${PREFIX_ANCHOR_WEIGHT}' FAMILY_DRO_WEIGHT='${FAMILY_DRO_WEIGHT}' \
        FORCED_ROUTE_ANSWER_WEIGHT='${FORCED_ROUTE_ANSWER_WEIGHT}' FORCED_ROUTE_DEPTH_WEIGHT='${FORCED_ROUTE_DEPTH_WEIGHT}' \
        EXTRA_ARGS='${EXTRA_ARGS}' \
        bash scripts/412_dgx_len20_prefix_anchor_gate.sh run-local > '${REMOTE_LOG}' 2>&1 < /dev/null & \
      pid=\$!; echo \"\$pid\" > '${REMOTE_PID}'; \
      echo \"submitted pid=\$pid\"; echo \"log=${REMOTE_LOG}\"; echo \"pid_file=${REMOTE_PID}\""
    ;;
  run)
    remote "env OUT_TAG='${OUT_TAG}' RESUME_FROM='${RESUME_FROM}' PROGRAM_LEN='${PROGRAM_LEN}' THINK_STEPS='${THINK_STEPS}' \
      STEPS='${STEPS}' TRAIN_CASES='${TRAIN_CASES}' EVAL_CASES='${EVAL_CASES}' EVAL_SEED='${EVAL_SEED}' \
      BATCH_SIZE='${BATCH_SIZE}' D_MODEL='${D_MODEL}' N_HEADS='${N_HEADS}' D_FF='${D_FF}' LR='${LR}' \
      SAVE_EVERY_STEPS='${SAVE_EVERY_STEPS}' LOG_EVERY='${LOG_EVERY}' EVAL_EVERY='${EVAL_EVERY}' \
      PREFIX_ANCHOR_WEIGHT='${PREFIX_ANCHOR_WEIGHT}' FAMILY_DRO_WEIGHT='${FAMILY_DRO_WEIGHT}' \
      FORCED_ROUTE_ANSWER_WEIGHT='${FORCED_ROUTE_ANSWER_WEIGHT}' FORCED_ROUTE_DEPTH_WEIGHT='${FORCED_ROUTE_DEPTH_WEIGHT}' \
      EXTRA_ARGS='${EXTRA_ARGS}' \
      bash scripts/412_dgx_len20_prefix_anchor_gate.sh run-local"
    ;;
  run-local)
    PYTHONPATH=src "${REMOTE_PYTHON}" scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py \
      --out-dir "local_eval/dgx_single_order_router_len20_prefix_anchor_seed${EVAL_SEED}_${OUT_TAG}" \
      --target-level "single-order-router len20 prefix-anchor seed${EVAL_SEED}" \
      --resume-from "${RESUME_FROM}" \
      --steps "${STEPS}" \
      --train-cases "${TRAIN_CASES}" \
      --eval-cases "${EVAL_CASES}" \
      --task-families 'modchain,revchain,modchain,revchain,checksum' \
      --eval-task-families 'modchain,revchain,checksum' \
      --eval-family-order-invariant \
      --include-family-tag \
      --program-len "${PROGRAM_LEN}" \
      --modulus 32 \
      --d-model "${D_MODEL}" \
      --n-heads "${N_HEADS}" \
      --d-ff "${D_FF}" \
      --batch-size "${BATCH_SIZE}" \
      --lr "${LR}" \
      --device cuda \
      --train-think-steps "${THINK_STEPS}" \
      --eval-think-steps "${THINK_STEPS}" \
      --backbone mha_etd \
      --think-structure single_order_router \
      --train-param-name-regex 'single_order_route1|trm_order_router' \
      --family-dro-loss-weight "${FAMILY_DRO_WEIGHT}" \
      --family-dro-temperature 1.0 \
      --depth-intermediate-family-dro \
      --depth-intermediate-family-dro-temperature 1.0 \
      --order-router-aux-loss-weight 0.08 \
      --order-router-aux-target-mode chain_vs_checksum \
      --forced-route-answer-loss-weight "${FORCED_ROUTE_ANSWER_WEIGHT}" \
      --forced-route-answer-route 1 \
      --forced-route-answer-families modchain,revchain \
      --forced-route-answer-max-cases 64 \
      --forced-route-answer-every 1 \
      --forced-route-depth-loss-weight "${FORCED_ROUTE_DEPTH_WEIGHT}" \
      --forced-route-depth-route 1 \
      --forced-route-depth-families modchain,revchain \
      --forced-route-depth-max-cases 64 \
      --forced-route-depth-every 1 \
      --forced-route-depth-min-depth 4 \
      --forced-route-depth-weight-power 1.0 \
      --forced-route-prefix-depth-anchor-loss-weight "${PREFIX_ANCHOR_WEIGHT}" \
      --forced-route-prefix-depth-anchor-route 1 \
      --forced-route-prefix-depth-anchor-families modchain,revchain \
      --forced-route-prefix-depth-anchor-max-cases 64 \
      --forced-route-prefix-depth-anchor-every 1 \
      --forced-route-prefix-depth-anchor-min-depth 1 \
      --forced-route-prefix-depth-anchor-weight-power 1.0 \
      --eval-seed "${EVAL_SEED}" \
      --eval-during-training-every "${EVAL_EVERY}" \
      --eval-during-training-cases "${EVAL_CASES}" \
      --periodic-eval-score-mode family_floor \
      --eval-initial-checkpoint \
      --restore-best-eval-checkpoint \
      --save-every-steps "${SAVE_EVERY_STEPS}" \
      --save-best-periodic-checkpoint \
      --eval-order-router-probe \
      --eval-order-router-route-ablation \
      --accept-min-exact 0.10 \
      --accept-min-depth-gain 0.06 \
      --accept-min-ablation-drop 0.06 \
      --accept-min-family-exact 0.06 \
      --accepted-decision "accepted_single_order_router_len20_prefix_anchor_seed${EVAL_SEED}" \
      --log-every "${LOG_EVERY}" \
      ${EXTRA_ARGS}
    ;;
  *)
    echo "unknown action: ${ACTION}" >&2
    usage >&2
    exit 2
    ;;
esac
