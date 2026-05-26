#!/usr/bin/env bash
set -euo pipefail

# Stage61A: donorless one-body baseline.
#
# Plain-language contract:
#   The model is born as one body: text reader -> recurrent thought -> LM-token
#   speaker. There is no Qwen donor, no typed renderer mouth, and no transplant
#   bridge. This answers whether the architecture itself can learn the task
#   when the thinking state and speaking state grow up together.

ACTION="${1:-plan}"
OUT_TAG="${OUT_TAG:-stage61a_onebody_fromscratch_$(date +%Y%m%d_%H%M%S)}"

export OUT_TAG
export RESUME_FROM=""
export INCLUDE_FAMILY_TAG="${INCLUDE_FAMILY_TAG:-1}"
export PROGRAM_LEN="${PROGRAM_LEN:-8}"
export THINK_STEPS="${THINK_STEPS:-8}"
export STEPS="${STEPS:-2400}"
export TRAIN_CASES="${TRAIN_CASES:-8192}"
export EVAL_CASES="${EVAL_CASES:-512}"
export BATCH_SIZE="${BATCH_SIZE:-96}"
export D_MODEL="${D_MODEL:-128}"
export N_HEADS="${N_HEADS:-8}"
export D_FF="${D_FF:-256}"
export ACCEPT_MIN_EXACT="${ACCEPT_MIN_EXACT:-0.15}"
export ACCEPT_MIN_DEPTH_GAIN="${ACCEPT_MIN_DEPTH_GAIN:-0.08}"
export ACCEPT_MIN_ABLATION_DROP="${ACCEPT_MIN_ABLATION_DROP:-0.08}"
export ACCEPT_MIN_FAMILY_EXACT="${ACCEPT_MIN_FAMILY_EXACT:-0.03}"
export EXTRA_ARGS="${EXTRA_ARGS:-} --eval-state-trace --eval-core-answer-probe --eval-core-step-probe --eval-operation-breakdown --save-every-steps 400 --eval-during-training-every 400 --restore-best-eval-checkpoint --save-best-periodic-checkpoint --periodic-eval-score-mode family_floor"

case "${ACTION}" in
  plan)
    cat <<PLAN
Stage61A DGX one-body from-scratch plan

Human contract:
  One model learns to read, think recurrently, and speak through the same
  LM-logit answer path from the beginning.

Technical runner:
  bash scripts/411_dgx_trm_raw_scaleout_gate.sh run

Default gate:
  PROGRAM_LEN=${PROGRAM_LEN}
  THINK_STEPS=${THINK_STEPS}
  full > think0 by >= ${ACCEPT_MIN_DEPTH_GAIN}
  full > worst destructive ablation by >= ${ACCEPT_MIN_ABLATION_DROP}

Run:
  bash scripts/launch_stage61a_dgx_onebody_fromscratch.sh run
PLAN
    ;;
  status|run)
    bash scripts/411_dgx_trm_raw_scaleout_gate.sh "${ACTION}"
    ;;
  *)
    echo "unknown action: ${ACTION}" >&2
    exit 2
    ;;
esac
