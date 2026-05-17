#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-run}"
DGX_HOST="${DGX_HOST:-dgx}"
DGX_REPO="${DGX_REPO:-/mnt/data4tb/qtrm_multimodal_memoryos}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/mnt/data4tb/ws_llm/.venv/bin/python}"
OUT_TAG="${OUT_TAG:-$(date +%Y%m%d_%H%M%S)}"

PROGRAM_LEN="${PROGRAM_LEN:-8}"
THINK_STEPS="${THINK_STEPS:-8}"
STEPS="${STEPS:-2400}"
TRAIN_CASES="${TRAIN_CASES:-8192}"
EVAL_CASES="${EVAL_CASES:-512}"
BATCH_SIZE="${BATCH_SIZE:-96}"
D_MODEL="${D_MODEL:-128}"
N_HEADS="${N_HEADS:-8}"
D_FF="${D_FF:-256}"
LR="${LR:-3.0e-4}"
RESUME_FROM="${RESUME_FROM:-}"
RESUME_ALLOW_MISSING="${RESUME_ALLOW_MISSING:-1}"
INCLUDE_FAMILY_TAG="${INCLUDE_FAMILY_TAG:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

usage() {
  cat <<USAGE
Usage:
  $0 [plan|status|run]

Purpose:
  DGX raw-reasoning scale-out gate for TRM-like QTRM claims.
  This tests whether the accepted len4 scoped reasoning signal scales to
  longer program lengths before spending more time on knowledge-heavy MMLU.

Key overrides:
  PROGRAM_LEN=${PROGRAM_LEN}
  THINK_STEPS=${THINK_STEPS}
  STEPS=${STEPS}
  TRAIN_CASES=${TRAIN_CASES}
  EVAL_CASES=${EVAL_CASES}
  BATCH_SIZE=${BATCH_SIZE}
  REMOTE_PYTHON=${REMOTE_PYTHON}
  RESUME_FROM=${RESUME_FROM:-<empty>}
  INCLUDE_FAMILY_TAG=${INCLUDE_FAMILY_TAG}
  EXTRA_ARGS=${EXTRA_ARGS:-<empty>}
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
TRM raw scale-out gate:

1. Train QTRM-native on modchain/revchain/checksum with longer program_len.
2. Use think_steps equal to the target reasoning depth.
3. Require full-depth generation exact, depth gain over think0, and destructive
   ablation drop. Also inspect eval-depth-sweep.
4. If len8/len12 fails, the breakthrough bottleneck is recurrent transition
   scaling, not MMLU answer rendering.
PLAN
    ;;
  status)
    remote "pwd; git status --short --branch; '${REMOTE_PYTHON}' - <<'PY'
import torch
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
PY"
    ;;
  run)
    extra_args=()
    if [[ -n "${RESUME_FROM}" ]]; then
      extra_args+=(--resume-from "'${RESUME_FROM}'")
      if [[ "${RESUME_ALLOW_MISSING}" == "1" ]]; then
        extra_args+=(--resume-allow-missing)
      fi
    fi
    if [[ "${INCLUDE_FAMILY_TAG}" == "1" ]]; then
      extra_args+=(--include-family-tag)
    fi
    remote "PYTHONPATH=src '${REMOTE_PYTHON}' scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py \
      --out-dir 'local_eval/dgx_trm_raw_scaleout_len${PROGRAM_LEN}_${OUT_TAG}' \
      --target-level 'TRM-like raw reasoning scale-out len${PROGRAM_LEN}' \
      ${extra_args[*]} \
      --steps '${STEPS}' \
      --train-cases '${TRAIN_CASES}' \
      --eval-cases '${EVAL_CASES}' \
      --task-families 'modchain,revchain,modchain,revchain,checksum' \
      --eval-task-families 'modchain,revchain,checksum' \
      --eval-family-order-invariant \
      --program-len '${PROGRAM_LEN}' \
      --modulus 32 \
      --d-model '${D_MODEL}' \
      --n-heads '${N_HEADS}' \
      --d-ff '${D_FF}' \
      --batch-size '${BATCH_SIZE}' \
      --lr '${LR}' \
      --device cuda \
      --train-think-steps '${THINK_STEPS}' \
      --eval-think-steps '${THINK_STEPS}' \
      --depth-intermediate-loss-weight 0.5 \
      --depth-intermediate-min-depth 1 \
      --active-len-curriculum \
      --eval-depth-sweep \
      --depth-counterfactual-loss-weight 0.10 \
      --depth-counterfactual-think-steps 0 \
      --depth-counterfactual-margin 1.0 \
      --state-reset-counterfactual-loss-weight 0.05 \
      --state-reset-counterfactual-margin 1.0 \
      --answer-margin-loss-weight 0.5 \
      --answer-margin 1.0 \
      --accept-min-exact '${ACCEPT_MIN_EXACT:-0.35}' \
      --accept-min-depth-gain '${ACCEPT_MIN_DEPTH_GAIN:-0.10}' \
      --accept-min-ablation-drop '${ACCEPT_MIN_ABLATION_DROP:-0.10}' \
      --accept-min-family-exact '${ACCEPT_MIN_FAMILY_EXACT:-0.15}' \
      --accepted-decision 'accepted_trm_raw_scaleout_len${PROGRAM_LEN}' \
      --log-every '${LOG_EVERY:-200}' \
      ${EXTRA_ARGS}"
    ;;
  *)
    echo "unknown action: ${ACTION}" >&2
    usage >&2
    exit 2
    ;;
esac
