#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-plan}"
DGX_HOST="${DGX_HOST:-dgx}"
DGX_REPO="${DGX_REPO:-/mnt/data4tb/wgram-lm}"
REMOTE_PYTHON="${REMOTE_PYTHON:-.venv/bin/python}"
OUT_TAG="${OUT_TAG:-$(date +%Y%m%d_%H%M%S)}"

usage() {
  cat <<USAGE
Usage:
  $0 [plan|status|sync|m7-final-token-repair|m7-depth-256|m7-fastlane]

Environment overrides:
  DGX_HOST=${DGX_HOST}
  DGX_REPO=${DGX_REPO}
  REMOTE_PYTHON=${REMOTE_PYTHON}
  OUT_TAG=${OUT_TAG}

Purpose:
  DGX fastlane for QTRM-native public-style core-depth scale-out repair.
  This keeps final inference native and uses DGX for faster training/eval.
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
DGX QTRM-native fastlane:

1. sync
   Pull the latest repo on DGX.

2. m7-final-token-repair
   Train the current QTRM-native LM path on public-style MCQ answer rendering
   without using a runtime donor.

3. m7-depth-256
   Evaluate depth0/1/2/4/8 on 256 held-out public-style cases.

4. accept only if deeper recurrence beats depth0 and shallow depths.

This is for core-depth scale-out repair, not MemoryOS/RAG or donor-sidecar wins.
PLAN
    ;;

  status)
    remote "pwd; git status --short --branch; nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader || true; bash scripts/407_qwen36_mtp_llama_server_dgx_local.sh status | sed -n '1,8p'"
    ;;

  sync)
    remote "git pull --ff-only"
    ;;

  m7-final-token-repair)
    remote "PYTHON='${REMOTE_PYTHON}' PYTHONPATH=src OUT_ROOT='local_eval/dgx_qtrm_native_m7_final_token_repair_${OUT_TAG}' INIT_CHECKPOINT='${INIT_CHECKPOINT:-local_eval/qtrm_native_m7a_final_token_space_mmluproval64_s300_20260516/last.pt}' DEVICE=cuda STEPS='${STEPS:-1200}' BATCH_SIZE='${BATCH_SIZE:-32}' LR='${LR:-1.0e-4}' MAX_TRAIN_CASES='${MAX_TRAIN_CASES:-256}' MAX_EVAL_CASES='${MAX_EVAL_CASES:-256}' THINK_STEPS='${THINK_STEPS:-8}' MULTI_DEPTH_CE_WEIGHT='${MULTI_DEPTH_CE_WEIGHT:-0.35}' MULTI_DEPTH_CE_DEPTHS='${MULTI_DEPTH_CE_DEPTHS:-4,8}' DEPTH_GAIN_WEIGHT='${DEPTH_GAIN_WEIGHT:-0.75}' DEPTH_GAIN_MARGIN='${DEPTH_GAIN_MARGIN:-0.25}' DEPTH_GAIN_SHALLOW_DEPTHS='${DEPTH_GAIN_SHALLOW_DEPTHS:-0,1,2,4}' TRAJECTORY_KL_WEIGHT='${TRAJECTORY_KL_WEIGHT:-0.05}' TRAJECTORY_KL_ANCHOR_DEPTH='${TRAJECTORY_KL_ANCHOR_DEPTH:-8}' TRAJECTORY_KL_COMPARE_DEPTHS='${TRAJECTORY_KL_COMPARE_DEPTHS:-6}' bash scripts/401_run_qtrm_native_m7a_final_token_healing.sh"
    ;;

  m7-depth-256)
    CHECKPOINT="${CHECKPOINT:-local_eval/qtrm_native_m7a_final_token_space_mmluproval64_s300_20260516/last.pt}"
    remote "PYTHON='${REMOTE_PYTHON}' PYTHONPATH=src OUT_ROOT='local_eval/dgx_qtrm_native_m7b_depth256_${OUT_TAG}' CHECKPOINT='${CHECKPOINT}' DEVICE=cuda MAX_CASES='${MAX_CASES:-256}' FULL_THINK_STEPS='${FULL_THINK_STEPS:-8}' SHALLOW_DEPTHS='${SHALLOW_DEPTHS:-1 2 4}' BASELINE_DEPTH='${BASELINE_DEPTH:-0}' MIN_GAIN_VS_BASELINE='${MIN_GAIN_VS_BASELINE:-0.03}' MIN_GAIN_VS_BEST_SHALLOW='${MIN_GAIN_VS_BEST_SHALLOW:-0.03}' bash scripts/403_run_qtrm_native_m7b_core_depth_gate.sh"
    ;;

  m7-fastlane)
    "$0" sync
    "$0" m7-final-token-repair
    CHECKPOINT="local_eval/dgx_qtrm_native_m7_final_token_repair_${OUT_TAG}/train/last.pt" "$0" m7-depth-256
    ;;

  *)
    echo "unknown action: ${ACTION}" >&2
    usage >&2
    exit 2
    ;;
esac
