#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/tripleyoung/qtrm-workspace/qtrm_multimodal_memoryos}"
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
CLEANED_DATA_PATH="${CLEANED_DATA_PATH:-/mnt/sdc1/datasets/hrm-text-data-io-cleaned-20260515}"
PTXAS="${PTXAS:-/usr/local/cuda-12.8/bin/ptxas}"

WORK_ROOT="${WORK_ROOT:-${ROOT}/local_eval/20260526_STAGE106_LOCAL_OPUS_GD_BLT_ABLATION}"
BASE_CKPT="${BASE_CKPT:-${ROOT}/local_eval/20260524_STAGE94Y_LOCAL_BLT2_TEACHEROFF_SEED9682_CONTROL1200/last.pt}"
SANITIZED_BASE_CKPT="${SANITIZED_BASE_CKPT:-${WORK_ROOT}/base_official_sanitized_model_only.pt}"
BASE_WARM_OUT="${BASE_WARM_OUT:-${WORK_ROOT}/base_warm_optimizer_seed}"
STATIC_WORK_DIR="${STATIC_WORK_DIR:-${WORK_ROOT}/static_first_window}"
OPUS_WORK_DIR="${OPUS_WORK_DIR:-${WORK_ROOT}/opus_gd_utility_window}"
EVAL_WORK_DIR="${EVAL_WORK_DIR:-${WORK_ROOT}/shared_eval_window}"
STATIC_SAMPLE="${STATIC_SAMPLE:-${STATIC_WORK_DIR}/sampled}"
OPUS_SAMPLE="${OPUS_SAMPLE:-${OPUS_WORK_DIR}/sampled}"
EVAL_SAMPLE="${EVAL_SAMPLE:-${EVAL_WORK_DIR}/sampled}"
STATIC_OUT="${STATIC_OUT:-${WORK_ROOT}/static_continue}"
OPUS_OUT="${OPUS_OUT:-${WORK_ROOT}/opus_gd_continue}"
BASE_WARM_LOG="${BASE_WARM_LOG:-${WORK_ROOT}/base_warm.log}"
STATIC_LOG="${STATIC_LOG:-${WORK_ROOT}/static_continue.log}"
OPUS_LOG="${OPUS_LOG:-${WORK_ROOT}/opus_gd_continue.log}"
BUILD_LOG="${BUILD_LOG:-${WORK_ROOT}/build.log}"
SUMMARY_JSON="${SUMMARY_JSON:-${WORK_ROOT}/summary.json}"

SOURCE_FILES="${SOURCE_FILES:-data/no_robots.jsonl data/natural_reasoning.jsonl data/webinstruct_verified.jsonl data/gsm8k_train.jsonl data/math_train.jsonl data/omnimath.jsonl data/Platypus/openbookqa.jsonl}"
SOURCE_GLOBS="${SOURCE_GLOBS:-}"
TRAIN_MAX_ROWS="${TRAIN_MAX_ROWS:-2048}"
EVAL_MAX_ROWS_SAMPLE="${EVAL_MAX_ROWS_SAMPLE:-512}"
MAX_ROWS_PER_FILE="${MAX_ROWS_PER_FILE:-256}"
MAX_SCAN_ROWS_PER_FILE="${MAX_SCAN_ROWS_PER_FILE:-2000}"
EPOCHS="${EPOCHS:-2}"
OPUS_CANDIDATE_MAX_ROWS="${OPUS_CANDIDATE_MAX_ROWS:-1024}"
OPUS_PROXY_JSONL="${OPUS_PROXY_JSONL:-${ROOT}/data/eval/prefixlm_language_heldout.jsonl ${ROOT}/data/eval/official_gdsuite_choice_probe.jsonl}"
OPUS_PROXY_MAX_ROWS="${OPUS_PROXY_MAX_ROWS:-0}"
OPUS_PROXY_MAX_ROWS_PER_GROUP="${OPUS_PROXY_MAX_ROWS_PER_GROUP:-8}"
OPUS_PROXY_GROUPING="${OPUS_PROXY_GROUPING:-source_file_bucket}"
OPUS_PROXY_SCORE_MODE="${OPUS_PROXY_SCORE_MODE:-minimax_mean}"
OPUS_PROXY_MEAN_WEIGHT="${OPUS_PROXY_MEAN_WEIGHT:-0.25}"
OPUS_PROJECTION_DIM="${OPUS_PROJECTION_DIM:-1024}"
OPUS_PRECONDITIONER="${OPUS_PRECONDITIONER:-adamw_state}"
OPUS_DEVICE="${OPUS_DEVICE:-cuda}"
DEFAULT_GD_SMOKE_PROBE="${ROOT}/local_eval/20260526_STAGE107L_LOCAL_ONLINE_OPUS_EFFECT_GDSUITE_SMOKE/official_gdsuite_choice_probe_2pertask.jsonl"
if [[ -z "${GD_PROBE_JSONL:-}" ]]; then
  if [[ -f "${DEFAULT_GD_SMOKE_PROBE}" ]]; then
    GD_PROBE_JSONL="${DEFAULT_GD_SMOKE_PROBE}"
  else
    GD_PROBE_JSONL="${ROOT}/data/eval/official_gdsuite_choice_probe.jsonl"
  fi
fi
GD_MAX_ROWS="${GD_MAX_ROWS:-0}"

START_STEP="${START_STEP:-1200}"
BASE_WARM_STEPS="${BASE_WARM_STEPS:-1280}"
TARGET_STEPS="${TARGET_STEPS:-1440}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-80}"
EVAL_EVERY="${EVAL_EVERY:-80}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
SEQ_LEN="${SEQ_LEN:-384}"
SEED="${SEED:-10601}"

ACTION="${1:-plan}"

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export REQUIRED_TRITON_PTXAS_PATH="${REQUIRED_TRITON_PTXAS_PATH:-${PTXAS}}"
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-${PTXAS}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

usage() {
  cat <<USAGE
Local OPUS/GD BLT ablation

Purpose:
  Compare a static byte/BLT data window against an OPUS/GD utility-selected
  byte/BLT window from the same starting checkpoint, same optimizer state, same
  model, same step budget.

Actions:
  plan
  status
  gates
  summarize
  run
  launch

Current contract:
  BASE_CKPT=${BASE_CKPT}
  SANITIZED_BASE_CKPT=${SANITIZED_BASE_CKPT}
  BASE_WARM_OUT=${BASE_WARM_OUT}
  STATIC_SAMPLE=${STATIC_SAMPLE}
  OPUS_SAMPLE=${OPUS_SAMPLE}
  EVAL_SAMPLE=${EVAL_SAMPLE}
  STATIC_OUT=${STATIC_OUT}
  OPUS_OUT=${OPUS_OUT}
  TARGET_STEPS=${TARGET_STEPS}
USAGE
}

status() {
  usage
  echo
  echo "paths:"
  for path in "${BASE_CKPT}" "${SANITIZED_BASE_CKPT}" "${BASE_WARM_OUT}/last.pt" "${STATIC_SAMPLE}/metadata.json" "${OPUS_SAMPLE}/metadata.json" "${EVAL_SAMPLE}/metadata.json" "${STATIC_OUT}/report.json" "${OPUS_OUT}/report.json" "${SUMMARY_JSON}"; do
    if [[ -e "${path}" ]]; then
      echo "  present ${path}"
    else
      echo "  missing ${path}"
    fi
  done
  echo
  echo "local training processes:"
  pgrep -af 'scripts/534_train_native_prefixlm_dataio.py|scripts/557_train_blt_d_prefixlm_dataio.py' || true
}

require_ready() {
  if [[ ! -x "${PYTHON}" ]]; then
    echo "missing python: ${PYTHON}" >&2
    exit 2
  fi
  if [[ ! -d "${CLEANED_DATA_PATH}" ]]; then
    echo "missing cleaned data: ${CLEANED_DATA_PATH}" >&2
    exit 2
  fi
  if [[ ! -f "${BASE_CKPT}" ]]; then
    echo "missing base checkpoint: ${BASE_CKPT}" >&2
    exit 2
  fi
  if [[ ! -x "${TRITON_PTXAS_PATH}" ]]; then
    echo "missing ptxas: ${TRITON_PTXAS_PATH}" >&2
    exit 2
  fi
}

wait_for_local_training_slot() {
  while pgrep -af 'scripts/534_train_native_prefixlm_dataio.py|scripts/557_train_blt_d_prefixlm_dataio.py' >/dev/null 2>&1; do
    echo "waiting_for_local_gpu_training_slot"
    pgrep -af 'scripts/534_train_native_prefixlm_dataio.py|scripts/557_train_blt_d_prefixlm_dataio.py' || true
    sleep 60
  done
}

build_sample() {
  local mode="$1"
  local work_dir="$2"
  local sampled_out="$3"
  local max_rows="$4"
  local opus_checkpoint="$5"
  if [[ -f "${sampled_out}/metadata.json" ]]; then
    echo "sample_ready mode=${mode} path=${sampled_out}"
    return 0
  fi
  mkdir -p "$(dirname "${sampled_out}")"
  (
    cd "${ROOT}"
    env \
      CLEANED_DATA_PATH="${CLEANED_DATA_PATH}" \
      WORK_DIR="${work_dir}" \
      SAMPLED_OUT="${sampled_out}" \
      SOURCE_FILES="${SOURCE_FILES}" \
      SOURCE_GLOBS="${SOURCE_GLOBS}" \
      MAX_ROWS="${max_rows}" \
      MAX_ROWS_PER_FILE="${MAX_ROWS_PER_FILE}" \
      MAX_SCAN_ROWS_PER_FILE="${MAX_SCAN_ROWS_PER_FILE}" \
      EPOCHS="${EPOCHS}" \
      SELECTION_MODE="${mode}" \
      OPUS_CHECKPOINT="${opus_checkpoint}" \
      OPUS_SCORE_OUT="${work_dir}/opus_projected_utility_scores.jsonl" \
      OPUS_REPORT_OUT="${work_dir}/opus_projected_utility_report.json" \
      OPUS_CANDIDATE_MAX_ROWS="${OPUS_CANDIDATE_MAX_ROWS}" \
      OPUS_CANDIDATE_MAX_SCAN_ROWS_PER_FILE="${MAX_SCAN_ROWS_PER_FILE}" \
      OPUS_PROXY_JSONL="${OPUS_PROXY_JSONL}" \
      OPUS_PROXY_MAX_ROWS="${OPUS_PROXY_MAX_ROWS}" \
      OPUS_PROXY_MAX_ROWS_PER_GROUP="${OPUS_PROXY_MAX_ROWS_PER_GROUP}" \
      OPUS_PROXY_GROUPING="${OPUS_PROXY_GROUPING}" \
      OPUS_PROXY_SCORE_MODE="${OPUS_PROXY_SCORE_MODE}" \
      OPUS_PROXY_MEAN_WEIGHT="${OPUS_PROXY_MEAN_WEIGHT}" \
      OPUS_PROJECTION_DIM="${OPUS_PROJECTION_DIM}" \
      OPUS_PRECONDITIONER="${OPUS_PRECONDITIONER}" \
      OPUS_DEVICE="${OPUS_DEVICE}" \
      PYTHON="${PYTHON}" \
      bash scripts/558_prepare_stage95_blt_foundation_byte_sample_dgx.sh build
  ) >>"${BUILD_LOG}" 2>&1
}

sanitize_base_checkpoint() {
  if [[ -f "${SANITIZED_BASE_CKPT}" ]]; then
    echo "sanitized_base_ready path=${SANITIZED_BASE_CKPT}"
    return 0
  fi
  mkdir -p "$(dirname "${SANITIZED_BASE_CKPT}")"
  "${PYTHON}" - <<PY
from pathlib import Path
import torch

src = Path("${BASE_CKPT}")
dst = Path("${SANITIZED_BASE_CKPT}")
payload = torch.load(src, map_location="cpu")
if not isinstance(payload, dict) or "model_state_dict" not in payload:
    raise SystemExit(f"unsupported checkpoint payload: {src}")

legacy_markers = (
    ".mixer.runtime_fallback.",
    ".mixer.impl.in_proj.",
    ".mixer.impl.gate_proj.",
    ".mixer.impl.out_proj.",
)
state = payload["model_state_dict"]
clean_state = {
    key: value
    for key, value in state.items()
    if not any(marker in key for marker in legacy_markers)
}
removed = len(state) - len(clean_state)

clean_payload = dict(payload)
clean_payload["model_state_dict"] = clean_state
clean_payload.pop("optimizer_state_dict", None)
clean_payload["checkpoint_includes_optimizer"] = False
clean_payload["stage106_sanitized_from"] = str(src)
clean_payload["stage106_removed_legacy_delta_fallback_keys"] = int(removed)
dst.parent.mkdir(parents=True, exist_ok=True)
torch.save(clean_payload, dst)
print(f"sanitized_base_written path={dst} removed_legacy_delta_fallback_keys={removed}")
PY
}

train_window() {
  local sampled_data="$1"
  local out_dir="$2"
  local log_path="$3"
  local resume_ckpt="$4"
  local load_optimizer="${5:-1}"
  local target_steps="${6:-${TARGET_STEPS}}"
  local strict_resume="${7:-1}"
  local resume_optimizer_args=()
  local resume_strict_args=(--resume-strict)
  if [[ "${load_optimizer}" == "1" ]]; then
    resume_optimizer_args+=(--resume-load-optimizer)
  fi
  if [[ "${strict_resume}" != "1" ]]; then
    resume_strict_args=(--no-resume-strict)
  fi
  if [[ -f "${out_dir}/report.json" ]]; then
    echo "train_complete out=${out_dir}"
    return 0
  fi
  mkdir -p "${out_dir}"
  (
    cd "${ROOT}"
    "${PYTHON}" scripts/557_train_blt_d_prefixlm_dataio.py \
      --sampled-data "${sampled_data}" \
      --eval-sampled-data "${EVAL_SAMPLE}" \
      --out-dir "${out_dir}" \
      --resume "${resume_ckpt}" \
      "${resume_strict_args[@]}" \
      "${resume_optimizer_args[@]}" \
      --steps "${target_steps}" \
      --checkpoint-every "${CHECKPOINT_EVERY}" \
      --batch-size "${BATCH_SIZE}" \
      --seq-len "${SEQ_LEN}" \
      --eval-every "${EVAL_EVERY}" \
      --eval-max-rows "${EVAL_MAX_ROWS_SAMPLE}" \
      --eval-batch-size "${EVAL_BATCH_SIZE}" \
      --eval-max-batches 0 \
      --lr 2.2e-4 \
      --lr-warmup-steps 500 \
      --adam-beta1 0.9 \
      --adam-beta2 0.95 \
      --weight-decay 0.1 \
      --grad-clip 1.0 \
      --amp-dtype bf16 \
      --matmul-precision high \
      --log-every 20 \
      --tensorboard-dir "${out_dir}/tensorboard" \
      --patch-size 2 \
      --patch-boundary-mode fixed \
      --dynamic-min-patch-size 2 \
      --dynamic-soft-patch-size 0 \
      --hbf-boundary-threshold 0.35 \
      --boundary-prior-weight 0.0 \
      --boundary-target-ratio 0.5 \
      --decoder-latent-mode add \
      --diffusion-weight 0.0 \
      --diffusion-mask-prob 0.0 \
      --d-model 384 \
      --n-heads 6 \
      --n-kv-heads 2 \
      --d-ff 1024 \
      --dropout 0.0 \
      --local-layers 2 \
      --local-heads 4 \
      --backbone trm_qwen35_3to1 \
      --think-structure trm_dual_z \
      --train-think-steps 2 \
      --delta-backend official_gated_delta2 \
      --strict-backends \
      --attention-backend sdpa \
      --seed "${SEED}" \
      --save-optimizer-checkpoint \
      --optimizer-checkpoint-every "${CHECKPOINT_EVERY}"
  ) >>"${log_path}" 2>&1
}

warm_base_checkpoint() {
  if [[ -f "${BASE_WARM_OUT}/last.pt" && -f "${BASE_WARM_OUT}/report.json" ]]; then
    echo "base_warm_ready out=${BASE_WARM_OUT}"
    return 0
  fi
  train_window "${STATIC_SAMPLE}" "${BASE_WARM_OUT}" "${BASE_WARM_LOG}" "${SANITIZED_BASE_CKPT}" 0 "${BASE_WARM_STEPS}" 0
}

best_checkpoint() {
  local out_dir="$1"
  if [[ -f "${out_dir}/best_eval_model.pt" ]]; then
    echo "${out_dir}/best_eval_model.pt"
  else
    echo "${out_dir}/last_model.pt"
  fi
}

run_gates() {
  local label="$1"
  local out_dir="$2"
  local ckpt
  ckpt="$(best_checkpoint "${out_dir}")"
  "${PYTHON}" "${ROOT}/scripts/567_eval_blt_generalization_dynamics_probe.py" \
    --checkpoint "${ckpt}" \
      --sampled-data "${EVAL_SAMPLE}" \
      --device cuda \
      --think-steps 2 \
      --max-rows "${GD_MAX_ROWS}" \
      --probe-jsonl "${GD_PROBE_JSONL}" \
    --out "${out_dir}/gdsuite_smoke_44_report.json" \
    --tensorboard-dir "${out_dir}/tensorboard" \
    --tensorboard-prefix "eval/generalization_dynamics_official" \
    >/dev/null
  "${PYTHON}" "${ROOT}/scripts/565_eval_blt_generation_gate.py" \
    --checkpoint "${ckpt}" \
    --sampled-data "${EVAL_SAMPLE}" \
    --device cuda \
    --think-steps 2 \
    --max-first-token-rows 128 \
    --first-token-batch-size 4 \
    --max-continuation-rows 128 \
    --continuation-batch-size 4 \
    --max-generation-rows 8 \
    --max-new-tokens 48 \
    --out "${out_dir}/generation_gate.json" \
    >/dev/null
  echo "gates_done label=${label} checkpoint=${ckpt}"
}

summarize() {
  "${PYTHON}" - <<PY
import json
from pathlib import Path

work = Path("${WORK_ROOT}")
static_out = Path("${STATIC_OUT}")
opus_out = Path("${OPUS_OUT}")

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def run_summary(out):
    report = load(out / "report.json")
    gd = load(out / "gdsuite_smoke_44_report.json")
    gen = load(out / "generation_gate.json")
    generation = gen.get("generation") or {}
    first_response = gen.get("first_response") or {}
    continuation = gen.get("response_continuation") or {}
    return {
        "final_eval_loss": report.get("final_eval_loss"),
        "best_eval_loss": report.get("best_eval_loss"),
        "best_eval_step": report.get("best_eval_step"),
        "gdsuite_smoke": gd.get("summary"),
        "generation_summary": {
            "first_response_accuracy": first_response.get("accuracy"),
            "continuation_accuracy": continuation.get("accuracy"),
            "continuation_eos_top1_accuracy": continuation.get("eos_top1_accuracy"),
            "free_generation_exact_fraction": generation.get("exact_fraction"),
            "free_generation_prefix_token_accuracy": generation.get("prefix_token_accuracy"),
            "free_generation_ended_with_eoa_fraction": generation.get("ended_with_eoa_fraction"),
            "free_generation_repeated_token_loop_fraction": generation.get("repeated_token_loop_fraction"),
        },
    }

summary = {
    "contract": "stage106_local_opus_gd_blt_ablation",
    "plain_language_read": (
        "This compares whether OPUS/GD-selected rows teach the same small BLT "
        "student better than a static first-seen window from the same base "
        "checkpoint and optimizer state."
    ),
    "base_checkpoint": "${BASE_CKPT}",
    "sanitized_base_checkpoint": "${SANITIZED_BASE_CKPT}",
    "branch_base_checkpoint": "${BASE_WARM_OUT}/last.pt",
    "static": run_summary(static_out),
    "opus_gd": run_summary(opus_out),
}
Path("${SUMMARY_JSON}").parent.mkdir(parents=True, exist_ok=True)
Path("${SUMMARY_JSON}").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
}

run() {
  require_ready
  wait_for_local_training_slot
  mkdir -p "${WORK_ROOT}"
  build_sample first "${EVAL_WORK_DIR}" "${EVAL_SAMPLE}" "${EVAL_MAX_ROWS_SAMPLE}" ""
  build_sample first "${STATIC_WORK_DIR}" "${STATIC_SAMPLE}" "${TRAIN_MAX_ROWS}" ""
  sanitize_base_checkpoint
  warm_base_checkpoint
  build_sample utility "${OPUS_WORK_DIR}" "${OPUS_SAMPLE}" "${TRAIN_MAX_ROWS}" "${BASE_WARM_OUT}/last.pt"
  train_window "${STATIC_SAMPLE}" "${STATIC_OUT}" "${STATIC_LOG}" "${BASE_WARM_OUT}/last.pt" 1 "${TARGET_STEPS}" 1
  train_window "${OPUS_SAMPLE}" "${OPUS_OUT}" "${OPUS_LOG}" "${BASE_WARM_OUT}/last.pt" 1 "${TARGET_STEPS}" 1
  run_gates static "${STATIC_OUT}"
  run_gates opus_gd "${OPUS_OUT}"
  summarize
}

launch() {
  local session="${SESSION:-stage106_local_opus_gd_blt_ablation}"
  mkdir -p "${WORK_ROOT}"
  tmux kill-session -t "${session}" 2>/dev/null || true
  tmux new-session -d -s "${session}" "cd '${ROOT}' && bash scripts/622_run_local_opus_gd_blt_ablation.sh run >> '${WORK_ROOT}/supervisor.log' 2>&1"
  echo "launched_session=${session}"
  echo "supervisor_log=${WORK_ROOT}/supervisor.log"
}

case "${ACTION}" in
  plan)
    usage
    ;;
  status)
    status
    ;;
  run)
    run
    ;;
  gates)
    require_ready
    run_gates static "${STATIC_OUT}"
    run_gates opus_gd "${OPUS_OUT}"
    ;;
  summarize)
    summarize
    ;;
  launch)
    launch
    ;;
  *)
    echo "unknown action: ${ACTION}" >&2
    usage >&2
    exit 2
    ;;
esac
