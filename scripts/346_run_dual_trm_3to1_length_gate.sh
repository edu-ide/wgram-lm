#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
PYTHONPATH_VALUE="${PYTHONPATH:-local_deps/mamba3_runtime:src}"
OUT_ROOT="${OUT_ROOT:-local_eval/qtrm_native_dual_path_reverse_length_gate_$(date +%Y%m%d_%H%M%S)}"
PROFILE="${PROFILE:-short}"
LENGTHS="${LENGTHS:-4,6,8}"
CANDIDATES="${CANDIDATES:-official,dual_path_reverse}"
THINK_STRUCTURE="${THINK_STRUCTURE:-trm_dual_z_interactive}"
TASK_FAMILIES="${TASK_FAMILIES:-checksum,modchain,revchain}"
EVAL_TASK_FAMILIES="${EVAL_TASK_FAMILIES:-}"
DELTA_BACKEND="${DELTA_BACKEND:-torch_gated_delta}"
STRICT_BACKENDS="${STRICT_BACKENDS:-0}"
if [[ "${STRICT_BACKENDS}" != "1" ]]; then
  export QTRM_DISABLE_LOCAL_MAMBA_REFERENCE="${QTRM_DISABLE_LOCAL_MAMBA_REFERENCE:-1}"
fi
DIFFUSIVE_LATENT_REFINE_LOSS_WEIGHT="${DIFFUSIVE_LATENT_REFINE_LOSS_WEIGHT:-0.1}"
DIFFUSIVE_LATENT_REFINE_NOISE_STD="${DIFFUSIVE_LATENT_REFINE_NOISE_STD:-0.01}"
DIFFUSIVE_LATENT_REFINE_DEPTH_WEIGHT_POWER="${DIFFUSIVE_LATENT_REFINE_DEPTH_WEIGHT_POWER:-0.5}"
DIFFUSIVE_LATENT_REFINE_FINAL_KL_WEIGHT="${DIFFUSIVE_LATENT_REFINE_FINAL_KL_WEIGHT:-0.05}"
DEVICE="${DEVICE:-cuda}"
D_MODEL="${D_MODEL:-64}"
D_FF="${D_FF:-128}"
N_HEADS="${N_HEADS:-4}"
N_KV_HEADS="${N_KV_HEADS:-2}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-4}"
LR_SCHEDULE="${LR_SCHEDULE:-constant}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"
LR_MIN_RATIO="${LR_MIN_RATIO:-0.1}"
SEED_BASE="${SEED_BASE:-613}"
EVAL_SEED="${EVAL_SEED:-9613}"
TOKENIZER_MODE="${TOKENIZER_MODE:-char}"
NUMBER_TOKENIZER_MAX_VALUE="${NUMBER_TOKENIZER_MAX_VALUE:-99}"
TRAIN_ACTIVE_LEN_MIN="${TRAIN_ACTIVE_LEN_MIN:-2}"
TRAIN_ACTIVE_LEN_MAX="${TRAIN_ACTIVE_LEN_MAX:--1}"
TRM_L_CYCLES="${TRM_L_CYCLES:-6}"
DEPTH_INTERMEDIATE_LOSS_WEIGHT="${DEPTH_INTERMEDIATE_LOSS_WEIGHT:-0.25}"
DEPTH_INTERMEDIATE_WEIGHT_POWER="${DEPTH_INTERMEDIATE_WEIGHT_POWER:-0.0}"
HALT_DEPTH_FINAL_LOSS_WEIGHT="${HALT_DEPTH_FINAL_LOSS_WEIGHT:-1.0}"
ADAPTIVE_HALT_EVAL="${ADAPTIVE_HALT_EVAL:-1}"
ADAPTIVE_HALT_LOSS_WEIGHT="${ADAPTIVE_HALT_LOSS_WEIGHT:-5.0}"
ACCEPT_MIN_EXACT="${ACCEPT_MIN_EXACT:-0.0}"
ACCEPT_MIN_DEPTH_GAIN="${ACCEPT_MIN_DEPTH_GAIN:--1.0}"
ACCEPT_MIN_ABLATION_DROP="${ACCEPT_MIN_ABLATION_DROP:--1.0}"
OPERATION_COUNTERFACTUAL_LOSS_WEIGHT="${OPERATION_COUNTERFACTUAL_LOSS_WEIGHT:-0.0}"
OPERATION_COUNTERFACTUAL_MARGIN="${OPERATION_COUNTERFACTUAL_MARGIN:-1.0}"
OPERATION_COUNTERFACTUAL_MAX_CASES="${OPERATION_COUNTERFACTUAL_MAX_CASES:-0}"
OPERATION_COUNTERFACTUAL_EVERY="${OPERATION_COUNTERFACTUAL_EVERY:-1}"
OPERATION_COUNTERFACTUAL_WARMUP_STEPS="${OPERATION_COUNTERFACTUAL_WARMUP_STEPS:-0}"
OPERATION_COUNTERFACTUAL_END_STEP="${OPERATION_COUNTERFACTUAL_END_STEP:--1}"
OPERATION_COUNTERFACTUAL_ACTIVE_LEN_MIN="${OPERATION_COUNTERFACTUAL_ACTIVE_LEN_MIN:-1}"
OPERATION_COUNTERFACTUAL_ACTIVE_LEN_MAX="${OPERATION_COUNTERFACTUAL_ACTIVE_LEN_MAX:--1}"
DEPTH_COUNTERFACTUAL_LOSS_WEIGHT="${DEPTH_COUNTERFACTUAL_LOSS_WEIGHT:-0.0}"
DEPTH_COUNTERFACTUAL_MARGIN="${DEPTH_COUNTERFACTUAL_MARGIN:-1.0}"
DEPTH_COUNTERFACTUAL_THINK_STEPS="${DEPTH_COUNTERFACTUAL_THINK_STEPS:-0}"
DEPTH_COUNTERFACTUAL_EVERY="${DEPTH_COUNTERFACTUAL_EVERY:-1}"
STATE_RESET_COUNTERFACTUAL_LOSS_WEIGHT="${STATE_RESET_COUNTERFACTUAL_LOSS_WEIGHT:-0.0}"
STATE_RESET_COUNTERFACTUAL_MARGIN="${STATE_RESET_COUNTERFACTUAL_MARGIN:-1.0}"
STATE_RESET_COUNTERFACTUAL_EVERY="${STATE_RESET_COUNTERFACTUAL_EVERY:-1}"
RETENTION_REFERENCE_CHECKPOINT="${RETENTION_REFERENCE_CHECKPOINT:-}"
RETENTION_KL_LOSS_WEIGHT="${RETENTION_KL_LOSS_WEIGHT:-0.0}"
RETENTION_ACTIVE_LEN_MIN="${RETENTION_ACTIVE_LEN_MIN:-1}"
RETENTION_ACTIVE_LEN_MAX="${RETENTION_ACTIVE_LEN_MAX:--1}"
RETENTION_MAX_CASES="${RETENTION_MAX_CASES:-0}"
RETENTION_EVERY="${RETENTION_EVERY:-1}"
RETENTION_TEMPERATURE="${RETENTION_TEMPERATURE:-1.0}"
ANSWER_SPACE_RANKING_LOSS_WEIGHT="${ANSWER_SPACE_RANKING_LOSS_WEIGHT:-0.0}"
ANSWER_SPACE_RANKING_MAX_CASES="${ANSWER_SPACE_RANKING_MAX_CASES:-0}"
ANSWER_SPACE_RANKING_EVERY="${ANSWER_SPACE_RANKING_EVERY:-1}"
ANSWER_SPACE_RANKING_TEMPERATURE="${ANSWER_SPACE_RANKING_TEMPERATURE:-1.0}"
FAMILY_DRO_LOSS_WEIGHT="${FAMILY_DRO_LOSS_WEIGHT:-0.0}"
FAMILY_DRO_TEMPERATURE="${FAMILY_DRO_TEMPERATURE:-0.0}"
ACTIVE_LEN_REPLAY_LOSS_WEIGHT="${ACTIVE_LEN_REPLAY_LOSS_WEIGHT:-0.0}"
ACTIVE_LEN_REPLAY_MIN="${ACTIVE_LEN_REPLAY_MIN:-${TRAIN_ACTIVE_LEN_MIN}}"
ACTIVE_LEN_REPLAY_MAX="${ACTIVE_LEN_REPLAY_MAX:-${TRAIN_ACTIVE_LEN_MAX}}"
ACTIVE_LEN_REPLAY_MAX_CASES="${ACTIVE_LEN_REPLAY_MAX_CASES:-0}"
ACTIVE_LEN_REPLAY_EVERY="${ACTIVE_LEN_REPLAY_EVERY:-1}"
STATE_TRACE_ANTI_COLLAPSE_LOSS_WEIGHT="${STATE_TRACE_ANTI_COLLAPSE_LOSS_WEIGHT:-0.0}"
STATE_TRACE_MIN_VARIANCE="${STATE_TRACE_MIN_VARIANCE:-0.6}"
STATE_TRACE_MIN_DELTA_NORM="${STATE_TRACE_MIN_DELTA_NORM:-3.5}"
CORE_STEP_CODEC_LOSS_WEIGHT="${CORE_STEP_CODEC_LOSS_WEIGHT:-0.0}"
CORE_STEP_CODEC_STATE_SOURCE="${CORE_STEP_CODEC_STATE_SOURCE:-both}"
CORE_STEP_CODEC_POOLING="${CORE_STEP_CODEC_POOLING:-last}"
RESUME_FROM="${RESUME_FROM:-}"
RESUME_ALLOW_MISSING="${RESUME_ALLOW_MISSING:-0}"
FAIL_FAST_ON_ACTIVE_REJECT="${FAIL_FAST_ON_ACTIVE_REJECT:-1}"
SKIP_EXISTING_CANDIDATES="${SKIP_EXISTING_CANDIDATES:-0}"

case "${PROFILE}" in
  smoke)
    STEPS="${STEPS:-12}"
    TRAIN_CASES="${TRAIN_CASES:-128}"
    EVAL_CASES="${EVAL_CASES:-48}"
    LOG_EVERY="${LOG_EVERY:-6}"
    ;;
  triage)
    STEPS="${STEPS:-120}"
    TRAIN_CASES="${TRAIN_CASES:-4096}"
    EVAL_CASES="${EVAL_CASES:-96}"
    LOG_EVERY="${LOG_EVERY:-60}"
    ;;
  short)
    STEPS="${STEPS:-800}"
    TRAIN_CASES="${TRAIN_CASES:-4096}"
    EVAL_CASES="${EVAL_CASES:-192}"
    LOG_EVERY="${LOG_EVERY:-200}"
    ;;
  standard)
    STEPS="${STEPS:-3000}"
    TRAIN_CASES="${TRAIN_CASES:-8192}"
    EVAL_CASES="${EVAL_CASES:-384}"
    LOG_EVERY="${LOG_EVERY:-500}"
    ;;
  *)
    echo "Unsupported PROFILE=${PROFILE}; expected smoke, triage, short, or standard" >&2
    exit 2
    ;;
esac

mkdir -p "${OUT_ROOT}"
rm -f "${OUT_ROOT}/fail_fast_stop.json"

IFS=',' read -r -a length_values <<< "${LENGTHS}"
IFS=',' read -r -a candidate_values <<< "${CANDIDATES}"

run_candidate() {
  local program_len="$1"
  local candidate="$2"
  local think_backbone=""
  local think_structure="${THINK_STRUCTURE}"
  local seed_offset=0
  local latent_refine_args=()
  local train_think_steps="${TRAIN_THINK_STEPS:-${program_len}}"
  local eval_think_steps="${EVAL_THINK_STEPS:-${train_think_steps}}"
  local train_active_len_max="${TRAIN_ACTIVE_LEN_MAX}"
  if [[ "${train_active_len_max}" == "-1" ]]; then
    train_active_len_max="${program_len}"
  fi

  case "${candidate}" in
    official)
      think_backbone="trm_official"
      seed_offset=0
      ;;
    qwen35_3to1)
      think_backbone="trm_qwen35_3to1"
      seed_offset=100
      ;;
    diffusive_trm)
      think_backbone="trm_official"
      think_structure="trm_dual_z_diffusive"
      seed_offset=200
      latent_refine_args+=(
        --latent-refine-loss-weight "${DIFFUSIVE_LATENT_REFINE_LOSS_WEIGHT}"
        --latent-refine-min-depth 1
        --latent-refine-noise-std "${DIFFUSIVE_LATENT_REFINE_NOISE_STD}"
        --latent-refine-depth-weight-power "${DIFFUSIVE_LATENT_REFINE_DEPTH_WEIGHT_POWER}"
        --latent-refine-final-kl-weight "${DIFFUSIVE_LATENT_REFINE_FINAL_KL_WEIGHT}"
      )
      ;;
    diffusive_gated_delta)
      think_backbone="trm_gated_delta"
      think_structure="trm_dual_z_diffusive"
      seed_offset=300
      latent_refine_args+=(
        --latent-refine-loss-weight "${DIFFUSIVE_LATENT_REFINE_LOSS_WEIGHT}"
        --latent-refine-min-depth 1
        --latent-refine-noise-std "${DIFFUSIVE_LATENT_REFINE_NOISE_STD}"
        --latent-refine-depth-weight-power "${DIFFUSIVE_LATENT_REFINE_DEPTH_WEIGHT_POWER}"
        --latent-refine-final-kl-weight "${DIFFUSIVE_LATENT_REFINE_FINAL_KL_WEIGHT}"
      )
      ;;
    diffusive_mamba3)
      think_backbone="trm_mamba3"
      think_structure="trm_dual_z_diffusive"
      seed_offset=400
      latent_refine_args+=(
        --latent-refine-loss-weight "${DIFFUSIVE_LATENT_REFINE_LOSS_WEIGHT}"
        --latent-refine-min-depth 1
        --latent-refine-noise-std "${DIFFUSIVE_LATENT_REFINE_NOISE_STD}"
        --latent-refine-depth-weight-power "${DIFFUSIVE_LATENT_REFINE_DEPTH_WEIGHT_POWER}"
        --latent-refine-final-kl-weight "${DIFFUSIVE_LATENT_REFINE_FINAL_KL_WEIGHT}"
      )
      ;;
    dual_path_reverse_diffusive|diffusive_reversed_hybrid_3to1)
      think_backbone="trm_official"
      think_structure="trm_dual_z_diffusive_reversed_hybrid_3to1"
      seed_offset=500
      latent_refine_args+=(
        --latent-refine-loss-weight "${DIFFUSIVE_LATENT_REFINE_LOSS_WEIGHT}"
        --latent-refine-min-depth 1
        --latent-refine-noise-std "${DIFFUSIVE_LATENT_REFINE_NOISE_STD}"
        --latent-refine-depth-weight-power "${DIFFUSIVE_LATENT_REFINE_DEPTH_WEIGHT_POWER}"
        --latent-refine-final-kl-weight "${DIFFUSIVE_LATENT_REFINE_FINAL_KL_WEIGHT}"
      )
      ;;
    dual_path_reverse_diffusive_joint_readout|diffusive_reversed_hybrid_3to1_joint_readout)
      think_backbone="trm_official"
      think_structure="trm_dual_z_diffusive_reversed_hybrid_3to1_joint_readout"
      seed_offset=700
      latent_refine_args+=(
        --latent-refine-loss-weight "${DIFFUSIVE_LATENT_REFINE_LOSS_WEIGHT}"
        --latent-refine-min-depth 1
        --latent-refine-noise-std "${DIFFUSIVE_LATENT_REFINE_NOISE_STD}"
        --latent-refine-depth-weight-power "${DIFFUSIVE_LATENT_REFINE_DEPTH_WEIGHT_POWER}"
        --latent-refine-final-kl-weight "${DIFFUSIVE_LATENT_REFINE_FINAL_KL_WEIGHT}"
      )
      ;;
    dual_path_reverse|reversed_hybrid_3to1)
      think_backbone="trm_official"
      think_structure="trm_dual_z_reversed_hybrid_3to1"
      seed_offset=600
      ;;
    *)
      echo "Unsupported candidate=${candidate}; expected official, dual_path_reverse, dual_path_reverse_diffusive, dual_path_reverse_diffusive_joint_readout, qwen35_3to1, diffusive_trm, diffusive_gated_delta, diffusive_mamba3, diffusive_reversed_hybrid_3to1, diffusive_reversed_hybrid_3to1_joint_readout, or reversed_hybrid_3to1" >&2
      exit 2
      ;;
  esac

  local out_dir="${OUT_ROOT}/len${program_len}_${candidate}"
  mkdir -p "${out_dir}"
  if [[ "${SKIP_EXISTING_CANDIDATES}" == "1" && -f "${out_dir}/report.json" ]]; then
    echo "=== length gate: len=${program_len} candidate=${candidate} existing report reused ==="
    echo "0" > "${out_dir}/exit_code.txt"
    LAST_OUT_DIR="${out_dir}"
    return 0
  fi
  local seed=$((SEED_BASE + seed_offset + program_len))
  local strict_args=()
  if [[ "${STRICT_BACKENDS}" == "1" ]]; then
    strict_args+=(--strict-backends)
  fi
  local resume_args=()
  if [[ -n "${RESUME_FROM}" ]]; then
    resume_args+=(--resume-from "${RESUME_FROM}")
    if [[ "${RESUME_ALLOW_MISSING}" == "1" ]]; then
      resume_args+=(--resume-allow-missing)
    fi
  fi
  local adaptive_halt_args=()
  if [[ "${ADAPTIVE_HALT_EVAL}" == "1" ]]; then
    adaptive_halt_args+=(
      --adaptive-halt-eval
      --halt-min-steps 1
      --adaptive-halt-loss-weight "${ADAPTIVE_HALT_LOSS_WEIGHT}"
      --adaptive-halt-target-mode active_len
      --adaptive-halt-active-len-target first_step
      --accept-max-adaptive-halt-exact-drop 1.0
      --accept-max-mean-halt-steps "${eval_think_steps}"
      --accept-min-halted-fraction 0.0
    )
  fi

  echo "=== length gate: len=${program_len} candidate=${candidate} think_backbone=${think_backbone} think_structure=${think_structure} ==="
  set +e
  PYTHONPATH="${PYTHONPATH_VALUE}" "${PYTHON_BIN}" \
    scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py \
    --out-dir "${out_dir}" \
    "${resume_args[@]}" \
    --target-level "dual-z TRM official vs Qwen3.5 3:1 vs Diffusive-TRM length scaling gate" \
    --steps "${STEPS}" \
    --train-cases "${TRAIN_CASES}" \
    --eval-cases "${EVAL_CASES}" \
    --task-families "${TASK_FAMILIES}" \
    --eval-task-families "${EVAL_TASK_FAMILIES}" \
    --tokenizer-mode "${TOKENIZER_MODE}" \
    --number-tokenizer-max-value "${NUMBER_TOKENIZER_MAX_VALUE}" \
    --program-len "${program_len}" \
    --modulus 32 \
    --d-model "${D_MODEL}" \
    --n-heads "${N_HEADS}" \
    --n-kv-heads "${N_KV_HEADS}" \
    --d-ff "${D_FF}" \
    --backbone trm_official \
    --encode-backbone mha_etd \
    --think-backbone "${think_backbone}" \
    --decode-backbone mha_etd \
    --think-structure "${think_structure}" \
    --trm-l-cycles "${TRM_L_CYCLES}" \
    --halt-pooling dedicated \
    --delta-backend "${DELTA_BACKEND}" \
    "${strict_args[@]}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --lr-schedule "${LR_SCHEDULE}" \
    --lr-warmup-steps "${LR_WARMUP_STEPS}" \
    --lr-min-ratio "${LR_MIN_RATIO}" \
    --weight-decay 0.01 \
    --grad-clip 1.0 \
    --family-dro-loss-weight "${FAMILY_DRO_LOSS_WEIGHT}" \
    --family-dro-temperature "${FAMILY_DRO_TEMPERATURE}" \
    --train-think-steps "${train_think_steps}" \
    --eval-think-steps "${eval_think_steps}" \
    "${adaptive_halt_args[@]}" \
    --depth-intermediate-loss-weight "${DEPTH_INTERMEDIATE_LOSS_WEIGHT}" \
    --depth-intermediate-weight-power "${DEPTH_INTERMEDIATE_WEIGHT_POWER}" \
    --halt-depth-final-loss-weight "${HALT_DEPTH_FINAL_LOSS_WEIGHT}" \
    --operation-counterfactual-loss-weight "${OPERATION_COUNTERFACTUAL_LOSS_WEIGHT}" \
    --operation-counterfactual-margin "${OPERATION_COUNTERFACTUAL_MARGIN}" \
    --operation-counterfactual-max-cases "${OPERATION_COUNTERFACTUAL_MAX_CASES}" \
    --operation-counterfactual-every "${OPERATION_COUNTERFACTUAL_EVERY}" \
    --operation-counterfactual-warmup-steps "${OPERATION_COUNTERFACTUAL_WARMUP_STEPS}" \
    --operation-counterfactual-end-step "${OPERATION_COUNTERFACTUAL_END_STEP}" \
    --operation-counterfactual-active-len-min "${OPERATION_COUNTERFACTUAL_ACTIVE_LEN_MIN}" \
    --operation-counterfactual-active-len-max "${OPERATION_COUNTERFACTUAL_ACTIVE_LEN_MAX}" \
    --depth-counterfactual-loss-weight "${DEPTH_COUNTERFACTUAL_LOSS_WEIGHT}" \
    --depth-counterfactual-margin "${DEPTH_COUNTERFACTUAL_MARGIN}" \
    --depth-counterfactual-think-steps "${DEPTH_COUNTERFACTUAL_THINK_STEPS}" \
    --depth-counterfactual-every "${DEPTH_COUNTERFACTUAL_EVERY}" \
    --state-reset-counterfactual-loss-weight "${STATE_RESET_COUNTERFACTUAL_LOSS_WEIGHT}" \
    --state-reset-counterfactual-margin "${STATE_RESET_COUNTERFACTUAL_MARGIN}" \
    --state-reset-counterfactual-every "${STATE_RESET_COUNTERFACTUAL_EVERY}" \
    --retention-reference-checkpoint "${RETENTION_REFERENCE_CHECKPOINT}" \
    --retention-kl-loss-weight "${RETENTION_KL_LOSS_WEIGHT}" \
    --retention-active-len-min "${RETENTION_ACTIVE_LEN_MIN}" \
    --retention-active-len-max "${RETENTION_ACTIVE_LEN_MAX}" \
    --retention-max-cases "${RETENTION_MAX_CASES}" \
    --retention-every "${RETENTION_EVERY}" \
    --retention-temperature "${RETENTION_TEMPERATURE}" \
    --answer-space-ranking-loss-weight "${ANSWER_SPACE_RANKING_LOSS_WEIGHT}" \
    --answer-space-ranking-max-cases "${ANSWER_SPACE_RANKING_MAX_CASES}" \
    --answer-space-ranking-every "${ANSWER_SPACE_RANKING_EVERY}" \
    --answer-space-ranking-temperature "${ANSWER_SPACE_RANKING_TEMPERATURE}" \
    --active-len-replay-loss-weight "${ACTIVE_LEN_REPLAY_LOSS_WEIGHT}" \
    --active-len-replay-min "${ACTIVE_LEN_REPLAY_MIN}" \
    --active-len-replay-max "${ACTIVE_LEN_REPLAY_MAX}" \
    --active-len-replay-max-cases "${ACTIVE_LEN_REPLAY_MAX_CASES}" \
    --active-len-replay-every "${ACTIVE_LEN_REPLAY_EVERY}" \
    --state-trace-anti-collapse-loss-weight "${STATE_TRACE_ANTI_COLLAPSE_LOSS_WEIGHT}" \
    --state-trace-min-variance "${STATE_TRACE_MIN_VARIANCE}" \
    --state-trace-min-delta-norm "${STATE_TRACE_MIN_DELTA_NORM}" \
    --core-step-codec-loss-weight "${CORE_STEP_CODEC_LOSS_WEIGHT}" \
    --core-step-codec-state-source "${CORE_STEP_CODEC_STATE_SOURCE}" \
    --core-step-codec-pooling "${CORE_STEP_CODEC_POOLING}" \
    "${latent_refine_args[@]}" \
    --active-len-batch-cycle \
    --eval-active-len-cycle \
    --active-len-cycle-min "${TRAIN_ACTIVE_LEN_MIN}" \
    --active-len-cycle-max "${program_len}" \
    --train-active-len-cycle-min "${TRAIN_ACTIVE_LEN_MIN}" \
    --train-active-len-cycle-max "${train_active_len_max}" \
    --seed "${seed}" \
    --eval-seed "${EVAL_SEED}" \
    --device "${DEVICE}" \
    --log-every "${LOG_EVERY}" \
    --max-examples 4 \
    --accept-min-exact "${ACCEPT_MIN_EXACT}" \
    --accept-min-depth-gain "${ACCEPT_MIN_DEPTH_GAIN}" \
    --accept-min-ablation-drop "${ACCEPT_MIN_ABLATION_DROP}" \
    --accept-min-family-exact 0.0
  local status="$?"
  set -e
  echo "${status}" > "${out_dir}/exit_code.txt"
  LAST_OUT_DIR="${out_dir}"
}

is_active_candidate() {
  case "$1" in
    dual_path_reverse|reversed_hybrid_3to1)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

record_fail_fast_if_rejected() {
  local program_len="$1"
  local candidate="$2"
  local report_path="${LAST_OUT_DIR}/report.json"
  local stop_path="${OUT_ROOT}/fail_fast_stop.json"

  PYTHONPATH="${PYTHONPATH_VALUE}" "${PYTHON_BIN}" - \
    "${program_len}" "${candidate}" "${report_path}" "${stop_path}" <<'PY'
import json
import sys
from pathlib import Path

program_len = int(sys.argv[1])
candidate = sys.argv[2]
report_path = Path(sys.argv[3])
stop_path = Path(sys.argv[4])

reasons = []
if not report_path.exists():
    reasons.append("missing_report")
    row = {}
else:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    train = report.get("train", {})
    key = f"think{train.get('eval_think_steps')}"
    metric = report.get("eval_metrics", {}).get(key, {})
    decisive = report.get("decisive_metrics", {})
    family_exact = [
        value.get("generation_exact")
        for value in metric.get("by_family", {}).values()
        if value.get("generation_exact") is not None
    ]
    target_active_len = str(program_len)
    target_active_len_exact = (
        metric.get("by_active_len", {})
        .get(target_active_len, {})
        .get("generation_exact")
    )
    min_family_exact = min(family_exact) if family_exact else 0.0
    row = {
        "full_generation_exact": metric.get("generation_exact"),
        "full_minus_think0": decisive.get("full_minus_think0"),
        "full_minus_worst_ablation": decisive.get("full_minus_worst_ablation"),
        "min_family_generation_exact": min_family_exact,
        "target_active_len_generation_exact": target_active_len_exact,
    }
    if row["full_generation_exact"] is None or row["full_generation_exact"] <= 0.0:
        reasons.append("full_generation_exact<=0")
    if row["full_minus_think0"] is None or row["full_minus_think0"] <= 0.0:
        reasons.append("full_minus_think0<=0")
    if row["full_minus_worst_ablation"] is None or row["full_minus_worst_ablation"] <= 0.0:
        reasons.append("full_minus_worst_ablation<=0")
    if row["min_family_generation_exact"] <= 0.0:
        reasons.append("min_family_generation_exact<=0")
    if (
        row["target_active_len_generation_exact"] is None
        or row["target_active_len_generation_exact"] <= 0.0
    ):
        reasons.append("target_active_len_generation_exact<=0")

if not reasons:
    raise SystemExit(0)

payload = {
    "program_len": program_len,
    "candidate": candidate,
    "report_path": str(report_path),
    "reject_reasons": reasons,
    "metrics": row,
}
stop_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"fail_fast_stop": payload}, ensure_ascii=False))
raise SystemExit(1)
PY
}

stop_requested=0
for program_len in "${length_values[@]}"; do
  for candidate in "${candidate_values[@]}"; do
    run_candidate "${program_len}" "${candidate}"
    if [[ "${FAIL_FAST_ON_ACTIVE_REJECT}" == "1" ]] && is_active_candidate "${candidate}"; then
      if ! record_fail_fast_if_rejected "${program_len}" "${candidate}"; then
        stop_requested=1
        break
      fi
    fi
  done
  if [[ "${stop_requested}" == "1" ]]; then
    break
  fi
done

PYTHONPATH="${PYTHONPATH_VALUE}" "${PYTHON_BIN}" - "${OUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for report_path in sorted(root.glob("len*_*/report.json")):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    train = report.get("train", {})
    key = f"think{train.get('eval_think_steps')}"
    metric = report.get("eval_metrics", {}).get(key, {})
    decisive = report.get("decisive_metrics", {})
    full_exact = metric.get("generation_exact")
    think0_exact = report.get("eval_metrics", {}).get("think0", {}).get("generation_exact")
    depth_gain = decisive.get("full_minus_think0")
    ablation_margin = decisive.get("full_minus_worst_ablation")
    family_exact = {
        name: value.get("generation_exact")
        for name, value in metric.get("by_family", {}).items()
    }
    active_len_exact = {
        name: value.get("generation_exact")
        for name, value in metric.get("by_active_len", {}).items()
    }
    target_active_len_exact = active_len_exact.get(str(train.get("program_len")))
    min_family_exact = min(family_exact.values()) if family_exact else 0.0
    strict_reject_reasons = []
    if full_exact is None or full_exact <= 0.0:
        strict_reject_reasons.append("full_generation_exact<=0")
    if depth_gain is None or depth_gain <= 0.0:
        strict_reject_reasons.append("full_minus_think0<=0")
    if ablation_margin is None or ablation_margin <= 0.0:
        strict_reject_reasons.append("full_minus_worst_ablation<=0")
    if min_family_exact <= 0.0:
        strict_reject_reasons.append("min_family_generation_exact<=0")
    if target_active_len_exact is None or target_active_len_exact <= 0.0:
        strict_reject_reasons.append("target_active_len_generation_exact<=0")
    strict_accepted = not strict_reject_reasons
    candidate = report_path.parent.name.split("_", 1)[1]
    rows.append(
        {
            "run": report_path.parent.name,
            "program_len": train.get("program_len"),
            "candidate": candidate,
            "think_backbone": train.get("think_backbone"),
            "full_generation_exact": full_exact,
            "think0_generation_exact": think0_exact,
            "full_minus_think0": depth_gain,
            "full_minus_worst_ablation": ablation_margin,
            "min_family_generation_exact": min_family_exact,
            "target_active_len_generation_exact": target_active_len_exact,
            "by_family": family_exact,
            "by_active_len": active_len_exact,
            "raw_report_accepted": report.get("accepted"),
            "raw_report_decision": report.get("decision"),
            "accepted": strict_accepted,
            "decision": "accepted_length_gate" if strict_accepted else "rejected_length_gate",
            "reject_reasons": strict_reject_reasons,
        }
    )
fail_fast_path = root / "fail_fast_stop.json"
fail_fast_stop = None
if fail_fast_path.exists():
    fail_fast_stop = json.loads(fail_fast_path.read_text(encoding="utf-8"))
summary = {"out_root": str(root), "rows": rows, "fail_fast_stop": fail_fast_stop}
(root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
