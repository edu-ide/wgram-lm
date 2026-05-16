#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
HF_HOME="${HF_HOME:-/mnt/nvme1n1p2/hf-cache-qtrm}"
PYTHONPATH="${PYTHONPATH:-src}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-2B-Base}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1200}"
ROOT_OUT_DIR="${ROOT_OUT_DIR:-local_eval/qwen35_integrated_ssot_revalidation_20260516}"
SUMMARY_JSONL="${SUMMARY_JSONL:-${ROOT_OUT_DIR}/summary.jsonl}"

export HF_HOME PYTHONPATH

mkdir -p "${ROOT_OUT_DIR}"
: > "${SUMMARY_JSONL}"

run_eval() {
  local name="$1"
  local suite_jsonl="$2"
  local checkpoint="$3"
  local max_cases="$4"
  local core_insertion_mode="$5"
  local core_insert_after_layer="$6"
  local core_adapter_dim="$7"
  local core_delta_adapter_mode="$8"
  local residual_scale="$9"
  local h_cycles="${10}"
  local l_cycles="${11}"
  local outer_steps="${12}"
  local halt_enabled="${13}"
  local step_conditioning_enabled="${14}"
  local min_core_gain="${15}"

  local out_dir="${ROOT_OUT_DIR}/${name}"
  mkdir -p "${out_dir}"
  echo "=== SSOT revalidation: ${name} ==="
  echo "checkpoint=${checkpoint}"
  echo "suite=${suite_jsonl}"

  local exit_code=0
  set +e
  CHECKPOINT="${checkpoint}" \
    SUITE_JSONL="${suite_jsonl}" \
    OUT_DIR="${out_dir}" \
    MAX_CASES="${max_cases}" \
    MIN_CASES="${max_cases}" \
    DEVICE="${DEVICE}" \
    DTYPE="${DTYPE}" \
    MODEL_ID="${MODEL_ID}" \
    CORE_INSERTION_MODE="${core_insertion_mode}" \
    CORE_INSERT_AFTER_LAYER="${core_insert_after_layer}" \
    CORE_ADAPTER_DIM="${core_adapter_dim}" \
    CORE_DELTA_ADAPTER_MODE="${core_delta_adapter_mode}" \
    RESIDUAL_SCALE="${residual_scale}" \
    H_CYCLES="${h_cycles}" \
    L_CYCLES="${l_cycles}" \
    OUTER_STEPS="${outer_steps}" \
    CORE_CONVERGENCE_HALT_ENABLED="${halt_enabled}" \
    CORE_CONVERGENCE_HALT_THRESHOLD="${CORE_CONVERGENCE_HALT_THRESHOLD:-0.2}" \
    CORE_CONVERGENCE_HALT_MIN_OUTER="${CORE_CONVERGENCE_HALT_MIN_OUTER:-1}" \
    CORE_STEP_CONDITIONING_ENABLED="${step_conditioning_enabled}" \
    timeout "${TIMEOUT_SECONDS}s" bash scripts/390_run_qwen35_integrated_m4_public_mcq.sh
  exit_code=$?
  set -e

  if [[ -f "${out_dir}/report.json" ]]; then
    jq -c \
      --arg name "${name}" \
      --argjson exit_code "${exit_code}" \
      --argjson strict_min_core_gain "${min_core_gain}" \
      '{
        name: $name,
        exit_code: $exit_code,
        decision,
        accepted,
        checkpoint,
        suite_jsonl,
        scorer,
        base_hits: .base_metrics.hits,
        core_hits: .core_metrics.hits,
        cases: .core_metrics.cases,
        core_gain_over_base,
        strict_min_core_gain: $strict_min_core_gain,
        strict_accepted: ((.core_gain_over_base // -999) >= $strict_min_core_gain),
        report_path: input_filename
      }' "${out_dir}/report.json" | tee -a "${SUMMARY_JSONL}"
  else
    jq -nc \
      --arg name "${name}" \
      --arg checkpoint "${checkpoint}" \
      --arg suite_jsonl "${suite_jsonl}" \
      --argjson exit_code "${exit_code}" \
      '{name:$name, checkpoint:$checkpoint, suite_jsonl:$suite_jsonl, exit_code:$exit_code, decision:"missing_report", accepted:false}' \
      | tee -a "${SUMMARY_JSONL}"
  fi
}

# Canonical mid-layer QTRM candidate: H=3/L=6, adapter-only, mid-layer suffix.
run_eval \
  "midlayer_external64" \
  "local_eval/m7_public_reasoning_suite/external_mcq_validation_pool_20260516.jsonl" \
  "local_eval/qwen35_integrated_midlayer_suffix_adapteronly_coretrain_langanchor_s160_20260516/last_core.pt" \
  "64" \
  "mid_layer_suffix" "11" "512" "adapter_only" "1.0" "3" "6" "3" "1" "1" "0.01"

run_eval \
  "midlayer_mmlupro64" \
  "local_eval/m7_public_reasoning_suite/mmlu_pro_validation_64.jsonl" \
  "local_eval/qwen35_integrated_midlayer_suffix_adapteronly_coretrain_langanchor_s160_20260516/last_core.pt" \
  "64" \
  "mid_layer_suffix" "11" "512" "adapter_only" "1.0" "3" "6" "3" "1" "1" "0.01"

run_eval \
  "optiononly_mmlupro64" \
  "local_eval/m7_public_reasoning_suite/mmlu_pro_validation_64.jsonl" \
  "local_eval/qwen35_integrated_midlayer_suffix_optiononly_mmlupro_s80_20260516/last_core.pt" \
  "64" \
  "mid_layer_suffix" "11" "512" "adapter_only" "1.0" "3" "6" "3" "1" "1" "0.01"

# Older public-MCQ healing checkpoint: final-residual default core settings.
# Earlier 391 training used padded-batch last-token logic, so these are
# revalidation targets, not canonical promotions.
run_eval \
  "public_coreonly_mmlu256" \
  "local_eval/m7_public_reasoning_suite/mmlu_pro_test_balanced_256.jsonl" \
  "local_eval/qwen35_integrated_public_mcq_healing_coreonly_val64_to_test256_s120_20260516/last_core.pt" \
  "256" \
  "final_residual" "-1" "128" "add" "0.05" "1" "1" "1" "0" "0" "0.01"

run_eval \
  "l23open_seed20260520_mmlu256" \
  "local_eval/m7_public_reasoning_suite/mmlu_pro_test_balanced_256.jsonl" \
  "local_eval/qwen35_integrated_public_mcq_healing_l23open_qwenlr0_seed20260520_s120_20260516/last_core.pt" \
  "256" \
  "final_residual" "-1" "128" "add" "0.05" "1" "1" "1" "0" "0" "0.01"

if [[ "${RUN_LARGE:-0}" == "1" ]]; then
  run_eval \
    "public_coreonly_mmlu512" \
    "local_eval/m7_public_reasoning_suite/mmlu_pro_test_balanced_512.jsonl" \
    "local_eval/qwen35_integrated_public_mcq_healing_coreonly_val64_to_test256_s120_20260516/last_core.pt" \
    "512" \
    "final_residual" "-1" "128" "add" "0.05" "1" "1" "1" "0" "0" "0.01"

  run_eval \
    "public_coreonly_mmlu512_resid0p06" \
    "local_eval/m7_public_reasoning_suite/mmlu_pro_test_balanced_512.jsonl" \
    "local_eval/qwen35_integrated_public_mcq_healing_coreonly_val64_to_test256_s120_20260516/last_core.pt" \
    "512" \
    "final_residual" "-1" "128" "add" "0.06" "1" "1" "1" "0" "0" "0.01"
fi

echo "summary=${SUMMARY_JSONL}"
