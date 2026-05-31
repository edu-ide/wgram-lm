#!/usr/bin/env bash
set -euo pipefail

QTRM_REPORT="${QTRM_REPORT:-local_eval/research_gate_runner/qtrm_native_l5_multifamily_standard/report.json}"
QWEN36_BASELINE_REPORT="${QWEN36_BASELINE_REPORT:-local_eval/m6_qwen36_scoped_baseline/report.json}"
M6_REPORT="${M6_REPORT:-local_eval/m6_scoped_raw_reasoning_manifest/report.json}"
M6_MD="${M6_MD:-local_eval/m6_scoped_raw_reasoning_manifest/report.md}"
STATUS_JSON="${STATUS_JSON:-local_eval/qtrm_native_27b_milestone_status/report.json}"
STATUS_MD="${STATUS_MD:-local_eval/qtrm_native_27b_milestone_status/report.md}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

BRIDGE_REPORT="${BRIDGE_REPORT:-local_eval/qwen_backbone_wgram_qwen_transition_hardv1_adapteronly_stepcond_ad128_s400_checksum_repair_from_select_20260515/report.json}"
BRIDGE_STABILITY_REPORT="${BRIDGE_STABILITY_REPORT:-local_eval/qwen_backbone_wgram_qwen_transition_hardv1_adapteronly_stepcond_ad128_checksum_repair_stability_20260515/report.json}"
NATIVE_REPORT="${NATIVE_REPORT:-local_eval/qtrm_native_language_bootstrap_bilingual_bpe16k_d192_repairv3_s4000_20260515/report.json}"
NATIVE_CORE_REPORT="${NATIVE_CORE_REPORT:-local_eval/qtrm_native_language_bootstrap_bilingual_bpe16k_d192_repairv3_core_ablation_20260515/report.json}"
EVAL_MANIFEST="${EVAL_MANIFEST:-local_eval/qwen36_public_target_manifest/report.json}"

PYTHONPATH="${PYTHONPATH:-src}" "${PYTHON_BIN}" scripts/376_build_m6_scoped_raw_reasoning_manifest.py \
  --qtrm-report "${QTRM_REPORT}" \
  --qwen36-baseline-report "${QWEN36_BASELINE_REPORT}" \
  --out-json "${M6_REPORT}" \
  --out-md "${M6_MD}"

PYTHONPATH="${PYTHONPATH:-src}" "${PYTHON_BIN}" scripts/372_qtrm_native_27b_milestone_status.py \
  --bridge-report "${BRIDGE_REPORT}" \
  --bridge-stability-report "${BRIDGE_STABILITY_REPORT}" \
  --native-report "${NATIVE_REPORT}" \
  --native-core-report "${NATIVE_CORE_REPORT}" \
  --core-reasoning-report "${QTRM_REPORT}" \
  --eval-manifest "${EVAL_MANIFEST}" \
  --m6-report "${M6_REPORT}" \
  --out-json "${STATUS_JSON}" \
  --out-md "${STATUS_MD}"
