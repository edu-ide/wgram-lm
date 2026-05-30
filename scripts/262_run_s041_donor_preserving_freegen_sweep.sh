#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="${CONFIG:-configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_loop_joint_decoder_s040.yaml}"
CHECKPOINT="${CHECKPOINT:-/mnt/nvme0n1p2/qtrm-runs/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_loop_joint_decoder_s040_from_selfrollout/last.pt}"
CASES="${CASES:-data/eval/pure_recursive_reasoning_heldout_72.jsonl}"
OUT_DIR="${OUT_DIR:-reports/s041_donor_preserving_freegen}"
MAX_CASES="${MAX_CASES:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
DEVICE="${DEVICE:-cuda}"
NO_REPEAT_NGRAM_SIZE="${NO_REPEAT_NGRAM_SIZE:-2}"
CONFLICT_QTRM_SCALE="${CONFLICT_QTRM_SCALE:-0.25}"

mkdir -p "${OUT_DIR}"

OUT_JSONL="${OUT_JSONL:-${OUT_DIR}/s041_conflict_gated_free_generation_smoke${MAX_CASES}.jsonl}"
OUT_SUMMARY_JSON="${OUT_SUMMARY_JSON:-${OUT_DIR}/s041_conflict_gated_free_generation_smoke${MAX_CASES}.summary.json}"
OUT_SUMMARY_MD="${OUT_SUMMARY_MD:-${OUT_DIR}/s041_conflict_gated_free_generation_smoke${MAX_CASES}.summary.md}"

MODES=(
  donor_only_no_evidence
  qtrm_core_off_no_evidence
  qtrm_core_steps_2_no_evidence
  qtrm_core_steps_4_no_evidence
  qtrm_core_steps_8_no_evidence
  qtrm_core_steps_2_qtrm_scale_0p25_donor_scale_1_no_evidence
  qtrm_core_steps_2_qtrm_scale_0p5_donor_scale_1_no_evidence
  qtrm_core_steps_2_qtrm_scale_1_donor_scale_1_no_evidence
  qtrm_core_steps_4_qtrm_scale_0p25_donor_scale_1_no_evidence
  qtrm_core_steps_4_qtrm_scale_0p5_donor_scale_1_no_evidence
  qtrm_core_steps_4_qtrm_scale_1_donor_scale_1_no_evidence
  qtrm_core_steps_8_qtrm_scale_0p25_donor_scale_1_no_evidence
  qtrm_core_steps_8_qtrm_scale_0p5_donor_scale_1_no_evidence
  qtrm_core_steps_8_qtrm_scale_1_donor_scale_1_no_evidence
)

MODE_ARGS=()
for mode in "${MODES[@]}"; do
  MODE_ARGS+=(--mode "${mode}")
done

PYTHONPATH=. python3 scripts/192_eval_raw_intelligence.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --cases "${CASES}" \
  --out "${OUT_JSONL}" \
  --device "${DEVICE}" \
  --scoring generation \
  --max-cases "${MAX_CASES}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --no-repeat-ngram-size "${NO_REPEAT_NGRAM_SIZE}" \
  --donor-qtrm-conflict-gate \
  --donor-qtrm-conflict-qtrm-scale "${CONFLICT_QTRM_SCALE}" \
  "${MODE_ARGS[@]}"

python3 scripts/analyze_s041_freegen_sweep.py \
  --input "${OUT_JSONL}" \
  --out-json "${OUT_SUMMARY_JSON}" \
  --out-md "${OUT_SUMMARY_MD}" \
  --title "S041 Donor-Preserving Conflict-Gated Free Generation Smoke${MAX_CASES}"

echo "S041 output: ${OUT_JSONL}"
echo "S041 summary: ${OUT_SUMMARY_MD}"
