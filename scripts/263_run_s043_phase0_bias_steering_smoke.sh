#!/usr/bin/env bash
set -euo pipefail

# S043 Phase 0 Smoke: Minimal donor-preserving residual steering bias
# Goal: Test whether a tiny number of parameters (single vocab-sized bias vector)
# can provide first-token / recovery signal on the donor mouth WITHOUT destroying fluency.
#
# Strict precautions:
# - Bias starts at (or very near) zero effect.
# - donor_correct_preservation_weight should be strong when training.
# - Always compare against pure donor_only baseline on the same cases.
# - Monitor repetition rate and donor-correct perplexity.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="${CONFIG:-configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_loop_joint_decoder_s040.yaml}"
CHECKPOINT="${CHECKPOINT:-/mnt/nvme0n1p2/qtrm-runs/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_loop_joint_decoder_s040_from_selfrollout/last.pt}"
CASES="${CASES:-data/eval/pure_recursive_reasoning_heldout_72.jsonl}"
OUT_DIR="${OUT_DIR:-reports/s043_phase0_bias_steering}"
MAX_CASES="${MAX_CASES:-8}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "${OUT_DIR}"

# Baseline: pure donor (no bias, no QTRM)
BASELINE_OUT="${OUT_DIR}/phase0_donor_only_smoke${MAX_CASES}.jsonl"
BASELINE_SUM="${OUT_DIR}/phase0_donor_only_smoke${MAX_CASES}.summary.md"

echo "=== S043 Phase 0: Donor-only baseline ==="
PYTHONPATH=. python3 scripts/192_eval_raw_intelligence.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --cases "${CASES}" \
  --out "${BASELINE_OUT}" \
  --device "${DEVICE}" \
  --max-cases "${MAX_CASES}" \
  --mode "donor_only_no_evidence" \
  --no-repeat-ngram-size 2 \
  2>&1 | tee "${OUT_DIR}/phase0_donor_only.log"

# Phase 0 with bias enabled (but still very small scale)
BIAS_OUT="${OUT_DIR}/phase0_bias_steering_smoke${MAX_CASES}.jsonl"
BIAS_SUM="${OUT_DIR}/phase0_bias_steering_smoke${MAX_CASES}.summary.md"

echo ""
echo "=== S043 Phase 0: With minimal residual steering bias (adaptive gate on) ==="
PYTHONPATH=. python3 scripts/192_eval_raw_intelligence.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --cases "${CASES}" \
  --out "${BIAS_OUT}" \
  --device "${DEVICE}" \
  --max-cases "${MAX_CASES}" \
  --donor-residual-steering-bias \
  --donor-residual-steering-bias-init-scale 0.01 \
  --donor-qtrm-conflict-gate \
  --donor-qtrm-conflict-gate-mode adaptive_margin \
  --donor-qtrm-conflict-qtrm-scale 0.25 \
  --mode "qtrm_core_steps_2_qtrm_scale_0p25_donor_scale_1_no_evidence" \
  --no-repeat-ngram-size 2 \
  2>&1 | tee "${OUT_DIR}/phase0_bias_steering.log"

echo ""
echo "S043 Phase 0 smoke complete."
echo ""
echo "=== MANDATORY COMPARISON ==="
echo "Baseline (donor-only): ${BASELINE_OUT}"
echo "With bias:             ${BIAS_OUT}"
echo ""
echo "Key metrics to compare manually or with:"
echo "  python -c 'import json; [print(r[\"id\"], r.get(\"hit\"), r.get(\"first_token_win_rate\")) for r in json.load(open(\"${BIAS_OUT}\"))]'"
echo ""
echo "When the bias run finishes, look for the 'S043 PHASE 0 DIAGNOSTIC BLOCK' in the output."
echo "It will automatically show first_token_win_rate, donor_correct signals, etc."
echo ""
echo "CRITICAL (S043 precautions - do not skip):"
echo "  1. first_token_win_rate improvement vs donor-only?"
echo "  2. Exact match improvement (especially on donor-wrong rows)?"
echo "  3. Repetition/collapse rate on donor-correct cases?"
echo "  4. Any regression in donor-correct fluency?"
echo ""
echo "If #4 shows regression → immediately stop and increase preservation weight."
echo ""
echo "Full execution guide + interpretation is here:"
echo "  docs/wiki/decisions/2026-05-30-s041-donor-preserving-freegen-smoke.md"
echo "  (search for 'Phase 0 - Recommended First Actual Smoke')"