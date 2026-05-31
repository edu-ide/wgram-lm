#!/usr/bin/env bash
set -euo pipefail
# S043 autonomous post-training eval for denoise recovery.
# Always runs donor-only baseline first, then steered with bias+gate+trained residual.
# Compares exact_match (free-gen) + first_token stats on the same heldout cases.
# Usage:
#   CHECKPOINT=runs/s043_denoise_recovery_real1/last.pt MAX_CASES=24 \
#   scripts/266_run_s043_denoise_recovery_eval.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="${CONFIG:-configs/s043_denoise_recovery_real1.yaml}"
CHECKPOINT="${CHECKPOINT:-runs/s043_denoise_recovery_real1/last.pt}"
CASES="${CASES:-data/eval/pure_recursive_reasoning_heldout_72.jsonl}"
OUT_DIR="${OUT_DIR:-reports/s043_denoise_recovery_real1}"
MAX_CASES="${MAX_CASES:-24}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "${OUT_DIR}"

echo "=== S043 Denoise Recovery Eval: Donor-only baseline (free-gen exact match) ==="
BASELINE_OUT="${OUT_DIR}/donor_only_${MAX_CASES}.jsonl"
PYTHONPATH=. python3 scripts/192_eval_raw_intelligence.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --cases "${CASES}" \
  --out "${BASELINE_OUT}" \
  --device "${DEVICE}" \
  --max-cases "${MAX_CASES}" \
  --mode "donor_only_no_evidence" \
  --no-repeat-ngram-size 2 \
  2>&1 | tee "${OUT_DIR}/donor_only.log"

echo ""
echo "=== S043 Denoise Recovery Eval: Steered (bias + adaptive gate + trained QTRM) ==="
STEERED_OUT="${OUT_DIR}/steered_recovery_${MAX_CASES}.jsonl"
PYTHONPATH=. python3 scripts/192_eval_raw_intelligence.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --cases "${CASES}" \
  --out "${STEERED_OUT}" \
  --device "${DEVICE}" \
  --max-cases "${MAX_CASES}" \
  --donor-residual-steering-bias \
  --donor-residual-steering-bias-init-scale 0.012 \
  --donor-qtrm-conflict-gate \
  --donor-qtrm-conflict-gate-mode adaptive_margin \
  --donor-qtrm-conflict-qtrm-scale 0.30 \
  --mode "qtrm_core_steps_2_qtrm_scale_0p25_donor_scale_1_no_evidence" \
  --no-repeat-ngram-size 2 \
  2>&1 | tee "${OUT_DIR}/steered.log"

echo ""
echo "=== MANDATORY FREE-GEN COMPARISON ==="
echo "Baseline (donor only): ${BASELINE_OUT}"
echo "Steered recovery:      ${STEERED_OUT}"
echo ""
echo "Quick numbers:"
python3 -c '
import json, sys
def count_hits(p):
    n = h = 0
    with open(p) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                n += 1
                if r.get("hit") or r.get("exact_match"):
                    h += 1
    return h, n, h / max(1, n)
b = count_hits(sys.argv[1])
s = count_hits(sys.argv[2])
print(f"exact_match baseline: {b[0]}/{b[1]} = {b[2]:.3f}")
print(f"exact_match steered : {s[0]}/{s[1]} = {s[2]:.3f}")
print(f"delta: {s[2]-b[2]:+.3f}")
' "${BASELINE_OUT}" "${STEERED_OUT}"

echo ""
echo "Full analysis + first_token / donor_preservation from the 192 diagnostic blocks above."
echo "If steered exact_match > baseline (and no repetition spike on donor-correct) → first real free-gen win."
echo "See wiki decision doc for contract."