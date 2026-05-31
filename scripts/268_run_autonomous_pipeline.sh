#!/usr/bin/env bash
set -euo pipefail

# Autonomous Pipeline Orchestrator
# Coordinates:
# 1. Wait for checkpoint upload (scp) to finish
# 2. Wait for remote harvester to finish
# 3. Evaluate the recovery checkpoint under scale=2.0 on DGX
# 4. Merge datasets and launch Mixed Denoise Recovery Training in the background on DGX

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "=== Autonomous Pipeline Orchestrator Launched ==="

# 1. Wait for local scp to finish
echo "[1/4] Waiting for local checkpoint scp transfer to complete..."
while ps aux | grep -v grep | grep -q "scp runs/s043_denoise_recovery_real1/last.pt"; do
  sleep 10
done
echo "Checkpoint file last.pt successfully uploaded to DGX!"

# 2. Wait for remote harvester to finish
echo "[2/4] Waiting for remote failure prefix harvester to complete on DGX..."
while ssh dgx "ps aux | grep -v grep | grep -q 'generate_real_bad_prefixes.py'"; do
  sleep 10
done
echo "Genuine failure prefixes successfully harvested on DGX!"

# 3. Evaluate the recovery checkpoint under scale=2.0 on DGX
echo "[3/4] Running steered recovery evaluation under scale=2.0 on DGX..."
ssh dgx "cd /mnt/data4tb/wgram-lm && PYTHONPATH=.:src /mnt/data4tb/venv_sglang_pr23000/bin/python scripts/192_eval_raw_intelligence.py --config configs/s043_denoise_recovery_real1.yaml --checkpoint runs/s043_denoise_recovery_real1/last.pt --cases data/eval/pure_recursive_reasoning_heldout_72.jsonl --out reports/s043_denoise_recovery_real1/test_recovery_scale2.jsonl --device cuda --max-cases 40 --donor-residual-steering-bias --donor-residual-steering-bias-init-scale 0.012 --donor-qtrm-conflict-gate --donor-qtrm-conflict-gate-mode adaptive_margin --donor-qtrm-conflict-qtrm-scale 0.25 --mode qtrm_core_steps_2_qtrm_scale_2_donor_scale_1_no_evidence --no-repeat-ngram-size 2 --scoring generation" > runs/dgx_eval_recovery_scale2.log 2>&1

echo "Steered recovery evaluation completed! Quick summary:"
ssh dgx "python3 -c \"import json; lines = [json.loads(l) for l in open('/mnt/data4tb/wgram-lm/reports/s043_denoise_recovery_real1/test_recovery_scale2.jsonl') if l.strip()]; hits = [l.get('hit', False) for l in lines]; ems = [l.get('exact_match', False) for l in lines]; print(f'Steered Recovery - Count: {len(lines)}, Hits: {sum(hits)}/{len(lines)} ({sum(hits)/len(lines):.3f}), EM: {sum(ems)}/{len(lines)}')\""

# 4. Concatenate and launch Mixed Denoise Recovery Training on DGX
echo "[4/4] Merging datasets and launching Mixed Denoise Recovery Training on DGX..."
ssh dgx "bash /mnt/data4tb/wgram-lm/scripts/267_run_s043_mixed_denoise_train.sh" > runs/dgx_mixed_train.log 2>&1 &

echo "Mixed Denoise Recovery Training launched in the background on DGX!"
echo "Check runs/dgx_mixed_train.log for progress."
echo "=== Autonomous Pipeline Orchestrator finished successfully! ==="
