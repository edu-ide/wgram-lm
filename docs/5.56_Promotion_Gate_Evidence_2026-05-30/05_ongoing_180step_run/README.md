# 05. Ongoing 180-step Real 642 Gold Run (Option A)

**Run Directory** (as of 2026-05-30 12:01):
`local_556_real642_long_180step_20260527_1201`

**Command Used**:
```bash
PYTHONPATH=. .venv/bin/python -u scripts/train_556_full_curriculum_minimal.py \
  --steps 180 \
  --d_model 64 \
  --batch 4 \
  --enable_stochastic_breadth \
  --log_every 10 \
  --save_dir local_556_real642_long_180step_20260527_1201 \
  --gold_path local_eval/642_adaptive_fine_tuned_200step/adaptive_phase2_checkpoint.pt
```

**Current Status at Package Creation Time**:
- Run successfully launched in background.
- Gold handling hardening is active (synthetic fallback with clear logging).
- Early logs showed strong stochastic diversity (~6.0).

---

## When the Run Finishes — Exact Steps

1. Confirm completion:
   ```bash
   tail -5 local_556_real642_long_180step_20260527_1201/full_output.log
   ```

2. Generate full analysis:
   ```bash
   python scripts/analyze_556_curriculum_metrics.py \
       local_556_real642_long_180step_20260527_1201/metrics.json \
       --output 05_ongoing_180step_run/180step_real642_final_analysis.md
   ```

3. Copy the final artifacts into this package:
   ```bash
   mkdir -p 05_ongoing_180step_run/final_artifacts
   cp local_556_real642_long_180step_20260527_1201/{metrics.json,best.pt,last.pt,full_output.log} \
      05_ongoing_180step_run/final_artifacts/
   cp 05_ongoing_180step_run/180step_real642_final_analysis.md \
      05_ongoing_180step_run/
   ```

4. Update the top-level `README.md` with the final numbers (decay range, stochastic diversity, any notable observations vs the 50-step run).

5. Decide whether to move to full ablation matrix or declare current evidence sufficient for Promotion Gate discussion.

---

**Helper Script** (run this when the 180-step run is done):

```bash
bash docs/5.56_Promotion_Gate_Evidence_2026-05-30/scripts/finalize_180step_run.sh
```

(This script will be created in the next step of package preparation.)
