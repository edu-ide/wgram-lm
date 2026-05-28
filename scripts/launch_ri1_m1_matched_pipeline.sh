#!/bin/bash
# =============================================================================
# RI-1 M1 Matched Pipeline Launcher (research-driven-architecture-debugging)
# =============================================================================
# Purpose: One-command execution of
#   1. M1 Variable Depth Training continuation (Huginn-style sampling)
#      on top of properly ported 3-track substrate (Workspaces + Attractor + Provenance)
#   2. Immediate Principle Gate + strict-B depth sweep (1/4/8) on the resulting artifact
#   3. Honest conditions-matched + Triple-Track (A/B/C) reporting
#
# Usage (on a machine with the full env + GPU):
#   bash scripts/launch_ri1_m1_matched_pipeline.sh \
#        --resume_from experiments/matched_port_evaluation_a9617cd8/continued_longer_50.pt \
#        --steps 25 \
#        --out_dir checkpoints/hybrid_ri4_ri1_m1_$(date +%Y%m%d)
#
# This script enforces:
# - proper three tracks (default)
# - RI-1 M1 variable depth sampling (randint + lognormal path)
# - loss curve + sampled depth logging (C-track)
# - automatic principle gate call after training
# - rich notes for matched conditions
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

RESUME_CKPT="experiments/matched_port_evaluation_a9617cd8/continued_longer_50.pt"
STEPS=25
OUT_DIR=""
D_MODEL=128
BATCH=1
RI1_DEPTH_MEAN=3
RI1_DEPTH_MAX=8
ENABLE_STOCHASTIC=true

while [[ $# -gt 0 ]]; do
  case $1 in
    --resume_from) RESUME_CKPT="$2"; shift 2 ;;
    --steps)       STEPS="$2"; shift 2 ;;
    --out_dir)     OUT_DIR="$2"; shift 2 ;;
    --d_model)     D_MODEL="$2"; shift 2 ;;
    --batch)       BATCH="$2"; shift 2 ;;
    --ri1_depth_mean) RI1_DEPTH_MEAN="$2"; shift 2 ;;
    --ri1_depth_max)  RI1_DEPTH_MAX="$2"; shift 2 ;;
    --disable_stochastic) ENABLE_STOCHASTIC=false; shift 1 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="checkpoints/hybrid_ri4_ri1_m1_$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p "$OUT_DIR"

STOCH_FLAG=""
if $ENABLE_STOCHASTIC; then
  STOCH_FLAG="--enable_stochastic_breadth"
fi

echo "=========================================================================="
echo "RI-1 M1 + Proper 3-Track Matched Continuation + Gate Pipeline"
echo "=========================================================================="
echo "Resume: $RESUME_CKPT"
echo "Steps:  $STEPS"
echo "Out:    $OUT_DIR"
echo "M1 sampling: randint (mean=$RI1_DEPTH_MEAN, max=$RI1_DEPTH_MAX)"
echo "3-track:  --all-three-tracks (Workspaces + Attractor + Provenance)"
echo "=========================================================================="

# 1. M1 training continuation (the critical step that exercises variable depth)
PYTHONPATH=. /home/tripleyoung/.local/bin/uv run python scripts/train_hybrid_ri4_real_continuation_minimal.py \
  --steps "$STEPS" \
  --d_model "$D_MODEL" \
  --batch "$BATCH" \
  --resume_from "$RESUME_CKPT" \
  --out_dir "$OUT_DIR" \
  $STOCH_FLAG \
  --all-three-tracks \
  --enable_ri1_variable_depth \
  --ri1_depth_sampling_mode randint \
  --ri1_depth_mean "$RI1_DEPTH_MEAN" \
  --ri1_depth_max "$RI1_DEPTH_MAX" \
  --save_every 5 \
  --heldout_answer_pressure_weight 0.8 \
  --trajectory_monotonic_weight 0.2 \
  2>&1 | tee "$OUT_DIR/run_m1.log"

# Find the final (or best) checkpoint
LATEST_CKPT=$(ls -t "$OUT_DIR"/step_*.pt 2>/dev/null | head -1 || echo "$OUT_DIR/last.pt")
if [[ ! -f "$LATEST_CKPT" ]]; then
  LATEST_CKPT="$OUT_DIR/last.pt"
fi

echo ""
echo "=========================================================================="
echo "M1 training done. Latest ckpt: $LATEST_CKPT"
echo "Now running immediate Principle Gate + depth sweep (strict B, all-three, honest notes)"
echo "=========================================================================="

# 2. Principle gate (full)
PYTHONPATH=. /home/tripleyoung/.local/bin/uv run python experiments/matched_port_evaluation_a9617cd8/validate_reasoning_test_principles.py \
  --ckpt "$LATEST_CKPT" \
  --mode full \
  --notes "RI-1 M1 continuation completed. Variable depth sampling (randint) active in trainer during training. 3-track proper port (default). conditions-matched from $RESUME_CKPT. Synthetic short base debt acknowledged. See run_m1.log + TB for C-track (loss descent + sampled depths)." \
  2>&1 | tee "$OUT_DIR/principle_gate_m1.log"

# 3. Depth sweep (B-track + RI-1 proxy) for 1/4/8
for d in 1 4 8; do
  echo "=== Depth sweep d=$d (post-M1) ===" | tee -a "$OUT_DIR/depth_sweep_post_m1.log"
  PYTHONPATH=. /home/tripleyoung/.local/bin/uv run python experiments/matched_port_evaluation_a9617cd8/compute_72_accuracy_from_base.py \
    --checkpoint "$LATEST_CKPT" \
    --all-three \
    --effective-depth "$d" \
    2>&1 | tee -a "$OUT_DIR/depth_sweep_post_m1.log"
  echo "" | tee -a "$OUT_DIR/depth_sweep_post_m1.log"
done

echo ""
echo "=========================================================================="
echo "PIPELINE COMPLETE"
echo "Artifacts:"
echo "  - $OUT_DIR/run_m1.log (loss curve + sampled_depth C-track)"
echo "  - $OUT_DIR/principle_gate_m1.log"
echo "  - $OUT_DIR/depth_sweep_post_m1.log (B-track strict-B on pure_72)"
echo "  - Fresh PRINCIPLES_GATE_*.json in the matched_port dir"
echo ""
echo "Next (manual or scripted):"
echo "  - Extract C-track numbers (start→end loss, % drop, convergence step, sampled depth stats)"
echo "  - Git tag (e.g. reasoning-pure72-ri1-m1-baseline-YYYYMMDD)"
echo "  - Update wiki with I→G→A status + falsification gate for next big-jump"
echo "=========================================================================="

# Optional: open TensorBoard hint
if [[ -d "$OUT_DIR/tb" ]]; then
  echo "TensorBoard: uv run tensorboard --logdir $OUT_DIR/tb"
fi
