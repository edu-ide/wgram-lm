#!/bin/bash
# launch_556_local_smoke.sh
#
# Easy launcher for the 5.56 Full Curriculum Minimal Trainer
# (with Stochastic Breadth - Reverse I→G→A)
#
# Usage examples:
#   Local (small smoke):
#       bash scripts/launch_556_local_smoke.sh --steps 50 --d_model 64 --batch 4
#
#   With stochastic breadth on:
#       bash scripts/launch_556_local_smoke.sh --steps 100 --enable_stochastic_breadth
#
#   DGX style (example):
#       bash scripts/launch_556_local_smoke.sh --steps 300 --d_model 256 --batch 16 --save_dir /mnt/data4tb/556_runs/run01

set -euo pipefail

# Default values
STEPS=100
BATCH=8
D_MODEL=128
ENABLE_STOCHASTIC_BREADTH=""
STOCHASTIC_ABLATION_ZERO="false"
LOG_EVERY=10
SAVE_DIR="local_556_smoke"
RESUME=""
GOLD_PATH=""
EXTRA_ARGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --batch)
            BATCH="$2"
            shift 2
            ;;
        --d_model)
            D_MODEL="$2"
            shift 2
            ;;
        --enable_stochastic_breadth)
            ENABLE_STOCHASTIC_BREADTH="--enable_stochastic_breadth"
            shift
            ;;
        --stochastic_ablation_zero)
            STOCHASTIC_ABLATION_ZERO="$2"
            shift 2
            ;;
        --log_every)
            LOG_EVERY="$2"
            shift 2
            ;;
        --save_dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        --resume)
            RESUME="--resume $2"
            shift 2
            ;;
        --gold_path)
            GOLD_PATH="--gold_path $2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

echo "=== Launching 5.56 Full Curriculum Minimal Trainer (Reverse I→G→A + 5.56 Gold Curriculum) ==="
echo "Steps: $STEPS | Batch: $BATCH | d_model: $D_MODEL"
echo "Stochastic Breadth: ${ENABLE_STOCHASTIC_BREADTH:-off} (ablation_zero=$STOCHASTIC_ABLATION_ZERO)"
echo "Save dir: $SAVE_DIR"
echo "Resume: ${RESUME:-none}"
echo "Gold path: ${GOLD_PATH:-'(synthetic proxy)'}"
echo "This launcher drives the full 5.56 Adaptive Rehearsal curriculum reconstruction"
echo "(scheduled decay 0.40→0.04 + attractor protection + gold structural bias + stochastic breadth)."
echo

mkdir -p "$SAVE_DIR"

# The actual command
# Use project's venv python + correct PYTHONPATH for this workspace
VENV_PYTHON="/home/tripleyoung/qtrm-workspace/wgram-lm/.venv/bin/python"
if [ ! -x "$VENV_PYTHON" ]; then
    VENV_PYTHON="python"
fi
echo "Using python: $VENV_PYTHON"
PYTHONPATH=. "$VENV_PYTHON" scripts/train_556_full_curriculum_minimal.py \
    --steps "$STEPS" \
    --batch "$BATCH" \
    --d_model "$D_MODEL" \
    $ENABLE_STOCHASTIC_BREADTH \
    --stochastic_ablation_zero "$STOCHASTIC_ABLATION_ZERO" \
    --log_every "$LOG_EVERY" \
    --save_dir "$SAVE_DIR" \
    $RESUME \
    $GOLD_PATH \
    $EXTRA_ARGS

echo ""
echo "Run finished. Artifacts in: $SAVE_DIR"
echo "  - best.pt / last.pt"
echo "  - metrics.json   (now contains 5.56 curriculum metrics: bind_weight, stochastic_diversity, gold_dist, ...)"
echo "  - config.json"
echo ""
echo "To reproduce a real 642 gold run (recommended next):"
echo "  bash scripts/launch_556_local_smoke.sh --steps 200 --enable_stochastic_breadth \\"
echo "      --gold_path local_eval/642_adaptive_fine_tuned_200step/adaptive_phase2_checkpoint.pt"