#!/bin/bash
set -e

echo "=== Finalizing 180-step Real 642 Gold Run into Promotion Gate Package ==="

cd "$(dirname "$0")/../.."

PKG_DIR="docs/5.56_Promotion_Gate_Evidence_2026-05-30"
RUN_DIR=$(ls -td local_556_real642_long_180step_* 2>/dev/null | head -1)

if [ -z "$RUN_DIR" ]; then
    echo "ERROR: No long 180-step run directory found."
    exit 1
fi

echo "Found run: $RUN_DIR"

mkdir -p "$PKG_DIR/05_ongoing_180step_run/final_artifacts"

# Copy artifacts
cp "$RUN_DIR"/{metrics.json,best.pt,last.pt,full_output.log,config.json} \
   "$PKG_DIR/05_ongoing_180step_run/final_artifacts/" 2>/dev/null || true

# Run analyzer
python scripts/analyze_556_curriculum_metrics.py \
    "$RUN_DIR/metrics.json" \
    --output "$PKG_DIR/05_ongoing_180step_run/180step_real642_final_analysis.md"

echo ""
echo "=== Analysis complete ==="
echo "Report: $PKG_DIR/05_ongoing_180step_run/180step_real642_final_analysis.md"
echo ""
echo "Next: Update the top-level README.md with the new numbers."
