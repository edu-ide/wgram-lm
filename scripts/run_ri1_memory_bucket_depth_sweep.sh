#!/bin/bash
#
# RI-1 + RI-4: Full Depth Scaling Sweep on Memory Requirement Buckets
#
# This script runs the separated evaluation:
# - Low memory demand bucket vs High memory demand bucket
# - Depths 1/4/8/12
# - Memory ON (dynamic slots) vs Memory OFF
#
# Usage examples:
#   # Quick version (small samples)
#   bash scripts/run_ri1_memory_bucket_depth_sweep.sh --max-cases 8
#
#   # Post-M1 evaluation on a fresh checkpoint
#   bash scripts/run_ri1_memory_bucket_depth_sweep.sh --ckpt checkpoints/hybrid_ri4_ri1_m1_.../hybrid_ri4_cont_stepXX.pt --max-cases 16 --depths 1,4,8,12
#
#   # More serious version
#   bash scripts/run_ri1_memory_bucket_depth_sweep.sh --max-cases 18
#
#   # Full on high bucket (be careful, slow)
#   bash scripts/run_ri1_memory_bucket_depth_sweep.sh --bucket high --max-cases 36

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH=src

# Defaults
MAX_CASES=16
BUCKET="both"   # low, high, both
DEPTHS=(1 4 8 12)

while [[ $# -gt 0 ]]; do
    case $1 in
        --max-cases)
            MAX_CASES="$2"
            shift 2
            ;;
        --bucket)
            BUCKET="$2"
            shift 2
            ;;
        --depths)
            IFS=',' read -ra DEPTHS <<< "$2"
            shift 2
            ;;
        --ckpt)
            CKPT="$2"
            shift 2
            ;;
        --config)
            CFG="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "================================================================"
echo "RI-1 + RI-4 Separated Depth Scaling Sweep"
echo "================================================================"
echo "Buckets : $BUCKET"
echo "Depths  : ${DEPTHS[*]}"
echo "Max cases per run : $MAX_CASES"
echo "Timestamp : $(date '+%Y-%m-%d %H:%M:%S')"
echo "Git HEAD  : $(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
echo "================================================================"

CKPT="checkpoints/diag_attractor_climb_v8/hybrid_ri4_cont_step50.pt"
CFG="configs/qwen35_2b_4090.yaml"

# Allow override from command line or env for post-M1 evaluation
if [[ -n "${RI1_CKPT:-}" ]]; then
    CKPT="$RI1_CKPT"
fi
if [[ -n "${RI1_CFG:-}" ]]; then
    CFG="$RI1_CFG"
fi

run_pair() {
    local bucket=$1
    local depth=$2
    local cases_file="data/eval/pure_reasoning_memory_${bucket}_bucket.jsonl"

    echo ""
    echo ">>> Bucket: $bucket | Depth: $depth | Memory ON"
    python scripts/192_eval_raw_intelligence.py \
        --config "$CFG" \
        --checkpoint "$CKPT" \
        --cases "$cases_file" \
        --mode "hybrid_sparse_slots_on_depth_${depth}_no_evidence" \
        --max-cases "$MAX_CASES" \
        --scoring forced_choice \
        --device cuda \
        --out "/tmp/ri1_${bucket}_d${depth}_on_$(date +%s).jsonl" || true

    echo ""
    echo ">>> Bucket: $bucket | Depth: $depth | Memory OFF"
    python scripts/192_eval_raw_intelligence.py \
        --config "$CFG" \
        --checkpoint "$CKPT" \
        --cases "$cases_file" \
        --mode "hybrid_sparse_slots_off_depth_${depth}_no_evidence" \
        --max-cases "$MAX_CASES" \
        --scoring forced_choice \
        --device cuda \
        --out "/tmp/ri1_${bucket}_d${depth}_off_$(date +%s).jsonl" || true
}

buckets_to_run=()
if [[ "$BUCKET" == "both" ]]; then
    buckets_to_run=("low" "high")
else
    buckets_to_run=("$BUCKET")
fi

for b in "${buckets_to_run[@]}"; do
    for d in "${DEPTHS[@]}"; do
        run_pair "$b" "$d"
    done
done

echo ""
echo "================================================================"
echo "Sweep finished."
echo "Results are in /tmp/ri1_* .jsonl files."
echo "Next: compare hit rates across depths within each bucket."
echo "Especially look at whether the gap between ON and OFF grows with depth in the HIGH bucket."
echo "================================================================"