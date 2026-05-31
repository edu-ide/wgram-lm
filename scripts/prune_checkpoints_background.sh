#!/usr/bin/env bash
# Background checkpoint pruner for long continuous runs.
# Keeps only the most recent N checkpoints + last/best.
# Safe to run in parallel with the trainer.

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <run_dir> <keep_count>"
    echo "Example: $0 local_eval/20260530_82M_SIMPLE_LONG 3"
    exit 1
fi

RUN_DIR="$1"
KEEP_COUNT="${2:-3}"

echo "[$(date)] Starting checkpoint pruner for ${RUN_DIR}, keeping last ${KEEP_COUNT} important checkpoints."

while true; do
    if [[ ! -d "${RUN_DIR}" ]]; then
        echo "[$(date)] Run dir disappeared, exiting pruner."
        exit 0
    fi

    # Find all step_*_model.pt files, sort by step number, keep only the newest N
    mapfile -t STEP_FILES < <(find "${RUN_DIR}" -maxdepth 1 -name 'step_*_model.pt' -printf '%T@ %p\n' 2>/dev/null | sort -n | cut -d' ' -f2- || true)

    TOTAL_STEP_FILES=${#STEP_FILES[@]}
    if (( TOTAL_STEP_FILES > KEEP_COUNT )); then
        TO_DELETE=$(( TOTAL_STEP_FILES - KEEP_COUNT ))
        echo "[$(date)] Found ${TOTAL_STEP_FILES} step checkpoints, deleting oldest ${TO_DELETE}..."
        for ((i=0; i<TO_DELETE; i++)); do
            FILE="${STEP_FILES[i]}"
            if [[ -f "$FILE" ]]; then
                rm -f "$FILE"
                echo "  Deleted: $FILE"
            fi
        done
    fi

    # Also clean up very old copy_* files if they accumulate (optional safety)
    # Keep only the newest last_model.pt / best_eval_model.pt (trainer already manages these well)

    sleep 900   # Check every 15 minutes
done
