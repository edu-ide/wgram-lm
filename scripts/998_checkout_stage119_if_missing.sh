#!/usr/bin/env bash
set -euo pipefail

REPO="/home/tripleyoung/qtrm-workspace/qtrm_multimodal_memoryos"
cd "$REPO"

CURRENT_BRANCH="$(git branch --show-current)"

echo "Current branch: $CURRENT_BRANCH"
echo

if [[ "$CURRENT_BRANCH" == "main" ]]; then
  echo "[INFO] You are on main. Stage119 experiment commits are on a separate branch."
  echo
  echo "Quick view (expected on main):"
  git log --oneline --max-count=5
  echo
  echo "If you need Stage119/Stage117~Stage119 commits, run:"
  echo "  git checkout ablation-opt2-isolate-memory"
  echo "  bash scripts/999_workspace_commit_guard.sh"
  echo
  echo "If you want a one-command quick check for all active data, run:"
  echo "  git log --oneline --decorate --all --since='2026-05-20' --until='2026-05-27'"
else
  echo "[INFO] You are already on an experiment branch."
  echo "Commit history preview:"
  git log --oneline --max-count=5
  echo
  echo "Re-run detailed view across branches at any time:"
  echo "  bash scripts/999_workspace_commit_guard.sh"
fi

