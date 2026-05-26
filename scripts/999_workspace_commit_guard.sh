#!/usr/bin/env bash
set -euo pipefail

REPO="/home/tripleyoung/qtrm-workspace/qtrm_multimodal_memoryos"
cd "$REPO"

printf "== HEAD ==\n"
git log --oneline --max-count=5 --decorate

printf "\n== all branches (latest 10) ==\n"
git for-each-ref --count=10 --sort=-creatordate --format='%(refname:short) %(objectname:short) %(committerdate:short) %(subject)' refs/heads refs/tags

printf "\n== dates: 2026-05-20 .. 2026-05-27 commits ==\n"
git log --oneline --decorate --since='2026-05-20' --until='2026-05-27' --all

printf "\n== workspace artifacts check ==\n"
for p in \
  docs/wiki/handoffs/2026-05-26-stage118-local-gd-preference-handoff.md \
  docs/wiki/decisions/2026-05-26-stage117-stage118-generated-algebra-traps.md \
  data/eval/stage117_algebra_trap_preference_train.jsonl \
  data/eval/stage118_fixed_parrot_algebra_trap_preference_train.jsonl \
  scripts/625_train_bpe_gd_preference.py \
  scripts/626_build_algebra_trap_preference_probe.py; do
  [ -e "$REPO/$p" ] && echo "[ok] $p" || echo "[miss] $p"
done

echo

echo "== note =="
echo "커밋은 브랜치마다 보이는 범위가 다릅니다. 보존 상태는 --all에서 확인하세요."
