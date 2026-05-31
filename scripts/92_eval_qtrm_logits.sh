#!/usr/bin/env bash
set -euo pipefail

cd ~/qtrm-workspace/wgram-lm
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi
export PYTHONPATH=${PYTHONPATH:-$PWD/src}
export HF_HOME=${HF_HOME:-~/.cache/huggingface}

python scripts/92_eval_qtrm_logits.py "$@"
