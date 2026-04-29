#!/usr/bin/env bash
set -euo pipefail
python -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools
pip install -r requirements.txt
python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda', torch.cuda.is_available())
PY
