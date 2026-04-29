#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=$PWD/src
python tests/test_forward.py
python tests/count_params.py --config configs/smoke_multimodal.yaml
