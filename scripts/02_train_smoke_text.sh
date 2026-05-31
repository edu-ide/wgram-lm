#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=$PWD/src
python -m wgram_lm.training.train --config configs/smoke_multimodal.yaml
