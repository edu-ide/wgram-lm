#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=$PWD/src
python -m qtrm_mm.training.train --config configs/smoke_multimodal.yaml --multimodal
