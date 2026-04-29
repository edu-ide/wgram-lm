#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=$PWD/src
PROFILE=${PROFILE:-4090}
TEXT_SAMPLES=${TEXT_SAMPLES:-20000}
MATH_SAMPLES=${MATH_SAMPLES:-4000}
MM_SAMPLES_PER_CONFIG=${MM_SAMPLES_PER_CONFIG:-1000}

bash scripts/01_download_datasets.sh
bash scripts/03_train_downloaded_multimodal.sh configs/smoke_multimodal.yaml
bash scripts/04_build_text_memory.sh || true
bash scripts/05_build_visual_memory.sh || true
