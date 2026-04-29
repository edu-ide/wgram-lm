#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=$PWD/src
PROFILE=${PROFILE:-dgx}
TEXT_SAMPLES=${TEXT_SAMPLES:-100000}
MATH_SAMPLES=${MATH_SAMPLES:-20000}
MM_SAMPLES_PER_CONFIG=${MM_SAMPLES_PER_CONFIG:-5000}

bash scripts/01_download_datasets.sh
bash scripts/03_train_downloaded_multimodal.sh configs/qwen35_2b_adapter.yaml
bash scripts/04_build_text_memory.sh || true
bash scripts/05_build_visual_memory.sh || true
