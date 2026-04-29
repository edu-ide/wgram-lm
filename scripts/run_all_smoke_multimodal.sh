#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=$PWD/src
PROFILE=${PROFILE:-smoke}
TEXT_SAMPLES=${TEXT_SAMPLES:-400}
MATH_SAMPLES=${MATH_SAMPLES:-200}
MM_SAMPLES_PER_CONFIG=${MM_SAMPLES_PER_CONFIG:-40}

bash scripts/01_download_datasets.sh
bash scripts/01_smoke_forward.sh
bash scripts/03_train_downloaded_multimodal.sh configs/smoke_multimodal.yaml
bash scripts/04_build_text_memory.sh || true
bash scripts/05_build_visual_memory.sh || true
bash scripts/06_infer_multimodal.sh configs/smoke_multimodal.yaml runs/smoke_multimodal/last.pt "Explain QTRM-MemoryOS." || true
bash scripts/07_retrieve.sh "recursive latent MemoryOS" || true
