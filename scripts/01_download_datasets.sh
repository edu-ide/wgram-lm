#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=${PYTHONPATH:-$PWD/src}
PROFILE=${PROFILE:-smoke}
DATA_DIR=${DATA_DIR:-data/raw}
TEXT_SAMPLES=${TEXT_SAMPLES:-0}
MATH_SAMPLES=${MATH_SAMPLES:-0}
MM_SAMPLES_PER_CONFIG=${MM_SAMPLES_PER_CONFIG:-0}
CAULDRON_CONFIGS=${CAULDRON_CONFIGS:-scienceqa,ai2d,chartqa,docvqa,textvqa}
INCLUDE_FALLBACKS=${INCLUDE_FALLBACKS:-1}
MAX_CHARS=${MAX_CHARS:-6000}

python -m wgram_lm.data.download_datasets \
  --out-dir "$DATA_DIR" \
  --profile "$PROFILE" \
  --text-samples "$TEXT_SAMPLES" \
  --math-samples "$MATH_SAMPLES" \
  --mm-samples-per-config "$MM_SAMPLES_PER_CONFIG" \
  --cauldron-configs "$CAULDRON_CONFIGS" \
  --max-chars "$MAX_CHARS" \
  $( [ "$INCLUDE_FALLBACKS" = "1" ] && echo --include-fallbacks )
