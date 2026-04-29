#!/usr/bin/env bash
set -euo pipefail
INPUT_DIR=${1:-data/images}
OUT_DIR=${2:-memory/visual}
MODEL_ID=${VISUAL_EMBED_MODEL:-google/siglip-base-patch16-224}
export PYTHONPATH=$PWD/src
mkdir -p "$INPUT_DIR"
python - <<'PY'
from pathlib import Path
from PIL import Image, ImageDraw
p=Path('data/images/example.png')
p.parent.mkdir(parents=True, exist_ok=True)
if not p.exists():
    im=Image.new('RGB',(224,224),(240,240,240))
    d=ImageDraw.Draw(im)
    d.rectangle([40,40,184,184], outline=(0,0,0), width=4)
    d.text((55,100),'QTRM', fill=(0,0,0))
    im.save(p)
PY
python -m qtrm_mm.memoryos.visual_index "$INPUT_DIR" "$OUT_DIR" --model-id "$MODEL_ID"
