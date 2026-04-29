#!/usr/bin/env bash
set -euo pipefail
MAX_JOBS=${MAX_JOBS:-4}
pip install flash-attn --no-build-isolation
pip install flash-linear-attention
python - <<'PY'
import importlib
for name in ['flash_attn', 'fla']:
    print(name, 'OK' if importlib.util.find_spec(name) else 'MISSING')
try:
    from fla.layers import GatedDeltaNet
    print('fla.layers.GatedDeltaNet OK')
except Exception as exc:
    print('fla.layers.GatedDeltaNet MISSING', type(exc).__name__, exc)
PY
