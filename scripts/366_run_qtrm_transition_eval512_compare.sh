#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="${PYTHONPATH:-src}"
export HF_HOME="${HF_HOME:-/mnt/nvme1n1p2/hf-cache-qtrm}"

QWEN_MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen3.5-2B-Base}"
OURO_MODEL_ID="${OURO_MODEL_ID:-/mnt/sdc1/models/ByteDance-Ouro-2.6B-Thinking}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-64}"
STEPS="${STEPS:-200}"
BATCH_SIZE="${BATCH_SIZE:-2}"
TRAIN_CASES="${TRAIN_CASES:-512}"
EVAL_CASES="${EVAL_CASES:-512}"
SEED="${SEED:-20260515}"

run_gate() {
  local out_dir="$1"
  shift
  echo "=== ${out_dir} ==="
  set +e
  .venv/bin/python scripts/362_train_qwen_backbone_qtrm_core_gate.py \
    --model-id "$QWEN_MODEL_ID" \
    --out-dir "$out_dir" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --steps "$STEPS" \
    --batch-size "$BATCH_SIZE" \
    --train-cases "$TRAIN_CASES" \
    --eval-cases "$EVAL_CASES" \
    --seed "$SEED" \
    --log-every 50 \
    "$@"
  local code="$?"
  set -e
  echo "exit_code=${code}"
}

run_gate "local_eval/qwen_backbone_qtrm_qwen_transition_eval512_s${STEPS}_20260515" \
  --core-impl qwen_layer_wrapped \
  --qwen-core-layer-indices 3 \
  --core-adapter-dim 64

run_gate "local_eval/qwen_backbone_qtrm_ouro_transition_l24_eval512_s${STEPS}_20260515" \
  --core-impl ouro_weight_wrapped \
  --ouro-model-id "$OURO_MODEL_ID" \
  --ouro-core-layer-indices 24 \
  --core-adapter-dim 64

.venv/bin/python - <<'PY'
import json
from pathlib import Path

paths = [
    Path("local_eval/qwen_backbone_qtrm_qwen_transition_eval512_s200_20260515/report.json"),
    Path("local_eval/qwen_backbone_qtrm_ouro_transition_l24_eval512_s200_20260515/report.json"),
]
rows = []
for path in paths:
    if not path.exists():
        continue
    report = json.loads(path.read_text())
    rows.append({
        "path": str(path),
        "core_impl": report.get("core_impl"),
        "core_layer_indices": report.get("core_layer_indices"),
        "accepted": report.get("accepted"),
        "gain": report.get("after_eval", {}).get("gain"),
        "core_accuracy": report.get("after_eval", {}).get("core_accuracy"),
        "language_top1": report.get("after_language", {}).get("top1_agreement"),
        "mean_abs_delta": report.get("after_language", {}).get("mean_abs_delta"),
    })
print(json.dumps(rows, ensure_ascii=False, indent=2))
PY
