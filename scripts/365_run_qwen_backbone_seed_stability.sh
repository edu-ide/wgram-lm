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
STEPS="${STEPS:-80}"
BATCH_SIZE="${BATCH_SIZE:-2}"
TRAIN_CASES="${TRAIN_CASES:-256}"
EVAL_CASES="${EVAL_CASES:-96}"
SEEDS="${SEEDS:-20260516 20260517}"

run_gate() {
  local name="$1"
  local seed="$2"
  shift 2
  local out_dir="local_eval/qwen_backbone_wgram_${name}_seed${seed}_s${STEPS}_20260515"
  echo "=== ${name} seed=${seed} ==="
  set +e
  .venv/bin/python scripts/362_train_qwen_backbone_wgram_core_gate.py \
    --model-id "$QWEN_MODEL_ID" \
    --out-dir "$out_dir" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --steps "$STEPS" \
    --batch-size "$BATCH_SIZE" \
    --train-cases "$TRAIN_CASES" \
    --eval-cases "$EVAL_CASES" \
    --seed "$seed" \
    --log-every 20 \
    "$@"
  local code="$?"
  set -e
  echo "exit_code=${code}"
}

for seed in $SEEDS; do
  run_gate "qwen_layer_wrapped" "$seed" \
    --core-impl qwen_layer_wrapped \
    --qwen-core-layer-indices 3 \
    --core-adapter-dim 64

  run_gate "ouro_weight_full_l24" "$seed" \
    --core-impl ouro_weight_wrapped \
    --ouro-model-id "$OURO_MODEL_ID" \
    --ouro-core-layer-indices 24 \
    --core-adapter-dim 64
done

.venv/bin/python - <<'PY'
import json
from pathlib import Path

paths = [
    Path("local_eval/qwen_backbone_wgram_qwen_layer_wrapped_train_gate_s80_20260515/report.json"),
    Path("local_eval/qwen_backbone_wgram_ouro_weight_full_l24_train_gate_s80_20260515/report.json"),
]
paths.extend(sorted(Path("local_eval").glob("qwen_backbone_wgram_qwen_layer_wrapped_seed*_s80_20260515/report.json")))
paths.extend(sorted(Path("local_eval").glob("qwen_backbone_wgram_ouro_weight_full_l24_seed*_s80_20260515/report.json")))

rows = []
for path in paths:
    if not path.exists():
        continue
    report = json.loads(path.read_text())
    rows.append(
        {
            "path": str(path),
            "core_impl": report.get("core_impl"),
            "ouro_layers": report.get("ouro_core_layer_indices"),
            "qwen_layers": report.get("qwen_core_layer_indices"),
            "accepted": report.get("accepted"),
            "gain": report.get("after_eval", {}).get("gain"),
            "core_accuracy": report.get("after_eval", {}).get("core_accuracy"),
            "language_top1": report.get("after_language", {}).get("top1_agreement"),
        }
    )

print(json.dumps(rows, ensure_ascii=False, indent=2))
PY
