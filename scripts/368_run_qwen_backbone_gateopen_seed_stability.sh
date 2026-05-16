#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="${PYTHONPATH:-src}"
export HF_HOME="${HF_HOME:-/mnt/nvme1n1p2/hf-cache-qtrm}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-2B-Base}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-80}"
STEPS="${STEPS:-300}"
BATCH_SIZE="${BATCH_SIZE:-2}"
TRAIN_CASES="${TRAIN_CASES:-768}"
EVAL_CASES="${EVAL_CASES:-512}"
SEEDS="${SEEDS:-20260515 20260516 20260517}"
FORCE="${FORCE:-0}"

for seed in $SEEDS; do
  out_dir="local_eval/qwen_backbone_qtrm_qwen_transition_hardv1_gateopen_ad128_seed${seed}_s${STEPS}_20260515"
  report="${out_dir}/report.json"
  if [[ "$FORCE" != "1" && -f "$report" ]]; then
    echo "skip existing ${report}"
    continue
  fi
  echo "=== seed ${seed} -> ${out_dir} ==="
  set +e
  .venv/bin/python scripts/362_train_qwen_backbone_qtrm_core_gate.py \
    --model-id "$MODEL_ID" \
    --out-dir "$out_dir" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --steps "$STEPS" \
    --batch-size "$BATCH_SIZE" \
    --train-cases "$TRAIN_CASES" \
    --eval-cases "$EVAL_CASES" \
    --seed "$seed" \
    --log-every 100 \
    --core-impl qwen_layer_wrapped \
    --qwen-core-layer-indices 3 \
    --core-adapter-dim 128 \
    --core-gate-init -2.0 \
    --residual-scale 0.5 \
    --case-mode hard_v1 \
    --min-reasoning-gain 0.05 \
    --min-language-top1-agreement 0.50 \
    --min-family-gain 0.01 \
    --min-family-core-accuracy 0.10
  code="$?"
  set -e
  echo "exit_code=${code}"
done

.venv/bin/python - <<'PY'
import json
from pathlib import Path

root = Path("local_eval")
paths = sorted(root.glob("qwen_backbone_qtrm_qwen_transition_hardv1_gateopen_ad128_seed*_s*_20260515/report.json"))
legacy = root / "qwen_backbone_qtrm_qwen_transition_hardv1_gateopen_ad128_s300_familyfloor_20260515/report.json"
if legacy.exists():
    paths.append(legacy)

rows = []
for path in paths:
    report = json.loads(path.read_text())
    after = report.get("after_eval", {})
    family = report.get("after_family_summary", {})
    rows.append(
        {
            "path": str(path),
            "seed": report.get("seed", path.parent.name),
            "accepted": report.get("accepted"),
            "gain": after.get("gain"),
            "core_accuracy": after.get("core_accuracy"),
            "min_family_gain": family.get("min_gain"),
            "min_family_core_accuracy": family.get("min_core_accuracy"),
            "core_gate_value": report.get("core_gate_value"),
            "language_top1": report.get("after_language", {}).get("top1_agreement"),
        }
    )

accepted = [row for row in rows if row["accepted"]]
summary = {
    "rows": rows,
    "num_reports": len(rows),
    "num_accepted": len(accepted),
    "all_accepted": len(rows) > 0 and len(accepted) == len(rows),
    "min_gain": min((row["gain"] for row in rows if row["gain"] is not None), default=None),
    "min_family_gain": min((row["min_family_gain"] for row in rows if row["min_family_gain"] is not None), default=None),
    "min_language_top1": min((row["language_top1"] for row in rows if row["language_top1"] is not None), default=None),
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
