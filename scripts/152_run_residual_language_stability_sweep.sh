#!/usr/bin/env bash
set -euo pipefail

cd ~/qtrm-workspace/qtrm_multimodal_memoryos
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi
export PYTHONPATH=${PYTHONPATH:-$PWD/src}
export HF_HOME=${HF_HOME:-~/.cache/huggingface}

CONFIG=${CONFIG:-configs/qwen35_2b_4090_donor_residual_s010_1000.yaml}
CHECKPOINT=${CHECKPOINT:-runs/qwen35_2b_4090_donor_residual_s010_1000/last.pt}
OUT_DIR=${OUT_DIR:-runs/language_stability/residual_sweep_$(date +%Y%m%d_%H%M%S)}
SCALES=${SCALES:-0.0 0.05 0.10}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-96}
NO_REPEAT_NGRAM_SIZE=${NO_REPEAT_NGRAM_SIZE:-2}
QTRM_RESIDUAL_CLAMP=${QTRM_RESIDUAL_CLAMP:-0.75}
MIN_NEW_TOKENS_BEFORE_STOP=${MIN_NEW_TOKENS_BEFORE_STOP:-16}

mkdir -p "$OUT_DIR"
SUMMARY_JSONL="$OUT_DIR/summary.jsonl"
: > "$SUMMARY_JSONL"

PROMPT_ARGS=(
  --prompt "양자 컴퓨팅이란 무엇인가요?"
  --prompt "Explain quantum entanglement in simple terms."
  --prompt "Solve step by step: if x + 3 = 7, what is x?"
  --prompt "한국어는 어떻게 발전했나요?"
)

echo "============================================================"
echo "Residual language stability sweep"
echo "Config: $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "Scales: $SCALES"
echo "Out: $OUT_DIR"
echo "============================================================"

for scale in $SCALES; do
  safe_scale=${scale//./p}
  jsonl="$OUT_DIR/scale_${safe_scale}.jsonl"
  summary="$OUT_DIR/scale_${safe_scale}_summary.json"
  stderr_log="$OUT_DIR/scale_${safe_scale}.stderr.log"

  echo
  echo "[scale=$scale] generating JSONL..."
  python scripts/92_eval_qtrm_logits.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --donor-logits-scale 1.0 \
    --qtrm-logits-scale "$scale" \
    --qtrm-residual-clamp "$QTRM_RESIDUAL_CLAMP" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --suppress-visible-reasoning-tokens \
    --no-repeat-ngram-size "$NO_REPEAT_NGRAM_SIZE" \
    --stop-after-sentence \
    --min-new-tokens-before-stop "$MIN_NEW_TOKENS_BEFORE_STOP" \
    --json \
    "${PROMPT_ARGS[@]}" \
    > "$jsonl" \
    2> "$stderr_log"

  python scripts/147_summarize_generation_format.py \
    --eval-jsonl "$jsonl" \
    --out "$summary"

  python - "$scale" "$jsonl" "$summary" "$SUMMARY_JSONL" <<'PY'
import json
import sys
from pathlib import Path

scale, jsonl, summary_path, summary_jsonl = sys.argv[1:]
summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
row = {
    "scale": float(scale),
    "jsonl": jsonl,
    "summary": summary_path,
    "records": summary["records"],
    "clean_rate": summary["clean_rate"],
    "repeat_failure_rate": summary["repeat_failure_rate"],
    "visible_reasoning_rate": summary["visible_reasoning_rate"],
    "answer_drift_rate": summary["answer_drift_rate"],
}
with Path(summary_jsonl).open("a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(json.dumps(row, ensure_ascii=False))
PY
done

echo
echo "Summary JSONL: $SUMMARY_JSONL"
