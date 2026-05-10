#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

export PYTHONPATH="${PYTHONPATH:-$PWD/src}"
export HF_HOME="${HF_HOME:-/mnt/nvme1n1p2/hf-cache-qtrm}"
export TMPDIR="${TMPDIR:-/mnt/nvme1n1p2/tmp}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_halt_head_s080.yaml}"
EVAL_CONFIG="${EVAL_CONFIG:-configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_halt_head_eval_gate.yaml}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_halt_head_s080_from_tail_s020/last.pt}"
TRAIN_DATA="${TRAIN_DATA:-data/filtered/pure_recursive_reasoning_train256_cases.jsonl}"
EVAL_CASES="${EVAL_CASES:-data/eval/pure_recursive_reasoning_heldout_72.jsonl}"
OUT_DIR="${OUT_DIR:-/mnt/nvme1n1p2/qtrm-runs/qwen35_2b_pure_recursive_reasoning_mixed_depth_act_s160_from_s080}"
STEPS="${STEPS:-160}"
LR="${LR:-5.0e-5}"
TRAINABLE_PARAM_POLICY="${TRAINABLE_PARAM_POLICY:-answer_state_loop_only}"
FINAL_LOGIT_CE_WEIGHT="${FINAL_LOGIT_CE_WEIGHT:-0.0}"
DEPTH_FINAL_CE_WEIGHT="${DEPTH_FINAL_CE_WEIGHT:-0.0}"
ALL_DEPTH_CE_WEIGHT="${ALL_DEPTH_CE_WEIGHT:-0.0}"
PROGRESS_MARGIN_WEIGHT="${PROGRESS_MARGIN_WEIGHT:-0.0}"
TERMINAL_DEPTH_CE_WEIGHT="${TERMINAL_DEPTH_CE_WEIGHT:-0.0}"
ANSWER_STATE_LOOP_HALT_CE_WEIGHT="${ANSWER_STATE_LOOP_HALT_CE_WEIGHT:-0.75}"
CHOICE_MARGIN_WEIGHT="${CHOICE_MARGIN_WEIGHT:-0.20}"
FAMILY_REPEAT="${FAMILY_REPEAT:-}"
STAGED_INTERNAL_SEQUENCE_CE_WEIGHT="${STAGED_INTERNAL_SEQUENCE_CE_WEIGHT:-0.0}"
STAGED_INTERNAL_SEQUENCE_MAX_TARGET_TOKENS="${STAGED_INTERNAL_SEQUENCE_MAX_TARGET_TOKENS:-6}"
EVAL_MAX_CASES="${EVAL_MAX_CASES:-8}"
EVAL_OUT="${EVAL_OUT:-${OUT_DIR}/mixed_depth_act_causal_forced_choice_smoke${EVAL_MAX_CASES}.jsonl}"
SUMMARY="${SUMMARY:-${EVAL_OUT%.jsonl}.summary.txt}"

mkdir -p "${OUT_DIR}" "${TMPDIR}"

echo "=== QTRM mixed-depth ACT S${STEPS} ==="
echo "train_config: ${TRAIN_CONFIG}"
echo "eval_config:  ${EVAL_CONFIG}"
echo "init:         ${INIT_CHECKPOINT}"
echo "train_data:   ${TRAIN_DATA}"
echo "eval_cases:   ${EVAL_CASES}"
echo "policy:       ${TRAINABLE_PARAM_POLICY}"
echo "family_repeat:${FAMILY_REPEAT:-none}"
echo "out:          ${OUT_DIR}"
echo

"${PYTHON_BIN}" scripts/196_train_pure_recursive_depth_supervised.py \
  --config "${TRAIN_CONFIG}" \
  --data-jsonl "${TRAIN_DATA}" \
  --init-checkpoint "${INIT_CHECKPOINT}" \
  --tokenizer-model-id Qwen/Qwen3.5-2B-Base \
  --steps "${STEPS}" \
  --lr "${LR}" \
  --trainable-param-policy "${TRAINABLE_PARAM_POLICY}" \
  --depth-steps 1,2,4,8 \
  --target-mode staged \
  --out-dir "${OUT_DIR}" \
  --final-logit-ce-weight "${FINAL_LOGIT_CE_WEIGHT}" \
  --depth-final-ce-weight "${DEPTH_FINAL_CE_WEIGHT}" \
  --all-depth-ce-weight "${ALL_DEPTH_CE_WEIGHT}" \
  --progress-margin-weight "${PROGRESS_MARGIN_WEIGHT}" \
  --terminal-depth-ce-weight "${TERMINAL_DEPTH_CE_WEIGHT}" \
  --answer-state-loop-halt-ce-weight "${ANSWER_STATE_LOOP_HALT_CE_WEIGHT}" \
  --causal-prefix-supervision \
  --causal-prefix-max-target-tokens 4 \
  --causal-prefix-later-token-weight 0.50 \
  --choice-margin-weight "${CHOICE_MARGIN_WEIGHT}" \
  --choice-margin 0.07 \
  --choice-margin-mode sequence \
  --family-repeat "${FAMILY_REPEAT}" \
  --staged-internal-sequence-ce-weight "${STAGED_INTERNAL_SEQUENCE_CE_WEIGHT}" \
  --staged-internal-sequence-max-target-tokens "${STAGED_INTERNAL_SEQUENCE_MAX_TARGET_TOKENS}" \
  --log-every 20

"${PYTHON_BIN}" scripts/192_eval_raw_intelligence.py \
  --config "${EVAL_CONFIG}" \
  --checkpoint "${OUT_DIR}/last.pt" \
  --cases "${EVAL_CASES}" \
  --max-cases "${EVAL_MAX_CASES}" \
  --max-length 512 \
  --scoring causal_forced_choice \
  --choice-score-normalization mean \
  --mode donor_only_no_evidence \
  --mode qtrm_core_off_no_evidence \
  --mode qtrm_core_steps_1_no_evidence \
  --mode qtrm_core_steps_2_no_evidence \
  --mode qtrm_core_steps_4_no_evidence \
  --mode qtrm_core_steps_8_no_evidence \
  --mode qtrm_core_steps_2_answer_halt_gate_off_no_evidence \
  --mode qtrm_core_steps_4_answer_halt_gate_off_no_evidence \
  --mode qtrm_core_steps_8_answer_halt_gate_off_no_evidence \
  --out "${EVAL_OUT}"

"${PYTHON_BIN}" - "${EVAL_OUT}" "${SUMMARY}" <<'PY'
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict

path, summary_path = sys.argv[1], sys.argv[2]
rows = [json.loads(line) for line in open(path, encoding="utf-8")]
by_mode = defaultdict(list)
for row in rows:
    by_mode[row["mode"]].append(row)

lines = ["MODE SUMMARY"]
for mode, mode_rows in by_mode.items():
    hits = sum(bool(row.get("hit")) for row in mode_rows)
    completions = Counter(row.get("completion") for row in mode_rows).most_common(5)
    lines.append(f"{mode}: {hits}/{len(mode_rows)} completions={completions}")

families = sorted({row.get("task_family", "") for row in rows})
lines.append("")
lines.append("FAMILY HIT MATRIX")
lines.append("mode," + ",".join(families))
for mode, mode_rows in by_mode.items():
    cells = []
    for family in families:
        family_rows = [row for row in mode_rows if row.get("task_family", "") == family]
        cells.append(f"{sum(bool(row.get('hit')) for row in family_rows)}/{len(family_rows)}")
    lines.append(mode + "," + ",".join(cells))

summary = "\n".join(lines)
open(summary_path, "w", encoding="utf-8").write(summary + "\n")
print(summary)
print(f"summary: {summary_path}")
PY
