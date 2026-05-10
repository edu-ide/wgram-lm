#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-src:.}"
export HF_HOME="${HF_HOME:-/mnt/nvme1n1p2/hf-cache-qtrm}"
export TMPDIR="${TMPDIR:-/mnt/nvme1n1p2/tmp}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

CONFIG="${CONFIG:-configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_halt_head_eval_gate.yaml}"
CHECKPOINT="${CHECKPOINT:-local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_halt_head_s080_from_tail_s020/last.pt}"
CASES="${CASES:-data/eval/pure_recursive_reasoning_heldout_72.jsonl}"
MAX_CASES="${MAX_CASES:-8}"
OUT_DIR="${OUT_DIR:-/mnt/nvme1n1p2/qtrm-eval/core_carry_mixed_depth_act}"
OUT="${OUT:-${OUT_DIR}/mixed_depth_act_eval_gate_causal_fc_smoke${MAX_CASES}.jsonl}"
SUMMARY="${SUMMARY:-${OUT%.jsonl}.summary.txt}"

mkdir -p "${OUT_DIR}" "${TMPDIR}"

echo "=== QTRM core carry mixed-depth ACT gate ==="
echo "config:     ${CONFIG}"
echo "checkpoint: ${CHECKPOINT}"
echo "cases:      ${CASES}"
echo "max_cases:  ${MAX_CASES}"
echo "out:        ${OUT}"
echo

"${PYTHON_BIN}" scripts/192_eval_raw_intelligence.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --cases "${CASES}" \
  --max-cases "${MAX_CASES}" \
  --max-length 512 \
  --scoring causal_forced_choice \
  --choice-score-normalization mean \
  --mode donor_only_no_evidence \
  --mode qtrm_core_off_no_evidence \
  --mode qtrm_core_steps_2_no_evidence \
  --mode qtrm_core_steps_4_no_evidence \
  --mode qtrm_core_steps_8_no_evidence \
  --mode qtrm_core_halt_carry_steps_2_no_evidence \
  --mode qtrm_core_halt_carry_steps_4_no_evidence \
  --mode qtrm_core_halt_carry_steps_8_no_evidence \
  --mode qtrm_core_steps_8_answer_halt_gate_off_no_evidence \
  --out "${OUT}"

"${PYTHON_BIN}" - "${OUT}" "${SUMMARY}" <<'PY'
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
    steps = [
        row.get("core_steps_actual_mean")
        for row in mode_rows
        if row.get("core_steps_actual_mean") is not None
    ]
    step_mean = sum(steps) / len(steps) if steps else None
    completions = Counter(row.get("completion") for row in mode_rows).most_common(6)
    lines.append(
        f"{mode}: {hits}/{len(mode_rows)}"
        + (f" steps={step_mean:.2f}" if step_mean is not None else "")
        + f" completions={completions}"
    )

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
