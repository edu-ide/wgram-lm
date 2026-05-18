#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-run}"
DGX_HOST="${DGX_HOST:-dgx}"
DGX_REPO="${DGX_REPO:-/mnt/data4tb/qtrm_multimodal_memoryos_gate}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/mnt/data4tb/ws_llm/.venv/bin/python}"
OUT_TAG="${OUT_TAG:-$(date +%Y%m%d_%H%M%S)}"

RESUME_FROM="${RESUME_FROM:-local_eval/qtrm_native_number_oprole_circular_ladder_len8_seed9338_posnone_finalonly_prefixanchor_len8_20260518_223748/best_periodic.pt}"
MODES="${MODES:-gru,encoded,state_mean,state_delta,encoded_state_mean}"
PROGRAM_LEN="${PROGRAM_LEN:-8}"
TRAIN_CASES="${TRAIN_CASES:-1024}"
EVAL_CASES="${EVAL_CASES:-512}"
EVAL_SEED="${EVAL_SEED:-9338}"
BATCH_SIZE="${BATCH_SIZE:-48}"
DEVICE="${DEVICE:-cuda}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-local_eval/runner_logs}"
REMOTE_LOG="${REMOTE_LOG:-${REMOTE_LOG_DIR}/carrier_mode_probe_${OUT_TAG}.log}"
REMOTE_PID="${REMOTE_PID:-${REMOTE_LOG_DIR}/carrier_mode_probe_${OUT_TAG}.pid}"

usage() {
  cat <<USAGE
Usage:
  $0 [plan|status|tail|submit|run|run-local]

Purpose:
  Eval-only DGX probe for the QTRM-native len8 carrier-state source. It resumes
  the current best len8 checkpoint and compares deterministic carrier modes
  without training, so route changes can be judged before architecture edits.
USAGE
}

remote() {
  ssh "${DGX_HOST}" "cd '${DGX_REPO}' && $*"
}

run_mode() {
  local mode="$1"
  local out_dir="local_eval/qtrm_native_len${PROGRAM_LEN}_carrier_${mode}_${OUT_TAG}"

  echo "=== QTRM len${PROGRAM_LEN} carrier mode probe: ${mode} ==="
  echo "resume_from=${RESUME_FROM}"
  echo "out_dir=${out_dir}"

  PYTHONPATH=src "${REMOTE_PYTHON}" scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py \
    --out-dir "${out_dir}" \
    --target-level "QTRM-native len${PROGRAM_LEN} carrier-mode probe ${mode}" \
    --resume-from "${RESUME_FROM}" \
    --resume-allow-missing \
    --steps 0 \
    --train-cases "${TRAIN_CASES}" \
    --eval-cases "${EVAL_CASES}" \
    --task-families 'modchain,revchain,modchain,revchain,checksum' \
    --eval-task-families 'modchain,revchain,checksum' \
    --eval-family-order-invariant \
    --include-family-tag \
    --tokenizer-mode number \
    --number-tokenizer-max-value 99 \
    --number-tokenizer-op-role-tokens \
    --value-codec circular \
    --program-len "${PROGRAM_LEN}" \
    --modulus 32 \
    --d-model 256 \
    --n-heads 8 \
    --d-ff 512 \
    --batch-size "${BATCH_SIZE}" \
    --lr 0 \
    --device "${DEVICE}" \
    --train-think-steps "${PROGRAM_LEN}" \
    --eval-think-steps "${PROGRAM_LEN}" \
    --backbone mha_etd \
    --think-structure trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier \
    --position-embedding-mode none \
    --carrier-state-mode "${mode}" \
    --eval-seed "${EVAL_SEED}" \
    --eval-during-training-every 1 \
    --eval-during-training-cases "${EVAL_CASES}" \
    --periodic-eval-score-mode family_floor \
    --eval-initial-checkpoint \
    --restore-best-eval-checkpoint \
    --save-every-steps 0 \
    --save-best-periodic-checkpoint \
    --eval-state-trace \
    --accept-min-exact -1 \
    --accept-min-depth-gain -1 \
    --accept-min-ablation-drop -1 \
    --accept-min-family-exact -1 \
    --accepted-decision "diagnostic_carrier_mode_probe" \
    --log-every 1
}

summarize() {
  "${REMOTE_PYTHON}" - "local_eval/qtrm_native_len${PROGRAM_LEN}_carrier_" "${OUT_TAG}" <<'PY'
import json
import pathlib
import sys

prefix = sys.argv[1]
tag = sys.argv[2]
rows = []
for report_path in sorted(pathlib.Path("local_eval").glob(f"{pathlib.Path(prefix).name}*_{tag}/report.json")):
    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)
    train = report.get("train", {})
    mode = train.get("carrier_state_mode", report_path.parent.name)
    decisive = report.get("decisive_metrics", {})
    by_family_raw = (
        report.get("eval_metrics", {})
        .get(f"think{train.get('eval_think_steps', '')}", {})
        .get("by_family", {})
    )
    by_family = {
        name: stats.get("generation_exact")
        for name, stats in by_family_raw.items()
        if isinstance(stats, dict)
    }
    rows.append(
        {
            "mode": mode,
            "full_generation_exact": decisive.get("full_generation_exact"),
            "min_family_generation_exact": decisive.get("min_family_generation_exact"),
            "full_minus_think0": decisive.get("full_minus_think0"),
            "full_minus_worst_ablation": decisive.get("full_minus_worst_ablation"),
            "full_minus_carrier_off": decisive.get("full_minus_carrier_off"),
            "by_family": by_family,
            "report": str(report_path),
        }
    )
rows.sort(
    key=lambda row: (
        row.get("min_family_generation_exact") or -1,
        row.get("full_minus_worst_ablation") or -1,
        row.get("full_generation_exact") or -1,
    ),
    reverse=True,
)
print(json.dumps({"out_tag": tag, "rows": rows}, ensure_ascii=False, indent=2))
PY
}

case "${ACTION}" in
  -h|--help|help)
    usage
    ;;
  plan)
    cat <<'PLAN'
Carrier mode probe:

1. Resume the current best len8 prefix-anchor checkpoint.
2. Evaluate carrier_state_mode in:
   gru, encoded, state_mean, state_delta, encoded_state_mean.
3. Use zero training steps and the same 512-case eval seed.
4. Rank by min-family exact, then ablation drop, then full exact.

This tests whether the len8 bottleneck is the learned GRU carrier itself or
the surrounding recurrent trajectory.
PLAN
    ;;
  status)
    remote "pwd; git status --short --branch; git rev-parse --short HEAD; \
      pgrep -af '424_dgx_len8_carrier_mode_probe|337_train' || true; \
      ls -dt local_eval/qtrm_native_len${PROGRAM_LEN}_carrier_*_${OUT_TAG} 2>/dev/null || true"
    ;;
  tail)
    remote "if [ -f '${REMOTE_LOG}' ]; then tail -120 '${REMOTE_LOG}'; else echo 'no log: ${REMOTE_LOG}'; fi"
    ;;
  submit)
    remote "mkdir -p '${REMOTE_LOG_DIR}'; \
      git pull --ff-only; \
      nohup env PYTHONUNBUFFERED=1 OUT_TAG='${OUT_TAG}' RESUME_FROM='${RESUME_FROM}' MODES='${MODES}' \
        PROGRAM_LEN='${PROGRAM_LEN}' TRAIN_CASES='${TRAIN_CASES}' EVAL_CASES='${EVAL_CASES}' \
        EVAL_SEED='${EVAL_SEED}' BATCH_SIZE='${BATCH_SIZE}' DEVICE='${DEVICE}' \
        bash scripts/424_dgx_len8_carrier_mode_probe.sh run-local > '${REMOTE_LOG}' 2>&1 < /dev/null & \
      pid=\$!; echo \"\$pid\" > '${REMOTE_PID}'; \
      echo \"submitted pid=\$pid\"; echo \"log=${REMOTE_LOG}\"; echo \"pid_file=${REMOTE_PID}\""
    ;;
  run)
    remote "env PYTHONUNBUFFERED=1 OUT_TAG='${OUT_TAG}' RESUME_FROM='${RESUME_FROM}' MODES='${MODES}' \
      PROGRAM_LEN='${PROGRAM_LEN}' TRAIN_CASES='${TRAIN_CASES}' EVAL_CASES='${EVAL_CASES}' \
      EVAL_SEED='${EVAL_SEED}' BATCH_SIZE='${BATCH_SIZE}' DEVICE='${DEVICE}' \
      bash scripts/424_dgx_len8_carrier_mode_probe.sh run-local"
    ;;
  run-local)
    IFS=',' read -r -a carrier_modes <<< "${MODES}"
    for mode in "${carrier_modes[@]}"; do
      run_mode "${mode}"
    done
    summarize
    ;;
  *)
    echo "unknown action: ${ACTION}" >&2
    usage >&2
    exit 2
    ;;
esac
