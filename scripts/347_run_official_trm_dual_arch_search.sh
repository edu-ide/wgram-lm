#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
OUT_ROOT="${OUT_ROOT:-local_eval/qtrm_native_official_trm_dual_arch_search_$(date +%Y%m%d_%H%M%S)}"
PROFILE="${PROFILE:-smoke}"
CANDIDATES="${CANDIDATES:-official,official_prenorm,coupled,gated_delta_l,mamba_h,gated_delta_mamba,router,qwen35_3to1,reversed_hybrid_3to1,reversed_hybrid_3to1_prenorm,reversed_hybrid_3to1_joint_readout,reversed_hybrid_3to1_core_gated_readout,tri_mixer}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-777}"
EVAL_SEED="${EVAL_SEED:-9777}"
VERBOSE="${VERBOSE:-0}"
ANSWER_LOSS_TYPE="${ANSWER_LOSS_TYPE:-cross_entropy}"

# Official TRM config uses H_cycles=3, L_cycles=6.  Keep this non-overridable
# for canonical dual-TRM architecture search; change only proposal/mixer blocks.
H_CYCLES="3"
L_CYCLES="6"

case "${PROFILE}" in
  smoke)
    STEPS="${STEPS:-8}"
    TRAIN_CASES="${TRAIN_CASES:-96}"
    EVAL_CASES="${EVAL_CASES:-48}"
    PROGRAM_LEN="${PROGRAM_LEN:-4}"
    D_MODEL="${D_MODEL:-32}"
    D_FF="${D_FF:-64}"
    BATCH_SIZE="${BATCH_SIZE:-12}"
    LOG_EVERY="${LOG_EVERY:-4}"
    ;;
  short)
    STEPS="${STEPS:-600}"
    TRAIN_CASES="${TRAIN_CASES:-4096}"
    EVAL_CASES="${EVAL_CASES:-192}"
    PROGRAM_LEN="${PROGRAM_LEN:-4}"
    D_MODEL="${D_MODEL:-64}"
    D_FF="${D_FF:-128}"
    BATCH_SIZE="${BATCH_SIZE:-32}"
    LOG_EVERY="${LOG_EVERY:-150}"
    ;;
  len8)
    STEPS="${STEPS:-1600}"
    TRAIN_CASES="${TRAIN_CASES:-8192}"
    EVAL_CASES="${EVAL_CASES:-384}"
    PROGRAM_LEN="${PROGRAM_LEN:-8}"
    D_MODEL="${D_MODEL:-96}"
    D_FF="${D_FF:-192}"
    BATCH_SIZE="${BATCH_SIZE:-32}"
    LOG_EVERY="${LOG_EVERY:-400}"
    ;;
  *)
    echo "Unsupported PROFILE=${PROFILE}; expected smoke, short, or len8" >&2
    exit 2
    ;;
esac

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONPATH="references/official/flash-linear-attention:local_deps/mamba3_runtime:src${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${OUT_ROOT}"

candidate_args() {
  local candidate="$1"
  case "${candidate}" in
    official)
      echo "--backbone trm_official --encode-backbone mha_etd --think-backbone trm_official --decode-backbone mha_etd --think-structure trm_dual_z"
      ;;
    official_prenorm)
      echo "--backbone trm_official_prenorm --encode-backbone mha_etd --think-backbone trm_official_prenorm --decode-backbone mha_etd --think-structure trm_dual_z"
      ;;
    coupled)
      echo "--backbone trm_official --encode-backbone mha_etd --think-backbone trm_official --decode-backbone mha_etd --think-structure trm_dual_z_coupled"
      ;;
    gated_delta_l)
      echo "--backbone trm_official --encode-backbone mha_etd --think-backbone trm_official --decode-backbone mha_etd --think-structure trm_dual_z_coupled_delta_l_only --delta-backend fla_gated_delta --delta-head-dim 16 --delta-num-v-heads 4 --delta-expand-v 1.0 --strict-backends"
      ;;
    mamba_h)
      echo "--backbone trm_official --encode-backbone mha_etd --think-backbone trm_official --decode-backbone mha_etd --think-structure trm_dual_z_coupled_mamba_h_only --strict-backends"
      ;;
    gated_delta_mamba)
      echo "--backbone trm_official --encode-backbone mha_etd --think-backbone trm_official --decode-backbone mha_etd --think-structure trm_dual_z_coupled_residual --delta-backend fla_gated_delta --delta-head-dim 16 --delta-num-v-heads 4 --delta-expand-v 1.0 --strict-backends"
      ;;
    router)
      echo "--backbone trm_official --encode-backbone mha_etd --think-backbone trm_official --decode-backbone mha_etd --think-structure trm_dual_z_coupled_hybrid_router --delta-backend fla_gated_delta --delta-head-dim 16 --delta-num-v-heads 4 --delta-expand-v 1.0 --strict-backends"
      ;;
    qwen35_3to1)
      echo "--backbone trm_qwen35_3to1 --encode-backbone mha_etd --think-backbone trm_qwen35_3to1 --decode-backbone mha_etd --think-structure trm_dual_z --delta-backend fla_gated_delta --delta-head-dim 16 --delta-num-v-heads 4 --delta-expand-v 1.0 --strict-backends"
      ;;
    reversed_hybrid_3to1)
      echo "--backbone trm_official --encode-backbone mha_etd --think-backbone trm_official --decode-backbone mha_etd --think-structure trm_dual_z_reversed_hybrid_3to1 --delta-backend fla_gated_delta --delta-head-dim 16 --delta-num-v-heads 4 --delta-expand-v 1.0 --strict-backends"
      ;;
    reversed_hybrid_3to1_prenorm)
      echo "--backbone trm_official_prenorm --encode-backbone mha_etd --think-backbone trm_official_prenorm --decode-backbone mha_etd --think-structure trm_dual_z_reversed_hybrid_3to1_prenorm --delta-backend fla_gated_delta --delta-head-dim 16 --delta-num-v-heads 4 --delta-expand-v 1.0 --strict-backends"
      ;;
    reversed_hybrid_3to1_joint_readout)
      echo "--backbone trm_official --encode-backbone mha_etd --think-backbone trm_official --decode-backbone mha_etd --think-structure trm_dual_z_reversed_hybrid_3to1_joint_readout --delta-backend fla_gated_delta --delta-head-dim 16 --delta-num-v-heads 4 --delta-expand-v 1.0 --strict-backends"
      ;;
    reversed_hybrid_3to1_core_gated_readout)
      echo "--backbone trm_official --encode-backbone mha_etd --think-backbone trm_official --decode-backbone mha_etd --think-structure trm_dual_z_reversed_hybrid_3to1_core_gated_readout --delta-backend fla_gated_delta --delta-head-dim 16 --delta-num-v-heads 4 --delta-expand-v 1.0 --strict-backends"
      ;;
    tri_mixer)
      echo "--backbone trm_tri_mixer --encode-backbone mha_etd --think-backbone trm_tri_mixer --decode-backbone mha_etd --think-structure trm_dual_z --delta-backend fla_gated_delta --delta-head-dim 16 --delta-num-v-heads 4 --delta-expand-v 1.0 --strict-backends"
      ;;
    *)
      echo "Unsupported candidate=${candidate}" >&2
      exit 2
      ;;
  esac
}

IFS=',' read -r -a candidate_values <<< "${CANDIDATES}"

for index in "${!candidate_values[@]}"; do
  candidate="${candidate_values[$index]}"
  out_dir="${OUT_ROOT}/${candidate}"
  mkdir -p "${out_dir}"
  read -r -a extra_args <<< "$(candidate_args "${candidate}")"
  candidate_seed=$((SEED + index * 101))

  echo "=== official TRM dual arch search: ${candidate} profile=${PROFILE} H=${H_CYCLES} L=${L_CYCLES} ==="
  command=(
  "${PYTHON_BIN}" scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py \
    --out-dir "${out_dir}" \
    --target-level "official TRM dual H3L6 architecture search" \
    --steps "${STEPS}" \
    --train-cases "${TRAIN_CASES}" \
    --eval-cases "${EVAL_CASES}" \
    --task-families "checksum,modchain,revchain" \
    --eval-task-families "checksum,modchain,revchain" \
    --program-len "${PROGRAM_LEN}" \
    --modulus 32 \
    --d-model "${D_MODEL}" \
    --n-heads 4 \
    --n-kv-heads 2 \
    --d-ff "${D_FF}" \
    "${extra_args[@]}" \
    --trm-l-cycles "${L_CYCLES}" \
    --halt-pooling dedicated \
    --batch-size "${BATCH_SIZE}" \
    --lr 1e-4 \
    --weight-decay 0.01 \
    --grad-clip 1.0 \
    --answer-loss-type "${ANSWER_LOSS_TYPE}" \
    --train-think-steps "${H_CYCLES}" \
    --eval-think-steps "${H_CYCLES}" \
    --depth-intermediate-loss-weight 0.25 \
    --answer-space-ranking-loss-weight 0.02 \
    --answer-space-ranking-max-cases 16 \
    --answer-space-ranking-every 2 \
    --active-len-batch-cycle \
    --eval-active-len-cycle \
    --active-len-cycle-min 2 \
    --active-len-cycle-max "${PROGRAM_LEN}" \
    --train-active-len-cycle-min 2 \
    --train-active-len-cycle-max "${PROGRAM_LEN}" \
    --seed "${candidate_seed}" \
    --eval-seed "${EVAL_SEED}" \
    --device "${DEVICE}" \
    --log-every "${LOG_EVERY}" \
    --max-examples 2 \
    --accept-min-exact 0.0 \
    --accept-min-depth-gain -1.0 \
    --accept-min-ablation-drop -1.0 \
    --accept-min-family-exact 0.0 \
    --accepted-decision "diagnostic_official_trm_dual_arch_search" \
    --eval-answer-space-argmax \
    --eval-core-step-probe
  )

  printf '%q ' "${command[@]}" > "${out_dir}/command.sh"
  printf '\n' >> "${out_dir}/command.sh"
  set +e
  if [[ "${VERBOSE}" == "1" ]]; then
    "${command[@]}" 2>&1 | tee "${out_dir}/run.log"
    status="${PIPESTATUS[0]}"
  else
    "${command[@]}" > "${out_dir}/run.log" 2>&1
    status="$?"
  fi
  set -e
  echo "${status}" > "${out_dir}/exit_code.txt"
  "${PYTHON_BIN}" - "${out_dir}" "${candidate}" <<'PY'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
candidate = sys.argv[2]
report_path = out_dir / "report.json"
if not report_path.exists():
    print(f"{candidate}: report missing")
    raise SystemExit(0)
report = json.loads(report_path.read_text(encoding="utf-8"))
metrics = report.get("decisive_metrics", {})
backend = report.get("backend_summary", {})
core_probe = (((report.get("eval_metrics") or {}).get("core_step_probe") or {}).get("core_step_probe_exact"))
print(
    f"{candidate}: full={metrics.get('full_generation_exact')} "
    f"gain={metrics.get('full_minus_think0')} "
    f"drop={metrics.get('full_minus_worst_ablation')} "
    f"core_probe={core_probe} "
    f"fla={backend.get('official_fla_delta_mixers', 0)} "
    f"mamba={backend.get('official_mamba3_mixers', 0)}"
)
PY
done

"${PYTHON_BIN}" - "${OUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for report_path in sorted(root.glob("*/report.json")):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metrics = report.get("decisive_metrics", {})
    train = report.get("train", {})
    rows.append({
        "candidate": report_path.parent.name,
        "decision": report.get("decision"),
        "accepted": bool(report.get("accepted")),
        "full_generation_exact": metrics.get("full_generation_exact"),
        "full_minus_think0": metrics.get("full_minus_think0"),
        "full_minus_worst_ablation": metrics.get("full_minus_worst_ablation"),
        "state_reset_generation_exact": metrics.get("state_reset_generation_exact"),
        "op_zero_generation_exact": metrics.get("op_zero_generation_exact"),
        "z_l_zero_generation_exact": metrics.get("z_l_zero_generation_exact"),
        "z_h_zero_generation_exact": metrics.get("z_h_zero_generation_exact"),
        "think_structure": train.get("think_structure"),
        "think_backbone": train.get("think_backbone"),
        "trm_l_cycles": train.get("trm_l_cycles"),
        "train_think_steps": train.get("train_think_steps"),
        "backend_summary": report.get("backend_summary", {}),
        "report_path": str(report_path),
    })
rows.sort(key=lambda row: (row["full_generation_exact"] is not None, row["full_generation_exact"] or -1), reverse=True)
summary = {
    "decision": "completed_official_trm_dual_arch_search",
    "accepted": True,
    "out_root": str(root),
    "ranking": rows,
}
(root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
