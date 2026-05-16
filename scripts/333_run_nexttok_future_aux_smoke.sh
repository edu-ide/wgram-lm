#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_core_state_only_kiss_mandatory_next_token_decoder_future_aux_s040.yaml}"
DATA="${DATA:-data/eval/pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_len1113_probe_eval60000_v6to7_len11_13_mixed_only.jsonl}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-/mnt/nvme0n1p2/qtrm-runs/research_gate_runner/core_state_only_kiss_mandatory_nexttok_finalpath_s020/last.pt}"
OUT_BASE="${OUT_BASE:-/mnt/nvme0n1p2/qtrm-runs/research_gate_runner/nexttok_future_aux_smoke}"
STEPS="${STEPS:-5}"
DEPTH_STEPS="${DEPTH_STEPS:-8}"
MAX_TARGET_TOKENS="${MAX_TARGET_TOKENS:-8}"
FUTURE_TARGET_TOKENS="${FUTURE_TARGET_TOKENS:-8}"
FUTURE_TOKEN_WEIGHT="${FUTURE_TOKEN_WEIGHT:-0.5}"
LATER_TOKEN_WEIGHT="${LATER_TOKEN_WEIGHT:-1.0}"
SELF_ROLLOUT_WEIGHT="${SELF_ROLLOUT_WEIGHT:-0.25}"
SAVE_EVERY="${SAVE_EVERY:-0}"
RUN_GATE="${RUN_GATE:-1}"
TRAIN_OUT="${TRAIN_OUT:-$OUT_BASE/train_s${STEPS}_future${FUTURE_TOKEN_WEIGHT}}"
GATE_OUT="${GATE_OUT:-$OUT_BASE/gate_1case_s${STEPS}_future${FUTURE_TOKEN_WEIGHT}}"

export HF_HOME="${HF_HOME:-/mnt/nvme1n1p2/hf-cache-qtrm}"
export PYTHONPATH="${PYTHONPATH:-src}"

.venv/bin/python scripts/196_train_pure_recursive_depth_supervised.py \
  --config "$CONFIG" \
  --data-jsonl "$DATA" \
  --init-checkpoint "$INIT_CHECKPOINT" \
  --steps "$STEPS" \
  --depth-steps "$DEPTH_STEPS" \
  --out-dir "$TRAIN_OUT" \
  --save-every "$SAVE_EVERY" \
  --target-logit-positions-only \
  --final-path-only-supervision \
  --depth-final-ce-weight 0 \
  --progress-margin-weight 0 \
  --all-depth-ce-weight 0 \
  --depth-trajectory-monotonic-weight 0 \
  --log-every 1 \
  --causal-prefix-supervision \
  --causal-prefix-skip-leading-whitespace-targets \
  --causal-prefix-max-target-tokens "$MAX_TARGET_TOKENS" \
  --causal-prefix-later-token-weight "$LATER_TOKEN_WEIGHT" \
  --causal-prefix-append-eos-target \
  --causal-prefix-self-rollout-weight "$SELF_ROLLOUT_WEIGHT" \
  --causal-prefix-self-rollout-max-target-tokens "$MAX_TARGET_TOKENS" \
  --answer-state-loop-future-token-ce-weight "$FUTURE_TOKEN_WEIGHT" \
  --answer-state-loop-future-token-max-target-tokens "$FUTURE_TARGET_TOKENS" \
  --core-state-zero-final-contrast-weight 0.25 \
  --core-state-zero-final-contrast-margin 0.05 \
  --core-state-zero-final-contrast-all-prefix-tokens \
  --answer-state-recurrent-final-contrast-weight 0.5 \
  --answer-state-recurrent-final-contrast-margin 0.05 \
  --answer-state-recurrent-final-contrast-all-prefix-tokens \
  --answer-next-token-decoder-final-contrast-weight 1.0 \
  --answer-next-token-decoder-final-contrast-margin 0.10 \
  --answer-next-token-decoder-final-contrast-all-prefix-tokens \
  --final-greedy-token-margin-weight 0.25 \
  --greedy-token-margin 0.25

if [[ "$RUN_GATE" == "0" ]]; then
  exit 0
fi

.venv/bin/python scripts/330_run_mixed_noncopy_lm_gate.py \
  --config "$CONFIG" \
  --checkpoint "$TRAIN_OUT/last.pt" \
  --max-cases 1 \
  --chunk-size 1 \
  --max-length 192 \
  --max-new-tokens 8 \
  --out-dir "$GATE_OUT" \
  --min-full-accuracy 0.0 \
  --min-donor-margin 0.0 \
  --min-core-off-margin 0.0 \
  --min-primitive-drop 0.0 \
  --min-source-slot-drop 0.0 \
  --min-source-binder-drop 0.0 \
  --min-bridge-drop 0.0 \
  --min-typed-value-bridge-drop 0.0 \
  --min-vocab-renderer-drop 0.0 \
  --min-core-state-zero-drop 0.0 \
  --min-answer-recurrent-drop 0.0 \
  --min-answer-next-token-decoder-drop 0.0 \
  --min-answer-free-transformer-latent-drop 0.0
