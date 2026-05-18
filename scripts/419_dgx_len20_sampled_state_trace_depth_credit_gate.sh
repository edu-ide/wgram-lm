#!/usr/bin/env bash
set -euo pipefail

export RUN_LABEL="${RUN_LABEL:-sampled_state_trace_depth_credit}"
export OUT_PREFIX="${OUT_PREFIX:-sampled_state_trace_depth_credit}"
export TARGET_LABEL="${TARGET_LABEL:-sampled-state-trace-depth-credit}"
export STATE_TRACE_DEPTH_SAMPLE_COUNT="${STATE_TRACE_DEPTH_SAMPLE_COUNT:-5}"
export STATE_TRACE_DEPTH_SAMPLE_MODE="${STATE_TRACE_DEPTH_SAMPLE_MODE:-uniform}"
export STATE_TRACE_DEPTH_WEIGHT="${STATE_TRACE_DEPTH_WEIGHT:-0.55}"

exec bash scripts/418_dgx_len20_state_trace_depth_credit_gate.sh "${1:-run}"
