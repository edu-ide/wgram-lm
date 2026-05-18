#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-run}"

STATE_TRACE_DEPTH_WEIGHT="${STATE_TRACE_DEPTH_WEIGHT:-0.45}"
STATE_TRACE_DEPTH_MIN_DEPTH="${STATE_TRACE_DEPTH_MIN_DEPTH:-4}"
STATE_TRACE_DEPTH_WEIGHT_POWER="${STATE_TRACE_DEPTH_WEIGHT_POWER:-1.0}"
STATE_TRACE_DEPTH_SOURCE="${STATE_TRACE_DEPTH_SOURCE:-h}"
STATE_TRACE_DEPTH_FAMILY_DRO_TEMP="${STATE_TRACE_DEPTH_FAMILY_DRO_TEMP:-1.0}"

case "${ACTION}" in
  -h|--help|help|plan)
    cat <<'PLAN'
Len20 state-trace depth-credit gate:

Purpose:
  Test an RLTT / latent-lookahead style objective without adding a side solver.
  The accepted single-order recurrent path stays canonical:

    token prompt -> QTRM recurrent core -> shared decode/norm/LM logits

Mechanism:
  During training, collect core_state_trace_h for every recurrent depth.
  Decode each traced state through the existing LM path and supervise it on the
  answer for the causal prefix available at that depth. This gives dense credit
  to the latent trajectory, not only final-answer CE.

Promotion:
  seed9338 min_family >= 0.06
  original-seed retention pass
  think0/state_reset/op_zero/route ablations still remove the gain

Implementation:
  This wrapper delegates execution to scripts/417_dgx_len20_time_conditioned_router_gate.sh
  with THINK_STRUCTURE=single_order_router and state-trace depth flags.
PLAN
    exit 0
    ;;
esac

export RUN_LABEL="${RUN_LABEL:-state_trace_depth_credit}"
export OUT_PREFIX="${OUT_PREFIX:-state_trace_depth_credit}"
export TARGET_LABEL="${TARGET_LABEL:-state-trace-depth-credit}"
export THINK_STRUCTURE="${THINK_STRUCTURE:-single_order_router}"
export TRAIN_PARAM_NAME_REGEX="${TRAIN_PARAM_NAME_REGEX:-single_order_route1|trm_order_router}"

STATE_TRACE_EXTRA_ARGS="--state-trace-depth-loss-weight ${STATE_TRACE_DEPTH_WEIGHT} \
--state-trace-depth-state-source ${STATE_TRACE_DEPTH_SOURCE} \
--state-trace-depth-min-depth ${STATE_TRACE_DEPTH_MIN_DEPTH} \
--state-trace-depth-weight-power ${STATE_TRACE_DEPTH_WEIGHT_POWER} \
--state-trace-depth-family-dro \
--state-trace-depth-family-dro-temperature ${STATE_TRACE_DEPTH_FAMILY_DRO_TEMP}"

export EXTRA_ARGS="${EXTRA_ARGS:-} ${STATE_TRACE_EXTRA_ARGS}"

exec bash scripts/417_dgx_len20_time_conditioned_router_gate.sh "${ACTION}"
