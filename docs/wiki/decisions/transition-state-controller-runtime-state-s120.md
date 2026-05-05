# Transition-State Controller Runtime-State S120

Status: rejected gate with useful causal signal, 2026-05-01.

## Why This Was Added

The first learned-state smoke passed because the trace rows still carried
phase-shaped state text. That was useful for proving wiring, but too close to a
trace scaffold. This run retrains the same learned-state controller with strict
runtime-style inputs:

```text
no trace-step oracle
no phase-specific state-summary
no previous-action input
no direct controller feature path
```

The controller must infer the loop state from QTRM row features and the current
runtime observation contract.

## Run

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src uv run python \
  scripts/158_train_transition_state_controller.py \
  --config configs/qwen35_2b_4090_controller_signal_s300.yaml \
  --checkpoint runs/qwen35_2b_4090_controller_signal_s300/last.pt \
  --train-jsonl data/filtered/asi_controller_trace_replay.jsonl \
  --eval-jsonl data/eval/asi_controller_trace_replay_heldout_72.jsonl \
  --out-pt runs/qwen35_2b_4090_transition_state_controller_runtime_state_s120/last.pt \
  --out-json docs/wiki/decisions/transition-state-controller-runtime-state-s120-summary.json \
  --max-train-sequences 128 \
  --max-eval-sequences 72 \
  --feature-batch-size 8 \
  --controller-batch-size 64 \
  --hidden-dim 128 \
  --state-predictor-hidden-dim 256 \
  --epochs 120 \
  --lr 3.0e-3 \
  --state-loss-weight 1.0 \
  --feature-scale 1.0 \
  --controller-feature-scale 0.0 \
  --reset-hidden \
  --learn-transition-state \
  --no-prev-action \
  --strict-runtime-state-inputs
```

## Result

```text
checkpoint: runs/qwen35_2b_4090_transition_state_controller_runtime_state_s120/last.pt
held-out action accuracy: 0.9630
held-out RETRIEVE_MEMORY accuracy: 0.8889
held-out VERIFY_EVIDENCE accuracy: 1.0000
held-out ANSWER accuracy: 1.0000
zero_transition_state accuracy: 0.3333
transition_state_drop: 0.6296
state_prediction_binary_accuracy: 0.8287
gate: rejected
```

## Interpretation

Positive signal:

- the strict runtime learned state is causally used;
- zeroing predicted transition state collapses the loop to the one-action
  baseline;
- `VERIFY_EVIDENCE` and `ANSWER` are stable once retrieval has occurred.

Negative signal:

- first-step retrieval is still imperfect: 8 of 72 held-out rows jumped to
  `ANSWER`;
- binary state reconstruction is below the old 0.90 state gate;
- this is still action-policy quality, not final task-level answer reward.

The old learned-state smoke should now be read as a scaffolded precursor. The
runtime-state run is the stricter controller metric.

## Follow-Up Gate

The task-level answer-loop gate using this checkpoint produced the first small
answer reward gain, but still rejected on action stability:

```text
report: docs/wiki/decisions/learned-state-answer-loop-runtime-state-gate.md
learned_state_qtrm: 5 / 8 = 0.6250
scripted_qtrm_answer_channel: 4 / 8 = 0.5000
scripted_donor_answer_channel: 4 / 8 = 0.5000
state_off: 2 / 8 = 0.2500
action_success_rate: 7 / 8 = 0.8750
gate: rejected
```

Next action should target first-step action stability, not another free-form
language loss.

Update:
`docs/wiki/decisions/transition-state-controller-runtime-actionfirst-s200.md`
does fix action stability. The remaining bottleneck is answer formation, tracked
in `docs/wiki/decisions/answer-formation-bottleneck-after-action-loop.md`.
