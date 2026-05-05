# Transition-State Controller Runtime Action-First S200

Status: accepted action-policy gate, rejected answer-reward gate, 2026-05-01.

## Run

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src uv run python \
  scripts/158_train_transition_state_controller.py \
  --config configs/qwen35_2b_4090_controller_signal_s300.yaml \
  --checkpoint runs/qwen35_2b_4090_controller_signal_s300/last.pt \
  --train-jsonl data/filtered/asi_controller_trace_replay.jsonl \
  --eval-jsonl data/eval/asi_controller_trace_replay_heldout_72.jsonl \
  --out-pt runs/qwen35_2b_4090_transition_state_controller_runtime_actionfirst_s200/last.pt \
  --out-json docs/wiki/decisions/transition-state-controller-runtime-actionfirst-s200-summary.json \
  --max-train-sequences 256 \
  --max-eval-sequences 72 \
  --feature-batch-size 8 \
  --controller-batch-size 64 \
  --hidden-dim 128 \
  --state-predictor-hidden-dim 256 \
  --epochs 200 \
  --lr 3.0e-3 \
  --state-loss-weight 0.2 \
  --feature-scale 1.0 \
  --controller-feature-scale 0.0 \
  --reset-hidden \
  --learn-transition-state \
  --no-prev-action \
  --strict-runtime-state-inputs
```

## Result

```text
held-out action accuracy: 1.0000
RETRIEVE_MEMORY: 72 / 72
VERIFY_EVIDENCE: 72 / 72
ANSWER: 72 / 72
zero_transition_state accuracy: 0.3333
transition_state_drop: 0.6667
state_prediction_binary_accuracy: 0.9913
gate: accepted
```

## Answer Gate

The same checkpoint fails the task-level answer-reward gate:

```text
report: docs/wiki/decisions/learned-state-answer-loop-runtime-actionfirst-gate.md
learned_state_qtrm: 0.5000
scripted_qtrm_answer_channel: 0.5000
scripted_donor_answer_channel: 0.5000
state_off: 0.2500
action_success_rate: 1.0000
gate: rejected
```

Interpretation:
the transition-state controller is now stable and causal, but a stable scripted
action sequence cannot improve answer quality unless the `VERIFY_EVIDENCE`
result changes the final answer renderer.
