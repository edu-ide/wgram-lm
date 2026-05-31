# Transition-State Controller Learned-State Smoke

Status: accepted narrow smoke with trace-scaffold caveat, 2026-05-01.

## Why This Was Added

The explicit-state smoke used a hand-built 9D transition vector. That proved
causal wiring, but not learned state extraction. This smoke adds a learned
state predictor:

```text
QTRM row feature -> TransitionStatePredictor -> predicted 9D loop state
predicted loop state -> TransitionStateController -> next action
```

The controller's direct QTRM feature path is disabled with
`controller_feature_scale=0.0`, and `previous_action` is disabled. Therefore the
controller must act through the predicted transition state.

## Implementation

Changed:

- `src/wgram_lm/agentic/transition_controller.py`
  - adds `TransitionStatePredictor`;
  - adds `transition_state_prediction_loss`;
  - reports state prediction loss, MAE, and binary accuracy.
- `scripts/158_train_transition_state_controller.py`
  - adds `--learn-transition-state`;
  - adds `--controller-feature-scale`;
  - trains predictor and controller jointly with action loss plus state loss;
  - requires held-out state prediction binary accuracy in the gate.
- `tests/test_transition_state_controller.py`
  - covers predictor training and loss behavior.

## Run

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src uv run python \
  scripts/158_train_transition_state_controller.py \
  --config configs/qwen35_2b_4090_controller_signal_s300.yaml \
  --checkpoint runs/qwen35_2b_4090_controller_signal_s300/last.pt \
  --train-jsonl data/filtered/asi_controller_trace_replay.jsonl \
  --eval-jsonl data/eval/asi_controller_trace_replay_heldout_72.jsonl \
  --out-pt runs/qwen35_2b_4090_transition_state_controller_learned_state_smoke/last.pt \
  --out-json docs/wiki/decisions/transition-state-controller-learned-state-smoke-summary.json \
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
  --no-prev-action
```

## Result

```text
feature_scale: 1.0
controller_feature_scale: 0.0
learn_transition_state: true
use_prev_action: false
reset_hidden: true

held-out eval_full accuracy: 1.0000
held-out state_prediction_binary_accuracy: 0.9974
eval_zero_transition_state accuracy: 0.3333
transition_state_drop: 0.6667
gate: accepted
```

## Interpretation

Important caveat, added after the strict runtime answer-loop check:
this run used trace rows whose input still carried phase-shaped state text. It
proved that a learned state predictor can drive the controller through a causal
state path, but it was too scaffolded to count as a runtime planner proof.
The stricter successor is
`docs/wiki/decisions/transition-state-controller-runtime-state-s120.md`.

What this proves:

- frozen QTRM row features contain enough information to predict the explicit
  loop-state scaffold on this trace task;
- the controller can act through predicted state rather than hand-built state;
- zeroing predicted transition state collapses the held-out loop to the
  one-action baseline;
- previous-action input and direct controller feature input are not required
  for this narrow trace loop.

What this does not prove:

- the state predictor is a general world model;
- answer quality or task-level reward improved;
- the verifier is judging factual correctness rather than reconstructing trace
  phase;
- latent reasoning is doing open-ended planning.

The next gate must leave action-phase prediction and move to task reward:

```text
prompt/evidence -> learned state -> action loop -> generated answer
score exact answer / UNKNOWN / conflict handling
ablate predicted state, world bit, verifier bit, and QTRM feature source
compare against scripted harness and donor harness
```

Only if the learned-state loop improves task-level answer reward over scripted
and donor harnesses should it be promoted from smoke test to main architecture.
