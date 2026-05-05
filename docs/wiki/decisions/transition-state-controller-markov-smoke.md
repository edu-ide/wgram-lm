# Transition-State Controller Markov Smoke

Status: accepted narrow smoke, 2026-05-01.

## Why This Was Added

The learned controller-signal experiments failed because they treated each trace
row independently:

```text
current row -> latent/readout -> two-bit signal -> action
```

That is not a closed-loop planner. The next architecture needs transition
state:

```text
state_t + previous_action_t + observation_t + verifier_t
-> next controller state
-> action_t+1
```

This smoke implements the first minimal version: an explicit previous-action
transition state controller.

## Implementation

Added:

- `src/qtrm_mm/agentic/transition_controller.py`
- `scripts/158_train_transition_state_controller.py`
- `tests/test_transition_state_controller.py`
- `return_features_only=True` in `QTRMMultimodalModel.forward` so feature
  extraction does not allocate full vocab logits.

The script now groups trace rows by `task_id + hash(prompt, workspace)`, sorts
by `step`, and deduplicates duplicate step rows. The hash matters because the
same task can have multiple augmented workspace variants that should not be
silently collapsed. The dedupe matters because the existing signal trace file
contains repeated `0,1,2` traces under the same task id. Without dedupe, the
transition sequence becomes `0,0,0,1,1,1,2,2,2`, which creates contradictory
previous-action targets.

## Run

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src uv run python \
  scripts/158_train_transition_state_controller.py \
  --config configs/qwen35_2b_4090_controller_signal_s300.yaml \
  --checkpoint runs/qwen35_2b_4090_controller_signal_s300/last.pt \
  --train-jsonl data/filtered/asi_controller_signal_trace_replay.jsonl \
  --eval-jsonl data/eval/asi_controller_signal_trace_replay_heldout_72.jsonl \
  --out-pt runs/qwen35_2b_4090_transition_state_controller_markov_smoke/last.pt \
  --out-json docs/wiki/decisions/transition-state-controller-markov-smoke-summary.json \
  --max-train-sequences 128 \
  --max-eval-sequences 72 \
  --feature-scale 0.0 \
  --reset-hidden
```

## Result

```text
controller_mode: explicit_markov_transition_state
feature_scale: 0.0
use_prev_action: true
reset_hidden: true

train_eval accuracy: 1.0000
held-out eval_full accuracy: 1.0000
eval_reset_transition_state accuracy: 0.3333
transition_state_drop: 0.6667
gate: accepted
```

## Interpretation

This is a real improvement over the failed per-row signal head, but the scope is
narrow.

What it proves:

- a transition-state action controller can learn the staged
  `RETRIEVE_MEMORY -> VERIFY_EVIDENCE -> ANSWER` loop;
- removing the explicit transition state collapses held-out accuracy to the
  one-action baseline;
- the trace-sequence data path and causal transition-state ablation are now
  wired.

What it does not prove:

- QTRM latent features are useful for reasoning;
- world-model/verifier predictions are learned;
- answer quality improves;
- ASI progress.

Important negative result:

```text
feature_scale=1.0 failed on held-out and collapsed to RETRIEVE_MEMORY.
hidden-only recurrent dynamics also failed on held-out.
```

The immediate architecture lesson is:

```text
First make the loop state explicit and causal.
Then add learned observations, verifier state, and world-model state into that
loop under separate ablations.
Do not rely on a single latent readout to infer the whole controller phase.
```

## Next Step

The next prototype should add explicit verifier/observation fields to the
transition input:

```text
previous_action
+ previous_observation_type
+ verifier_status
+ verifier_reward
+ optional world_model_state
-> next action
```

Only after this explicit loop is stable should QTRM latent features be scaled
back in and required to improve over the explicit-state baseline.
