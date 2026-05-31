# Transition-State Controller Explicit-State Smoke

Status: accepted narrow smoke, 2026-05-01.

## Why This Was Added

The previous Markov smoke proved that `previous_action` can carry the staged
loop:

```text
RETRIEVE_MEMORY -> VERIFY_EVIDENCE -> ANSWER
```

That was useful but too weak. A real controller should also react to the
previous observation and verifier/world state. This smoke removes
`previous_action`, zeros QTRM latent features, and asks whether explicit
transition-state features alone can drive the loop.

## Transition-State Features

The controller input can now include a 9-dimensional explicit state vector:

```text
has_previous_observation
previous_observation_has_memory_evidence
previous_observation_has_candidate_answer
previous_observation_has_verified_candidate_answer
previous_observation_mentions_unknown
previous_observation_source_count_norm
previous_reward
previous_world_model_signal
previous_verifier_signal
```

This is intentionally small and inspectable. It is not a learned world model
yet. It is a causal wiring smoke: if the state vector is zeroed, later actions
must fail.

## Implementation

Changed:

- `src/wgram_lm/agentic/transition_controller.py`
  - accepts `transition_state_dim`;
  - concatenates normalized explicit state features with QTRM features and
    optional previous-action embeddings;
  - supports `zero_transition_state` during autoregressive evaluation.
- `scripts/158_train_transition_state_controller.py`
  - builds explicit state features from trace rows;
  - adds `--use-transition-state` and `--transition-state-scale`;
  - reports `eval_zero_transition_state` and `eval_force_start_prev_action`;
  - groups sequence variants by `task_id + hash(prompt, workspace)` so
    augmented traces are not silently collapsed.
- `tests/test_transition_state_controller.py`
  - covers explicit state input, state collation, and variant-preserving
    sequence grouping.

## Run

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src uv run python \
  scripts/158_train_transition_state_controller.py \
  --config configs/qwen35_2b_4090_controller_signal_s300.yaml \
  --checkpoint runs/qwen35_2b_4090_controller_signal_s300/last.pt \
  --train-jsonl data/filtered/asi_controller_trace_replay.jsonl \
  --eval-jsonl data/eval/asi_controller_trace_replay_heldout_72.jsonl \
  --out-pt runs/qwen35_2b_4090_transition_state_controller_explicit_state_smoke/last.pt \
  --out-json docs/wiki/decisions/transition-state-controller-explicit-state-smoke-summary.json \
  --max-train-sequences 128 \
  --max-eval-sequences 72 \
  --feature-batch-size 8 \
  --controller-batch-size 64 \
  --hidden-dim 128 \
  --epochs 80 \
  --lr 3.0e-3 \
  --feature-scale 0.0 \
  --reset-hidden \
  --use-transition-state \
  --no-prev-action
```

## Result

```text
controller_mode: explicit_markov_transition_state
feature_scale: 0.0
use_transition_state: true
transition_state_dim: 9
use_prev_action: false
reset_hidden: true

held-out eval_full accuracy: 1.0000
eval_zero_transition_state accuracy: 0.3333
eval_reset_transition_state accuracy: 0.3333
zero_transition_state_drop: 0.6667
transition_state_drop: 0.6667
gate: accepted
```

## Interpretation

What this proves:

- the controller can use previous observation/verifier state without
  `previous_action`;
- the explicit state is causally necessary under the ablation;
- the transition loop can be represented as inspectable state, not only hidden
  recurrence.

What this does not prove:

- QTRM latent reasoning is useful;
- the world-model and verifier signals are learned end to end;
- answer generation quality improves;
- ASI-level planning has been achieved.

The immediate next gate was completed in
`transition-state-controller-learned-state-smoke.md`: hand-built state was
replaced with a learned state predictor while keeping the same state-zeroing
ablation. The remaining gate must move from trace-phase prediction to
task-level answer reward:

```text
prompt/evidence -> learned state -> action loop -> generated answer
state/world/verifier ablations
scripted/donor harness comparisons
```

Only if the learned state improves held-out task-level reward should this
become part of the main QTRM architecture.
