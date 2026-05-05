# Verified Reasoning LeWM Core S200

Date: 2026-05-03

## Question

Does adding a LeWorldModel-style next-latent prediction loss to the recursive
core improve prompt-only raw reasoning?

## Setup

Compared two checkpoints initialized from the same canonical recursive baseline:

```text
init:
  runs/qwen35_2b_4090_pure_recursive_answer_state_loop_causal_prefix_s160/last.pt

data:
  data/filtered/verified_reasoning_train256.jsonl

eval:
  data/eval/verified_reasoning_eval120.jsonl
  max_cases: 5
  scoring: causal_forced_choice
  no retrieval
  no MemoryOS shortcut
```

CE-only run:

```text
checkpoint:
  runs/qwen35_2b_4090_verified_reasoning_s200/last.pt
eval:
  runs/eval/verified_reasoning_s200_interleaved_max5.jsonl
```

CE + LeWM core-world-model run:

```text
config:
  configs/qwen35_2b_4090_verified_reasoning_lewm_core_s200.yaml
checkpoint:
  runs/qwen35_2b_4090_verified_reasoning_lewm_core_s200/last.pt
eval:
  runs/eval/verified_reasoning_lewm_core_s200_interleaved_max5.jsonl
```

## Implementation Change

The raw recursive trainer now supports LeWM-style core-world-model loss:

```text
scripts/196_train_pure_recursive_depth_supervised.py
  --core-world-model-weight
  passes core_world_model_actions into QTRMMultimodalModel
  adds jepa_world_model_loss over core_world_model_pred/target

src/qtrm_mm/training/train.py
  trainable_param_policy=core_and_answer_state_loop_and_world_model

src/qtrm_mm/qtrm_model.py
  core_world_model predictor max length now covers dynamic depth schedules up
  to core_step_conditioning_max_steps
```

## Result

```text
baseline:
  total: 2/30
  donor_only_no_evidence: 2/5
  qtrm_core_off_no_evidence: 0/5
  qtrm_core_steps_1_no_evidence: 0/5
  qtrm_core_steps_2_no_evidence: 0/5
  qtrm_core_steps_4_no_evidence: 0/5
  qtrm_core_steps_8_no_evidence: 0/5

CE-only S200:
  total: 2/30
  donor_only_no_evidence: 2/5
  all QTRM/core modes: 0/5 each

CE + LeWM core S200:
  total: 2/30
  donor_only_no_evidence: 2/5
  all QTRM/core modes: 0/5 each
```

The LeWM path is wired and trainable. The smoke run showed nonzero
`core_world_model` loss for depth 2/4/8, but the held-out raw-reasoning gate did
not improve.

## Decision

Do not promote LeWM core-world-model loss as canonical raw-intelligence
architecture yet.

Current status:

```text
LeWM module present: yes
LeWM loss wired into raw trainer: yes
LeWM causal gain on this gate: no
Canonical raw-reasoning promotion: rejected
```

## Next Step

The failure is likely not "LeWM is useless"; it is that next-latent prediction
over the current core trajectory is too weakly aligned with answer correctness.

Next candidates:

1. Train LeWM on synthetic state-transition tasks where the next latent state
   has a verifiable symbolic target.
2. Add an eval that directly measures world-model prediction quality and checks
   whether lower prediction error correlates with answer correctness.
3. Keep answer-token CE as the readout gate, but add a separate transition-state
   gate before claiming LeWM improves raw reasoning.
