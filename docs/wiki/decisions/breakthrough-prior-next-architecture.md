# Breakthrough Prior Next Architecture

Date: 2026-05-05

## Why Local MLP Patches Are No Longer Enough

The last two local fixes failed:

```text
core_state_carry:
  full fine-tune: 80/624 value
  carry-only:    158/624 value

core_role_value_delta_only:
  lr 1e-4:       112/624 value
  lr 1e-5:       184/624 value, baseline tie

baseline:
  value:         184/624
  step exact:     16/256
  trace exact:     0/32
  action-code:    32/32
```

Conclusion:

```text
Another free hidden-state MLP is not a breakthrough. It can preserve or perturb
the accepted action-code path, but it does not create exact value-state
execution.
```

## Prior Work That Changes The Architecture

### 1. Looped / Latent Reasoning

Relevant sources:

```text
Scaling up Test-Time Compute with Latent Reasoning, 2025
https://arxiv.org/abs/2502.05171

Reasoning with Latent Thoughts: On the Power of Looped Transformers, 2025
https://arxiv.org/abs/2502.17416

Scaling Latent Reasoning via Looped Language Models / Ouro, 2025
https://arxiv.org/abs/2510.25741
local: references/papers/recurrent_depth/ouro_looplm_2510.25741.pdf

LoopFormer, 2026
https://arxiv.org/abs/2602.11451
local: references/papers/recurrent_depth/loopformer_2602.11451.pdf
official code: references/official/loopformer @ 59a8ae8

Prioritize the Process, Not Just the Outcome / RLTT, 2026
https://arxiv.org/abs/2602.10520
local: references/papers/recurrent_depth/rltt_latent_thought_trajectory_2602.10520.pdf

LoopRPT: Reinforcement Pre-Training for Looped Language Models, 2026
https://arxiv.org/abs/2603.19714
local: references/papers/recurrent_depth/looprpt_2603.19714.pdf

A Mechanistic Analysis of Looped Reasoning Language Models, 2026
https://arxiv.org/abs/2604.11791

A Formal Comparison Between Chain of Thought and Latent Thought, 2025/2026
https://arxiv.org/abs/2509.25239
```

Takeaway for QTRM:

```text
Looped recurrence is still right for latent reasoning, but it must be trained
as a stable trajectory. Independent per-step CE is too weak.

Use:
  variable-depth training;
  time/step-size conditioning;
  shortcut consistency between short and full loop routes;
  trajectory-level credit rather than only final-state reward;
  depth-wise causal probes.
```

Architecture implication:

```text
The next QTRM improvement should not be "another head on the final hidden
state." It should make each recurrent step learnable and accountable:

  core step t state
  -> predicted transition/action/value state
  -> short-route/full-route consistency
  -> held-out score must improve as loop budget grows
```

### 2. Neural Algorithmic Reasoning

Relevant sources:

```text
CLRS Algorithmic Reasoning Benchmark
https://arxiv.org/abs/2205.15659
https://github.com/deepmind/clrs
local code: references/official/clrs @ bfd042f

Transformers meet Neural Algorithmic Reasoners, 2024
https://arxiv.org/pdf/2406.09308
local: references/papers/role_value_slots/transnar_transformers_meet_nar_2406.09308.pdf

Discrete Neural Algorithmic Reasoning, ICML 2025
https://openreview.net/pdf?id=Inrv8EXylW
local: references/papers/role_value_slots/discrete_neural_algorithmic_reasoning_icml2025.pdf
official code: references/official/dnar @ 12f3f0b

Neural Algorithmic Reasoning for Hypergraphs with Looped Transformers, 2025
https://arxiv.org/abs/2501.10688
```

Takeaway for QTRM:

```text
Exact value-state tasks need algorithmic execution bias:
  finite states;
  hard/discrete transitions;
  separation of discrete execution state from continuous text features;
  step/hint supervision;
  OOD length/value-range gates.
```

This directly explains our failure: role-value logits are continuous readouts,
but exact arithmetic/list state needs a discrete execution trace.

### 3. Adaptive Compute / Halting

Relevant sources:

```text
PonderNet
https://arxiv.org/abs/2107.05407

AdaPonderLM, 2026
https://arxiv.org/abs/2603.01914
```

Takeaway for QTRM:

```text
Early exit is useful only after the recurrent state actually improves with
more steps. Halting is not the next blocker. First prove a depth-improving
state machine.
```

### 4. Verifiable Synthetic Curriculum

Relevant source:

```text
SLR: Automated Synthesis Framework for Scalable Logical Reasoning, 2025
https://huggingface.co/papers/2506.15787
```

Takeaway for QTRM:

```text
Use generated tasks with executable validators. Do not rely on language loss or
teacher traces alone for exact reasoning. Every state update should be
checkable.
```

## Recommended Architecture Pivot

Name:

```text
QTRM Discrete Algorithmic Executor
```

Path:

```text
canonical prompt/donor states
-> prompt parser / role binder
-> mandatory recurrent core
-> discrete algorithmic state:
     action_code
     value_delta_code
     role_pointer
     terminality
-> hard/straight-through update over typed registers
-> role-value logits
```

Key rule:

```text
The recurrent core may propose the next execution state, but exact value update
must pass through a finite typed state bottleneck. Continuous hidden MLP deltas
alone are demoted to probe-only.
```

## Implementation Scaffold

Implemented on 2026-05-05:

```text
config fields:
  core_value_delta_code_enabled
  core_value_delta_codebook_size
  core_value_delta_code_gate_init_bias
  core_value_delta_code_gate_min

model outputs:
  core_value_delta_code_logits
  core_value_delta_code_gate_mean

trainable policy:
  core_value_delta_code_only

training hook:
  scripts/196_train_pure_recursive_depth_supervised.py
  --core-value-delta-code-ce-weight

eval ablation:
  scripts/238_eval_qtrm_algorithmic_value_state.py
  --disable-core-value-delta-code

config:
  configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_core_value_delta_code_only_s120.yaml
```

Status:

```text
Implemented and tested as `core_value_delta_code_only_s120`. Rejected as
canonical because it ties the 184/624 value baseline, code-off is identical,
direct code-logit readout drops to 63/624, and depth 8 does not beat depth 1.

Decision record:
  docs/wiki/decisions/core-value-delta-code-only-s120.md
```

## Smallest Falsifiable Experiment

Add a new discrete value-delta state path:

```text
core_depth_state
-> value_delta_code_logits
-> straight-through one-hot value_delta_code
-> typed register update features
-> role-value logits
```

Add a trajectory credit path:

```text
core_depth_state[short route]
-> code/value prediction
-> align with core_depth_state[full route] and final verified target
```

Train only the new value-delta state path first:

```text
trainable policy:
  core_value_delta_code_only

freeze:
  core.*
  action-code / transition-state heads
  core_role_value_state_embed/head
  answer_state_loop
```

Acceptance gate:

```text
held-out value accuracy > 184/624
step exact             > 16/256
trace exact            > 0/32
action-code exact      = 32/32

delta-code-off or code-shuffle must drop below full.

loop-depth sweep must show at least one strict improvement:
  depth 8 > depth 1 on held-out value/step metrics
```

Kill criterion:

```text
If finite value-delta code also only ties baseline, stop local patches and
rebuild the value-state path as a TransNAR-style external neural algorithmic
reasoner with cross-attention glue.
```

Kill criterion result:

```text
Triggered. The direct finite value-delta code path is rejected as canonical.
The next candidate must execute typed register transitions rather than attach
another independent code/readout head.
```

## Design Decision

The next architecture is not another latent MLP. It is:

```text
Discrete NAR-style typed execution bottleneck
+ LoopFormer/Ouro/RLTT-style trajectory credit
+ QTRM mandatory recursive core
```

This is the first prior-backed candidate that directly attacks the current
root failure: QTRM has an action trajectory but not an exact value-state
execution trajectory.

## Ranked Candidate Shortlist

### 1. Discrete Value-Delta Executor

Limitation solved:
exact value-state updates are currently continuous readouts and fail held-out
trace exact.

Prior mechanism:
DNAR discrete/continuous flow split; CLRS hint/state supervision; TransNAR
Transformer-to-algorithmic-reasoner glue.

Architecture change:
add finite `value_delta_code` and straight-through typed register features
inside the mandatory recurrent core path.

Minimal prototype:
train `core_value_delta_code_only` from the accepted baseline and evaluate
code-off/code-shuffle.

Reject if:
it ties or loses to 184/624 value accuracy, or code ablation does not drop.

SSOT/KISS/YAGNI/DRY:
one canonical prompt stream, one recurrent state path, one new discrete code,
one eval ablation.

### 2. LoopFormer Shortcut-Consistency Core Training

Limitation solved:
current depth states do not reliably improve as loop budget grows.

Prior mechanism:
LoopFormer time/dt conditioning and shortcut consistency; Ouro learned depth
allocation.

Architecture change:
keep the core architecture but train short and full loop schedules in the same
batch; align short-route predictions to full-route representations/targets.

Minimal prototype:
add a depth-pair loss over existing `core_depth_states` without adding a new
answer channel.

Reject if:
depth 8 still does not beat depth 1 on held-out reasoning, or core-off matches
full.

SSOT/KISS/YAGNI/DRY:
same prompt, same core, no MemoryOS, no duplicated evidence path.

### 3. RLTT/LoopRPT-Style Trajectory Credit

Limitation solved:
final CE does not teach intermediate latent steps what good reasoning states
look like.

Prior mechanism:
RLTT distributes reward across latent thought trajectories; LoopRPT assigns
step-wise reinforcement signals using an EMA/reference teacher and noisy latent
rollouts.

Architecture change:
add a verified process reward for each depth state, starting from supervised
synthetic validators before any teacher-only online distillation.

Minimal prototype:
use executable synthetic tasks to assign per-depth rewards/logit margins.

Reject if:
process reward improves training metrics but not held-out depth sweep.

SSOT/KISS/YAGNI/DRY:
the verifier labels state quality; the model still has one causal answer path.
