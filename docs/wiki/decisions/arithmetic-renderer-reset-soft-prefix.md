# Arithmetic Renderer Reset Soft-Prefix

Date: 2026-05-08

Status: accepted L1 renderer reset scaffold, not L3 promotion.

## Why

Mixed-list arithmetic generation stayed at 0 despite:

```text
final-hidden ReFT-lite
donor-internal soft-prefix
scheduled-sampling soft-prefix
post-trained Qwen3.5-2B donor baseline
```

The fastest useful reset is to prove the renderer on a shorter answer surface
before returning to mixed-list multi-step composition.

## Data

Created arithmetic-only subsets:

```text
data/filtered/pure_recursive_reasoning_arith_chain_train64.jsonl
data/eval/pure_recursive_reasoning_arith_chain_heldout4.jsonl
data/eval/pure_recursive_reasoning_arith_chain_heldout18.jsonl
data/eval/pure_recursive_solver_trace_arith_heldout128.jsonl
```

These are generated from existing raw-intelligence JSONL files by filtering
`task_family == "arithmetic_chain"`.

## Donor Baseline

Post-trained donor baseline:

```text
model: Qwen/Qwen3.5-2B
prompt: chat template + numeric_strict
eval: data/eval/pure_recursive_hard_family_heldout200_cases.jsonl
result: 0/8
```

Base donor baseline:

```text
model: Qwen/Qwen3.5-2B-Base
prompt: numeric_strict
eval: data/eval/pure_recursive_hard_family_heldout200_cases.jsonl
result: 0/8
```

Conclusion:

```text
Donor replacement/prompt formatting alone does not solve answer-only arithmetic
generation.
```

## Accepted L1 Smoke: Heldout4

Command family:

```text
scripts/304_train_core_soft_prefix_donor.py
config: configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_s120.yaml
checkpoint: local_eval/qwen35_2b_pure_recursive_transition_joint_dense_terminal_v2_allfamilies_s120_from_oodstress/last.pt
train: arithmetic_chain train64
eval: arithmetic_chain heldout4
steps: 240
core_steps: 4
```

Artifact:

```text
local_eval/research_gate_runner/core_soft_prefix_arith_reset_s240/report.json
```

Result:

```text
decision: accepted_l1_soft_prefix
teacher-forced full:     0.9375
teacher-forced core_off: 0.0625
generation donor:        0/4
generation core_off:     0/4
generation full:         3/4
```

Examples:

```text
417  -> 422   miss
632  -> 632   hit
851  -> 851   hit
1074 -> 1074  hit
```

Interpretation:

```text
This is the first donor-internal renderer experiment where QTRM core state
causally improves greedy generation over donor and core_off.
```

## Broader Check: Heldout18

Artifact:

```text
local_eval/research_gate_runner/core_soft_prefix_arith_reset_s240_eval18/report.json
```

Result:

```text
decision: accepted_l1_soft_prefix
teacher-forced full:     0.5789
teacher-forced core_off: 0.0702
generation donor:        0/18
generation core_off:     0/18
generation full:         2/18
```

Interpretation:

```text
The signal survives a wider held-out set but is not yet strong enough for L2/L3.
```

## Depth Comparison

Depth 8 variant:

```text
artifact: local_eval/research_gate_runner/core_soft_prefix_arith_reset_s240_eval18_depth8/report.json
generation full: 0/18
```

Decision:

```text
Use core_steps=4 for this renderer reset path. More depth is not automatically
better for the current checkpoint and adapter.
```

## Next Fastest Step

```text
1. Keep arithmetic renderer reset as the short-loop gate.
2. Scale from heldout18 to heldout128 only after heldout18 improves above 50%.
3. Try small architecture changes only inside this reset gate:
   - adapter rank/scale sweep
   - prefix token count sweep
   - first-token + full-token mixed loss
   - small scheduled-sampling ratio after teacher-forced warmup
4. Promote back to mixed-list only after arithmetic greedy generation is stable.
```

Promotion rule:

```text
L2: heldout18 full >= 0.50 and core_off/donor stay 0.
L3 candidate: heldout128 full >= 0.50 with core_off drop >= 0.25.
```

## Shortest-Path Follow-Up: Scheduled Sampling

Artifact:

```text
local_eval/research_gate_runner/core_soft_prefix_arith_reset_s480_sched03_eval18/report.json
```

Result:

```text
scheduled_sampling_prob: 0.3
teacher-forced full:     0.6140
teacher-forced core_off: 0.0702
generation donor:        0/18
generation core_off:     1/18
generation full:         4/18
```

Decision:

```text
partial improvement, not L2. This is the best current heldout18 renderer result
but still below the 9/18 promotion target.
```

Rejected variants:

```text
scheduled_sampling_prob: 0.3, warmup_steps: 120
artifact: local_eval/research_gate_runner/core_soft_prefix_arith_reset_s480_sched03_warm120_eval18/report.json
generation full: 3/18
generation full exact: 3/18

scheduled_sampling_prob: 0.6
artifact: local_eval/research_gate_runner/core_soft_prefix_arith_reset_s480_sched06_eval18/report.json
generation full: 0/18

scheduled_sampling_prob: 0.3, steps: 960
artifact: local_eval/research_gate_runner/core_soft_prefix_arith_reset_s960_sched03_eval18/report.json
generation full: 1/18
```

Interpretation:

```text
Rollout pressure helps only when mild. Warm-up improves exact cleanliness but
does not improve total heldout18 hits. Too much or too long destabilizes greedy
numeric rendering.
```

## Rejected Alternative: Donor Layer Residual Hook

The true internal layer hook was implemented and rejected:

```text
docs/wiki/decisions/core-layer-residual-donor-renderer-reject.md
```

Shortest-path conclusion:

```text
Do not spend more runs on broad layer sweeps now. Continue from soft-prefix
scheduled_sampling=0.3 with warm-up scheduling and stricter exact/first-number
telemetry.
```

## L2 Local Threshold: Train-Range Reset

The previous train split was range-skewed:

```text
train64:    arith-chain-100..163, answers mostly 200..600+
heldout18:  arith-chain-000..017, answers 17..122
heldout4:   arith-chain-200..203, answers 417..1074
```

That made the heldout18 failure ambiguous: it mixed renderer failure with a
numeric range shift. A new non-overlapping train split was generated:

```text
data/filtered/pure_recursive_reasoning_arith_chain_train64_start18.jsonl
range: arith-chain-018..081
```

Artifact:

```text
local_eval/research_gate_runner/core_soft_prefix_arith_train64_start18_s480_sched03_eval18/report.json
```

Result:

```text
decision: accepted_l1_soft_prefix
teacher-forced full:     0.8070
teacher-forced core_off: 0.0702
generation donor:        0/18
generation core_off:     1/18
generation full:         9/18
generation exact:        3/18
```

Interpretation:

```text
This is the first heldout18 run to reach the L2 local hit threshold of 50%.
However, it is not a clean L3 renderer solution:

- many successful rows emit the correct first answer number plus trailing
  symbols or prompt fragments;
- core_off has one loose hit;
- exact answer-only generation is still 3/18.
```

Therefore record it as:

```text
accepted L2-local first-answer-number scaffold
rejected strict answer-only renderer
not L3 canonical renderer
```

Shorter decoding did not solve the strict renderer:

```text
artifact:
  local_eval/research_gate_runner/core_soft_prefix_arith_train64_start18_s480_sched03_maxnew4_eval18/report.json

generation full: 6/18
generation exact: 1/18
```

Next bottleneck:

```text
Turn first-answer-number correctness into strict answer-only stopping without
using an external answer extractor.
```

## L2 Local Strict Answer-Only Gate: EOS Target

A minimal canonical-path stopping fix was added to the soft-prefix trainer:

```text
scripts/304_train_core_soft_prefix_donor.py --append-eos-target
```

This appends the donor EOS token after the answer target, so the frozen donor
LM path can learn to stop after the answer instead of emitting prompt fragments
or math symbols.

Artifact:

```text
local_eval/research_gate_runner/core_soft_prefix_arith_train64_start18_s480_eos_nosched_eval18/report.json
```

Result:

```text
train: data/filtered/pure_recursive_reasoning_arith_chain_train64_start18.jsonl
eval:  data/eval/pure_recursive_reasoning_arith_chain_heldout18.jsonl
scheduled_sampling_prob: 0.0
append_eos_target: true

teacher-forced full:     0.8933
teacher-forced core_off: 0.2000

generation donor:        0/18 exact 0/18
generation core_off:     1/18 exact 0/18
generation full:        10/18 exact 10/18
```

Examples:

```text
17  -> 17
32  -> 32
51  -> 51
42  -> 42
77  -> 77
106 -> 106
105 -> 105
50  -> 50
```

Decision:

```text
accepted L2-local strict answer-only arithmetic renderer scaffold
not L3 canonical renderer
```

Why not L3:

```text
- this is arithmetic-only heldout18;
- the train/eval numeric range was deliberately repaired;
- heldout128 and mixed-list generation are not yet passed;
- the component is still a soft-prefix adapter around a frozen donor, not the
  final integrated QTRM renderer path.
```

Next promotion gate:

```text
heldout128 strict exact >= 0.50
donor exact = 0
core_off exact <= 0.10
then retry mixed-list primitive generation
```

## Larger-Scope Rejection: Range Generalization

The L2-local strict result does not generalize to larger or interleaved
arithmetic ranges.

Adjacent heldout128:

```text
train:
  data/filtered/pure_recursive_reasoning_arith_chain_train64_start18.jsonl

eval:
  data/eval/pure_recursive_reasoning_arith_chain_heldout128_start82.jsonl

artifact:
  local_eval/research_gate_runner/core_soft_prefix_arith_train64_start18_s480_eos_nosched_eval128_start82/report.json

generation donor:    0/128 exact 0/128
generation core_off: 0/128 exact 0/128
generation full:     0/128 exact 0/128
```

Interleaved same-range split:

```text
builder:
  scripts/307_build_arithmetic_interleaved_range_split.py

train:
  data/filtered/pure_recursive_reasoning_arith_chain_interleaved_train64_start18.jsonl

eval:
  data/eval/pure_recursive_reasoning_arith_chain_interleaved_eval64_start18.jsonl

artifact:
  local_eval/research_gate_runner/core_soft_prefix_arith_interleaved64_start18_s480_eos_nosched_eval64/report.json

generation donor:    0/64 exact 0/64
generation core_off: 0/64 exact 0/64
generation full:     0/64 exact 0/64
```

Decision:

```text
reject L3 promotion.
The soft-prefix renderer can be trained into a narrow heldout18 answer-only
scaffold, but it has not learned a range-general arithmetic renderer.
```

New bottleneck:

```text
The core/renderer must expose algorithmic state, not only a memorized numeric
soft prompt. Return to the primitive/core-state path and make the final answer
renderer condition on explicit latent transition/value state before broad
generation claims.
```

## Follow-Up: Explicit State-Conditioned Prefix Rejected

The direct follow-up was run in
`docs/wiki/decisions/state-conditioned-soft-prefix-reject.md`.

Result:

```text
core_role_value_state_logits -> soft-prefix -> donor LM

arithmetic smoke:
  full generation exact: 0/4
  state_off teacher-forced > full

mixed native distribution:
  full generation exact: 0/16
  state_off teacher-forced >= full
```

Updated bottleneck:

```text
Do not keep enlarging soft-prefix adapters.
The value-state codec itself must preserve compositional numeric values before
the renderer can be expected to produce range-general answers.
```
