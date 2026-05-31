# Algorithmic Value-State S480

Date: 2026-05-05

## Question

Can the accepted mixed-composition latent action controller be extended with a
structured value-state path that learns algorithmic intermediate values without
damaging action-code behavior?

## Architecture Tested

Added a separate factorized recurrent value-state path:

```text
accepted QTRM core/action trajectory
  -> factorized value slots
  -> kind head: list/scalar
  -> generic slot head: relative offsets / coefficient / residual
```

The target intentionally avoids raw digit strings:

```text
depth 1 list:   50002,50004,50006       -> kind=list,   slots=[2,4,6]
depth 2 list:   100004,100008,100012    -> kind=list,   slots=[3,7,11]
depth 3 scalar: 300024                  -> kind=scalar, slots=[7,19]
depth 4 scalar: 300015                  -> kind=scalar, slots=[7,10]
```

## Artifacts

```text
shared target/scorer:
  src/wgram_lm/algorithmic_value_state.py

model:
  src/wgram_lm/wgram_model.py

training:
  scripts/196_train_pure_recursive_depth_supervised.py

evaluator:
  scripts/238_eval_qtrm_algorithmic_value_state.py

config:
  configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_algorithmic_value_state_s120.yaml

mixed-only train slice:
  data/filtered/pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_train40000_v0to5_mixed_only.jsonl
```

## Checkpoints

```text
pad-including CE:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_algorithmic_value_state_s480_from_mixed_s720/last.pt

content-slot CE:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_algorithmic_value_state_content_s480_from_mixed_s720/last.pt
```

## Results

Pad-including CE learned the trivial all-pad solution:

```text
held-out rows:          32
kind accuracy:          1.0000
slot accuracy:          0.6875
content slot accuracy:  0.0000
step exact:             0/256
trace exact:            0/32

transition-state off:
  kind accuracy:         0.0000
  content slot accuracy: 0.0000
```

Content-slot-only CE removed the pad shortcut but still failed value content:

```text
held-out rows:          32
kind accuracy:          1.0000
content slot accuracy:  0.0500
step exact:             0/256
trace exact:            0/32

action-code preservation:
  exact:                 32/32
  halted exact:          32/32
```

## Decision

Reject as a value-bearing state architecture.

The path preserves the accepted action controller and learns the coarse
list/scalar phase, but it does not learn exact numeric content. The generic
slot head collapses to repeated per-step guesses or pad-like shortcuts instead
of binding slot role, prompt number, and transition operation.

## Root Cause

The current factorized value-state path asks one generic slot vocabulary to
represent multiple different semantics:

```text
list element offsets
doubled element offsets
scalar coefficient
scalar residual
final residual after subtraction
padding
```

This violates the neural-algorithmic state contract. The model can classify
phase from the action trajectory, but it is not forced to bind each field to a
specific state variable.

## Next Candidate

Replace generic value slots with typed algorithmic fields:

```text
kind head
list_offset_heads[slot_i]
doubled_offset_heads[slot_i]
scalar_coeff_head
scalar_residual_head
final_residual_head
valid_mask/finality head
```

Acceptance gate:

```text
held-out len7/9 value trace exact > 0/32
content slot accuracy clearly above majority baseline
action-code exact remains 32/32
field/head-off ablation drops the value metric
```

Kill criterion:

If typed fields still give 0/32 trace exact after a short run, stop training
readout heads and move the value transition into the recurrent core state
itself rather than adding more probes.
