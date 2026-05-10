# Transition Joint Mixed Composition Len1113 S080

Date: 2026-05-07

Status: accepted Stage 1 length-transfer extension.

## Claim

The accepted dynamic-halt mixed-composition checkpoint should degrade
gracefully when the held-out mixed list-to-arithmetic task moves beyond list
lengths 7/9 to lengths 11/13.

This remains a synthetic latent-action gate. It does not prove open-ended
natural-language reasoning or autoregressive answer generation.

## Probe Split

Builder:

```text
scripts/235_build_mixed_family_composition_gate.py
```

Generated artifacts:

```text
train:
  data/filtered/pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_len1113_probe_train40000_v0to5.jsonl

eval:
  data/eval/pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_len1113_probe_eval60000_v6to7_len11_13.jsonl

summary:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_balanced_interleave_s720_from_s240/mixed_composition_len1113_probe_summary.json
```

Split:

```text
train:
  list variants: 0,1,2,3,4,5
  list lengths:  5,7,9
  rows:          384

eval:
  list variants: 6,7
  list lengths:  11,13
  rows:           32
  value range:    60000+
```

## Baseline S720 Probe

Checkpoint:

```text
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_balanced_interleave_s720_from_s240/last.pt
```

Result on length 11/13:

```text
trace exact:    32/32
step acc:       1.0000
finality acc:   0.9922
halted exact:  30/32
```

Failure:

```text
Two length-13 variant 6/7 cases predicted the correct action-code trace but
marked depth 2 as terminal. This is a halt/finality transfer failure, not an
action-code failure.
```

## Joint-Only Recovery

Training intentionally used only transition-state joint CE. Final-answer CE and
depth-answer CE were disabled.

Checkpoint:

```text
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_len1113_jointonly_s080_from_s720/last.pt
```

Training:

```text
init:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_balanced_interleave_s720_from_s240/last.pt

steps:
  80

loss:
  --transition-state-joint-ce-weight 1.0
  --final-logit-ce-weight 0.0
  --depth-final-ce-weight 0.0
  --all-depth-ce-weight 0.0
  --progress-margin-weight 0.0
```

Note:

```text
An earlier attempt omitted --depth-final-ce-weight 0.0 and was stopped as an
invalid contaminated run before any checkpoint was saved.
```

## Results

Length 11/13 eval:

```text
full:
  trace exact:    32/32
  step acc:       1.0000
  finality acc:   1.0000
  halted exact:  32/32

transition-state-off:
  trace exact:     0/32
  step acc:        0.1250
  finality acc:    0.3750
  halted exact:   0/32

code shuffle, swap code 1 and 2:
  trace exact:     0/32
  step acc:        0.7500
  finality acc:    1.0000
  halted exact:   0/32

code dropout to hold code 4:
  trace exact:     0/32
  step acc:        0.5000
  finality acc:    1.0000
  halted exact:   0/32
```

Canonical length 7/9 regression check:

```text
full:
  trace exact:    32/32
  step acc:       1.0000
  finality acc:   1.0000
  halted exact:  32/32
```

## Decision

Accept as a Stage 1 length-transfer extension.

The result shows that the dynamic-halt transition-state path can be extended
from held-out lengths 7/9 to 11/13 without regressing the previous gate, and
that the result is still causally dependent on transition state and action-code
semantics.

Boundary:

```text
This is still supervised latent-action execution. It is not Stage 2 neural
state-content prediction and not a general LM answer-generation result.
```

## Next

```text
1. Add a second mixed composition order, such as arithmetic -> list decision.
2. Add held-out composition depth greater than four transitions.
3. Start Stage 2 neural transition-state content prediction only after the
   mixed-action code path remains stable under those gates.
```
