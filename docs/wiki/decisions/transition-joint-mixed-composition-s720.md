# Transition Joint Mixed Composition S720

Date: 2026-05-05

## Claim

After accepting within-family list length/value transfer, test the next harder
Stage 1 gate: list state must feed an arithmetic composition step.

The task is:

```text
list -> keep evens -> double kept values -> sum doubled values -> subtract offset
```

This is still a synthetic primitive gate, not open-ended reasoning. It tests
whether the recurrent transition-state path can compose two primitive families
instead of stopping after the list transform.

## Architecture Fix

The previous `terminal_v2` codebook encoded terminality into action names. That
is too rigid for composition, because the same list-compose action can be final
in a pure list task but non-final inside a mixed task.

Implemented a dynamic-halt codebook:

```text
dynamic_halt_v3:
  0 extract_or_unary_transform
  1 compose_from_previous
  2 aggregate_from_previous
  3 final_compose_from_previous
  4 hold_final

finality:
  supervised separately by answer_match
```

The accepted mixed trace is:

```text
codes:    0, 1, 2, 3, 4, 4, 4, 4
finality: 0, 0, 0, 1, 1, 1, 1, 1
```

## Artifacts

```text
builder:
  scripts/235_build_mixed_family_composition_gate.py

codebook:
  scripts/229_build_pure_recursive_latent_action_codebook_cases.py

dense target support:
  scripts/232_build_dense_transition_targets.py

config:
  configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_s120.yaml

train:
  data/filtered/pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_balanced_interleave_train40000_v0to5.jsonl

eval:
  data/eval/pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_eval50000_v6to7_len7_9.jsonl

checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_balanced_interleave_s720_from_s240/last.pt

evals:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_balanced_interleave_s720_from_s240/eval_mixed_composition_full.json
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_balanced_interleave_s720_from_s240/eval_mixed_composition_transition_off.json
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_balanced_interleave_s720_from_s240/eval_mixed_composition_code_shuffle.json
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_balanced_interleave_s720_from_s240/eval_mixed_composition_code_dropout_to_hold.json
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_balanced_interleave_s720_from_s240/eval_mixed_composition_train_only_full.json
```

## Split

```text
train:
  rows: 480
  families:
    arithmetic_chain
    boolean_logic
    list_transform
    mixed_list_arithmetic
    symbolic_binding
  mixed rows: 240
  train list variants: 0, 1, 2, 3, 4, 5
  train list length: 5

eval:
  rows: 32
  family:
    mixed_list_arithmetic
  eval list variants: 6, 7
  eval list lengths: 7, 9
  value range:
    starts at 50001+
```

Important builder fix:

```text
Training script consumes rows deterministically.
Putting all mixed rows after primitive rows meant S120/S240 did not see the
mixed rows soon enough. The accepted builder interleaves primitive and mixed
rows so small training budgets actually exercise the composition examples.
```

## Result

```text
full held-out:
  step_acc:      1.0000
  finality_acc:  1.0000
  exact:         32/32
  halted exact:  32/32

transition-state-off:
  step_acc:      0.1250
  finality_acc:  0.3750
  exact:         0/32
  halted exact:  0/32

code shuffle, swap code 1 and 2:
  step_acc:      0.7500
  finality_acc:  1.0000
  exact:         0/32
  halted exact:  0/32

code dropout to hold code 4:
  step_acc:      0.5000
  finality_acc:  1.0000
  exact:         0/32
  halted exact:  0/32

train mixed only:
  step_acc:      1.0000
  finality_acc:  1.0000
  exact:         240/240
  halted exact:  240/240
```

## Decision

Accept as a Stage 1 mixed-family composition gate:

```text
The dense joint transition-state path can learn a dynamic-halt recurrent trace
where a list intermediate state continues into an arithmetic aggregation and
final subtraction. The accepted held-out result depends causally on the
transition-state path and action-code semantics.
```

Boundary:

```text
This is still a synthetic latent-action gate with supervised trace targets.
It does not prove open-ended neural state-transition content prediction,
unseen operation invention, natural-language answer generation, or ASI.
```

## Next Gate

Promote only after one of these passes:

```text
1. Add mixed-family tasks with two different composition orders.
2. Add longer arithmetic/list chains and held-out composition depth.
3. Replace action-code trace supervision with neural state-content prediction.
4. Show that core depth improves answer/state quality, not only code accuracy.
```
