# List Transform Process S096 Reject

Date: 2026-05-07

## Question

Does adding value-bearing staged sequence supervision for list-transform
intermediate states fix the current list failure without retrieval or MemoryOS?

## Failure Ledger

Baseline checkpoint:

```text
/mnt/nvme1n1p2/qtrm-runs/qwen35_2b_pure_recursive_reasoning_mixed_depth_act_core_joint_ce_s160_from_s080/last.pt
```

Baseline smoke8 list-transform failures at `qtrm_core_steps_8_no_evidence`:

```text
hits: 0/2
by_error:
  filtered_state_selected: 1
  reversed_final_selected: 1
```

Ledger artifacts:

```text
docs/wiki/decisions/list-transform-failure-ledger-smoke8.md
/mnt/nvme1n1p2/qtrm-runs/qwen35_2b_pure_recursive_reasoning_mixed_depth_act_core_joint_ce_s160_from_s080/list_transform_failure_ledger_smoke8.json
```

## Experiment

Runner:

```text
scripts/258_run_qtrm_mixed_depth_act_s160.sh
```

Setup:

```text
init: /mnt/nvme1n1p2/qtrm-runs/qwen35_2b_pure_recursive_reasoning_mixed_depth_act_core_joint_ce_s160_from_s080/last.pt
steps: 96
lr: 1.0e-5
policy: core_and_answer_state_loop
family_repeat: list_transform=6
final_logit_ce_weight: 0.03
depth_final_ce_weight: 0.08
terminal_depth_ce_weight: 0.15
staged_internal_sequence_ce_weight: 0.45
staged_internal_sequence_max_target_tokens: 8
choice_margin_weight: 0.20
answer_state_loop_halt_ce_weight: 0.50
```

The intent was to make depth1/depth2 list states causal enough for the final LM
path:

```text
depth1: filtered even list
depth2: doubled list in original order
```

## Result

Smoke8:

```text
donor_only:              2/8
core_off:                0/8
core_steps1:             2/8
core_steps2:             3/8
core_steps4:             3/8
core_steps8:             3/8
halt_gate_off steps8:    3/8
```

Family matrix for `core_steps8`:

```text
arithmetic_chain: 1/2
boolean_logic:    1/2
list_transform:   0/2
symbolic_binding: 1/2
```

List failure ledger after the run:

```text
hits: 0/2
by_error:
  filtered_state_selected: 1
  reversed_final_selected: 1
```

Artifacts:

```text
/mnt/nvme1n1p2/qtrm-runs/qwen35_2b_pure_recursive_reasoning_list_process_s096_from_core_joint_s160/list_process_causal_fc_smoke8.jsonl
/mnt/nvme1n1p2/qtrm-runs/qwen35_2b_pure_recursive_reasoning_list_process_s096_from_core_joint_s160/list_transform_failure_ledger_smoke8.json
```

The rejected `last.pt` was deleted after evaluation to save disk.

## Decision

Reject.

Reasons:

```text
1. list_transform stays 0/2.
2. the exact same two error classes remain: filtered-state selection and
   reversed-order final selection.
3. overall depth8 regresses from 4/8 to 3/8 relative to core-joint CE S160.
4. halt_gate_off still matches gate-on, so this does not create ACT causality.
```

## Interpretation

Staged internal sequence CE is useful telemetry, but in the current architecture
it is still an auxiliary depth-readout pressure. It does not force the final
answer logits to use an order-preserving list state.

Root hypothesis:

```text
The list operation requires an order-preserving select/map/copy mechanism in the
causal answer path. A side depth-readout can learn or expose intermediate text
without becoming the mechanism used by final token scoring.
```

## Next Candidate

Do not increase this loss blindly. The next falsifiable candidate should be one
of:

```text
1. list-specific causal path contrast:
   correct ordered final list must beat reversed final and filtered state at
   every causal prefix, with held-out low-value and high-value splits.

2. order-preserving pointer/copy bottleneck:
   an internal learned selector writes ordered kept items into the same hidden
   state that produces LM logits; ablate it and require list accuracy to drop.

3. data split correction gate:
   separate train/eval by operation pattern and value range to check whether the
   current low-number list heldout is mostly token-distribution shift rather
   than recursive reasoning failure.
```
