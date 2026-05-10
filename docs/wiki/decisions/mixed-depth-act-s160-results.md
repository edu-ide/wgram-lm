# Mixed-Depth ACT S160 Results

Date: 2026-05-07

## Question

Can the accepted S080 recurrent answer checkpoint become a real mixed-depth ACT
controller on `pure_recursive_reasoning_heldout_72`, without retrieval or
MemoryOS shortcuts?

## Loader And Loss Fixes

Before training, the mixed-depth dataset exposed two training bugs:

```text
1. pure_recursive_reasoning_train256_cases.jsonl stores the canonical answer in
   answer_aliases, not chosen/answer. The depth-supervised loader now accepts
   answer_aliases[0] as the canonical answer.
2. the dataset has depth_targets but no transition_finality_targets. Terminal
   depth masks now fall back to depths whose depth_targets value matches the
   final answer.
3. staged choice-margin training now excludes the current staged answer from
   rejected choices, preventing self-negative margins.
```

Runner:

```text
scripts/258_run_qtrm_mixed_depth_act_s160.sh
```

Eval telemetry now records answer-state halt logits so hard-first halt behavior
can be checked separately from argmax depth.

## Run A: Answer-Loop-Only S160

Setup:

```text
init:   local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_halt_head_s080_from_tail_s020/last.pt
train:  data/filtered/pure_recursive_reasoning_train256_cases.jsonl
eval:   data/eval/pure_recursive_reasoning_heldout_72.jsonl
policy: answer_state_loop_only
steps:  160
lr:     5.0e-5
```

Smoke8 result:

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

Halt telemetry on the 32-record probe:

```text
qtrm_core_steps8 choice observations: 28
no positive halt logit:               20
first positive halt step:             step2 on 8 observations
```

Decision:

```text
reject as mixed-depth ACT. The checkpoint remains above donor/core-off on this
smoke, but halt_gate_off matches gate-on. The gate is not producing the gain.
```

Artifacts:

```text
/mnt/nvme1n1p2/qtrm-runs/qwen35_2b_pure_recursive_reasoning_mixed_depth_act_s160_from_s080/mixed_depth_act_causal_forced_choice_smoke8.jsonl
/mnt/nvme1n1p2/qtrm-runs/qwen35_2b_pure_recursive_reasoning_mixed_depth_act_s160_from_s080/mixed_depth_act_halt_telemetry_smoke8.jsonl
```

The rejected answer-loop-only `last.pt` was removed after evaluation to save
disk; the eval JSONL and runner preserve the result.

## Run B: Core Joint CE S160

Setup:

```text
policy:                   core_and_answer_state_loop
steps:                    160
lr:                       2.0e-5
final_logit_ce_weight:    0.05
depth_final_ce_weight:    0.10
terminal_depth_ce_weight: 0.20
halt_ce_weight:           0.75
choice_margin_weight:     0.20
```

Smoke8 result:

```text
donor_only:              2/8
core_off:                0/8
core_steps1:             2/8
core_steps2:             3/8
core_steps4:             3/8
core_steps8:             4/8
halt_gate_off steps8:    4/8
```

Family matrix for `core_steps8`:

```text
arithmetic_chain: 2/2
boolean_logic:    1/2
list_transform:   0/2
symbolic_binding: 1/2
```

Decision:

```text
partial accept as a fixed-depth raw-core improvement smoke, reject as ACT.
Core-joint CE improves depth8 from 3/8 to 4/8 and fixes both arithmetic smoke
cases, but the halt gate is still not causal because gate-off equals gate-on.
```

Artifacts:

```text
/mnt/nvme1n1p2/qtrm-runs/qwen35_2b_pure_recursive_reasoning_mixed_depth_act_core_joint_ce_s160_from_s080/last.pt
/mnt/nvme1n1p2/qtrm-runs/qwen35_2b_pure_recursive_reasoning_mixed_depth_act_core_joint_ce_s160_from_s080/mixed_depth_act_core_joint_ce_causal_fc_smoke8.jsonl
```

## Root Interpretation

The current blocker is not just "halt head needs more steps." The answer-loop
halt gate can learn some depth signal, but the recurrent states are not yet
semantically reliable across families. In particular, list-transform remains
0/2 even after core-joint CE.

The promoted claim must remain narrower:

```text
QTRM core can beat donor/core-off on this smoke and core-joint training can
improve fixed-depth depth8, but mixed-depth ACT is still unproven.
```

## Next Bottleneck

Focus on list-transform and true halt causality:

```text
1. evaluate core-joint CE on a larger held-out slice before promotion;
2. add list-transform-specific failure ledger using forced-choice score gaps;
3. train with process-state targets that make list order/filter/map operations
   visible in the causal LM path;
4. promote ACT only if gate-on beats best fixed depth and halt_gate_off drops.
```

## List Follow-Up

The immediate list-process follow-up rejected auxiliary staged sequence CE:

```text
docs/wiki/decisions/list-transform-process-s096-reject.md
```

Result:

```text
list_transform stayed 0/2
core_steps8 regressed from 4/8 to 3/8
```

This shifts the next candidate from "more process-state auxiliary CE" to a
causal order-preserving list select/map/copy path or a data split correction
gate.
