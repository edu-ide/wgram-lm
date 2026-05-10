# Core Carry ACT Smoke4

Date: 2026-05-07

## Question

Does the new `QTRMCoreCarry` eval harness improve raw recursive reasoning, or
does it damage the accepted answer-halt baseline?

## Setup

Checkpoint:

```text
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_halt_head_s080_from_tail_s020/last.pt
```

Correct eval config:

```text
configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_halt_head_eval_gate.yaml
```

Important correction:

```text
Do not use the S080 training config for this checkpoint gate.
The accepted checkpoint requires answer_state_loop_halt_gate_enabled=true at
eval time.
```

Scoring:

```text
causal_forced_choice
choice_score_normalization=mean
max_cases=4
no retrieval
no MemoryOS
```

## Results

Depth-8 carry comparison:

```text
donor_only_no_evidence:                         0/4
qtrm_core_off_no_evidence:                      0/4
qtrm_core_steps_8_no_evidence:                  4/4
qtrm_core_steps_8_answer_halt_gate_off:         0/4
qtrm_core_halt_carry_steps_8_no_evidence:       2/4
```

Depth-4 carry comparison:

```text
qtrm_core_steps_4_no_evidence:                  4/4
qtrm_core_halt_carry_steps_4_no_evidence:       4/4
```

Artifacts:

```text
/mnt/nvme1n1p2/qtrm-eval/core_carry_smoke/carry_vs_nocarry_eval_gate_causal_fc_smoke4.jsonl
/mnt/nvme1n1p2/qtrm-eval/core_carry_smoke/carry_vs_nocarry_eval_gate_causal_fc_smoke4_depth4.jsonl
```

## Interpretation

Accepted:

```text
The carry harness is wired and can preserve the accepted raw-recursive answer
path when the recurrent depth matches the learned useful depth.
```

Rejected:

```text
Depth-8 carry is not promoted. It over-continues the state across token-prefix
forwards and partially drifts to the pre-subtraction intermediate answer.
```

Confirmed:

```text
The accepted S080 checkpoint's causal path is still the answer halt gate.
Turning answer_halt_gate off collapses the same smoke to 0/4.
```

## Next Gate

The next real ACT claim must evaluate variable-depth tasks:

```text
qtrm_core_steps_4_no_evidence
qtrm_core_steps_8_no_evidence
qtrm_core_halt_carry_steps_4_no_evidence
qtrm_core_halt_carry_steps_8_no_evidence
answer_halt_gate_off
```

Promotion requires carry/ACT to choose or preserve the useful depth across
mixed-depth tasks, not merely match a fixed depth-4 smoke.
