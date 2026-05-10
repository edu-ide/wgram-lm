# Core Carry Mixed-Depth ACT Gate

Date: 2026-05-07

## Question

Does the accepted S080 answer-halt checkpoint support a real mixed-depth ACT
claim, where the model preserves or selects useful recurrent depth across
different reasoning families?

## Setup

Checkpoint:

```text
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_halt_head_s080_from_tail_s020/last.pt
```

Eval config:

```text
configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_halt_head_eval_gate.yaml
```

Eval data:

```text
data/eval/pure_recursive_reasoning_heldout_72.jsonl
```

Scoring:

```text
causal_forced_choice
choice_score_normalization=mean
no retrieval
no MemoryOS
max_cases=8
```

Runner:

```text
scripts/257_run_core_carry_mixed_depth_act_gate.sh
```

## Smoke4 Result

```text
donor_only:                       1/4
core_off:                         0/4
core_steps2:                      2/4
core_steps4:                      2/4
core_steps8:                      2/4
core_halt_carry_steps2:           3/4
core_halt_carry_steps4:           2/4
core_halt_carry_steps8:           2/4
answer_halt_gate_off:             1/4
```

The smoke4 result showed a possible carry signal: `carry_steps2` was the only
mode to solve the first list-transform case.

## Smoke8 Result

```text
donor_only:                       2/8
core_off:                         0/8
core_steps2:                      3/8
core_steps4:                      3/8
core_steps8:                      2/8
core_halt_carry_steps2:           3/8
core_halt_carry_steps4:           3/8
core_halt_carry_steps8:           2/8
answer_halt_gate_off:             2/8
```

Family matrix:

```text
mode,arithmetic_chain,boolean_logic,list_transform,symbolic_binding
donor_only,1/2,1/2,0/2,0/2
core_off,0/2,0/2,0/2,0/2
core_steps2,1/2,1/2,0/2,1/2
core_steps4,1/2,1/2,0/2,1/2
core_steps8,0/2,1/2,0/2,1/2
core_halt_carry_steps2,0/2,1/2,1/2,1/2
core_halt_carry_steps4,1/2,1/2,0/2,1/2
core_halt_carry_steps8,0/2,1/2,0/2,1/2
answer_halt_gate_off,0/2,1/2,0/2,1/2
```

Artifacts:

```text
/mnt/nvme1n1p2/qtrm-eval/core_carry_smoke/mixed_depth_act_eval_gate_causal_fc_smoke4.jsonl
/mnt/nvme1n1p2/qtrm-eval/core_carry_smoke/mixed_depth_act_eval_gate_causal_fc_smoke8.jsonl
```

## Decision

Reject as a promoted mixed-depth ACT checkpoint.

Reasons:

```text
1. No carry mode beats the best fixed-depth modes at smoke8.
2. carry_steps2 transfers one list-transform case from wrong to correct, but
   loses both arithmetic-chain cases and one case in symbolic/boolean families.
3. core_steps8 degrades relative to core_steps2/4, confirming that more
   recurrent compute is not monotonically useful without a learned depth policy.
4. answer_halt_gate_off is still weak, so the answer-halt path remains causal,
   but the current halt policy is not yet a general mixed-depth ACT controller.
```

## Interpretation

Current S080 is still best described as:

```text
fixed useful-depth answer halt behavior on the trained mixed-list arithmetic
surface, not general adaptive computation.
```

The carry harness is useful as a diagnostic. It can preserve fixed-depth-4
behavior and can reveal when continuation drifts, but this checkpoint was not
trained to use carry as a general stateful ACT policy.

## Next Architecture Step

Train a dedicated mixed-depth ACT checkpoint instead of only evaluating carry:

```text
core_halt_loss_mode=q_value
core_halt_q_value_gamma=0.9
core_halt_exploration_prob>0
answer_state_loop_halt_gate_enabled=false during halt-head training
answer_state_loop_halt_gate_enabled=true during eval
training data must mix depth2 and depth4 finality targets
gate on pure_recursive_reasoning_heldout_72, not only dynamic_halt_v3 depth4
```

Promotion requirement:

```text
mixed-depth ACT > best fixed depth
mixed-depth ACT > donor/core_off
halt_gate_off drops
no retrieval or MemoryOS shortcut
```

## Follow-Up

The immediate S160 follow-up is recorded in:

```text
docs/wiki/decisions/mixed-depth-act-s160-results.md
```

Outcome:

```text
answer-loop-only S160: reject as ACT
core-joint CE S160: partial fixed-depth depth8 improvement, reject as ACT
```

The q-value core halt path remains available, but should not be promoted until
the recurrent states themselves are reliable enough that a halt/off ablation can
show a causal accuracy drop.
