# Ouro Donor-Guided Hard-Negative S080 Reject

Date: 2026-05-07

## Purpose

After the final-only donor-guided adapter failed, this run tested whether
token-local hard negatives could make the Qwen donor renderer choose final
answers instead of intermediate trace strings.

The runtime path stayed universal and autoregressive:

```text
prompt tokens -> Qwen donor logits
prompt tokens -> QTRM answer-state loop -> low-rank LM-adapter delta
final logits = donor_logits + bounded QTRM delta
```

No MemoryOS or retrieval context was used.

## Implementation

```text
scripts/253_run_qtrm_ouro_donor_guided_hardneg_s080.sh
configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_donor_guided_adapter_s060.yaml
```

Training used final-only causal-prefix supervision plus hard negatives:

```text
depth_steps: 8
target_mode: final
choice_margin_mode: sequence
choice negatives: row.choices excluding accepted aliases
tail negatives: preterminal trace strings
subtract-tail counterfactuals: final +/- 1 and related numeric near misses
trainable_param_policy: answer_state_loop_lm_adapter_only
qtrm_logits_scale: 0.0
donor_logits_scale: 1.0
```

## Evaluation

Artifacts:

```text
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_donor_guided_hardneg_s080_from_halt_s080/generation_smoke8.jsonl
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_donor_guided_hardneg_s080_from_halt_s080/causal_forced_choice_smoke4.jsonl
```

Generation smoke8:

```text
donor_only:       0/8
core_off:         0/8
core_steps_8:     0/8
delta_off:        0/8
halt_gate_off:    0/8
```

Causal forced-choice smoke4:

```text
donor_only:       0/4
core_off:         0/4
core_steps_8:     0/4
delta_off:        0/4
halt_gate_off:    0/4
```

Representative core8 generation failures:

```text
expected 300015 -> completion 100002
expected 300015 -> completion 50001
expected 400037 -> completion 100000
expected 400037 -> completion 50004
```

Gold-token rank probe on the same 4 held-out cases:

```text
accepted halt-head baseline:
  core8 first_unique@1: 4/4
  core8 all<=10:        0/4
  halt_gate_off all<=10: 0/4

hard-negative adapter:
  donor_only all<=10:   3/4
  core_off all<=10:     3/4
  delta_off all<=10:    3/4
  core8 all<=10:        1/4
  halt_gate_off all<=10: 3/4
```

Rank artifacts:

```text
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_halt_head_s080_from_tail_s020/gold_token_rank_probe4_eval_gate_20260507.jsonl
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_donor_guided_hardneg_s080_from_halt_s080/gold_token_rank_probe4_20260507.jsonl
```

## Decision

Reject.

Hard negatives are not enough when the only trainable renderer component is a
low-rank adapter over the answer-state hidden. The model stops preferring the
old exact intermediate strings in some cases, but it still falls back to
prompt/list-local numeric tokens rather than computing and rendering the final
answer.

## Interpretation

The current bottleneck is deeper than final-answer-vs-trace discrimination.
The answer-state hidden is not renderer-ready for autoregressive numeric
generation under this adapter-only policy.

The rank probe is the decisive diagnostic: accepted halt-head core8 can make
the first answer token unique top-1, but it does not keep the whole answer
sequence in the top-10. The hard-negative adapter then removes even that
first-token causal signal and makes full-sequence rank readiness worse than
donor/delta-off.

Do not continue blind sweeps of:

```text
adapter rank
adapter learning rate
hard-negative weights
short step count
```

The next falsifier should test renderer-readiness directly before training:

```text
same-prefix gold-token rank probe at each answer position
answer-state hidden -> frozen donor unembedding score
compare core8 vs core_off vs halt_gate_off
promote only if core8 improves gold-token rank before any adapter tuning
```

If rank readiness is still absent, the next architecture must train an
in-loop answer-token state, not a post-hoc LM-adapter delta.
