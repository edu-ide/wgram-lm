# Ouro Answer Recurrent Rebuild S080 Reject

Date: 2026-05-08

## Question

After the short-term typed-state L2 retry failed, can we quickly rebuild the
earlier Ouro/LoopLM-style recurrent answer checkpoint from the remaining stable
checkpoint?

Target level:

```text
L2 local reproduction gate
```

Required signal:

```text
action-code exact remains 32/32
full core8 causal-forced-choice > donor_only
full core8 causal-forced-choice > core_off
full core8 causal-forced-choice > answer_state_recurrent_off
```

## Run

Init checkpoint:

```text
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_core_role_value_state_joint_len579_s240_from_joint_s240/last.pt
```

Output checkpoint:

```text
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_recurrent_rebuild_s080_from_len579_s240/last.pt
```

Training:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src \
  .venv/bin/python scripts/196_train_pure_recursive_depth_supervised.py \
  --config configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_recurrent_s080.yaml \
  --data-jsonl data/filtered/pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_len579_train40000_v0to5.jsonl \
  --init-checkpoint local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_core_role_value_state_joint_len579_s240_from_joint_s240/last.pt \
  --steps 80 \
  --lr 2.0e-5 \
  --depth-steps 1,2,4,8 \
  --target-mode staged \
  --final-logit-ce-weight 1.0 \
  --depth-final-ce-weight 1.0 \
  --all-depth-ce-weight 0.0 \
  --progress-margin-weight 0.0
```

Training CE improved, but this is not the acceptance metric:

```text
final_path_ce roughly 10.0 -> 3.6 on logged depth4 batches
```

## Results

Action-code preservation passed:

```text
rows:          32
exact:         32/32
step_acc:       1.0000
finality_acc:   1.0000
halted_exact:  32/32
```

Causal forced-choice smoke8 rejected:

```text
donor_only:                         0/8
core_off:                           0/8
core_steps_8:                       0/8
core_steps_8 answer_recurrent_off:  0/8
```

Completion audit:

```text
donor_only:
  list-like distractors such as 100004,100008,100012

core_off:
  forced-choice tie on all 8 cases

full core8:
  list-like distractors and near-final distractors such as 400040

answer_recurrent_off:
  list-like distractors
```

## Decision

Reject this rebuild as L2 progress.

The action controller is still intact, but the recurrent answer path did not
reproduce the earlier held-out causal answer gain. This means the prior S80
signal is not currently stable enough to use as the canonical L2 base.

## Slow Variant Note

A causal-prefix S80 variant was started with:

```text
--causal-prefix-supervision
--causal-prefix-max-target-tokens 8
--causal-prefix-later-token-weight 0.65
```

It was stopped after 4/80 steps because each step took roughly 18-21 seconds.
That path may still be relevant, but it should be run only through a
validation-gated script that saves short checkpoints and evaluates early.

## Next Constraint

Do not keep accepting lower training CE as progress.

The next recurrent-answer experiment must use validation-gated checkpoint
selection:

```text
train short slices
save step checkpoints
run smoke4 or smoke8 forced-choice on each checkpoint
keep only checkpoints where full > donor/core_off/recurrent_off
stop immediately when the validation gate regresses
```

This is now a checkpoint-selection and objective-alignment problem, not a
plain "train longer" problem.
