# Ouro Causal Talker-Only S080 Reject

Date: 2026-05-07

## Purpose

The S040 causal Talker attempt trained the whole answer-state loop and destroyed
the accepted forced-choice signal. This follow-up tested a narrower falsifier:
freeze the existing accepted answer-halt checkpoint and train only the new
causal Talker parameters.

Canonical runtime path stayed unchanged:

```text
prompt tokens
-> donor hidden states
-> QTRM recursive answer-state loop
-> causal Talker block over answer hidden + latent trajectory summary
-> LM head
-> autoregressive text
```

The Talker remained inside the LM-logit path. It was not a hidden answer solver
or side answer channel.

## Implementation

```text
src/wgram_lm/wgram_model.py
src/wgram_lm/training/train.py
scripts/192_eval_raw_intelligence.py
scripts/196_train_pure_recursive_depth_supervised.py
tests/test_core_halting.py
tests/test_training_checkpoint_init.py
tests/test_pure_recursive_depth_supervised_train_script.py
tests/test_raw_intelligence_eval_script.py
configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_causal_talker_only_s080.yaml
scripts/250_run_qtrm_ouro_causal_talker_only_s080.sh
```

New training policy:

```text
trainable_param_policy: answer_state_loop_talker_only
trainable tensors: 12
trainable params: 3,147,777
```

Training started from:

```text
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_halt_head_s080_from_tail_s020/last.pt
```

The teacher-preservation path used the same accepted checkpoint and added final
logit KL while the teacher ran with the new random Talker disabled.

## Evaluation

Artifacts:

```text
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_causal_talker_only_s080_from_halt_s080/generation_smoke8.jsonl
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_causal_talker_only_s080_from_halt_s080/causal_forced_choice_smoke4.jsonl
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_causal_talker_only_s080_from_halt_s080/causal_forced_choice_smoke4_core4_talker_off.jsonl
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_causal_talker_only_s080_from_halt_s080/generation_smoke8_eval_gate.jsonl
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_causal_talker_only_s080_from_halt_s080/causal_forced_choice_smoke4_eval_gate.jsonl
```

Generation smoke8 with answer halt gate enabled:

```text
donor_only:       0/8
core_off:         0/8
core_steps_4:     0/8
core_steps_8:     0/8
core8_halt_off:   0/8
core8_talker_off: 0/8
```

Causal forced-choice smoke4 with answer halt gate enabled:

```text
donor_only:        0/4
core_off:          0/4
core_steps_4:      4/4
core_steps_8:      4/4
core8_halt_off:    0/4
core8_talker_off:  4/4
```

## Decision

Reject as a promoted renderer checkpoint.

The narrow training preserves the accepted halt-gated latent scoring signal,
but the key ablation shows that the Talker is not the causal source: disabling
the Talker at depth 8 still gives 4/4 when the answer halt gate is enabled.
Disabling the halt gate collapses depth 8 back to 0/4. The new Talker also
fails the main renderer goal because greedy generation remains 0/8 for every
mode.

This is still useful diagnostically:

```text
answer_halt_gate is still the causal forced-choice component
Talker-only does not add causal forced-choice gain over the halt-gated baseline
Talker-only does not repair multi-token local LM-token stability
```

## Next Bottleneck

Do not keep scaling this Talker-only direction as-is. The failure is not just a
small renderer capacity problem; the learned answer-ready state remains good
for sequence scoring under the halt gate but is not locally stable for
autoregressive numeric continuation.

Next candidates, ranked:

```text
1. Keep answer halt-gate eval/runtime as mandatory for forced-choice gates;
   never compare depth-8 checkpoints with the gate accidentally disabled.
2. Donor residual-stream hook/ReFT-style intervention: let QTRM write a bounded
   residual into the donor decoder state rather than replacing the renderer.
3. Broader language-continuation warmup for the Talker before reasoning-specific
   training, with strict donor-KL and talker-off ablation.
```

Promotion gate for the next renderer attempt:

```text
generation smoke8 > 0/8
forced-choice smoke8 does not regress from accepted halt-head baseline
core/talker/depth ablation loses the gain
no retrieval or side answer channel
```
