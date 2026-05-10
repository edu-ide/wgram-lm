# Ouro Donor-Guided Adapter S060 Reject

Date: 2026-05-07

## Purpose

After Talker-only failed to repair greedy generation, this falsifier tested the
donor-preserving direction:

```text
prompt tokens
-> Qwen donor hidden/logits
-> QTRM answer-halt recursive state
-> low-rank answer-state LM adapter
-> bounded QTRM delta over donor logits
-> autoregressive text
```

This keeps Qwen as the language renderer. The private QTRM LM head is disabled:

```text
qtrm_logits_scale: 0.0
donor_logits_scale: 1.0
qtrm_residual_clamp: 2.0
trainable_param_policy: answer_state_loop_lm_adapter_only
```

## Implementation

The depth-supervised training script previously did not pass donor logits into
the QTRM model, so donor-preserving configs could not train the actual
`donor_logits + QTRM_delta` path. This was fixed first.

```text
scripts/196_train_pure_recursive_depth_supervised.py
configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_donor_guided_adapter_s060.yaml
scripts/251_run_qtrm_ouro_donor_guided_adapter_s060.sh
tests/test_pure_recursive_depth_supervised_train_script.py
tests/test_training_checkpoint_init.py
```

Training started from:

```text
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_halt_head_s080_from_tail_s020/last.pt
```

Missing keys were expected:

```text
answer_state_loop_lm_adapter_down.weight
answer_state_loop_lm_adapter_up.weight
```

Only the adapter was trainable:

```text
trainable params: 1,990,656
trainable tensors: 2
```

## Evaluation

Artifacts:

```text
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_donor_guided_adapter_s060_from_halt_s080/generation_smoke8.jsonl
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_donor_guided_adapter_s060_from_halt_s080/causal_forced_choice_smoke4.jsonl
```

Generation smoke8:

```text
donor_only:       0/8
core_off:         0/8
core_steps_4:     0/8
core_steps_8:     0/8
delta_off:        0/8
halt_gate_off:    0/8
```

Causal forced-choice smoke4:

```text
donor_only:       0/4
core_off:         0/4
core_steps_4:     0/4
core_steps_8:     0/4
delta_off:        0/4
halt_gate_off:    0/4
```

Representative failure:

```text
expected: 300015
preferred/generated: 100004,100008,100012
```

## Decision

Reject as a promoted checkpoint.

This experiment validated the plumbing fix: donor logits now participate in the
training forward pass. But the low-rank donor-guided adapter did not repair
generation and also destroyed the accepted halt-gated forced-choice signal.

The failure class is now sharper:

```text
The model learns to make intermediate trace tokens locally plausible.
It still does not learn a stable final-answer emission policy.
```

## Next Bottleneck

Do not continue by only increasing adapter rank or steps. The objective is too
weakly final-answer-specific and mean forced-choice still rewards long
intermediate trace strings.

Next candidate:

```text
Final-answer-only renderer gate:
  train a tiny answer-mode/finality-conditioned delta that activates only after
  the halt gate selects a terminal state, with hard negatives built from
  intermediate trace strings and +/- final-answer distractors.
```

Required promotion gate:

```text
generation smoke8 > 0/8
causal forced-choice smoke8 recovers the accepted halt-head baseline
delta_off and halt_gate_off lose the gain
donor_only/core_off remain lower
```
