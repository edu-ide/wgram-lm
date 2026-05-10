# Ouro Causal Talker S040 Reject

Date: 2026-05-07

## Purpose

After the future-token auxiliary failed to repair greedy rendering, the next
candidate moved the renderer into the canonical LM path:

```text
prompt tokens
-> donor hidden states
-> QTRM recursive answer-state loop
-> causal Talker block over answer hidden + latent trajectory summary
-> LM head
-> autoregressive text
```

This follows the JEPA-Reasoner/PonderLM-2/latent-lookahead lesson that a latent
reasoner needs a causal text renderer, not only a side auxiliary head.

## Implementation

```text
src/qtrm_mm/config.py
src/qtrm_mm/qtrm_model.py
tests/test_core_halting.py
tests/test_training_checkpoint_init.py
configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_causal_talker_s040.yaml
scripts/249_run_qtrm_ouro_causal_talker_s040.sh
```

New model knobs:

```text
answer_state_loop_talker_enabled
answer_state_loop_talker_layers
answer_state_loop_talker_gate_init_bias
answer_state_loop_talker_gate_min
```

The Talker is not a hidden answer channel. It updates
`answer_state_loop_hidden`, then the normal LM head produces
`answer_state_loop_logits`, which remains the runtime answer path.

## Training

Started from:

```text
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_halt_head_s080_from_tail_s020/last.pt
```

Command:

```text
scripts/249_run_qtrm_ouro_causal_talker_s040.sh
```

Train summary:

```text
steps: 40
trainable policy: answer_state_loop_only
trainable tensors: 35
trainable params: 7,346,692
```

Loss moved, but this was not sufficient:

```text
initial loss: ~5.80
late loss:    ~2.48
```

## Evaluation

Artifacts:

```text
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_causal_talker_s040_from_halt_s080/generation_smoke8.jsonl
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_causal_talker_s040_from_halt_s080/causal_forced_choice_smoke4.jsonl
```

Generation:

```text
donor_only:       0/8
core_off:         0/8
core_steps_4:     0/8
core_steps_8:     0/8
```

Causal forced-choice:

```text
donor_only:       0/4
core_off:         0/4
core_steps_4:     0/4
core_steps_8:     0/4
halt_gate_off:    0/4
```

## Decision

Reject S040 causal Talker training as a promoted checkpoint.

The architecture is cleaner than the auxiliary future-token head because it
preserves the universal LM path, but the short answer-loop-only training does
not learn a usable renderer and destroys the accepted forced-choice signal.

## Next Bottleneck

The renderer still lacks enough language-model-compatible training pressure.
The next candidate should not add another small head. Use one of:

```text
1. full autoregressive teacher-forcing over Talker runtime logits with stronger
   baseline-preservation KL from the accepted halt-head checkpoint;
2. donor residual-stream hook/ReFT-style intervention so QTRM writes into the
   donor decoder state instead of replacing the donor renderer;
3. train Talker from scratch on a broader language continuation set before
   reintroducing the reasoning gate.
```

The most falsifiable next step is candidate 1 if staying inside the current
QTRM code path, or candidate 2 if prioritizing decoder compatibility.
