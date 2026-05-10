# Ouro Finality Selector Zeroshot

Status: rejected probe, 2026-05-06.

## Question

The S80 Ouro recurrent checkpoint learns the transition-state joint trace
perfectly on the 32-case action-code smoke, but its final LM answer path only
reaches 2/8 on the held-out forced-choice smoke. This probe tests whether the
already-correct transition finality signal can causally select a better answer
loop depth without retraining.

## Change

Added an answer-state-loop finality selector:

```text
answer_state_loop_depth_hidden[depth]
+ transition_state_joint_logits[depth]
-> finality score
-> select depth hidden
-> LM head
```

Two modes were checked:

- `soft`: softmax over depth finality scores.
- `hard_first`: choose the first depth whose finality score is positive;
  fall back to max-finality depth if no positive finality exists.

## Artifacts

```text
config:
  configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_finality_selector_zeroshot.yaml
  configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_finality_selector_hardfirst_zeroshot.yaml

checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_recurrent_len579_s080_from_len579_s240/last.pt

eval:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_finality_selector_zeroshot_s080/lm_causal_forced_choice_smoke8.jsonl
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_finality_selector_hardfirst_zeroshot_s080/lm_causal_forced_choice_smoke8.jsonl
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_finality_selector_hardfirst_zeroshot_s080/action_code_eval32.json
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_finality_selector_hardfirst_zeroshot_s080/gate_selection.json
```

## Result

```text
soft selector:
  full:         2/8
  selector off: 2/8

hard_first selector:
  full:         2/8
  selector off: 2/8
  action-code:  32/32

gate selector:
  s080_recurrent_baseline:          accepted smoke baseline
  s080_hardfirst_finality_zeroshot: rejected, ablation_drop_below_min
```

## Decision

Reject as a canonical architecture upgrade.

The finality selector preserves the learned transition controller, but it does
not increase answer accuracy and has no ablation drop against selector-off.
This means finality selection is not the missing causal bridge from the
transition-state trace to answer-token logits.

## Next

Do not add more post-hoc depth selectors first. The next falsifiable candidate
should make the answer-state loop read the transition code/state at every
recurrent step, so the answer hidden state is computed from the learned
transition trace rather than selected after the fact.
