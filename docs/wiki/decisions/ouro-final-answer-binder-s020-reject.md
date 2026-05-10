# Ouro Final-Answer Binder S020 Reject

Status: rejected probe, 2026-05-06.

## Question

The accepted tail-negative checkpoint still has this failure:

```text
full core8:
  correct_final:     4
  pre_subtract_sum:  4
```

The model learns the transition/action trace, but the normal LM answer path
often emits a preterminal value. This probe tested whether a finality-weighted
answer binder can connect the final latent transition to answer logits.

## Implementation

Added an ablatable final-answer binder:

```text
transition_state_joint_logits
-> finality weights over core depths
-> final answer delta
-> answer_state_loop hidden
-> LM logits
```

Runtime ablation:

```text
qtrm_core_steps_8_transition_final_answer_binder_off_no_evidence
```

Two content variants were tested:

```text
v1: finality-weighted joint-code probability embedding
v2: finality-weighted core_depth_state embedding
```

The v2 state version is architecturally better because the answer delta reads
the latent core state, not only the low-bandwidth code/finality class.

## Artifacts

```text
config:
  configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_final_answer_binder_s020.yaml

v1 checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_final_answer_binder_s020_from_tail_s020/last.pt

v1 eval:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_final_answer_binder_s020_from_tail_s020/lm_causal_forced_choice_smoke8.jsonl
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_final_answer_binder_s020_from_tail_s020/tail_error_summary.json

v2 checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_final_state_binder_s020_from_tail_s020/last.pt

v2 eval:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_final_state_binder_s020_from_tail_s020/lm_causal_forced_choice_smoke8.jsonl
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_final_state_binder_s020_from_tail_s020/tail_error_summary.json
```

## Result

v1:

```text
donor_only:        0/8
core_off:          0/8
full core8:        2/8
binder_off:        2/8
joint_bridge_off:  2/8

full tail:
  correct_final:     2
  doubled_list:      4
  pre_subtract_sum:  2
```

v2:

```text
donor_only:        0/8
core_off:          0/8
full core8:        2/8
binder_off:        2/8
joint_bridge_off:  2/8

full tail:
  correct_final:     2
  doubled_list:      4
  pre_subtract_sum:  2

binder_off tail:
  correct_final:     2
  doubled_list:      3
  pre_subtract_sum:  3
```

The accepted tail-negative baseline remains stronger:

```text
tail-negative baseline full core8: 4/8
```

## Decision

Reject both binders as canonical architecture upgrades.

The binder path is wired and ablatable in unit tests, but on the held-out smoke
it has no positive causal delta. More importantly, training with a randomly
initialized answer delta perturbs the answer path and regresses from the
accepted 4/8 baseline to 2/8. This confirms that the next improvement should
not be another answer-side patch.

## Research Update

Current LoopLM/latent-reasoning prior points away from post-hoc answer binders:

```text
Ouro: recurrent latent computation is built into pretraining, with learned
depth allocation.

Looped Transformers: gains come from effective recurrent depth and latent
thoughts, not from a final readout adapter alone.

COCONUT: latent reasoning uses hidden-state recurrence plus staged CoT-to-latent
training.

LoopRPT/LoopFormer: recent variants emphasize latent-step credit assignment,
variable-depth training, and shortcut consistency.
```

References:

- <https://arxiv.org/abs/2510.25741>
- <https://arxiv.org/abs/2502.17416>
- <https://openreview.net/pdf?id=Itxz7S4Ip3>
- <https://papers.cool/arxiv/2603.19714>
- <https://huggingface.co/papers/2602.11451>

## Next

Stop trying answer-binder-only fixes first. The next falsifiable candidate
should train the latent trajectory itself:

```text
1. freeze the accepted answer path as a baseline-preserving control;
2. add variable-depth / shortcut-consistency loss so shorter and longer loops
   stay aligned instead of drifting;
3. assign process credit directly to latent steps, not only final answer tokens;
4. keep promotion gate strict: full core8 must beat 4/8 and binder/bridge/core
   ablations must drop.
```
