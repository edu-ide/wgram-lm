# Ouro Donor-Guided Adapter Final-Only S060 Reject

Date: 2026-05-07

## Purpose

The first donor-guided adapter run used staged targets, so the adapter could
learn intermediate trace strings as if they were answer targets. This follow-up
kept the same donor-renderer architecture but trained only final-answer tokens:

```text
depth_steps: 8
target_mode: final
qtrm_logits_scale: 0.0
donor_logits_scale: 1.0
trainable_param_policy: answer_state_loop_lm_adapter_only
```

## Implementation

```text
scripts/252_run_qtrm_ouro_donor_guided_adapter_final_s060.sh
```

Training started from the accepted answer-halt checkpoint and used the same
low-rank adapter config:

```text
configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_donor_guided_adapter_s060.yaml
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_halt_head_s080_from_tail_s020/last.pt
```

## Evaluation

Artifacts:

```text
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_donor_guided_adapter_final_s060_from_halt_s080/generation_smoke8.jsonl
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_donor_guided_adapter_final_s060_from_halt_s080/causal_forced_choice_smoke4.jsonl
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

## Decision

Reject.

Final-only supervision removes the obvious staged-target contamination, but it
does not recover the accepted halt-gated forced-choice signal and does not open
greedy generation.

## Interpretation

The root bottleneck is no longer just target contamination. A low-rank delta on
top of donor logits is too weak or too unconstrained to convert the latent
answer-ready state into stable final-answer tokens on these numeric tasks.

The next architecture should not be another adapter-rank or step-count sweep.
It needs a different causal target:

```text
token-local final-answer discriminator / scorer
trained with hard negatives:
  intermediate trace strings
  near numeric final answers
  empty answer
then use it either as:
  a reranking/verifier probe, or
  a differentiable contrastive loss on the donor-guided final logits
```

Promotion still requires greedy generation above 0 and ablation loss under
delta-off/halt-off/core-off.
