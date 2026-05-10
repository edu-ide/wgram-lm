# Final Answer Bridge S120

Date: 2026-05-05

Status: rejected.

## Failure

The current best value-state checkpoint still fails the universal LLM causal
path on the mixed list-to-arithmetic heldout set:

```text
LM forced-choice, 32 rows x 6 modes:
  donor_only:   0/32
  core_off:     0/32
  core_steps_1: 0/32
  core_steps_2: 0/32
  core_steps_4: 0/32
  core_steps_8: 0/32
```

The model ranks intermediate state strings such as:

```text
100004,100008,100012
```

above the final scalar answer. That means action-code and role-value probes are
not yet enough; the canonical LM answer path has not learned to render the
final state.

## Hypothesis

Train the answer-state loop directly on final answer tokens while preserving
the accepted dynamic-halt joint transition code:

```text
prompt tokens
-> frozen donor hidden context
-> latent workspace
-> mandatory recursive core
-> transition_state_joint logits
-> answer_state_loop LM logits
-> final answer text
```

No role-value or typed-register answer channel is active in this experiment.

## Config

```text
configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_final_answer_bridge_s120.yaml
```

Initial checkpoint:

```text
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_core_role_value_state_joint_len579_s240_from_joint_s240/last.pt
```

## Training Objective

```text
causal-prefix final answer CE
+ final-depth answer CE
+ transition_state_joint CE
```

The causal-prefix path prevents the core from seeing future answer tokens while
learning the answer renderer.

## Acceptance

Smoke gate:

```text
LM forced-choice core_steps_8 > donor_only
LM forced-choice core_steps_8 > core_off
transition joint/action-code exact remains 32/32 on the 32-row heldout
```

Promote only if the improvement survives causal forced-choice scoring. A
non-causal forced-choice or runtime executor result is diagnostic only.

## Result

Checkpoint:

```text
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_final_answer_bridge_len579_s120_from_len579_s240/last.pt
```

Training signal:

```text
final/depth CE: 9.78 -> 3.75 on logged core_steps=4 batches
transition joint: preserved at 1.0 acc on the final logged core_steps=4 batch
```

Held-out action-code gate:

```text
rows:          32
exact:         32/32
step_acc:       1.0000
finality_acc:   1.0000
halted_exact:  32/32
```

Held-out LM causal forced-choice smoke8:

```text
donor_only:   0/8
core_off:     0/8
core_steps_1: 0/8
core_steps_8: 0/8
```

The model still ranks intermediate state strings under mean-normalized
causal forced-choice. Under sum-normalized rescoring from the same logits,
QTRM modes collapse to `EMPTY`.

## Decision

Reject.

Direct final-answer CE can lower training loss without transferring to the
held-out value range/length gate. This confirms the root bottleneck: the model
does not have a reliable neural value-state transition that the LM answer path
can consume. The action policy is still solved; the value computation is not.

## Next

Do not keep adding answer-renderer CE alone. The next candidate must make the
computed value state causal before rendering:

```text
prompt tokens
-> recurrent core
-> neural value-state transition
-> state-conditioned answer_state_loop
-> LM logits
```

The executor may remain as a teacher/verifier only. Promotion requires
held-out LM answer accuracy to drop under value-state-off/core-off ablations.
