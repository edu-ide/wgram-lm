# Pure Recursive Single-Trace Rebuild S160 Failure Ledger

Date: 2026-05-03

## Failure

The LeWM-free canonical single-trace TRM rebuild does not pass the 72-case raw
recursive reasoning gate.

## Evidence

Run:

```text
config:
  configs/qwen35_2b_4090_pure_recursive_answer_state_loop_causal_prefix_s160.yaml
init:
  /mnt/nvme1n1p2/qtrm-local-checkpoints/metacog_pair_rebuild/no_warmup_rebuilt_s001/last.pt
checkpoint:
  /mnt/nvme1n1p2/qtrm-local-checkpoints/pure_recursive_answer_state_loop_causal_prefix_rebuild_s160/last.pt
eval:
  local_eval/pure_recursive_answer_state_loop_causal_prefix_rebuild_s160_depth_gate_72.jsonl
summary:
  docs/wiki/decisions/pure-recursive-answer-state-loop-causal-prefix-rebuild-s160-depth-gate-72-summary.json
```

Result:

```text
status: rejected
donor_only: 22/72
core_off: 0/72
core1: 19/72
core2: 18/72
core4: 18/72
core8: 18/72
depth outputs identical: 69/72
```

The core is necessary because `core_off` collapses to `0/72`, but depth does
not scale. The deepest core also loses to donor-only by 4 hits.

## Known Limitation Class

Single-trace recurrent state is active but not learning useful iterative
computation. The model has a causal core path, yet most depth modes emit the
same answer.

## Root Architecture Hypothesis

The answer-state loop currently learns a shallow classifier over the prompt and
first recurrent update. It is not forced to encode distinct intermediate
states, such as arithmetic partial sums, symbolic binding updates, or list
transform states.

## Could The Big Structure Be Wrong?

Possibly, but not proven yet.

The current big structure still has a plausible causal path:

```text
prompt tokens -> latent workspace -> mandatory recursive core -> answer-state loop -> logits
```

What is weak is the training pressure. The held-out failure says:

```text
mandatory core exists
but extra recurrent depth is mostly behaviorally inert
```

## Information Path Needed

```text
prompt tokens
-> depth-specific recurrent state update
-> intermediate symbolic target is readable at the matching depth
-> final answer improves with deeper depth
```

## Current Information Path

```text
prompt tokens
-> recursive core state
-> final answer CE and progress margin
```

The current path does not strongly require intermediate states to be different.

## Local Fix Budget Already Spent

- LeWM next-latent auxiliary loss: rejected for semantic reasoning.
- Single-trace TRM answer-state loop: core-causal but rejected on 72-case scale.

## Likely Component

Training objective for recursive depth, not MemoryOS, answer formatting, or
LeWM.

## Alternative Explanations

- The warm-start checkpoint is a rebuild from a metacognitive random-init pair,
  not the old exact accepted S160 checkpoint.
- The current 160-step training may be too short for list transforms and
  arithmetic.
- The first-token target is too weak for list outputs.

## Architecture Candidates

1. Semantic depth pressure:
   add staged internal first-token CE so labelled depths must predict their
   own symbolic targets.

2. All-depth final answer pressure:
   apply CE to every depth state so all recurrent states are answer-readable,
   then rely on progress margin to make the final depth better.

3. Recurrent-state delta objective:
   penalize identical depth outputs or train depth-specific state deltas before
   answer readout.

## Recommended Candidate

Start with candidate 1 because it directly targets the observed failure:
depth states are not semantically differentiated.

## Smallest Next Experiment

Train from the S160 rebuild checkpoint with:

```text
STAGED_INTERNAL_FIRST_TOKEN_CE_WEIGHT=0.5
ALL_DEPTH_CE_WEIGHT=0.10
MAX_CASES=16
CORE_WORLD_MODEL_WEIGHT=0.0
```

Acceptance for the smoke:

```text
core8 must beat donor-only and core1 on 16 held-out cases,
and changed depth outputs should increase materially above 3/72 scaled rate.
```

## Kill Criterion

If staged internal CE still leaves depth outputs mostly identical and core8 does
not beat donor-only, stop local loss tuning and redesign the core as an
explicit recurrent state machine with supervised state variables.

## Follow-Up: Semantic Depth S120 Probe

Run:

```text
init:
  /mnt/nvme1n1p2/qtrm-local-checkpoints/pure_recursive_answer_state_loop_causal_prefix_rebuild_s160/last.pt
checkpoint:
  /mnt/nvme1n1p2/qtrm-local-checkpoints/pure_recursive_answer_state_loop_semantic_depth_s120/last.pt
heldout eval:
  local_eval/pure_recursive_answer_state_loop_semantic_depth_s120_depth_gate_16.jsonl
heldout summary:
  docs/wiki/decisions/pure-recursive-answer-state-loop-semantic-depth-s120-depth-gate-16-summary.json
train-slice eval:
  local_eval/pure_recursive_answer_state_loop_semantic_depth_s120_train_slice_16.jsonl
train-slice summary:
  docs/wiki/decisions/pure-recursive-answer-state-loop-semantic-depth-s120-train-slice-16-summary.json
transition eval:
  local_eval/pure_recursive_answer_state_loop_semantic_depth_s120_symbolic_transition_train16.jsonl
transition summary:
  docs/wiki/decisions/pure-recursive-answer-state-loop-semantic-depth-s120-symbolic-transition-train16-summary.json
```

Held-out result:

```text
status: rejected
donor_only: 5/16
core_off: 0/16
core1/core2/core4/core8: 2/16, 2/16, 2/16, 2/16
depth outputs identical: 16/16
```

Train-distribution slice result:

```text
status: rejected
donor_only: 3/16
core_off: 0/16
core1/core2/core4/core8: 4/16, 4/16, 4/16, 4/16
depth outputs identical: 16/16
```

Symbolic transition train-slice result:

```text
accuracy: 17/64 = 0.2656
depth1/depth2/depth4/depth8: 4/16, 5/16, 4/16, 4/16
arithmetic_chain: 3/16
symbolic_binding: 4/16
boolean_logic: 10/16
list_transform: 0/16
```

Interpretation:

The semantic-depth probe did not satisfy the kill criterion. It slightly beats
donor on the train-distribution slice, but it still has no depth scaling and no
depth-output diversity. The transition gate confirms the core is not yet
learning reliable intermediate state targets.

One additional local probe is still justified before replacing the core:
the previous training used causal-prefix supervision with only the first answer
token. That makes `list_transform` and other multi-token choices structurally
undertrained while forced-choice scoring evaluates the full answer string.

Next bounded probe:

```text
CAUSAL_PREFIX_MAX_TARGET_TOKENS=6
FAMILY_REPEAT=list_transform=4,arithmetic_chain=3,symbolic_binding=2
STAGED_INTERNAL_FIRST_TOKEN_CE_WEIGHT=0.8
CHOICE_MARGIN_WEIGHT=0.3
CORE_WORLD_MODEL_WEIGHT=0.0
```

Acceptance:

```text
On heldout smoke16, core8 must beat donor-only and core1,
depth outputs must change on at least 4/16 cases,
and list_transform must rise above 0/4.
```

Kill criterion:

```text
If this multi-token probe still has identical depth outputs or list_transform
stays at 0, stop local loss tuning and move to an explicit recurrent
state-machine core with supervised state variables.
```

## Follow-Up: Multi-Token Depth S080 Probe

Run:

```text
init:
  /mnt/nvme1n1p2/qtrm-local-checkpoints/pure_recursive_answer_state_loop_semantic_depth_s120/last.pt
checkpoint:
  /mnt/nvme1n1p2/qtrm-local-checkpoints/pure_recursive_answer_state_loop_multitoken_depth_s080/last.pt
heldout eval:
  local_eval/pure_recursive_answer_state_loop_multitoken_depth_s080_depth_gate_16.jsonl
heldout summary:
  docs/wiki/decisions/pure-recursive-answer-state-loop-multitoken-depth-s080-depth-gate-16-summary.json
```

Training changes:

```text
CAUSAL_PREFIX_MAX_TARGET_TOKENS=6
FAMILY_REPEAT=list_transform=4,arithmetic_chain=3,symbolic_binding=2
STAGED_INTERNAL_FIRST_TOKEN_CE_WEIGHT=0.8
CHOICE_MARGIN_WEIGHT=0.3
CORE_WORLD_MODEL_WEIGHT=0.0
```

Held-out result:

```text
status: rejected
donor_only: 5/16
core_off: 0/16
core1/core2/core4/core8: 3/16, 4/16, 4/16, 4/16
depth outputs changed: 1/16
list_transform core8: 0/4
```

Interpretation:

This probe removed the first-token-only defect and produced a small depth
ladder change, but it still failed the actual raw-intelligence objective:
core8 does not beat donor-only, list-transform remains at zero, and depth
outputs are still identical on 15/16 cases.

Decision:

```text
Stop local loss tuning on the current answer-state-loop-only design.
Do not reintroduce LeWM for this failure.
Do not add more answer-format or threshold fixes.
Move to an explicit recurrent transition-state core candidate.
```

Replacement architecture candidate:

```text
prompt token stream
-> frozen donor hidden context
-> latent workspace
-> mandatory recursive core
-> explicit transition-state predictor over core_depth_states
-> recurrent state controller / answer renderer
-> LM logits
```

The key difference is that the intermediate state is no longer only implicit in
answer logits. The model must expose and train a state variable per step, and
ablations must show that removing this state drops both transition-state
accuracy and final-answer accuracy.
