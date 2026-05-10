# Ouro Selective Context S020

Date: 2026-05-06

Status: implemented probe; S020 and dense-alignment S020 rejected as causal
architecture gain.

## Prior

This is the first minimal implementation of the SubQ/SSA-style selective
context router plan.

The safe interpretation was:

```text
answer hidden state
+ core/workspace trajectory states
+ prompt hidden states
-> learned top-k selector
-> answer-state loop cross-attention
-> recurrent answer block
-> LM logits
```

This keeps the universal LLM causal path. It is not MemoryOS, RAG, a hidden
solver, or an external answer channel.

## Implementation

Added:

```text
answer_state_loop_selective_context_enabled
answer_state_loop_selective_context_top_k
disable_answer_state_loop_selective_context
force_answer_state_loop_dense_context
qtrm_core_steps_N_answer_selective_context_off_no_evidence
```

The router computes a learned query from the current answer hidden state,
scores candidate core/workspace plus prompt states, keeps top-k states, and
feeds only those states to the answer-state cross-attention.

Config:

```text
configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_selective_context_s080.yaml
```

Training:

```text
init:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_recurrent_len579_s080_from_len579_s240/last.pt

out:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_selective_context_len579_s020_from_s080/step_000020.pt

policy:
  answer_state_loop_only

steps:
  20
```

## Result

Training CE moved down on logged batches:

```text
final_path_ce: 4.3327 -> 2.0460
```

Held-out action-code was preserved:

```text
rows:          32
exact:         32/32
step_acc:       1.0000
finality_acc:   1.0000
halted_exact:  32/32
```

LM causal forced-choice smoke8:

```text
full:       2/8
router_off: 2/8
```

Ablation-aware selector:

```text
S80 baseline:
  full:            2/8
  recurrent_off:   0/8
  ablation_drop:   2
  accepted:        true

selective_s020:
  full:            2/8
  router_off:      2/8
  ablation_drop:   0
  accepted:        false
```

Report:

```text
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_selective_context_len579_s020_from_s080/checkpoint_gate_selection_with_ablation.json
```

## Dense-Alignment V2

The first failure hypothesis was that answer CE alone did not teach the router
what to select. V2 added a dense teacher path:

```text
sparse router path:
  top-k(state + prompt) -> answer loop -> LM logits

dense teacher path:
  all(state + prompt) -> answer loop -> LM logits

loss:
  answer CE + 0.5 * KL(sparse final logits || dense final logits)
```

Implementation:

```text
--answer-selective-context-alignment-weight
--answer-selective-context-alignment-temperature
answer_selective_context_alignment_loss
```

Result:

```text
checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_selective_context_align_len579_s020_from_s080/step_000020.pt

training:
  final_path_ce: 4.3567 -> 2.0881 on logged batches
  answer_selective_context_alignment_kl: 0.0228 -> 0.0198

action-code heldout:
  rows:          32
  exact:         32/32
  step_acc:       1.0000
  finality_acc:   1.0000
  halted_exact:  32/32

LM causal forced-choice smoke8:
  full:       2/8
  router_off: 2/8
```

Ablation-aware selector:

```text
S80 baseline:
  full:            2/8
  recurrent_off:   0/8
  ablation_drop:   2
  accepted:        true

selective_align_s020:
  full:            2/8
  router_off:      2/8
  ablation_drop:   0
  accepted:        false
```

Report:

```text
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_selective_context_align_len579_s020_from_s080/checkpoint_gate_selection_with_ablation.json
```

## Decision

Reject S020 as a causal architecture improvement. It preserves the S80 score
and action-code controller, but the selective router itself is not doing useful
work yet because disabling it does not reduce held-out answer accuracy.

Reject dense-alignment S020 for the same reason. The alignment loss is wired and
the action controller is preserved, but router-off still ties full mode.

Keep the implementation as a probe scaffold. Do not promote it to canonical
until a router-off ablation drop appears on held-out LM causal forced-choice.

## Next

```text
1. Keep S80 Ouro recurrent as the canonical baseline.
2. Use ablation-aware checkpoint selection for all future architecture probes.
3. Do not accept equal-score probes unless the new component has a causal
   ablation drop.
4. If continuing selective routing, do not spend more on short S020 CE/KL. The
   next router attempt needs explicit router supervision, a longer validation
   sweep with snapshot selection, or a different root answer-value path.
```
