# Factorized Value-State S480

Date: 2026-05-05

Status: rejected.

## Claim

After value-readout-only failed, test a minimal state-factorized candidate:
separate value slots are recurrently updated from prompt context and the frozen
accepted action-state trajectory.

This is inspired by:

```text
Neural Algorithmic Reasoning: recurrent latent processor with intermediate state.
Dreamer/RSSM: separate recurrent/action/value latent signals.
Factored Latent Action World Models: factorized state/action transition.
```

## Implementation

New config:

```text
configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_factorized_value_state_s120.yaml
```

New model path:

```text
factorized_value_state_init
factorized_value_state_step_embed
factorized_value_state_action_proj
factorized_value_state_prompt_cross
factorized_value_state_update
factorized_value_state_head
```

The evaluator reuses:

```text
transition_value_state_logits
```

so the same value-state held-out gate applies.

## Training

```text
init checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_balanced_interleave_s720_from_s240/last.pt

output checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_factorized_value_state_s480_from_mixed_s720/last.pt

trainable:
  factorized_value_state_* only
```

## Held-Out Result

Value-state:

```text
rows:          32
trace exact:    0/32
step exact:     0.0000
token acc:      0.3713
```

Action-code preservation:

```text
rows:          32
trace exact:   32/32
halted exact:  32/32
step acc:       1.0000
finality acc:   1.0000
```

## Decision

Reject.

The factorized value-state path preserves the accepted action controller, but
it still does not recover exact intermediate value states.

Interpretation:

```text
The failure is no longer action-policy contamination.
The failure is value representation/training: string-level digit sequence CE is
not creating exact algorithmic state.
```

## Next

Move from free-form digit sequence targets to structured neural-algorithmic
state targets:

```text
list slots:       element_0..element_N plus mask
accumulator slot: integer value or bounded categorical/scalar representation
operator slot:    current operation/finality
transition loss:  state_t -> state_{t+1}, not only decode(state_t)
causal gate:      disabling structured value_state must reduce final success
```

Kill criterion:

```text
If structured slots still preserve action 32/32 but remain value exact 0/32,
replace the current QTRM value path with a processor-style Neural Algorithmic
Reasoning core.
```
