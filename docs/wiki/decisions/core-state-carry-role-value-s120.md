# Core State-Carry Role-Value S120

Date: 2026-05-05

## Question

Can a dedicated recurrent carry update on the core role-value tokens improve
held-out value-state reasoning without damaging the accepted latent action-code
policy?

## Failure Ledger

Current accepted baseline:

```text
checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_core_role_value_state_joint_len579_s240_from_joint_s240/last.pt

held-out len7/9 mixed composition, 32 cases:
  value accuracy: 184/624 = 0.2949
  step exact:      16/256 = 0.0625
  trace exact:      0/32

action-code preservation:
  exact:      32/32
  step acc:  256/256
```

Failure class:

```text
Role-value tokens are inside the mandatory recurrent core, but increasing
recurrent depth still does not create a robust value-state trace. Side heads can
read partial value information, while full exact traces remain 0/32.
```

Hypothesis tested:

```text
Add an explicit gated recurrent update on the role-value token slice:

previous role-value latent tokens
  -> state_carry_update
  -> gated residual update
  -> next recurrent core step
  -> core_role_value_state_logits
```

This is stricter than a side auxiliary because the updated role tokens are fed
back into both `z_h` and `z_l` inside the recurrent core.

Acceptance gate:

```text
value accuracy > 184/624
step exact     > 16/256
trace exact    > 0/32
action-code exact must not regress from 32/32
```

## Implementation

Added disabled-by-default config fields:

```text
core_state_carry_enabled
core_state_carry_hidden_dim
core_state_carry_gate_init_bias
core_state_carry_gate_min
```

Added recurrent-core telemetry:

```text
core_state_carry_gate_mean
```

Added eval ablation:

```text
scripts/238_eval_qtrm_algorithmic_value_state.py --disable-core-state-carry
```

Added trainable policy:

```text
trainable_param_policy: core_state_carry_only
```

The policy trains only:

```text
core.state_carry_norm.*
core.state_carry_update.*
core.state_carry_gate.*
```

## Experiment A: Core And Role-Value State Fine-Tune

Config:

```text
configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_core_role_value_state_carry_joint_s120.yaml
```

Training:

```text
init:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_core_role_value_state_joint_len579_s240_from_joint_s240/last.pt

out:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_core_role_value_state_carry_len579_s120_from_len579_s240/last.pt

steps: 120
lr: 1e-5
trainable policy: core_and_role_value_state
role-value CE: 1.0
transition-state joint CE: 1.0
```

Held-out result:

```text
full:
  value accuracy: 80/624 = 0.1282
  step exact:      0/256
  trace exact:     0/32

disable core_state_carry at eval:
  value accuracy: 80/624 = 0.1282
  step exact:      0/256
  trace exact:     0/32
```

Interpretation:

```text
The regression is not caused only by the active carry path at inference. The
120-step update damaged shared core/role-value/head parameters.
```

Decision: reject.

## Experiment B: Carry-Only Fine-Tune

Config:

```text
configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_core_state_carry_only_s120.yaml
```

Training:

```text
init:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_core_role_value_state_joint_len579_s240_from_joint_s240/last.pt

out:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_core_state_carry_only_len579_s120_from_len579_s240/last.pt

steps: 120
lr: 1e-4
trainable policy: core_state_carry_only
trainable params: 526,337
role-value CE: 1.0
transition-state joint CE: 1.0
```

Held-out value-state result:

```text
value accuracy: 158/624 = 0.2532
step exact:      16/256 = 0.0625
trace exact:      0/32
```

Action-code preservation:

```text
exact:         32/32
step accuracy: 256/256 = 1.0
finality acc:  256/256 = 1.0
halted exact:  32/32
```

Interpretation:

```text
Freezing the accepted baseline avoids catastrophic damage and preserves the
action-code policy, but this carry update still fails the value-state promotion
gate because it does not beat 184/624.
```

Decision: reject as canonical; keep as a probe-only recurrent update mechanism.

## Architecture Consequence

The useful lesson is not "add more recurrent modules." The useful lesson is:

```text
Accepted action code path:
  stable and causally usable.

Current value path:
  partial role-value readout, not a reliable recursive state machine.

State-carry MLP:
  too unconstrained; it can perturb role tokens but does not impose the exact
  symbolic/value transition needed by the task.
```

Next architecture should keep the KISS/SSOT constraint:

```text
one canonical prompt/donor stream
mandatory recurrent core
explicit typed transition state
small trainable delta module only if baseline-preserving
promotion only by held-out value + step + trace gates
```

Next candidates:

1. Supervise a compact recurrent value delta code instead of direct value slots.
2. Train only a value-delta adapter while freezing the accepted action-code path.
3. Require a state-transition ablation where zeroing the value-delta state
   lowers held-out value accuracy, not just side-head logits.

