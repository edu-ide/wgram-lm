# L3 Blocked By Value-State Gate

Date: 2026-05-08

## Status

L3 is blocked. The previous Ouro recurrent L2 acceptance cannot be used as a
promotion base because the raw-intelligence eval script was loading
trainable-only checkpoints incorrectly.

## Root Cause

`scripts/192_eval_raw_intelligence.py` loaded checkpoint tensors directly and
ignored `base_checkpoint` metadata. For trainable-only delta checkpoints, that
left the non-trainable base path randomly initialized.

Fix:

```text
scripts/192_eval_raw_intelligence.py now uses load_initial_checkpoint(...)
tests/test_raw_intelligence_eval_script.py guards against direct torch.load
```

Verification:

```bash
PYTHONPATH=src .venv/bin/python -m unittest \
  tests.test_raw_intelligence_eval_script \
  tests.test_ouro_validation_runner
```

Result:

```text
4 tests OK
```

## Corrected Ouro Result

Re-evaluated:

```text
local_eval/research_gate_runner/ouro_answer_recurrent_validation_gated_seed0_s40_eval4_fixedload
```

Corrected decision:

```text
rejected
```

Held-out causal forced-choice:

```text
step_000010 full=0/4
step_000020 full=0/4
step_000030 full=0/4
step_000040 full=0/4
last       full=0/4
donor=0/4
core_off=0/4
recurrent_off=0/4
```

Action-code eval still reaches `32/32`, but that only proves operation/finality
control. It does not prove exact numeric value-state or final LM answer
generation.

## Renderer Retry

A minimal recurrent LM adapter was tested:

```text
configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_recurrent_lm_adapter_s160.yaml
local_eval/research_gate_runner/ouro_recurrent_lm_adapter_renderer_s64_seed0
```

Result:

```text
donor_only: 0/4
core_off:   0/4
full:       0/4
recurrent_off: 0/4
```

Decision:

```text
rejected
```

Interpretation: renderer-only repair is the wrong next bottleneck. The latent
state is not carrying the exact value needed for the final answer.

## Typed Register Retry

Ran the prior recommended candidate:

```text
operation CE
+ typed-register transition CE
+ role-value CE
```

Run:

```text
local_eval/research_gate_runner/typed_register_operation_transition_value_s160_seed0
```

Original typed-register readout:

```text
full: 136/624 value accuracy
off:  184/624 value accuracy
step exact full: 0/256
step exact off:  16/256
trace exact: 0/32
```

Decision:

```text
rejected
```

Transition-readout variant:

```text
configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_typed_register_executor_transition_readout_s160.yaml
```

Result:

```text
full value accuracy: 205/624 = 0.3285
off value accuracy:  184/624 = 0.2949

full step exact: 5/256
off step exact:  16/256

trace exact: 0/32
```

Decision:

```text
partial, not L2
```

This is a useful causal value-accuracy signal, but it fails the L2 promotion
contract because full step exact is worse than the ablated baseline and trace
exact remains zero.

## Current Bottleneck

The current bottleneck is:

```text
exact value-state transition, not renderer
```

The model can learn operation/finality codes and can improve per-role value
accuracy in one variant, but it does not yet keep the whole role vector correct
at each recurrent step.

## Next Shortest Path

Do not proceed to L3 until L2 passes again.

Tested L2 candidate:

```text
transition-readout typed register
+ operation CE
+ transition CE
+ role-value CE
+ row-level step margin
```

The row-level margin was added in:

```text
scripts/196_train_pure_recursive_depth_supervised.py
--core-typed-register-step-margin-weight
--core-typed-register-step-margin
```

Initial result:

```text
run: local_eval/research_gate_runner/typed_register_transition_readout_step_margin_s120_seed0
best preserved: best_partial.pt -> step_000120.pt

full value accuracy: 218/624
off value accuracy:  184/624

full step exact: 26/256
off step exact:  16/256

trace exact: 0/32
action-code exact: 32/32
```

Decision:

```text
partial, not L2
```

The row-level margin produced the first corrected value-state gain where full
beats the typed-register-off baseline on both per-value accuracy and step exact,
while preserving action-code exactness. It still fails promotion because no
held-out row has a fully exact recurrent trace.

## Prompt-Extract Step-Margin Result

The next shortest variant enabled prompt-derived role-value extraction before
the recurrent typed-register executor:

```text
config:
configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_typed_register_transition_readout_prompt_stepmargin_s160.yaml

run:
local_eval/research_gate_runner/typed_register_prompt_stepmargin_s160_seed0

best preserved:
best_partial.pt -> step_000120.pt
```

Held-out result:

```text
full value accuracy: 229/624
off value accuracy:  167/624

full step exact: 34/256
off step exact:  0/256

trace exact: 0/32
action-code exact: 32/32
```

Decision:

```text
partial, not L2
```

This is the strongest corrected partial value-state signal so far. It proves
the mandatory recursive typed-register path can affect held-out state accuracy
causally, but it still does not solve whole-trace exactness.

## Prompt-CE / Parity / Self-Condition / Mandatory-Update Results

Direct prompt role-value supervision was added to test whether the failure was
mostly prompt-state initialization:

```text
core_role_value_state_prompt_logits
--core-role-value-prompt-ce-weight
```

Run:

```text
local_eval/research_gate_runner/typed_register_prompt_ce_s120_seed0
```

Held-out result:

```text
full value accuracy: 226/624
off value accuracy:  166/624

full step exact: 35/256
off step exact:  0/256

trace exact: 0/32
action-code exact: 32/32
```

Direct prompt-state eval:

```text
baseline prompt state: 0/120 values, 0/32 rows
prompt-CE state:      64/120 values, 16/32 rows
```

Decision:

```text
partial, not L2
```

Interpretation: prompt-state initialization improved, but only for half the
held-out rows. The recurrent trace remains non-exact.

Three follow-up architecture hypotheses were tested and rejected:

```text
1. prompt parity bottleneck
   config: configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_typed_register_transition_readout_prompt_parity_s120.yaml
   run:    local_eval/research_gate_runner/typed_register_prompt_parity_s120_seed0
   best:   205/624 values, 29/256 steps, 0/32 trace
   direct prompt state: 56/120 values, 0/32 rows

2. prompt value-state self-conditioning
   config: configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_typed_register_transition_readout_prompt_self_condition_s120.yaml
   run:    local_eval/research_gate_runner/typed_register_prompt_self_condition_s120_seed0
   best:   202/624 values, 26/256 steps, 0/32 trace
   direct prompt state: 56/120 values, 0/32 rows

3. mandatory typed-register update gate
   config: configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_typed_register_transition_readout_prompt_mandatory_update_s120.yaml
   run:    local_eval/research_gate_runner/typed_register_prompt_mandatory_update_s120_seed0
   best:   202/624 values, 22/256 steps, 0/32 trace
```

Decision:

```text
all rejected
```

These results matter because they rule out three tempting shortcuts:

```text
parity-only binding is too narrow and unstable
soft self-conditioning does not preserve exact state
forcing the typed-register gate open is not enough
```

At this point the strongest value-accuracy partial was:

```text
local_eval/research_gate_runner/typed_register_prompt_stepmargin_s160_seed0/best_partial.pt
```

## Updated Promotion Gate

L2 must still pass all of this before L3 work is legitimate:

```text
full value accuracy > 184/624
full step exact    > 16/256
trace exact        > 0/32
typed-register-off < full on value accuracy and step exact
action-code exact  = 32/32
```

The current best passes the first, second, fourth, and fifth checks, but fails
`trace exact > 0/32`.

## Next Architecture Reset

The remaining bottleneck is not action selection and not the final renderer. It
is prompt-bound exact state update:

```text
prompt-derived problem fields
-> recurrent typed-register state
-> exact per-step value vector
```

The next architecture-level reset should replace the current soft
operation-conditioned register update with a stricter learned state codec where
operation, source roles, destination roles, scalar constants, parity/base
bindings, and updated values are separate internal fields, all derived from the
same token stream and all feeding the causal LM path.

Before another integrated QTRM training run, run an oracle-initial-state
separation test:

```text
gold prompt state -> recurrent typed-register executor -> trace exact
```

If oracle initial state fails, the recurrent executor/state codec is the blocker.
If oracle initial state passes, the prompt-derived state extractor is the
blocker. This split is now required before claiming another L2 attempt.

## Oracle-Prefix Separation Diagnostic

Added reusable diagnostic:

```text
scripts/310_analyze_value_trace_oracle_prefix.py
tests/test_value_trace_oracle_prefix.py
```

This is an L0 diagnostic. It does not alter the model. It asks how many leading
predicted value-state steps must be replaced with gold targets before an already
computed recurrent trace becomes exact.

Canonical partial:

```text
input:
local_eval/research_gate_runner/typed_register_prompt_stepmargin_s160_seed0/value_eval_step_000120_full.json

output:
local_eval/research_gate_runner/typed_register_prompt_stepmargin_s160_seed0/oracle_prefix_value_eval_step_000120_full.json
```

Result:

```text
raw trace exact: 0/32
oracle first 1: 0/32
oracle first 2: 0/32
oracle first 3: 0/32
oracle first 4: 2/32
oracle first 8: 32/32

min prefix histogram:
  4 steps: 2 rows
  8 steps: 30 rows
```

Other rejected variants:

```text
prompt-CE:        first1=0, first4=2, histogram={2:2, 8:30}
prompt parity:    first1=0, first4=2, histogram={4:2, 8:30}
self-condition:  first1=0, first4=2, histogram={2:2, 8:30}
mandatory update: first1=0, first4=0, histogram={8:32}
```

## Value-Feedback Codec Result

The next smallest architecture change made the typed-register state feed back
its own predicted value distribution through a learned value embedding:

```text
registers -> value logits -> soft value embedding -> gated register feedback
```

Implementation:

```text
core_typed_register_value_feedback_enabled
core_typed_register_value_feedback_gate_init_bias
core_typed_register_value_feedback_gate_min
```

Config:

```text
configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_typed_register_value_feedback_s120.yaml
```

Run:

```text
local_eval/research_gate_runner/typed_register_value_feedback_s120_seed0
```

Best checkpoint:

```text
step_000120.pt
best_partial.pt
```

Held-out result:

```text
full value accuracy: 226/624
off value accuracy:  163/624

full step exact: 42/256
off step exact:   0/256

trace exact: 0/32
```

Oracle-prefix diagnostic:

```text
raw trace exact: 0/32
oracle first 1: 0/32
oracle first 2: 0/32
oracle first 3: 2/32
oracle first 4: 2/32
oracle first 8: 32/32

min prefix histogram:
  3 steps: 2 rows
  8 steps: 30 rows
```

Decision:

```text
partial, not L2
```

This is the best corrected step-exact partial so far, improving step exact from
`34/256` to `42/256` while preserving a strong ablation drop. It still fails
the L2 contract because no held-out row has a fully exact recurrent trace.

Interpretation:

```text
The blocker is not only prompt initialization.
Most rows need every value-state step replaced by oracle gold to recover.
Therefore the integrated QTRM typed-register transition path is not yet a
reliable recurrent state machine.
```

Next action:

```text
Reset to the accepted L1 transition-table/direct-CE probe and port that
transition mechanism back into QTRM as a stricter state codec, instead of adding
more prompt heads or soft gates.
```

## Trace-Margin / Template-Codec Follow-Up

Trace-level margin was added after the row-level step margin to prevent one bad
role from being hidden by average CE:

```text
scripts/196_train_pure_recursive_depth_supervised.py
--core-typed-register-trace-margin-weight
--core-typed-register-trace-margin
```

Run:

```text
local_eval/research_gate_runner/role_value_template_trace_margin_s220_seed0
```

Best held-out result:

```text
last.pt
full value accuracy: 220/624
full step exact:     42/256
trace exact:          0/32
```

Decision:

```text
partial, not L2
```

The step-exact score tied the value-feedback partial, but trace exact remained
zero. Error concentration moved to the scalar coeff/residual roles, especially
role 9.

## Factorized Template Codec Result

The next attempt replaced the flat template classifier with a factorized
template code:

```text
template_id = length_slot * 14 + base_parity * 7 + offset_slot
```

Implementation:

```text
core_role_value_template_factorized_enabled
core_role_value_template_length_classes
core_role_value_template_parity_classes
core_role_value_template_offset_classes
```

Config:

```text
configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_role_value_template_factorized_s160.yaml
```

Run:

```text
local_eval/research_gate_runner/role_value_template_factorized_s360_seed0
```

Best held-out result:

```text
step_000180.pt
full value accuracy: 226/624
off value accuracy:  167/624
full step exact:      22/256
off step exact:        0/256
trace exact:           0/32
```

Decision:

```text
rejected for L2
```

The factorized codec proves a causal full-vs-off value gain, but it is weaker
than the existing prompt-stepmargin and value-feedback partials. The factorized
template classifier did not become a reliable prompt-to-state code under this
short training budget.

## Scalar-Role CE Reweight Result

Because the held-out failures were concentrated in the last two role-value
fields, the core typed-register CE received a scalar-role multiplier:

```text
--core-typed-register-scalar-role-ce-multiplier
```

Run:

```text
local_eval/research_gate_runner/typed_register_scalar_weight_s240_seed0
```

Stopped early after 120 steps because the loss became unstable.

Held-out result:

```text
step_000040: values  48/624, steps  0/256, rows 0/32
step_000080: values  60/624, steps  0/256, rows 0/32
step_000120: values 158/624, steps 10/256, rows 0/32
```

Decision:

```text
rejected
```

Interpretation:

```text
Simple scalar-role reweighting is not the missing architecture.
It can move in-batch metrics, but it destabilizes held-out recurrent state.
The bottleneck is the transition mechanism, not only role weighting.
```

## Current Shortest Path

The remaining L2 blocker is still:

```text
exact recurrent value-state transition
```

Rejected shortcuts:

```text
more prompt heads
parity-only binding
soft self-conditioning
mandatory update gate
value feedback
flat template table
factorized template code
scalar-role CE reweighting
```

Next candidate should port the state-machine result from
`docs/wiki/decisions/pure-recursive-solver-state-machine-probe-s240.md` into the
QTRM path:

```text
prompt tokens
-> mandatory recursive core
-> transition metadata logits
-> primitive/state-machine transition update
-> explicit value state
-> answer renderer
```

This is not yet a final universal-LLM claim. It is the shortest falsifiable
path to prove that the recurrent core can drive a causal state transition before
returning to broader LLM generation.

## Primitive + Typed Register Selector Result

Question tested:

```text
Can the primitive/state-machine readout be combined with the stronger
typed-register step-coherence path inside the same universal LLM causal path?
```

Implementation:

```text
core_primitive_typed_selector_enabled
core_primitive_typed_selector_init_bias
core_primitive_typed_selector_gate
--core-primitive-typed-selector-bce-weight
```

The selector is internal to QTRM:

```text
prompt/donor hidden
-> recursive core
-> primitive role-value logits + typed-register value logits
-> learned selector gate
-> core_role_value_state_logits
-> answer path
```

It is therefore a valid LLM-compatible scaffold, not an external calculator.

Merge/oracle diagnostic:

```text
typed baseline:                  226/624 values, 42/256 steps
primitive-only on typed prompt:  237/624 values, 11/256 steps
sum blend:                       232/624 values, 24/256 steps
confidence switch:               240/624 values, 32/256 steps
step-confidence switch:          245/624 values, 28/256 steps
oracle per-role source choice:   300/624 values, 50/256 steps
```

Trainable selector runs:

```text
closed selector, bias -8:
  run: primitive_typed_selector_s80_lr1e2_seed5
  best: 226/624 values, 42/256 steps, 0/32 rows

open selector, bias 0:
  run: primitive_typed_selector_open_s80_lr1e2_seed6
  best: 232/624 values, 24/256 steps, 0/32 rows

source-supervised selector BCE:
  run: primitive_typed_selector_bce_s120_lr1e2_seed7
  best: 232/624 values, 24/256 steps, 0/32 rows
```

Decision:

```text
rejected for L2
```

Interpretation:

```text
Primitive logits contain complementary value signal, but they damage recurrent
step coherence when mixed naively. A selector over confidence/margin features
does not recover the oracle source-choice gap.
```

Updated bottleneck:

```text
The blocker is copy-preserving state transition, not only source selection.
The next architecture should make unchanged roles stay copied by construction
and let the recursive core write only the operation-conditioned role deltas.
```

Next shortest candidate:

```text
prompt tokens / donor hidden
-> recursive core
-> operation + destination/source metadata
-> copy-preserving state codec
-> role-delta write
-> value-state logits
-> answer renderer
```

## Copy-Preserving Update-Gate Result

Question tested:

```text
Is the remaining L2 failure mainly that unchanged role values are not copied
across recurrent steps?
```

Implementation:

```text
core_primitive_role_value_update_gate_enabled
core_primitive_role_value_update_gate_init_bias
core_primitive_role_value_update_gate_min
core_primitive_role_value_update_gate
--core-primitive-role-value-update-gate-bce-weight
```

The update gate is still inside the universal LLM causal path:

```text
prompt/donor hidden
-> recursive core
-> primitive value-state logits
-> learned copy/write update gate
-> value-state logits
-> answer path
```

It is a valid scaffold because it does not call an external solver and does not
receive hidden gold answers at inference.

Gate-only run:

```text
run:
local_eval/research_gate_runner/primitive_update_gate_copy_bce_s120_lr1e2_seed8

trainable policy:
core_primitive_role_value_update_gate_only

best primitive-only held-out checkpoint:
step_000030.pt

primitive-only result:
239/624 values, 11/256 steps, 0/32 rows

later checkpoints:
step_000060: 219/624 values,  3/256 steps, 0/32 rows
step_000090: 171/624 values,  3/256 steps, 0/32 rows
step_000120: 112/624 values, 0/256 steps, 0/32 rows
```

Decision:

```text
rejected for L2
```

Interpretation:

```text
The copy/write target is learnable, but gate-only tuning over-closes the state
update and destroys the actual value transition. Copy gating alone is not the
missing state codec.
```

Joint primitive state-machine run:

```text
run:
local_eval/research_gate_runner/primitive_state_machine_copy_bce_s80_lr1e4_seed9

init checkpoint:
local_eval/research_gate_runner/primitive_typed_merge_20260508/typed_prompt.pt

init SHA256:
9286294b0d98e8688588b8f5f07d6e7126c8e6265de25bd59bb18d7465378c2e

trainable policy:
primitive_role_value_state_machine

losses:
core primitive role-value CE
+ primitive role-value step margin
+ primitive role-value update-gate BCE
+ primitive transition operation CE
```

Primitive-only held-out result:

```text
step_000020: 194/624 values,  0/256 steps, 0/32 rows
step_000040: 178/624 values, 48/256 steps, 0/32 rows
step_000060: 168/624 values, 32/256 steps, 0/32 rows
step_000080: 169/624 values, 16/256 steps, 0/32 rows
```

Selected typed+primitive held-out result:

```text
step_000020: 226/624 values, 26/256 steps, 0/32 rows
step_000040: 206/624 values, 18/256 steps, 0/32 rows
step_000060: 216/624 values, 32/256 steps, 0/32 rows
step_000080: 226/624 values, 42/256 steps, 0/32 rows
```

Checkpoint SHA256:

```text
step_000020: f1880fbcfeffcc9d516d9cb7b20ad280b46140dd8b29e035cdcb1b57393d37ea
step_000040: 64d49e81ea9343d8c4311d0555e6e57f436de3be6dd61fe2a7b7751e1e54a356
step_000060: 04e3b9283d2e615da8802ae91c4bdc4c962959510ee73b64c78e6a053b9429cc
step_000080: f8e48f888809318e4fcf21c9eb97c676c939c8bd373912660e4dc74dd8f9bacf
```

Decision:

```text
rejected for L2
```

Interpretation:

```text
The primitive-only path can briefly improve step exactness to 48/256, but only
by losing value accuracy. The selected typed+primitive path falls back to the
typed baseline at 226/624 values and 42/256 steps. No checkpoint reaches
trace_exact > 0/32, and none beats the previous confidence-switch diagnostic
of 245/624 values.
```

Updated bottleneck:

```text
The blocker is not just copying unchanged roles. It is the learned recurrent
state codec itself: the model needs an internal representation that separates
state carry, operation metadata, and role-delta writes before producing
value-state logits.
```

Next replacement candidate:

```text
prompt tokens / donor hidden
-> recursive core
-> explicit latent state codec:
   carry_state, op_code, source_role, dest_role, scalar_delta
-> role-delta writer with copy-by-default residual update
-> value-state logits
-> autoregressive answer path
```

Promotion rule for the replacement:

```text
Do not accept another selector/gate-only improvement.
Pass requires held-out trace_exact > 0/32, value and step exact above the
typed value-feedback partial, and an ablation drop when the state codec is
disabled.
```

## Residual-Delta Codec Result

Question tested:

```text
Does changing the primitive role-value head from absolute next-state prediction
to residual delta prediction make copy-by-default state updates stable enough
to pass L2?
```

Implementation:

```text
core_primitive_role_value_residual_delta_enabled

absolute mode:
next_logits = head(hidden)
next_logits = current_logits + gate * (next_logits - current_logits)

residual-delta mode:
delta_logits = head(hidden)
next_logits = current_logits + gate * delta_logits
```

This preserves the universal LLM causal path:

```text
prompt/donor hidden
-> recursive core
-> primitive operation logits
-> residual-delta state codec
-> value-state logits
-> answer path
```

Regression test:

```text
tests.test_core_halting.CoreHaltingTests
  .test_core_primitive_residual_delta_preserves_state_when_delta_is_zero
```

Run:

```text
config:
configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_primitive_residual_delta_codec_s160.yaml

run:
local_eval/research_gate_runner/primitive_residual_delta_codec_s120_lr5e4_seed10

init:
local_eval/research_gate_runner/primitive_typed_merge_20260508/typed_prompt.pt

trainable policy:
primitive_role_value_state_machine
```

Primitive-only held-out result:

```text
step_000030: 132/624 values, 16/256 steps, 0/32 rows
step_000060: 183/624 values, 15/256 steps, 0/32 rows
step_000090: 193/624 values, 26/256 steps, 0/32 rows
step_000120: 120/624 values, 32/256 steps, 0/32 rows
```

Selected typed+primitive held-out result:

```text
step_000030: 215/624 values, 31/256 steps, 0/32 rows
step_000060: 178/624 values,  8/256 steps, 0/32 rows
step_000090: 168/624 values, 10/256 steps, 0/32 rows
step_000120: 178/624 values, 16/256 steps, 0/32 rows
```

Checkpoint SHA256:

```text
step_000030: c178b76d0bb06522659113c53400c4b3684d4b898bd0ee6aabb00a3776e37906
step_000060: 198bd11e2797d0e0e00d222bdfbb66f421ead9099b197b4d9fad8f6f75124d0c
step_000090: 7434e53bca7d5d60cfb0d150382d0abb30e0e7b6c70379777bdff2eab0944fe9
step_000120: d936be1a73ec0964b172b4867cf5c2b7281194c16a7c9fbba2eebdde0c39d5ff
```

Decision:

```text
rejected for L2
```

Interpretation:

```text
Residual-delta prediction fixes the formal copy-by-default property, but it
does not create a reliable learned state codec. The primitive-only path still
trades value accuracy against step exactness, and the selected path never
beats the typed value-feedback partial. Trace exact remains 0/32.
```

Updated kill rule:

```text
Stop testing scalar gate/selector/delta surface changes. The next useful test
must separate the transition into explicit fields or use an oracle-executor
split to prove whether the failure is the operation-field extraction, the
state codec, or the delta writer.
```

## Operation-Field Separation Check

After the residual-delta rejection, the primitive operation head was evaluated
directly on the same held-out rows:

```text
script:
scripts/230_eval_qtrm_latent_action_codebook.py

prediction source:
primitive

run:
local_eval/research_gate_runner/primitive_residual_delta_codec_s120_lr5e4_seed10
```

Held-out result:

```text
step_000030: action trace 32/32 rows, 256/256 steps, finality 256/256
step_000060: action trace 32/32 rows, 256/256 steps, finality 256/256
step_000090: action trace 32/32 rows, 256/256 steps, finality 256/256
step_000120: action trace 32/32 rows, 256/256 steps, finality 256/256
```

Decision:

```text
operation-field extraction is not the current blocker
```

Interpretation:

```text
The model already predicts the held-out operation/action schedule perfectly.
The remaining L2 failure is specifically the operation-conditioned value-state
transition: state codec + role delta writer.
```

Next architecture candidate is therefore narrower:

```text
keep:
  prompt/donor hidden -> recursive core -> primitive operation logits

replace:
  monolithic primitive value-state MLP

with:
  explicit field factorization:
    current raw-list slots
    current doubled-list slots
    scalar coeff
    scalar residual
    destination field
    delta/value write
  then compile those fields back into role-value logits for the answer path
```

Residual-delta oracle-prefix diagnostic:

```text
step_000030: raw 0/32, oracle first4 0/32, oracle first8 32/32, hist {8:32}
step_000060: raw 0/32, oracle first4 1/32, oracle first8 32/32, hist {4:1,5:1,8:30}
step_000090: raw 0/32, oracle first4 0/32, oracle first8 32/32, hist {8:32}
step_000120: raw 0/32, oracle first4 0/32, oracle first8 32/32, hist {8:32}
```

Decision:

```text
Residual-delta did not reduce prefix dependency. Most rows still require all
8 value-state steps to be replaced by gold targets.
```

This further narrows the next change:

```text
Do not tune the recurrent value-state MLP further.
Build an explicit field compiler / delta-writer diagnostic where operation
metadata is held fixed and only the value-field transition is learned or
verified.
```

Role-value error summary for the best residual-delta primitive checkpoint:

```text
script:
scripts/311_summarize_role_value_state_errors.py

input:
local_eval/research_gate_runner/primitive_residual_delta_codec_s120_lr5e4_seed10/value_eval_primitive_step_000090_full.json

joined rows:
data/eval/pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_len579_eval50000_v6to7_len7_9.jsonl
```

Top role errors:

```text
scalar_residual: 192/192 errors = 1.0000
scalar_coeff:    126/192 errors = 0.6563
doubled_list_1:   17/32  errors = 0.5313
doubled_list_0:   16/32  errors = 0.5000
doubled_list_2:   16/32  errors = 0.5000
raw_list_0:       16/32  errors = 0.5000
raw_list_1:       16/32  errors = 0.5000
raw_list_2:       16/32  errors = 0.5000
```

Top action/step errors:

```text
action 4 hold_final: 212/256 errors = 0.8281
action 2 aggregate:   53/64  errors = 0.8281
action 3 subtract:    53/64  errors = 0.8281
depths 3-8:           53/64  errors per depth = 0.8281
```

Interpretation:

```text
The list extraction phases are partially learned, but the scalar path is
broken. Once the trace reaches aggregate/subtract/hold_final, the wrong scalar
residual is carried through every later recurrent step. The next architecture
should isolate scalar_coeff/scalar_residual as first-class fields rather than
forcing them through the same role-value MLP as list offsets.
```

## Field-Specific Primitive Heads Result

Question tested:

```text
Can separating list-field and scalar-field output heads fix the scalar path
without leaving the universal LLM causal path?
```

Implementation:

```text
core_primitive_role_value_field_specific_heads_enabled

list roles:
  core_primitive_role_value_list_head

scalar roles:
  core_primitive_role_value_scalar_head

fallback/tail roles:
  core_primitive_role_value_head
```

This is still internal to QTRM:

```text
prompt/donor hidden
-> recursive core
-> primitive operation logits
-> residual-delta codec hidden
-> field-specific role-value heads
-> value-state logits
```

Run:

```text
config:
configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_primitive_field_heads_delta_codec_s160.yaml

run:
local_eval/research_gate_runner/primitive_field_heads_delta_codec_s90_lr5e4_seed11
```

Held-out primitive-only result:

```text
step_000030: 209/624 values, 17/256 steps, 0/32 rows
step_000060: 171/624 values,  9/256 steps, 0/32 rows
step_000090: 136/624 values, 16/256 steps, 0/32 rows
```

Top field errors:

```text
step_000030:
  doubled_list_3: 24/24 errors
  scalar_residual: 191/192 errors

step_000060:
  doubled_list_3: 24/24 errors
  scalar_residual: 173/192 errors

step_000090:
  scalar_coeff: 192/192 errors
  scalar_residual: 184/192 errors
```

Decision:

```text
rejected for L2
```

Interpretation:

```text
Field-specific heads slightly reduce scalar_residual errors at one checkpoint,
but they do not improve the recurrent trace and they destabilize other fields.
This rejects "separate scalar/list output heads" as the missing architecture by
itself.
```

Updated bottleneck:

```text
The scalar failure is not only output-head interference. The model must carry
or derive the scalar residual through a causal state variable before the
role-value logits are produced.
```

## Oracle-Role Separation Diagnostic

Added diagnostic:

```text
scripts/312_analyze_value_trace_oracle_roles.py
tests/test_value_trace_oracle_roles.py
```

Purpose:

```text
Ask which role groups would need to be replaced by oracle gold values before
the predicted recurrent trace becomes exact.
```

Results:

```text
primitive_residual_delta step_000090:
  scalar_residual:             1/32 rows recovered
  scalar:                     10/32 rows recovered
  raw_list,doubled_list:       0/32 rows recovered
  raw_list,doubled_list,scalar: 32/32 rows recovered

primitive_field_heads step_000030:
  scalar_residual:             0/32 rows recovered
  scalar:                      8/32 rows recovered
  raw_list,doubled_list:       0/32 rows recovered
  raw_list,doubled_list,scalar: 32/32 rows recovered

primitive_state_machine_copy_bce step_000040:
  scalar_residual:             8/32 rows recovered
  scalar:                     16/32 rows recovered
  raw_list,doubled_list:       0/32 rows recovered
  raw_list,doubled_list,scalar: 32/32 rows recovered

typed_register_value_feedback step_000120:
  scalar_residual:             0/32 rows recovered
  scalar:                      0/32 rows recovered
  raw_list,doubled_list:       0/32 rows recovered
  raw_list,doubled_list,scalar: 32/32 rows recovered
```

Decision:

```text
The bottleneck is not isolated to scalar_residual alone. The trace needs both
list-phase fields and scalar-phase fields correct. Since operation/action
metadata is already 32/32, the missing component is an operation-conditioned
field compiler / delta writer, not another global role-value head.
```

Next candidate:

```text
prompt/donor hidden
-> recursive core
-> primitive operation logits
-> operation-conditioned phase compiler:
   action 0: raw-list extraction fields
   action 1: doubled-list fields
   action 2: scalar aggregate fields
   action 3: scalar subtract fields
   action 4: scalar hold/carry fields
-> role-value logits
-> answer path
```

Promotion rule remains:

```text
This can only be accepted as L2 if held-out trace_exact > 0/32 and ablations
show the operation-conditioned compiler is causal. It is not L3/L4 until the
same state improves the universal LM answer path.
```

## Operation-Specific Phase Compiler Result

Tested candidate:

```text
config:
configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_primitive_operation_phase_compiler_s160.yaml

run:
local_eval/research_gate_runner/primitive_operation_phase_compiler_s90_lr5e4_seed12

init checkpoint:
local_eval/research_gate_runner/primitive_typed_merge_20260508/typed_prompt.pt
```

Training kept the primitive operation path solved:

```text
primitive_transition_operation_acc = 1.0
```

Held-out value-state eval:

```text
step_000030: values=224/624, steps=32/256, trace=0/32
step_000060: values=170/624, steps=26/256, trace=0/32
step_000090: values=133/624, steps=0/256,  trace=0/32
```

Top role errors:

```text
step_000030: scalar_residual=192/192, scalar_coeff=96/192
step_000060: scalar_residual=192/192, scalar_coeff=144/192
step_000090: scalar_residual=192/192, scalar_coeff=186/192
```

Oracle-role diagnostic:

```text
step_000030:
  scalar_residual=8/32
  scalar=16/32
  raw_list,doubled_list=0/32
  raw_list,doubled_list,scalar=32/32
  all=32/32

step_000060:
  scalar_residual=0/32
  scalar=10/32
  raw_list,doubled_list=0/32
  raw_list,doubled_list,scalar=32/32
  all=32/32

step_000090:
  scalar_residual=0/32
  scalar=0/32
  raw_list,doubled_list=0/32
  raw_list,doubled_list,scalar=32/32
  all=32/32
```

Decision:

```text
rejected for L2/L3/L4
```

Interpretation:

```text
Operation-specific output heads do not solve the bottleneck. The operation
router is already correct, but the learned value-state codec still fails to
carry both list-phase fields and scalar-phase fields through the recurrent
trace. The next fix should not be another head variant. It should make the
intermediate field state causal and typed: list fields, scalar fields, and
residual/carry fields must be maintained as explicit learned latent state and
then compiled into role-value logits.
```

## Trace-Margin Rescue Result

To test whether the failure was mostly an optimization objective problem rather
than an architecture problem, the best operation-specific partial checkpoint was
continued with trace-level margin pressure:

```text
init:
local_eval/research_gate_runner/primitive_operation_phase_compiler_s90_lr5e4_seed12/step_000030.pt

run:
local_eval/research_gate_runner/primitive_trace_margin_rescue_s60_from_op30_lr2e4_seed13

loss:
core_primitive_role_value_state_ce_weight = 1.0
core_primitive_role_value_step_margin_weight = 0.5
core_primitive_role_value_trace_margin_weight = 2.0
core_primitive_role_value_update_gate_bce_weight = 1.0
```

Held-out value-state eval:

```text
step_000020: values=106/624, steps=7/256, trace=0/32
step_000040: values=120/624, steps=0/256, trace=0/32
step_000060: values=183/624, steps=2/256, trace=0/32
```

Top role errors:

```text
step_000020: scalar_coeff=192/192, scalar_residual=191/192
step_000040: scalar_coeff=192/192, scalar_residual=191/192
step_000060: scalar_residual=185/192, scalar_coeff=144/192
```

Decision:

```text
rejected for L2/L3/L4
```

Interpretation:

```text
Trace-level loss pressure does not rescue the operation-specific primitive
codec. It worsens the best partial result and keeps trace_exact at 0/32.
This rejects the "just add more exactness loss" path. The remaining move is a
root-architecture change: replace monolithic role-value logits with a causal
typed field state/carry that preserves list fields and scalar fields as state
variables before compiling them to role-value logits.
```

## Typed Primitive-Conditioned Field-State Result

Implemented a root-architecture probe:

```text
typed_algorithmic_value_state_primitive_conditioning_enabled

primitive operation logits
-> typed recurrent field-state
-> kind / raw-list / doubled-list / scalar-coeff / scalar-residual / final-residual logits
```

Code/config:

```text
src/qtrm_mm/config.py
src/qtrm_mm/qtrm_model.py
src/qtrm_mm/training/train.py
configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_typed_primitive_conditioned_field_state_s160.yaml
```

Run:

```text
local_eval/research_gate_runner/typed_primitive_conditioned_field_state_s80_lr4e4_seed14
init: primitive_operation_phase_compiler_s90_lr5e4_seed12/step_000030.pt
```

Held-out typed-field eval:

```text
step_000020 full:          fields=416/928, content=416/880, steps=0/256, trace=0/32
step_000020 recurrent_off: fields=40/928,  content=8/880,   steps=0/256, trace=0/32

step_000040 full:          fields=418/928, content=418/880, steps=0/256, trace=0/32
step_000040 recurrent_off: fields=40/928,  content=8/880,   steps=0/256, trace=0/32

step_000060 full:          fields=369/928, content=369/880, steps=0/256, trace=0/32
step_000060 recurrent_off: fields=63/928,  content=32/880,  steps=0/256, trace=0/32

step_000080 full:          fields=330/928, content=330/880, steps=0/256, trace=0/32
step_000080 recurrent_off: fields=37/928,  content=8/880,   steps=0/256, trace=0/32
```

Best field breakdown for `step_000040 full`:

```text
kind:             192/256
raw_list_offsets: 62/128
doubled_offsets:  64/128
scalar_coeff:     88/192
scalar_residual:  10/192
final_residual:    2/32
```

Primitive-conditioning ablation on the same checkpoint:

```text
primitive_conditioning_off:
fields=419/928, content=419/880, steps=0/256, trace=0/32
```

Decision:

```text
partial L1 scaffold, rejected for L2/L3/L4
```

Interpretation:

```text
The new typed field-state path is causal with respect to its recurrent state:
recurrent_off collapses from ~418/928 to ~40/928 fields. However, primitive
conditioning itself is not yet causal, because turning it off does not reduce
the score. The learned state is still too constant-like and fails exact scalar
residual/final residual fields, so it cannot be promoted.

Next shortest candidate:
keep the recurrent typed field-state path, but add a scalar-residual focused
follow-up where residual/final fields receive stronger causal pressure. If that
still cannot move scalar_residual/final_residual, replace the generic carried
hidden vector with typed scalar/list subregisters instead of one shared carried
state.
```

## Scalar-Residual Focus Follow-Up

Run:

```text
local_eval/research_gate_runner/typed_field_state_scalar_residual_focus_s80_from_step40_seed15
init: typed_primitive_conditioned_field_state_s80_lr4e4_seed14/step_000040.pt

loss:
typed_algorithmic_kind_ce_multiplier = 0.1
typed_algorithmic_list_ce_multiplier = 0.25
typed_algorithmic_scalar_ce_multiplier = 24.0
```

Held-out typed-field eval:

```text
step_000020: fields=380/928, content=380/880, steps=8/256, trace=0/32
step_000040: fields=322/928, content=322/880, steps=0/256, trace=0/32
step_000060: fields=320/928, content=320/880, steps=0/256, trace=0/32
step_000080: fields=368/928, content=368/880, steps=0/256, trace=0/32
```

Best field breakdown for `step_000020`:

```text
kind:             192/256
raw_list_offsets: 64/128
doubled_offsets:  66/128
scalar_coeff:     48/192
scalar_residual:  10/192
final_residual:    0/32
```

Decision:

```text
rejected for L2/L3/L4
```

Interpretation:

```text
The scalar-focused loss can create a small step-exact signal, but it does not
move scalar_residual or final_residual on held-out cases. This rejects
readout-only scalar weighting as the next fix.

Next root-architecture doubt:
the core hidden state used by the typed field readout may not contain enough
causal numeric/offset information because only the typed readout was trained.
The next A/B should train the recursive core and typed field-state together.
If that still fails, the shared carried hidden vector should be replaced by
typed list/scalar subregisters with separate update paths.
```

## Core + Typed Field-State Scalar Probe Result

Run:

```text
local_eval/research_gate_runner/core_typed_field_state_scalar_probe_s60_from_typed40_seed16
init: typed_primitive_conditioned_field_state_s80_lr4e4_seed14/step_000040.pt

trainable-param-policy:
core_and_typed_algorithmic_value_state

loss:
typed_algorithmic_kind_ce_multiplier = 0.25
typed_algorithmic_list_ce_multiplier = 0.75
typed_algorithmic_scalar_ce_multiplier = 12.0
```

Held-out typed-field eval:

```text
step_000020: fields=382/928, content=375/880, steps=8/256, trace=0/32
step_000040: fields=373/928, content=373/880, steps=0/256, trace=0/32
step_000060: fields=311/928, content=295/880, steps=0/256, trace=0/32
```

Best field breakdown for `step_000020`:

```text
kind:             192/256
raw_list_offsets: 66/128
doubled_offsets:  64/128
scalar_coeff:     48/192
scalar_residual:  10/192
final_residual:    2/32
```

Decision:

```text
rejected for L2/L3/L4
```

Interpretation:

```text
Jointly training the recursive core plus typed field-state readout does not
solve the value-state bottleneck. It recovers the same weak step-exact signal
as scalar-focused readout training but leaves trace_exact at 0/32 and keeps
scalar_residual/final_residual nearly unchanged.

This rejects "just unfreeze the core around the same shared hidden carry" as
the next fix. The next root-architecture candidate should stop treating all
algorithmic state as one generic hidden vector. It should split the recurrent
state into typed subregisters:

prompt/donor hidden
-> recursive core
-> typed list/scalar subregisters
   - raw list slots
   - transformed list slots
   - scalar coefficient slots
   - scalar residual slots
   - final residual slot
-> operation/phase-conditioned neural update
-> LM-compatible answer path

Promotion remains blocked until held-out trace exact improves and component
ablations prove that the recursive typed state causally carries the answer.
```

## Typed Subregister Field-State Candidate

Target level:

```text
L1 scaffold -> possible L2 local gate if held-out trace or step exact improves
```

Major bottleneck:

```text
The shared recurrent hidden vector does not preserve separate list/scalar/final
algorithmic state variables well enough to close the value-state loop.
```

Architecture change:

```text
typed_algorithmic_value_state_subregisters_enabled

prompt/donor hidden
-> recursive core
-> primitive/joint/step-conditioned recurrent input
-> separate neural subregisters:
   list_subregister
   scalar_subregister
   final_subregister
-> field heads:
   raw/doubled list heads read list_subregister
   scalar coeff/residual heads read scalar_subregister
   final residual head reads final_subregister
   kind head reads the averaged register state
```

Baseline to beat:

```text
typed_primitive_conditioned_field_state_s80_lr4e4_seed14/step_000040
fields=418/928, content=418/880, steps=0/256, trace=0/32

core_typed_field_state_scalar_probe_s60_from_typed40_seed16/step_000020
fields=382/928, content=375/880, steps=8/256, trace=0/32
```

Promotion/kill rule:

```text
Promote only if held-out trace_exact becomes non-zero or step_exact/content
improve with a matching recurrent/subregister ablation drop.

Reject if trace stays 0/32 and scalar_residual/final_residual remain near the
previous 10/192 and 2/32 range.
```

Result:

```text
local_eval/research_gate_runner/typed_subregister_field_state_s80_from_typed40_seed17
init: typed_primitive_conditioned_field_state_s80_lr4e4_seed14/step_000040.pt

step_000020 full:          fields=460/928, content=428/880, steps=10/256, trace=0/32
step_000020 recurrent_off: fields=80/928,  content=48/880,  steps=0/256,  trace=0/32

step_000040 full:          fields=447/928, content=415/880, steps=16/256, trace=0/32
step_000040 recurrent_off: fields=80/928,  content=48/880,  steps=0/256,  trace=0/32

step_000060 full:          fields=439/928, content=407/880, steps=0/256,  trace=0/32
step_000080 full:          fields=428/928, content=396/880, steps=10/256, trace=0/32
```

Best field breakdown for `step_000020 full`:

```text
kind:             224/256
raw_list_offsets: 64/128
doubled_offsets:  63/128
scalar_coeff:     97/192
scalar_residual:  10/192
final_residual:    2/32
```

Preserved checkpoints:

```text
best_field_local_l2.pt = step_000020
best_step_local_l2.pt  = step_000040
```

Decision:

```text
accepted as L2-local field-state improvement
rejected for L3/L4 complete reasoning trace
```

Interpretation:

```text
Typed subregisters are the first variant in this branch to clearly beat the
shared-carry typed field-state on held-out field/content/step metrics while
collapsing under recurrent_off ablation. This means the separate recurrent
subregister path is causally carrying useful state.

It does not solve the major bottleneck yet. scalar_residual remains at 10/192
and final_residual remains at 2/32, so trace exact stays 0/32. The next shortest
move is not another generic readout head; it is a residual-focused phase on the
subregister architecture, because scalar_coeff improved from 88/192 to 97/192
but residual/final still do not move.
```

## Typed Subregister Residual-Focus Result

Run:

```text
local_eval/research_gate_runner/typed_subregister_residual_focus_s80_from_l2field_seed18
init: typed_subregister_field_state_s80_from_typed40_seed17/best_field_local_l2.pt

loss:
typed_algorithmic_kind_ce_multiplier = 0.1
typed_algorithmic_list_ce_multiplier = 0.25
typed_algorithmic_scalar_ce_multiplier = 24.0
typed_algorithmic_value_state_pad_ce_weight = 0.0
```

Held-out typed-field eval:

```text
step_000020 full:          fields=498/928, content=466/880, steps=6/256, trace=0/32
step_000020 recurrent_off: fields=80/928,  content=48/880,  steps=0/256, trace=0/32

step_000040 full:          fields=396/928, content=364/880, steps=0/256, trace=0/32
step_000060 full:          fields=432/928, content=400/880, steps=0/256, trace=0/32
step_000080 full:          fields=384/928, content=352/880, steps=0/256, trace=0/32
```

Best field breakdown for `step_000020 full`:

```text
kind:             256/256
raw_list_offsets: 64/128
doubled_offsets:  64/128
scalar_coeff:    102/192
scalar_residual:  10/192
final_residual:    2/32
```

Preserved checkpoint:

```text
best_field_local_l2.pt = step_000020
```

Decision:

```text
accepted as stronger L2-local field/content improvement
rejected for L3/L4 complete reasoning trace
```

Interpretation:

```text
The subregister architecture continues to improve causal held-out field/content
metrics and the recurrent_off ablation still collapses the score, so the
learned subregister state is real. However, scalar_residual and final_residual
are unchanged despite a heavy scalar loss. This rejects "more scalar CE on the
same scalar register" as the path to L3.

The remaining bottleneck is not field classification in general. It is the
missing residual variable transition: the model learns kind/list/coeff fields
but does not learn the residual variable that closes the scalar calculation.
The next architecture must make residual an explicit recurrent latent variable
with its own transition target or delta path, while still keeping it inside the
universal LLM causal path.
```

## Residual Feedback Probe Result

Implemented optional residual self-feedback:

```text
typed_algorithmic_value_state_residual_feedback_enabled

previous scalar_residual / final_residual probability distributions
-> residual feedback projection
-> next recurrent typed-state update
```

Run:

```text
local_eval/research_gate_runner/typed_subregister_residual_feedback_s80_from_residualfocus_seed19
init: typed_subregister_residual_focus_s80_from_l2field_seed18/best_field_local_l2.pt
```

Held-out typed-field eval:

```text
step_000020: fields=432/928, content=400/880, steps=0/256, trace=0/32
step_000040: fields=384/928, content=352/880, steps=0/256, trace=0/32
step_000060: fields=418/928, content=386/880, steps=0/256, trace=0/32
step_000080: fields=385/928, content=353/880, steps=0/256, trace=0/32
```

Best field breakdown for `step_000020`:

```text
kind:             256/256
raw_list_offsets: 64/128
doubled_offsets:  64/128
scalar_coeff:     48/192
scalar_residual:   0/192
final_residual:    0/32
```

Decision:

```text
rejected
```

Interpretation:

```text
Naive residual self-feedback degrades the accepted L2-local subregister result.
It likely feeds poorly calibrated residual distributions back into the state
and amplifies the wrong attractor. This rejects probability-feedback as the
next L3 path.

The next candidate should avoid feeding predicted residuals back as free-form
probabilities. It should instead supervise a compact residual-transition latent
directly: residual_delta / residual_class / final_residual_class as a state
variable, with an ablation proving that this latent affects the final LM path.
```

## Finality Target Fix Result

Bug found:

```text
typed_algorithmic_field_targets_from_row labelled final_residual only at the
last labelled depth. In dynamic-halt traces, final/hold begins earlier:
transition_finality_targets = 1 for depths 4..8, but only depth 8 had a
final_residual target.
```

Fix:

```text
If transition_finality_targets is present, label final_residual at every depth
where finality == 1. Fall back to last labelled depth for older rows.
```

Unit coverage:

```text
tests.test_qtrm_algorithmic_value_state_eval.
  QTRMAlgorithmicValueStateEvalTests.
  test_typed_algorithmic_targets_label_all_finality_residuals
```

Baseline re-eval after the target fix:

```text
typed_subregister_residual_focus_s80_from_l2field_seed18/best_field_local_l2.pt
fields=506/1056, content=474/1008, steps=6/256, trace=0/32

breakdown:
kind=256/256
raw_list_offsets=64/128
doubled_offsets=64/128
scalar_coeff=102/192
scalar_residual=10/192
final_residual=10/160
```

Follow-up run:

```text
local_eval/research_gate_runner/typed_subregister_finality_targetfix_s80_from_residualfocus_seed20
init: typed_subregister_residual_focus_s80_from_l2field_seed18/best_field_local_l2.pt
```

Held-out eval:

```text
step_000020 full:          fields=409/1056, content=377/1008, steps=2/256,  trace=0/32
step_000040 full:          fields=449/1056, content=417/1008, steps=32/256, trace=0/32
step_000040 recurrent_off: fields=82/1056,  content=50/1008,  steps=0/256,  trace=0/32
step_000060 full:          fields=416/1056, content=384/1008, steps=16/256, trace=0/32
step_000080 full:          fields=372/1056, content=340/1008, steps=4/256,  trace=0/32
```

Best step breakdown for `step_000040 full`:

```text
kind=256/256
raw_list_offsets=64/128
doubled_offsets=64/128
scalar_coeff=48/192
scalar_residual=0/192
final_residual=17/160
```

Preserved checkpoint:

```text
best_step_local_l2.pt = step_000040
```

Decision:

```text
accepted as target-correctness fix and L2-local recurrent signal
rejected for L3/L4
```

Interpretation:

```text
The target fix makes final_residual supervision less sparse and produces the
best held-out step-exact count so far under the corrected target definition:
32/256. The recurrent_off ablation again collapses, so the signal is causal.

However, scalar_residual falls to 0/192 and final_residual is only 17/160.
This means the corrected finality supervision helps some list/kind/phase exact
steps but still does not teach the scalar residual transition. The L3 blocker is
now specifically scalar residual computation/generalization, not finality label
sparsity.
```

## Residual Delta Latent Candidate

Target level:

```text
L2 local gate first; promote to L3 only if scalar_residual/final_residual and
step-exactness improve under held-out eval and collapse under recurrent-off.
```

Major bottleneck:

```text
The current subregister core can keep list/kind fields but does not learn the
scalar residual transition. The most likely local bottleneck is that absolute
residual classes are too sparse/unstable for the carried scalar register.
```

Architecture change:

```text
typed_algorithmic_value_state_residual_delta_enabled

prompt/donor hidden
-> recursive core
-> primitive/joint/step-conditioned recurrent input
-> scalar_subregister
-> scalar_residual_delta_logits
-> scalar_residual / final_residual heads
-> LM answer path
```

This remains inside the universal LLM causal path. The delta head is an
auxiliary latent-state pressure, not an external calculator.

Promotion rule:

```text
accept L2 if scalar_residual improves materially over the corrected-target
baseline and recurrent_off loses the gain.
accept L3 only if trace/step exactness improves on held-out cases and the same
latent state causally affects the final answer path.
```

Kill rule:

```text
If scalar_residual stays near 0/192 or gains come without recurrent ablation
drop, reject residual-delta as another local scaffold and escalate to a root
state-codec redesign rather than adding more heads.
```

Run:

```text
local_eval/research_gate_runner/typed_subregister_residual_delta_s80_from_targetfix_seed21
init: typed_subregister_finality_targetfix_s80_from_residualfocus_seed20/best_step_local_l2.pt
```

Held-out eval:

```text
step_000020 full: fields=452/1056, content=420/1008, steps=6/256, trace=0/32
step_000040 full: fields=404/1056, content=372/1008, steps=0/256, trace=0/32
step_000060 full: fields=434/1056, content=402/1008, steps=0/256, trace=0/32
step_000080 full: fields=434/1056, content=402/1008, steps=0/256, trace=0/32
```

Best field breakdown for `step_000020 full`:

```text
kind=256/256
raw_list_offsets=64/128
doubled_offsets=64/128
scalar_coeff=48/192
scalar_residual=10/192
scalar_residual_delta=0/160
final_residual=10/160
```

Decision:

```text
rejected
```

Interpretation:

```text
Residual-delta supervision does not move the scalar transition. The apparent
field-count gain over the corrected-target run is small and comes with a large
step-exactness regression from 32/256 to 6/256. The new delta latent is itself
0/160, so this is not a usable causal state-codec improvement.

This exhausts the local "add another residual head" direction. The next
candidate must redesign the scalar state codec or task factorization, not add
more auxiliary scalar heads to the same carried representation.
```

## Scalar Ordinal And Operand Codec Attempts

Failure signal:

```text
The held-out scalar_residual predictions collapse to a few class attractors
instead of tracking numeric class distance. For the residual-delta run,
predicted scalar_residual was almost entirely {10, 13}, while targets covered
10..41. Train data covers the eval class range, so this is not a simple
unseen-class split.
```

Architecture attempts:

```text
1. scalar ordinal expected-class loss
   --typed-algorithmic-scalar-ordinal-weight

2. scalar operand slot
   typed_algorithmic_value_state_scalar_offset_enabled
   scalar_offset target = mixed_offset + 1

3. core-open operand slot
   trainable_param_policy = core_and_typed_algorithmic_value_state
```

The ordinal loss is still canonical because it only changes training pressure
on the same scalar logits. The offset slot is an operand-extraction codec, not
an external calculator; it is valid only if it improves the recurrent answer
path.

Held-out results:

```text
scalar_ordinal_s80_from_targetfix_seed22:
  step_000020 full: fields=437/1056, content=405/1008, steps=32/256, trace=0/32
  step_000040 full: fields=432/1056, content=400/1008, steps=0/256,  trace=0/32
  step_000060 full: fields=451/1056, content=419/1008, steps=32/256, trace=0/32
  step_000080 full: fields=441/1056, content=409/1008, steps=14/256, trace=0/32
  best breakdown:
    kind=256/256 raw=64/128 doubled=64/128
    scalar_coeff=49/192 scalar_residual=8/192 final_residual=10/160

scalar_offset_s80_from_targetfix_seed23:
  step_000020 full: fields=413/1056, content=381/1008, steps=1/256, trace=0/32
  step_000040 full: fields=431/1056, content=399/1008, steps=0/256, trace=0/32
  step_000060 full: fields=427/1056, content=395/1008, steps=0/256, trace=0/32
  step_000080 full: fields=432/1056, content=400/1008, steps=0/256, trace=0/32
  offset slot at step_000080: offset=24/192, scalar_residual=0/192,
  final_residual=0/160

core_typed_scalar_offset_s80_from_targetfix_seed24:
  step_000020 full: fields=384/1056, content=352/1008, steps=0/256, trace=0/32
  step_000040 full: fields=436/1056, content=404/1008, steps=0/256, trace=0/32
  step_000060 full: fields=415/1056, content=383/1008, steps=7/256, trace=0/32
  step_000080 full: fields=372/1056, content=340/1008, steps=0/256, trace=0/32
```

Decision:

```text
all rejected for L2/L3/L4 promotion
```

Interpretation:

```text
The ordinal loss gives a tiny field-count gain but worsens scalar_residual
itself and does not beat the corrected-target baseline on step exactness.
The scalar_offset slot does not learn robust held-out operand extraction, and
opening the core for 80 steps does not rescue it.

The current bottleneck is now root-level: the typed subregister codec can
memorize field shape and some list offsets, but it does not learn the scalar
arithmetic transition under the held-out surface/length split. The next
candidate must change the task/codec to a falsifiable smaller scalar arithmetic
gate before re-integrating with the full mixed-list transition trace.
```

## Scalar-Only Arithmetic Codec Gate

Question:

```text
Is typed-register + primitive already a universal LLM method, or only a
diagnostic scaffold?
```

Canonical answer:

```text
typed-register + primitive is not accepted as the final universal LLM
architecture. It is a scaffold/probe that is allowed only when it is derived
from the canonical prompt/token stream, trained as an internal latent-state
bottleneck, and shown to causally improve the LM answer path.

If it directly behaves like an external state-machine, calculator, or hidden
answer channel, it is not a universal LLM method.
```

Scalar-only gate:

```text
data/filtered/scalar_affine_arithmetic_codec_train40000_v0to5_cases512.jsonl
data/eval/scalar_affine_arithmetic_codec_eval50000_v6to7_cases64.jsonl

local_eval/research_gate_runner/scalar_affine_codec_typed_s120_from_targetfix_seed25
init: typed_subregister_finality_targetfix_s80_from_residualfocus_seed20/best_step_local_l2.pt
```

Held-out eval:

```text
step_000040 full: fields=1308/3968, content=1308/3968, steps=7/1024, trace=0/128
step_000080 full: fields=1320/3968, content=1320/3968, steps=2/1024, trace=0/128
step_000120 full: fields=1339/3968, content=1339/3968, steps=9/1024, trace=0/128
```

Best field breakdown for `step_000120 full`:

```text
kind=1024/1024
scalar_coeff=242/1024
scalar_offset=0/1024
scalar_residual=33/1024
final_residual=40/896
```

Decision:

```text
rejected for L2/L3/L4 promotion
```

Interpretation:

```text
The model learns type/shape identification but not the numeric value
transition. The scalar-only factorization makes the bottleneck clearer:
exact value computation is the failure, not only mixed-list prompt complexity.

This confirms typed-register + primitive should remain a diagnostic scaffold.
It is useful because it exposes whether the recursive core can carry and update
state, but it is not proof of a general LLM reasoning architecture until the
same state causally improves autoregressive LM answers over donor-only,
core-off, and primitive/register-off baselines.
```

## Continuous Scalar Codec Attempt

Hypothesis:

```text
Exact scalar class CE may be too brittle. A continuous normalized scalar value
head might learn numeric distance first, then eval can round that value back to
the discrete scalar class.
```

Implementation:

```text
typed_algorithmic_value_state_scalar_regression_enabled
typed_algorithmic_scalar_{coeff,offset,residual,final_residual}_value
--typed-algorithmic-scalar-regression-weight
--use-typed-scalar-regression-values
```

Run:

```text
local_eval/research_gate_runner/scalar_affine_codec_continuous_s120_from_targetfix_seed26
config: configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_typed_subregister_scalar_regression_s160.yaml
trainable: typed_algorithmic_value_state_only
loss: kind CE 0.1 + scalar regression 128.0, scalar class CE 0.0
```

Held-out rounded-regression eval:

```text
step_000040 full: fields=1083/3968, steps=0/1024, trace=0/128
step_000080 full: fields=1053/3968, steps=0/1024, trace=0/128
step_000120 full: fields=1048/3968, steps=0/1024, trace=0/128
```

Best field breakdown for `step_000040 full`:

```text
kind=1024/1024
scalar_coeff=44/1024
scalar_offset=3/1024
scalar_residual=5/1024
final_residual=10/896
```

Decision:

```text
rejected for L2/L3/L4 promotion
```

Interpretation:

```text
Continuous scalar regression trains in-sample enough to reduce scalar
regression MAE, but held-out rounded value accuracy is worse than scalar class
CE. Because kind is perfect while coeff/offset/residual are nearly absent, the
more likely root bottleneck is not only scalar arithmetic; the frozen core
states are not carrying the prompt operands into the typed scalar path.

The next falsification should open the recursive core together with the scalar
codec. If core-open regression still cannot extract coeff/offset/residual, the
architecture needs a prompt-operand binding redesign rather than another value
head.
```

## Core-Open Continuous Scalar Codec

Hypothesis:

```text
The scalar codec may be failing because the frozen recursive core does not
carry prompt operands into the typed scalar state. Opening the core together
with the scalar codec should improve held-out coeff/offset/residual if this is
only a frozen-core transport problem.
```

Run:

```text
local_eval/research_gate_runner/scalar_affine_codec_continuous_coreopen_s120_from_targetfix_seed27
config: configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_typed_subregister_scalar_regression_s160.yaml
trainable: core_and_typed_algorithmic_value_state
lr: 1.0e-4
loss: kind CE 0.1 + scalar regression 128.0, scalar class CE 0.0
```

Held-out rounded-regression eval:

```text
step_000040 full: fields=1056/3968, steps=0/1024, trace=0/128
step_000080 full: fields=1052/3968, steps=0/1024, trace=0/128
step_000120 full: fields=1054/3968, steps=0/1024, trace=0/128
```

Important metric note:

```text
The eval summary above was produced before scalar_offset was added to the
typed-field scorer. Manual breakdown showed scalar_offset=0/1024. The scorer
has now been fixed so future typed summaries include scalar_offset in field and
step-exact metrics.
```

Manual best breakdown for `step_000040 full`:

```text
kind=1024/1024
scalar_coeff=0/1024
scalar_offset=0/1024
scalar_residual=0/1024
final_residual=32/896
```

Decision:

```text
rejected for L2/L3/L4 promotion
```

Interpretation:

```text
Opening the core for 120 steps does not recover operand extraction or scalar
transition. This makes continued local variants of typed scalar CE, ordinal,
delta, offset, or continuous heads low-value.

The next architecture candidate should target prompt-to-latent operand binding
directly, with a gate that first proves coeff/offset/residual extraction from
the canonical token stream before asking the recurrent core to transform those
values. If that binder becomes a hidden parser or external calculator, it must
remain probe-only and cannot be promoted to L4.
```

## Prompt Operand Binding Diagnostics

Hypothesis:

```text
The scalar affine gate is failing because the model cannot reliably bind
operands from the canonical prompt/token stream into recurrent latent state.
Before adding more arithmetic heads, prove whether a small learned binder can
extract scalar_coeff, scalar_offset, and scalar_initial_residual from the same
input stream.
```

Probe A:

```text
local_eval/research_gate_runner/prompt_operand_binder_donorhidden_s120_seed28
input: frozen Qwen donor hidden states
best exact_acc: 0.000000
step_120 eval: exact=0.000000, coeff=0.250000, offset=0.164062, residual=0.031250
```

Probe B:

```text
local_eval/research_gate_runner/prompt_operand_binder_tokenembed_s240_seed29
input: learned token embeddings
best exact_acc: 0.046875
step_240 eval: exact=0.046875, coeff=0.640625, offset=0.390625, residual=0.132812
```

Interpretation:

```text
The token embedding path carries more operand signal than the frozen donor
hidden path, but neither gives a reliable exact operand binding solution. This
supports treating typed-register and primitive heads as diagnostic scaffolds,
not as canonical LLM architecture, until the universal prompt/token -> latent
state -> LM logits path improves.
```

## Token-Embedding Core-Typed Scalar Codec

Hypothesis:

```text
If operand binding is the blocker, opening token embeddings together with the
recursive core and typed state heads should improve held-out scalar affine
trace accuracy over the previous donor-hidden/core-only attempts.
```

Run:

```text
local_eval/research_gate_runner/scalar_affine_codec_tokenembed_coretyped_s80_from_targetfix_seed30
config: configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_typed_subregister_scalar_regression_s160.yaml
trainable: token_embed_core_and_typed_algorithmic_value_state
steps: 80
loss: kind CE 0.1 + scalar class CE 2.0 + scalar regression 32.0
```

Storage note:

```text
The run reached step 80 and wrote step_000040.pt and step_000080.pt, but the
final last.pt write failed because the root filesystem was full. The partial
last.pt was removed. Step checkpoints were still evaluated.
```

Held-out eval:

```text
step_000040 class:      fields=1448/4992, steps=0/1024, trace=0/128
step_000040 regression: fields=1178/4992, steps=0/1024, trace=0/128
step_000080 class:      fields=1503/4992, steps=0/1024, trace=0/128
step_000080 regression: fields=1026/4992, steps=0/1024, trace=0/128
```

Decision:

```text
rejected for L2/L3/L4 promotion
```

Conclusion:

```text
Opening token embeddings improves field accuracy modestly relative to some
earlier scalar-head variants, but exact step and trace accuracy remain zero.
This is not a solved recursive reasoning gate. The next candidate should not
add another typed scalar head; it should redesign the canonical prompt-binding
and recurrent state update so the core is trained end-to-end to route operands
into the same causal path that produces LM logits.
```

## Positional Prompt Binder Probe

Hypothesis:

```text
The previous prompt operand binder was too weak because it had no positional
signal. Operand binding in scalar prompts depends on token order, so a small
field-query Transformer with learned position embeddings should separate
numeric binding from QTRM core/recurrent failures.
```

Implementation:

```text
scripts/314_train_prompt_operand_binder_probe.py
added --binder-kind transformer
added learned position embeddings for transformer binder
fixed checkpoint_payload to save token_embed for token_embedding probes
```

Same-surface held-out probe:

```text
local_eval/research_gate_runner/prompt_operand_transformer_pos_tokenembed_same_surface_s800_seed32
train: variants 0-5, base_start 40000
eval:  variants 0-5, base_start 50000
best exact_acc: 1.000000
step_400/600/800 eval: exact=1.000000, coeff=1.000000, offset=1.000000, residual=1.000000
```

Surface-OOD held-out probe:

```text
local_eval/research_gate_runner/prompt_operand_transformer_pos_tokenembed_surface_ood_s800_seed33
train: variants 0-5, base_start 40000
eval:  variants 6-7, base_start 50000
best exact_acc: 0.398438
step_800 eval: exact=0.398438, coeff=0.750000, offset=0.523438, residual=0.515625
```

Decision:

```text
accepted as L1 diagnostic only
rejected for L2/L3/L4 promotion
```

Interpretation:

```text
Operand binding is learnable from the canonical token stream when positional
information is present and the prompt surface is familiar. The current scalar
affine L2 gate is therefore mixing at least two bottlenecks:

1. canonical prompt-token positional binding
2. surface-form generalization to unseen arithmetic phrasings

This does not prove recursive reasoning. It only proves that the old binder and
some QTRM paths were underpowered for prompt binding.
```

## Typed Prompt-Context QTRM Attempt

Hypothesis:

```text
If positional prompt binding is the blocker, adding direct prompt-context
cross-attention to typed_algorithmic_value_state should improve QTRM held-out
scalar affine fields and create nonzero exact-step accuracy.
```

Implementation:

```text
src/qtrm_mm/config.py
src/qtrm_mm/qtrm_model.py
configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_typed_subregister_promptctx_scalar_regression_s160.yaml

New config flags:
typed_algorithmic_value_state_prompt_context_enabled
typed_algorithmic_value_state_prompt_gate_init_bias
typed_algorithmic_value_state_prompt_gate_min
```

Run:

```text
local_eval/research_gate_runner/typed_promptctx_scalar_affine_s160_from_targetfix_seed34
init: local_eval/research_gate_runner/typed_subregister_finality_targetfix_s80_from_residualfocus_seed20/best_step_local_l2.pt
trainable: typed_algorithmic_value_state_only
steps: 160
loss: kind CE 0.1 + scalar class CE 2.0 + scalar regression 16.0
```

Held-out eval:

```text
same_surface step_000080 class: fields=4464/14976, steps=1/3072, trace=0/384
same_surface step_000160 class: fields=4392/14976, steps=0/3072, trace=0/384
surface_ood  step_000080 class: fields=1515/4992,  steps=0/1024, trace=0/128
surface_ood  step_000160 class: fields=1504/4992,  steps=7/1024, trace=0/128
```

Decision:

```text
rejected for L2/L3/L4 promotion
```

Conclusion:

```text
Direct prompt-context cross-attention into typed_algorithmic_value_state does
not transfer the standalone positional binder success into QTRM's recurrent
typed state. This suggests the next candidate must make prompt binding a
first-class recurrent state initialization/update path, not a late auxiliary
head over core_depth_states.
```

## Learned Core Depth Readout Attempt

Hypothesis:

```text
The QTRM core may already bind prompt information into non-first workspace
slots, but typed_algorithmic_value_state reads only state[:, 0, :] from every
recurrent depth. A learned query readout over the full recurrent latent set
should recover more prompt-bound information than the first-token readout.
```

Implementation:

```text
src/qtrm_mm/config.py
src/qtrm_mm/qtrm_model.py
src/qtrm_mm/training/train.py
configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_depthreadout_promptctx_scalar_regression_s160.yaml

New flag:
core_depth_readout_enabled

The readout is internal: a learned query cross-attends over each recurrent
workspace state and produces the per-depth state consumed by downstream heads.
```

Run:

```text
local_eval/research_gate_runner/depthreadout_promptctx_scalar_affine_s160_from_targetfix_seed35
init: local_eval/research_gate_runner/typed_subregister_finality_targetfix_s80_from_residualfocus_seed20/best_step_local_l2.pt
trainable: core_and_typed_algorithmic_value_state
steps: 160
loss: kind CE 0.1 + scalar class CE 2.0 + scalar regression 16.0
```

Surface-OOD held-out eval:

```text
step_000080 class: fields=1363/4992, steps=0/1024, trace=0/128
step_000160 class: fields=1500/4992, steps=0/1024, trace=0/128
```

Decision:

```text
rejected for L2/L3/L4 promotion
```

Conclusion:

```text
The first-token readout was a plausible bottleneck, but replacing it with a
learned full-workspace readout did not improve held-out exact steps or trace
accuracy. The stronger standalone probe still points to missing trainable
token-level positional/numeric binding in the canonical QTRM path, not merely
to the downstream depth-state readout.
```

## Position-Aware Token Path QTRM Attempt

Hypothesis:

```text
The standalone prompt binder only reached exact same-surface binding after
adding learned token positions. QTRM's scalar configs route text through
trainable embeddings and prelude/core blocks whose prelude attention may be
absent or weak; therefore the canonical token path may need an explicit learned
text_position_embed before the recursive core can bind prompt operands.
```

Implementation:

```text
src/qtrm_mm/config.py
src/qtrm_mm/qtrm_model.py
src/qtrm_mm/training/train.py
tests/test_training_checkpoint_init.py

New flag:
text_position_embed_enabled

Trainable policy updated:
token_embed_core_and_typed_algorithmic_value_state opens text_embed,
text_position_embed, core, core_depth_readout, transition_state_joint, and
typed_algorithmic_* parameters.
```

Verification:

```text
PYTHONPATH=src .venv/bin/python -m py_compile \
  src/qtrm_mm/config.py src/qtrm_mm/qtrm_model.py src/qtrm_mm/training/train.py

PYTHONPATH=src .venv/bin/python -m unittest \
  tests.test_training_checkpoint_init.TrainingCheckpointInitTests.test_token_embed_core_and_typed_policy_opens_token_path \
  tests.test_training_checkpoint_init.TrainingCheckpointInitTests.test_core_and_typed_algorithmic_policy_trains_core_and_field_heads \
  tests.test_core_halting.CoreHaltingTests.test_model_exposes_typed_algorithmic_value_state_logits

result: OK
```

Run:

```text
local_eval/research_gate_runner/textpos_depthreadout_promptctx_scalar_affine_s080_from_targetfix_seed36
init: local_eval/research_gate_runner/typed_subregister_finality_targetfix_s80_from_residualfocus_seed20/best_step_local_l2.pt
trainable: token_embed_core_and_typed_algorithmic_value_state
steps: 80
loss: kind CE 0.1 + scalar class CE 2.0 + scalar regression 16.0
```

Train signal:

```text
step 1:  loss=68.6047, typed_acc=0.2821, content_acc=0.0968, scalar_regression_mae=34.8387
step 40: loss=36.8874, typed_acc=0.2051, content_acc=0.0000, scalar_regression_mae=15.8065
step 80: content_acc remained 0.0000 in the logged training window
```

Surface-OOD held-out eval:

```text
last.pt: fields=1354/4992, steps=0/1024, trace=0/128
field_accuracy=0.2712, step_exact_accuracy=0.0000, trace_exact_accuracy=0.0000
```

Decision:

```text
rejected for L2/L3/L4 promotion
```

Conclusion:

```text
Adding position embeddings to the integrated QTRM token path was not enough to
transfer the standalone binder result into recursive typed-state reasoning.
This strengthens the structural doubt: the failure is not only missing position
encoding. The next candidate should isolate the donor-hidden/projector mixture
and test a cleaner token-only recursive path before adding more typed heads or
loss terms.
```

## Donor-Context-Off Token Path A/B

Hypothesis:

```text
The integrated scalar gate may be failing because frozen Qwen donor hidden
states are injected through the multimodal projector before QTRM's trainable
token/core path, creating a mixed context that is poor for synthetic numeric
binding. If so, the same position-aware token path should improve when donor
hidden context is disabled.
```

Implementation:

```text
scripts/196_train_pure_recursive_depth_supervised.py
scripts/238_eval_qtrm_algorithmic_value_state.py

New CLI flag:
--disable-donor-context

The donor tokenizer and optional donor logits remain available, but donor
hidden states are not injected into QTRM. This preserves the canonical
tokenizer -> token embedding -> recursive core path for raw-intelligence gates.
```

Verification:

```text
PYTHONPATH=src .venv/bin/python -m py_compile \
  scripts/196_train_pure_recursive_depth_supervised.py \
  scripts/238_eval_qtrm_algorithmic_value_state.py

PYTHONPATH=src .venv/bin/python -m unittest \
  tests.test_qtrm_algorithmic_value_state_eval \
  tests.test_pure_recursive_depth_supervised_train_script

result: OK
```

Run:

```text
local_eval/research_gate_runner/textpos_donorctxoff_scalar_affine_s080_from_targetfix_seed37
init: local_eval/research_gate_runner/typed_subregister_finality_targetfix_s80_from_residualfocus_seed20/best_step_local_l2.pt
trainable: token_embed_core_and_typed_algorithmic_value_state
steps: 80
flag: --disable-donor-context
loss: kind CE 0.1 + scalar class CE 2.0 + scalar regression 16.0
```

Train signal:

```text
step 1:  loss=100.0917, typed_acc=0.2051, content_acc=0.0000, scalar_regression_mae=29.9355
step 40: loss=32.7514,  typed_acc=0.2051, content_acc=0.0000, scalar_regression_mae=8.5806
step 80: content_acc remained 0.0000 in the logged training window
```

Surface-OOD held-out eval with donor context disabled:

```text
last.pt: fields=1496/4992, steps=1/1024, trace=0/128
field_accuracy=0.2997, step_exact_accuracy=0.0010, trace_exact_accuracy=0.0000
```

Decision:

```text
rejected for L2/L3/L4 promotion
```

Conclusion:

```text
Disabling donor hidden context gives a small field/step movement but does not
produce a viable recursive reasoning gate. The donor/projector mixture is not
the only bottleneck. The next architecture candidate should move from late
typed readout supervision toward causal recurrent state initialization/update:
the prompt-derived token representation must seed the recursive core state
directly, and depth-to-depth transitions must be trained/evaluated before
relying on auxiliary typed heads.
```

## Position-Aware Full-Attention Prelude A/B

Hypothesis:

```text
The scalar gate may fail because the prelude uses attn_every=4 with only two
prelude layers, so prompt tokens are not sufficiently mixed before the latent
workspace reads them. A more standard LLM-style prompt mixer with attn_every=1
should improve token-to-workspace binding if this is the bottleneck.
```

Implementation:

```text
configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_textpos_attn1_donorctxoff_scalar_regression_s040.yaml

Changed from previous token-path config:
attn_every: 1
steps: 40
flag: --disable-donor-context
```

Run:

```text
local_eval/research_gate_runner/textpos_attn1_donorctxoff_scalar_affine_s040_from_targetfix_seed38
init: local_eval/research_gate_runner/typed_subregister_finality_targetfix_s80_from_residualfocus_seed20/best_step_local_l2.pt
trainable: token_embed_core_and_typed_algorithmic_value_state
steps: 40
loss: kind CE 0.1 + scalar class CE 2.0 + scalar regression 16.0
```

Train signal:

```text
step 1:  loss=96.6176, typed_acc=0.1538, content_acc=0.0000, scalar_regression_mae=21.6129
step 20: loss=26.1627, typed_acc=0.2051, content_acc=0.0000, scalar_regression_mae=22.0645
step 40: content_acc remained 0.0000 in the logged training window
```

Surface-OOD held-out eval:

```text
class argmax:       fields=1326/4992, steps=0/1024, trace=0/128, field_accuracy=0.2656
scalar regression: fields=1127/4992, steps=0/1024, trace=0/128, field_accuracy=0.2258
```

Decision:

```text
rejected for L2/L3/L4 promotion
```

Conclusion:

```text
A standard attention-heavy token mixer does not recover the recursive scalar
reasoning gate. This rejects "prelude self-attention is the only missing
ingredient." The bottleneck is now localized more tightly: prompt information
can be bound by a standalone positional transformer probe, but the integrated
QTRM path does not causally preserve/update that bound state across recursive
depth. The next replacement candidate should train the core as a recurrent
state transition model over prompt-seeded latent slots before exposing typed
readouts.
```

## Trainable Input Path Fix

Bug / structural finding:

```text
The token_embed_core_and_typed_algorithmic_value_state policy opened
text_embed, text_position_embed, core, and typed heads, but did not open the
prelude or latent workspace. Therefore previous token-path experiments trained
the token embedding while leaving the main prompt reader / token-to-workspace
binder mostly random-frozen.
```

Fix:

```text
src/qtrm_mm/training/train.py
tests/test_training_checkpoint_init.py

The token_embed_core_and_typed_algorithmic_value_state policy now trains:
text_embed, text_position_embed, prelude, workspace, core, core_depth_readout,
transition_state_joint, and typed_algorithmic_*.
```

Verification:

```text
PYTHONPATH=src .venv/bin/python -m py_compile src/qtrm_mm/training/train.py

PYTHONPATH=src .venv/bin/python -m unittest \
  tests.test_training_checkpoint_init.TrainingCheckpointInitTests.test_token_embed_core_and_typed_policy_opens_token_path \
  tests.test_training_checkpoint_init.TrainingCheckpointInitTests.test_core_and_typed_algorithmic_policy_trains_core_and_field_heads

result: OK
```

Run:

```text
local_eval/research_gate_runner/textpos_attn1_inputpath_coretyped_s040_from_targetfix_seed39
config: configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_textpos_attn1_donorctxoff_scalar_regression_s040.yaml
init: local_eval/research_gate_runner/typed_subregister_finality_targetfix_s80_from_residualfocus_seed20/best_step_local_l2.pt
trainable: token_embed_core_and_typed_algorithmic_value_state
trainable params: 175,877,144
steps: 40
flag: --disable-donor-context
```

Train signal:

```text
step 1:  loss=74.9808, typed_acc=0.3846, content_acc=0.2258, scalar_regression_mae=56.7742
step 20: loss=34.5759, typed_acc=0.2051, content_acc=0.0000, scalar_regression_mae=20.9355
step 40: content_acc remained 0.0000 in the logged training window
```

Surface-OOD held-out eval:

```text
class argmax:       fields=1500/4992, steps=0/1024, trace=0/128, field_accuracy=0.3005
scalar regression: fields=1241/4992, steps=0/1024, trace=0/128, field_accuracy=0.2486
```

Decision:

```text
partial diagnostic only; rejected for L2/L3/L4 promotion
```

Conclusion:

```text
Opening the input path produces the first clear training-side content signal
in the integrated QTRM path and slightly improves OOD field accuracy, so the
previous policy was a real bottleneck. However the signal collapses during
training and does not yield exact recurrent steps. The next fix should address
stability and supervision alignment: avoid overwhelming discrete class CE and
train a held-out-selected checkpoint schedule, or freeze fewer random modules
only after the prompt-to-workspace binder has a stable local gate.
```

## Regression-Only Input-Path Smoke

Hypothesis:

```text
The opened input path briefly produces content signal, then collapses under
mixed discrete scalar CE plus regression. If discrete scalar class CE is
overpowering or misaligning the continuous value path, disabling scalar class
CE and relying on scalar regression should preserve the content signal.
```

Run:

```text
local_eval/research_gate_runner/textpos_attn1_inputpath_regonly_s040_from_targetfix_seed40
config: configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_textpos_attn1_donorctxoff_scalar_regression_s040.yaml
init: local_eval/research_gate_runner/typed_subregister_finality_targetfix_s80_from_residualfocus_seed20/best_step_local_l2.pt
trainable: token_embed_core_and_typed_algorithmic_value_state
steps: 40
lr: 5e-5
flag: --disable-donor-context
loss: kind CE 0.1 + scalar class CE 0.0 + scalar regression 16.0
```

Train signal:

```text
step 1:  loss=22.6250, typed_acc=0.1795, content_acc=0.0000, scalar_regression_mae=46.8387
step 20: loss=0.9120,  typed_acc=0.3333, content_acc=0.1613, scalar_regression_mae=10.1290
step 40: content_acc remained 0.1613 in the logged training window
```

Held-out eval:

```text
surface-OOD class argmax:       fields=1329/4992,  steps=0/1024, trace=0/128, field_accuracy=0.2662
surface-OOD scalar regression: fields=1058/4992,  steps=0/1024, trace=0/128, field_accuracy=0.2119
same-surface class argmax:     fields=3970/14976, steps=0/3072, trace=0/384, field_accuracy=0.2651
```

Decision:

```text
rejected for L2/L3/L4 promotion
```

Conclusion:

```text
Regression-only training stabilizes the training-side content signal but does
not transfer to same-surface or OOD exact recurrent steps. This rejects the
idea that scalar CE/regression mismatch is the primary blocker. The remaining
bottleneck is deeper: QTRM can receive trainable prompt signals, but the
current late typed-state objective does not force a correct recurrent state
machine through the core trajectory. The next candidate must supervise or
constrain the depth-to-depth latent transition directly, not only the final
readout fields.
```

## Balanced Input-Path Long Smoke

Hypothesis:

```text
The 40-step input-path smokes may be too short and too loss-heavy. After fixing
the trainable policy so prelude/workspace are actually trainable, a longer
balanced run may let the canonical token path bind the prompt into the
recursive state and recover a local L2 signal.
```

Run:

```text
local_eval/research_gate_runner/textpos_attn1_inputpath_balanced_s800_from_targetfix_seed41
config: configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_textpos_attn1_donorctxoff_scalar_regression_s040.yaml
init: local_eval/research_gate_runner/typed_subregister_finality_targetfix_s80_from_residualfocus_seed20/best_step_local_l2.pt
trainable: token_embed_core_and_typed_algorithmic_value_state
trainable params: 175,877,144
steps: 800
lr: 5e-5
flag: --disable-donor-context
loss: kind CE 0.1 + scalar class CE 0.5 + scalar regression 4.0
```

Train signal:

```text
step 1:   loss=28.8887, typed_acc=0.2051, content_acc=0.0000, scalar_regression_mae=49.7097
mid run:  loss=5.5882,  typed_acc=0.2051, content_acc=0.0000, scalar_regression_mae=9.4516
step 700: loss=6.1711,  typed_acc=0.4103, content_acc=0.2581, scalar_regression_mae=4.3871
final:    typed_acc=0.4103, content_acc=0.2581, step_exact=0.0000
```

Held-out eval:

```text
surface-OOD class argmax:       fields=2087/4992,  steps=1/1024, trace=0/128, field_accuracy=0.4181
surface-OOD scalar regression: fields=1074/4992,  steps=0/1024, trace=0/128, field_accuracy=0.2151
same-surface class argmax:     fields=6859/14976, steps=8/3072, trace=0/384, field_accuracy=0.4580
```

Decision:

```text
partial diagnostic; rejected for L2/L3/L4 promotion
```

Conclusion:

```text
The input-path trainability fix is real: field accuracy improves materially
over the frozen-workspace runs, and exact step count becomes nonzero for the
first time in this branch. However trace accuracy remains zero, same-surface
step exact is only 8/3072, and regression decode collapses. The blocker is not
just prompt ingestion or longer training. The typed field head is still a
weakly causal probe rather than a stable recurrent answer path.

Next architecture move: stop treating typed-register/state output as the main
candidate for canonical LLM promotion. Use it only as an internal diagnostic.
The next L2 attempt should force the recursive core to improve the canonical
LM logits directly, with state supervision used as auxiliary depth pressure
instead of a separate answer channel.
```

## Ouro Recurrent Answer Path Validation Gate

Hypothesis:

```text
If typed value-state heads are only diagnostic, the more canonical next path is
the Ouro/LoopLM-style recurrent answer loop: prompt/donor hidden states feed
QTRM recursion, then the recurrent answer loop must improve LM-token scoring
over donor-only, core-off, and recurrent-off baselines. Latent action/state
accuracy alone is not enough.
```

Run:

```text
local_eval/research_gate_runner/ouro_answer_recurrent_validation_gated_seed42
config: configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_recurrent_s080.yaml
init: local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_core_role_value_state_joint_len579_s240_from_joint_s240/last.pt
seed: 42
steps: 40
save_every: 20
trainable: core_and_answer_state_loop
trainable params: 35,817,486
loss: final-logit CE 1.0 + depth-final CE 1.0 + causal-prefix supervision
gate: causal forced-choice over donor-only, core-off, full core, recurrent-off
```

Training/logging:

```text
missing keys: 12
unexpected keys: 5
saved step_000020.pt, step_000040.pt, last.pt
```

Gate result:

```text
decision: rejected

step_000020:
  donor_only hits: 0/4
  core_off hits: 0/4, ties: 4/4
  full_core hits: 0/4
  recurrent_off hits: 0/4
  full_margin_over_best_baseline: 0

step_000040:
  donor_only hits: 0/4
  core_off hits: 0/4, ties: 4/4
  full_core hits: 0/4
  recurrent_off hits: 0/4
  full_margin_over_best_baseline: 0

last:
  donor_only hits: 0/4
  core_off hits: 0/4, ties: 4/4
  full_core hits: 0/4
  recurrent_off hits: 0/4
  full_margin_over_best_baseline: 0
```

Action-code diagnostic:

```text
best candidate: step_000020
latent action/code eval: exact_rows=16/16, step_accuracy=1.0, trace_exact_accuracy=1.0
```

Observed failure pattern:

```text
Full core often breaks the donor/core-off tie, but it selects an intermediate
list-state candidate such as "100004,100008,100012" instead of the final scalar
answer such as "300015". This means the latent recurrent state/action path can
represent the task trace, but the LM answer path is currently aligned to
intermediate state features rather than final-answer semantics.
```

Decision:

```text
rejected for L2/L3/L4 promotion
diagnostic value: high
preserve best.pt as a diagnostic seed; remove duplicate rejected step/last files
```

Conclusion:

```text
The current bottleneck is no longer simply "does the recursive core learn a
trace?" On this branch, the trace/action-code path can be perfect while the
canonical LM scoring path remains at zero. The next architecture move should
not add more typed heads. It should add final-answer binding pressure:

1. distinguish intermediate-state tokens from final-answer tokens in the
   recurrent answer loop;
2. train a finality-conditioned readout so only final state can influence LM
   answer logits;
3. add negative candidates for intermediate states during forced-choice CE so
   the model is penalized for preferring trace artifacts over final answers.
```

## Final-Answer Binding Loss Runner Patch And Rejection

Patch:

```text
scripts/309_run_validation_gated_ouro_recurrent.py now forwards final-answer
binding options into scripts/196_train_pure_recursive_depth_supervised.py:

--terminal-depth-ce-weight
--answer-state-loop-halt-ce-weight
--choice-margin-weight
--choice-margin
--choice-margin-mode
--tail-negative-margin-weight
--tail-negative-margin
--tail-negative-family-filter
--subtract-tail-counterfactual-margin-weight
--subtract-tail-counterfactual-margin
--subtract-tail-counterfactual-family-filter
```

Verification:

```text
PYTHONPATH=src .venv/bin/python -m py_compile scripts/309_run_validation_gated_ouro_recurrent.py
PYTHONPATH=src .venv/bin/python -m unittest tests.test_ouro_validation_runner

result: OK
```

Failed setup:

```text
local_eval/research_gate_runner/ouro_final_answer_binding_validation_seed43
decision: train_failed
reason: answer-state-loop halt CE requires model.answer_state_loop_halt_enabled=true
fix: rerun with --answer-state-loop-halt-ce-weight 0.0
```

Run:

```text
local_eval/research_gate_runner/ouro_final_answer_binding_validation_seed44
seed: 44
steps: 40
save_every: 9999
target_mode: final
loss additions:
  causal-prefix max target tokens 8
  terminal-depth CE 1.0
  choice-margin sequence 0.4 @ margin 0.12
  tail-negative margin 0.3 @ margin 0.08
  subtract-tail counterfactual margin 0.2 @ margin 0.05
```

Gate result:

```text
decision: rejected

donor_only hits: 0/4
core_off hits: 0/4, ties: 4/4
full_core hits: 0/4
recurrent_off hits: 0/4
full_margin_over_best_baseline: 0
```

Action-code diagnostic:

```text
seed42 baseline action-code eval: exact_rows=16/16, step_accuracy=1.0
seed44 final-binding eval:       exact_rows=3/16,  step_accuracy=0.796875
```

Observed effect:

```text
The final-answer binding pressure improved the scalar answer's relative
logprob on some held-out examples, but intermediate list states still ranked
above the final scalar answer. At the same time, the latent action-code trace
collapsed from perfect to 3/16 exact rows.
```

Decision:

```text
rejected for L2/L3/L4 promotion
delete seed44 weights after documenting; keep report/eval JSON only
```

Conclusion:

```text
Naively pushing final-answer loss through the same recurrent answer path is
not the right architecture fix. It creates destructive interference: the core
loses the latent trace while still failing to render final answers. The next
candidate should separate concerns inside the canonical path:

1. freeze/preserve the trace/action-code path that already works;
2. train only a small finality-conditioned answer binder/readout on top of the
   frozen final latent state;
3. keep intermediate-state negatives as readout-level constraints, not as
   pressure on the whole recurrent core.
```

## Core-Carry ACT Rerun And List Failure Ledger

Date: 2026-05-09

Rerun:

```text
checkpoint:
  /mnt/nvme1n1p2/qtrm-runs/qwen35_2b_pure_recursive_reasoning_mixed_depth_act_core_joint_ce_s160_from_s080/last.pt

eval:
  scripts/257_run_core_carry_mixed_depth_act_gate.sh

out:
  /mnt/nvme1n1p2/qtrm-eval/core_carry_mixed_depth_act_core_joint_s160_rerun/mixed_depth_act_eval_gate_causal_fc_smoke8.jsonl
```

Smoke8 result:

```text
donor_only:                       2/8
core_off:                         0/8
core_steps2:                      3/8
core_steps4:                      3/8
core_steps8:                      4/8
core_halt_carry_steps2:           3/8
core_halt_carry_steps4:           3/8
core_halt_carry_steps8:           2/8
answer_halt_gate_off:             4/8
```

Family matrix:

```text
mode,arithmetic_chain,boolean_logic,list_transform,symbolic_binding
donor_only,1/2,1/2,0/2,0/2
core_off,0/2,0/2,0/2,0/2
core_steps2,1/2,1/2,0/2,1/2
core_steps4,1/2,1/2,0/2,1/2
core_steps8,2/2,1/2,0/2,1/2
core_halt_carry_steps2,1/2,1/2,0/2,1/2
core_halt_carry_steps4,1/2,1/2,0/2,1/2
core_halt_carry_steps8,0/2,1/2,0/2,1/2
answer_halt_gate_off,2/2,1/2,0/2,1/2
```

List failure ledgers:

```text
/mnt/nvme1n1p2/qtrm-eval/core_carry_mixed_depth_act_core_joint_s160_rerun/list_transform_failure_ledger_core8.json
/mnt/nvme1n1p2/qtrm-eval/core_carry_mixed_depth_act_core_joint_s160_rerun/list_transform_failure_ledger_halt_carry8.json
/mnt/nvme1n1p2/qtrm-eval/core_carry_mixed_depth_act_core_joint_s160_rerun/list_transform_failure_ledger_donor.json
```

Core8 list ledger:

```text
hits: 0/2
by_error:
  filtered_state_selected: 1
  reversed_final_selected: 1
mean_correct_minus_selected_score: -0.609418954168047
```

The same error types appear for donor-only and halt-carry8:

```text
donor_only:    filtered_state_selected=1, reversed_final_selected=1
halt_carry8:   filtered_state_selected=1, reversed_final_selected=1
```

Decision:

```text
Reject core-carry/ACT as the active L3 fix.
```

Interpretation:

```text
More recurrence or detached carry does not address the active list bottleneck.
The model still prefers either:

1. the depth-1 filtered intermediate state, or
2. the reversed final list.

Therefore the missing mechanism is not simply more loop depth. It is an
order-preserving list state that causally feeds the LM answer path.
```

Next smallest falsifiable candidate:

```text
Target level:
  L2 local gate, not L3/L4.

Major bottleneck:
  list/order-preserving answer path.

Baseline to beat:
  core_steps8 on the S160 checkpoint, which is 0/2 on list_transform and 4/8
  overall on smoke8.

Required score:
  list_transform > 0/2 without reducing overall smoke8 below 4/8.

Required ablation drop:
  disabling the new order/list path must reduce list_transform back to 0/2 or
  reduce correct-list logprob margin.

Promotion decision if pass:
  keep as an L2 local list-path candidate and scale to heldout72.

Kill decision if fail:
  stop answer-side loss tuning and move to a root transition architecture that
  represents ordered select/map/copy state inside the recurrent core.
```

## List-Order LM Gate Tooling

Added:

```text
scripts/311_summarize_list_order_lm_gate.py
tests/test_list_order_lm_gate_summary.py
```

Purpose:

```text
Automatically reject list/order "improvements" that are not causal. A run must
improve list_transform over donor/core-off and must drop when the selected
transition/order path is disabled.
```

Verification:

```text
PYTHONPATH=src .venv/bin/python -m unittest tests.test_list_order_lm_gate_summary

result: OK
```

Core-carry S160 rerun gate:

```text
report:
  /mnt/nvme1n1p2/qtrm-eval/core_carry_mixed_depth_act_core_joint_s160_rerun/list_order_gate.json

decision:
  rejected

metrics:
  core_overall_hits: 4/8
  core_list_hits: 0/2
  ablation_list_hits: 0/2
  best_baseline_list_hits: 0/2

reject reasons:
  core list hits do not beat best baseline list hits
  ablation ties or beats core list hits
```

List-transfer-long checkpoint raw-LM smoke:

```text
eval:
  /mnt/nvme1n1p2/qtrm-eval/list_transfer_long_checkpoint_raw_lm_smoke8.jsonl

report:
  /mnt/nvme1n1p2/qtrm-eval/list_transfer_long_checkpoint_raw_lm_smoke8.list_order_gate.json

decision:
  rejected

metrics:
  donor_only: 2/8
  core_off: 0/8
  core_steps8: 3/8
  transition_state_off: 3/8
  core_list_hits: 1/2
  transition_state_off_list_hits: 1/2
```

Important caveat:

```text
The list-transfer-long checkpoint reported 36 missing keys under the current
config, so it is not a clean promotion candidate. Even with that caveat, the
decisive result is non-causal: transition_state_off ties core on list_transform.
```

Conclusion:

```text
The existence of older list-transfer checkpoints that solve transition/action
state does not solve the current L3/L4 bottleneck. The missing piece is a
causal order-preserving answer path: disabling that path must drop list accuracy
or correct-list margin.
```

## Answer-Loop-Only List-Order Contrast S080 Rejection

Run:

```text
out:
  /mnt/nvme1n1p2/qtrm-runs/qwen35_2b_list_order_answer_loop_s080_from_core_joint_s160

init:
  /mnt/nvme1n1p2/qtrm-runs/qwen35_2b_pure_recursive_reasoning_mixed_depth_act_core_joint_ce_s160_from_s080/last.pt

config:
  configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_halt_head_eval_gate.yaml

train data:
  data/filtered/pure_recursive_reasoning_smallrange_train256_cases.jsonl

policy:
  answer_state_loop_only

steps:
  80

loss:
  target_mode=final
  final_logit_ce=0.05
  terminal_depth_ce=0.25
  choice_margin_sequence=0.80 @ margin 0.12
  transition_joint_answer_bridge_contrast=0.50 @ margin 0.05
  family_repeat=list_transform=8
```

Gate:

```text
eval:
  /mnt/nvme1n1p2/qtrm-runs/qwen35_2b_list_order_answer_loop_s080_from_core_joint_s160/list_order_answer_loop_s080_causal_fc_smoke8.jsonl

report:
  /mnt/nvme1n1p2/qtrm-runs/qwen35_2b_list_order_answer_loop_s080_from_core_joint_s160/list_order_gate.json
```

Smoke8:

```text
donor_only:        2/8
core_off:          0/8
core_steps2:       3/8
core_steps4:       4/8
core_steps8:       5/8
bridge_off8:       5/8
```

Family matrix for core8:

```text
arithmetic_chain: 2/2
boolean_logic:    1/2
list_transform:   0/2
symbolic_binding: 2/2
```

List ledger:

```text
core8 list_transform: 0/2
by_error:
  filtered_state_selected: 2
```

Decision:

```text
Reject for the active list-order L2/L3 bottleneck.
```

Interpretation:

```text
Freezing the core and training only answer_state_loop avoids destroying the
base core signal and can raise overall smoke8 from 4/8 to 5/8. However, it does
not solve order-preserving list answers, and the transition-joint answer bridge
is not causal because bridge_off ties full. The training pressure shifts list
errors toward the depth-1 filtered state instead of producing the final ordered
map state.
```

Cleanup:

```text
Delete rejected last.pt after this record. Keep eval/report/ledger artifacts.
```

## Plain List-Transform Target Codec Fix

Date: 2026-05-09

Root issue:

```text
Plain list_transform rows in the smoke gate do not contain list_value_start.
The previous role/value and typed target builders assumed a base offset for
list values. As a result, base-less list_transform rows could produce all -100
targets, so the exact bottleneck family had little or no state supervision in
some training paths.
```

Fix:

```text
src/qtrm_mm/algorithmic_value_state.py

absolute_list_value_classes(values, max_slots, slot_vocab_size)
role_value_targets_from_row(...)
algorithmic_targets_from_row(...)
typed_algorithmic_field_targets_from_row(...)
```

Behavior:

```text
base-less list values now map to absolute value classes as value + 1
depth 1 represents the filtered/raw kept list
depth > 1 represents the doubled final list
```

Verification:

```text
PYTHONPATH=src .venv/bin/python -m unittest \
  tests.test_qtrm_algorithmic_value_state_eval.\
QTRMAlgorithmicValueStateEvalTest.\
test_plain_list_transform_targets_use_absolute_value_slots \
  tests.test_qtrm_algorithmic_value_state_eval.\
QTRMAlgorithmicValueStateEvalTest.\
test_plain_list_transform_typed_targets_use_raw_then_doubled_roles

result: OK
```

Interpretation:

```text
This is a target-correctness fix, not an L2/L3 acceptance. It removes one
false-negative training condition so later list-order experiments receive a
real ordered-list state target.
```

## Role-Value Answer Bridge S040 Rejection

Question:

```text
Can typed-register + primitive be promoted from diagnostic scaffold into a
universal LLM path by making role/value state feed the answer loop through a
causal bridge?
```

Answer:

```text
Not yet. In the current implementation, typed-register + primitive remains a
diagnostic/training scaffold. It is only universal-LLM-compatible if the state
is derived from the token stream and causally improves LM logits. This run did
not prove that.
```

Run:

```text
out:
  /mnt/nvme1n1p2/qtrm-runs/qwen35_2b_list_order_role_value_answer_bridge_s040_from_primitive_s90

init:
  local_eval/research_gate_runner/primitive_field_heads_delta_codec_s90_lr5e4_seed11/last.pt

config:
  configs/qwen35_2b_4090_list_order_role_value_answer_bridge_s060.yaml

train data:
  data/filtered/pure_recursive_reasoning_smallrange_train256_cases.jsonl

policy:
  primitive_role_value_answer_bridge_loop

steps:
  40
```

New training pressure:

```text
core_role_value_state_answer_bridge_enabled = true
core_role_value_answer_bridge_contrast = 0.50 @ margin 0.05
core_primitive_role_value_state_ce = 1.0
core_primitive_role_value_step_margin = 0.20 @ margin 0.05
core_primitive_role_value_trace_margin = 0.50 @ margin 0.10
choice_sequence_margin = 0.80 @ margin 0.12
family_repeat=list_transform=8
```

Implementation note:

```text
The bridge-off contrast forward pass must run under no_grad. Without that, the
extra forward pass OOMs on the 4090 while evaluating the counterfactual.
```

Gate:

```text
eval:
  /mnt/nvme1n1p2/qtrm-runs/qwen35_2b_list_order_role_value_answer_bridge_s040_from_primitive_s90/role_value_answer_bridge_s040_causal_fc_smoke8.jsonl

report:
  /mnt/nvme1n1p2/qtrm-runs/qwen35_2b_list_order_role_value_answer_bridge_s040_from_primitive_s90/list_order_gate.json

ledger:
  /mnt/nvme1n1p2/qtrm-runs/qwen35_2b_list_order_role_value_answer_bridge_s040_from_primitive_s90/list_transform_failure_ledger_core8.json
```

Smoke8:

```text
donor_only:   2/8
core_off:     0/8
core_steps8:  2/8
bridge_off8:  2/8
```

Family matrix for core8:

```text
arithmetic_chain: 0/2
boolean_logic:    1/2
list_transform:   0/2
symbolic_binding: 1/2
```

List ledger:

```text
core8 list_transform: 0/2
by_error:
  filtered_state_selected: 1
  reversed_final_selected: 1
```

Decision:

```text
rejected for L2/L3/L4 promotion
```

Why rejected:

```text
The core path does not beat donor-only overall, does not improve list_transform,
and bridge_off ties full. Therefore the role/value answer bridge is not yet a
causal universal-LM answer improvement.
```

Architecture implication:

```text
typed-register + primitive is acceptable only as an internal state/transition
training probe. It must not become the canonical final-answer mechanism unless
the canonical path:

prompt/chat template -> tokenizer -> donor hidden or token embeddings ->
recursive QTRM core -> LM logits -> autoregressive text

improves over donor/core-off and drops under bridge/register/primitive ablation.
```

Kill decision:

```text
Delete this rejected last.pt after preserving report artifacts. Stop adding
answer-side bridges for this bottleneck unless the next candidate changes the
root transition architecture or proves ordered select/map/copy state before LM
answer supervision.
```

## Source-Position Pointer Codec L2-Local Acceptance

Date: 2026-05-09

Root cause discovered after the role-value answer bridge rejection:

```text
Plain list_transform rows have no list_value_start/base_value. The first
correctness fix encoded those rows with absolute value classes:

  raw 4,2 -> classes 5,3
  final 8,4 -> classes 9,5

This is not a general reasoning state. The train split uses list values around
19..56 while heldout starts around 1..16, so the classifier is asked to predict
unseen numeric classes. That makes the task look like a reasoning failure when
the real issue is the state codec.
```

Architecture correction:

```text
For base-less list_transform rows, encode list state as source-position
pointers into the prompt list, not as absolute numeric values.

Example:
  prompt list: [1, 4, 2, 7, 3]
  raw kept:    4,2   -> source-position classes 2,3
  doubled:     8,4   -> source-position classes 2,3

This directly represents ordered select/map/copy state:

prompt/donor hidden
-> recursive core
-> primitive operation logits
-> primitive role-value recurrent writer
-> source-position pointer state
```

Files changed:

```text
src/qtrm_mm/algorithmic_value_state.py
  row_input_list(...)
  source_position_list_classes(...)
  base-less list targets now prefer source positions and only fall back to
  absolute values when the input list cannot be parsed.

scripts/238_eval_qtrm_algorithmic_value_state.py
  primitive-off ablation with empty logits is now scored as all-wrong instead
  of crashing.

tests/test_qtrm_algorithmic_value_state_eval.py
  plain list_transform target tests now assert source-position classes.
```

Verification:

```text
PYTHONPATH=src .venv/bin/python -m unittest \
  tests.test_qtrm_algorithmic_value_state_eval

result: 22 tests OK

PYTHONPATH=src .venv/bin/python -m py_compile \
  src/qtrm_mm/algorithmic_value_state.py \
  scripts/238_eval_qtrm_algorithmic_value_state.py

result: OK
```

Training run:

```text
out:
  /mnt/nvme1n1p2/qtrm-runs/primitive_field_heads_source_pointer_s120_seed16

init:
  local_eval/research_gate_runner/primitive_field_heads_delta_codec_s90_lr5e4_seed11/last.pt

init sha256:
  9a9204a9b01001713772294afcf30ae5753b0e3cd3877adabb83918caf52747d

accepted checkpoint:
  /mnt/nvme1n1p2/qtrm-runs/primitive_field_heads_source_pointer_s120_seed16/accepted_l2_source_pointer_step_000040.pt

accepted sha256:
  719598db223aeec17cb62054179fe375ee21b2f2bc4dfbec307b22598acc542b
```

Run settings:

```text
config:
  configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_primitive_field_heads_delta_codec_s160.yaml

steps:
  120

selected step:
  step_000040

loss:
  primitive_transition_operation_ce = 0.50
  core_primitive_role_value_state_ce = 1.0
  core_primitive_role_value_step_margin = 0.50 @ margin 0.05
  core_primitive_role_value_trace_margin = 1.0 @ margin 0.10
  core_primitive_role_value_update_gate_bce = 0.50
  family_repeat = list_transform=16
```

Heldout72 source-pointer state result:

```text
full step_000040:
  rows:          72
  exact rows:     9/72
  value acc:     77/180 = 0.4278
  step exact:    36/72  = 0.5000
  trace exact:    9/72  = 0.1250

primitive-off ablation:
  exact rows:     0/72
  value acc:      0/180
  step exact:     0/72
  trace exact:    0/72

old s90 baseline under source-pointer targets:
  exact rows:     0/72
  value acc:     52/180 = 0.2889
  step exact:     0/72
  trace exact:    0/72
```

Family scope:

```text
Only list_transform has labelled source-pointer role/value targets in this
gate. Arithmetic, symbolic, and boolean rows are present but have no labelled
role/value values for this specific state metric.
```

Decision:

```text
accepted as L2-local ordered select/map/copy state signal
rejected for L3/L4 promotion
```

Why accepted only at L2:

```text
The source-position pointer codec produces heldout trace-exact rows and drops
to zero under primitive-off ablation. This proves the primitive recurrent state
can causally learn an ordered list pointer state.

It is not L3/L4 because this state does not yet improve the canonical LM answer
path. It still has to be compiled into:

prompt/chat template -> tokenizer -> donor hidden/token embeddings ->
recursive QTRM core -> LM logits -> autoregressive text
```

Next promotion candidate:

```text
Use accepted_l2_source_pointer_step_000040.pt as the frozen/initialized state
base, then train the answer path to consume the source-pointer state. The next
gate must be a raw LM forced-choice list_transform gate:

  full core+source-pointer answer path > donor/core_off
  source-pointer/primitive-off drops list_transform accuracy or margin
  no degradation of the non-list smoke families
```

## Source-Pointer Dense Answer Bridge Rejection

Date: 2026-05-09

Question:

```text
Can the accepted L2 source-position pointer state improve the canonical LM
answer path if the answer loop can also attend over prompt tokens?
```

Design:

```text
prompt/chat template -> tokenizer -> donor hidden states
-> QTRM recursive core / accepted source-pointer state
-> role-value answer bridge + dense prompt context in answer loop
-> donor-preserving LM logits
```

Run:

```text
config:
  configs/qwen35_2b_4090_list_order_source_pointer_dense_answer_bridge_s080.yaml

init:
  /mnt/nvme1n1p2/qtrm-runs/primitive_field_heads_source_pointer_s120_seed16/accepted_l2_source_pointer_step_000040.pt

out:
  /mnt/nvme1n1p2/qtrm-runs/qwen35_2b_list_order_source_pointer_dense_answer_bridge_s080_from_l2_pointer

trainable:
  role_value_answer_bridge_loop_only

seed:
  17
```

Heldout smoke gate:

```text
cases:
  first 8 list_transform rows from pure_recursive_reasoning_heldout_72

scoring:
  causal forced-choice, mean normalized
```

Results:

```text
step_000040:
  donor_only:                   1/8
  qtrm_core_off:                1/8
  full qtrm_core_steps_8:       2/8
  role_value_answer_bridge_off: 2/8
  primitive_role_value_off:     2/8
  selective_context_off:        2/8

step_000080:
  donor_only:                   1/8
  qtrm_core_off:                1/8
  full qtrm_core_steps_8:       2/8
  role_value_answer_bridge_off: 2/8
  primitive_role_value_off:     2/8
  selective_context_off:        2/8
```

Decision:

```text
Rejected for L3/L4.
```

Why rejected:

```text
The full path beats donor/core-off by one smoke case, but bridge-off,
primitive-off, and selective-context-off all tie the full path. Therefore the
gain is not caused by the accepted source-pointer state or by prompt-context
answer routing. This is a renderer/residual bias effect, not a recursive-core
reasoning gain.
```

Architectural implication:

```text
Adding prompt context to the answer loop is necessary but insufficient. The
current answer bridge embeds the predicted state classes and leaves the answer
loop to rediscover how those classes bind to prompt spans. There is no strong
causal binding objective from source-pointer state -> selected prompt span ->
LM logits.

Next candidate should train or probe a pointer-conditioned prompt-span binder
inside the LM path, and the gate must require:

  full > donor/core_off
  primitive/source-pointer off drops
  span-binder off drops
```

## Source-Pointer Final-Contrast Rejection

Date: 2026-05-09

Follow-up hypothesis:

```text
The dense answer bridge failed because the training objective did not force
the final LM answer logits to depend on primitive/source-pointer state. Add
final-path ablation contrast directly:

  full final target logp > bridge_off final target logp
  full final target logp > primitive_off final target logp
```

Implementation:

```text
scripts/196_train_pure_recursive_depth_supervised.py

added:
  final_path_ablation_contrastive_loss
  --core-role-value-answer-bridge-final-contrast-weight
  --core-role-value-answer-bridge-final-contrast-margin
  --core-primitive-role-value-answer-final-contrast-weight
  --core-primitive-role-value-answer-final-contrast-margin
  --core-primitive-role-value-answer-final-contrast-all-prefix-tokens
```

Run:

```text
out:
  /mnt/nvme1n1p2/qtrm-runs/qwen35_2b_list_order_source_pointer_final_contrast_s040_from_l2_pointer

init:
  /mnt/nvme1n1p2/qtrm-runs/primitive_field_heads_source_pointer_s120_seed16/accepted_l2_source_pointer_step_000040.pt

seed:
  18

steps:
  40

trainable:
  role_value_answer_bridge_loop_only

loss:
  final_logit_ce_weight = 0.5
  core_role_value_answer_bridge_final_contrast_weight = 0.50
  core_primitive_role_value_answer_final_contrast_weight = 1.0
```

Training signal:

```text
The primitive-off final target log-prob delta stayed approximately 0.0 through
the run. This means the current final answer path still does not materially
depend on primitive/source-pointer state even when pressured.
```

Heldout smoke:

```text
step_000040:
  donor_only:                   1/8
  qtrm_core_off:                1/8
  full qtrm_core_steps_8:       2/8
  role_value_answer_bridge_off: 2/8
  primitive_role_value_off:     2/8
  selective_context_off:        2/8
```

Decision:

```text
Rejected for L3/L4.
```

Implication:

```text
This is no longer a simple loss-weight problem. The current answer bridge can
create small tensor differences, but not enough causal information flow to
change answer selection under held-out forced-choice. The next architecture
candidate must explicitly bind pointer/source state to prompt spans or use an
answer-renderable latent value state, while still feeding the canonical LM
logits path.
```

## Source-Pointer Prompt-Bound Answer Bridge Rejection

Date: 2026-05-09

Hypothesis:

```text
The previous dense answer bridge may have failed because source-pointer
classes were not re-bound to the prompt hidden sequence before entering the
answer loop.

Add a prompt-context cross-attention inside
core_role_value_state_answer_bridge so role/value bridge tokens can attend
back to the donor/token hidden stream before feeding answer_state_loop.
```

Implementation:

```text
src/qtrm_mm/config.py
src/qtrm_mm/qtrm_model.py
tests/test_core_halting.py

added:
  core_role_value_state_answer_prompt_context_enabled
  core_role_value_state_answer_prompt_gate_init_bias
  core_role_value_state_answer_prompt_gate_min

config:
  configs/qwen35_2b_4090_list_order_source_pointer_prompt_bound_answer_bridge_s060.yaml
```

Verification:

```text
PYTHONPATH=src .venv/bin/python -m unittest \
  tests.test_core_halting \
  tests.test_model_config \
  tests.test_training_checkpoint_init \
  tests.test_raw_intelligence_eval_script \
  tests.test_pure_recursive_depth_supervised_train_script

result:
  326 tests OK

PYTHONPATH=src .venv/bin/python -m py_compile \
  src/qtrm_mm/config.py \
  src/qtrm_mm/qtrm_model.py \
  scripts/196_train_pure_recursive_depth_supervised.py \
  scripts/192_eval_raw_intelligence.py

result:
  OK
```

Run:

```text
out:
  /mnt/nvme1n1p2/qtrm-runs/qwen35_2b_list_order_source_pointer_prompt_bound_bridge_s030_from_l2_pointer

init:
  /mnt/nvme1n1p2/qtrm-runs/primitive_field_heads_source_pointer_s120_seed16/accepted_l2_source_pointer_step_000040.pt

seed:
  19

steps:
  30

trainable:
  role_value_answer_bridge_loop_only
```

Operational note:

```text
The first s060 attempt accidentally expanded --out-dir to an empty shell value
and wrote under runs/, which points to /mnt/sdb1. That mount is currently in
emergency_ro mode, so final checkpoint save failed and the partial step_000030
checkpoint was unreadable. The valid rerun used a literal /mnt/nvme1n1p2 path.
```

Heldout smoke:

```text
file:
  prompt_bound_bridge_step30_list_smoke8.causal_fc.jsonl

partial eval was intentionally killed after the causal decision was already
determined because the full 6-mode smoke was running too slowly.

completed modes:
  donor_only:                   1/8
  qtrm_core_off:                1/8
  full qtrm_core_steps_8:       2/8
  role_value_answer_bridge_off: 2/8
```

Decision:

```text
Rejected for L3/L4.
```

Reason:

```text
The full model again ties the bridge-off ablation. Therefore the apparent
2/8 gain over donor/core_off is not causally attributable to the new
prompt-bound answer bridge.
```

Implication:

```text
Simply giving the answer bridge more prompt access is insufficient. The
remaining bottleneck is answer-renderable state: the recurrent core can expose
narrow pointer/value signals, but the canonical LM logits path is still not
forced to render the transformed values through a component whose ablation
causes the held-out answer gain to disappear.
```

## Source-Pointer Final Binder Rejection

Date: 2026-05-09

Hypothesis:

```text
The answer bridge may be failing because it exposes all recurrent role/value
steps, so the LM path can prefer an intermediate filtered list state. Add a
final-only role/value answer binder that injects only the last recurrent
role/value bridge state into the answer loop before LM logits.
```

Implementation:

```text
src/qtrm_mm/config.py
src/qtrm_mm/qtrm_model.py
scripts/192_eval_raw_intelligence.py
tests/test_core_halting.py

added:
  core_role_value_state_answer_final_binder_enabled
  core_role_value_state_answer_final_gate_init_bias
  core_role_value_state_answer_final_gate_min
  disable_core_role_value_answer_final_binder
  qtrm_core_steps_N_core_role_value_answer_final_binder_off_no_evidence

config:
  configs/qwen35_2b_4090_list_order_source_pointer_final_binder_s040.yaml
```

Verification:

```text
PYTHONPATH=src .venv/bin/python -m unittest \
  tests.test_core_halting \
  tests.test_model_config \
  tests.test_training_checkpoint_init \
  tests.test_raw_intelligence_eval_script \
  tests.test_pure_recursive_depth_supervised_train_script

result:
  327 tests OK

PYTHONPATH=src .venv/bin/python -m py_compile \
  src/qtrm_mm/config.py \
  src/qtrm_mm/qtrm_model.py \
  scripts/192_eval_raw_intelligence.py \
  scripts/196_train_pure_recursive_depth_supervised.py

result:
  OK
```

Run:

```text
out:
  /mnt/nvme1n1p2/qtrm-runs/qwen35_2b_list_order_source_pointer_final_binder_s040_from_l2_pointer

init:
  /mnt/nvme1n1p2/qtrm-runs/primitive_field_heads_source_pointer_s120_seed16/accepted_l2_source_pointer_step_000040.pt

seed:
  20

steps:
  40
```

Heldout smoke4:

```text
donor_only:             0/4
qtrm_core_off:          0/4
full qtrm_core_steps_8: 1/4
final_binder_off:       1/4
bridge_off:             1/4
```

Decision:

```text
Rejected for L2/L3/L4.
```

Reason:

```text
The full model ties the final-binder-off and bridge-off ablations. Therefore
the small gain over donor/core_off is not caused by the final binder.
```

Implication:

```text
Final-state injection alone is still too weak. The active bottleneck is now
more clearly upstream of answer binding: the recurrent state must represent an
order-preserving select/map/copy structure whose removal changes the final LM
choice. More answer-side bridges without a stronger recurrent ordered-list
state are unlikely to produce L3 progress.
```
