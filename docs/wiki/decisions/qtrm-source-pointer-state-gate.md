
## Status 2026-05-09: Validation-Gated Reject

The source-position pointer refresh gate was rerun with checkpoint selection
enabled. This addresses the failure mode where `last.pt` can be worse than an
earlier recurrent-state checkpoint.

```text
run: /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/qtrm_source_pointer_state_select_best
train: data/filtered/qtrm_absolute_ordered_state_train512_v0to31.jsonl
eval: data/eval/qtrm_absolute_ordered_state_eval128_v0to31.jsonl
mode: source_position
candidate checkpoints: step_000100, step_000200, step_000300, last
best checkpoint: step_000200
```

Candidate full-eval results:

```json
[
  {"checkpoint": "step_000100", "trace_exact_accuracy": 0.0, "value_accuracy": 0.013888888888888888, "step_exact_accuracy": 0.0},
  {"checkpoint": "step_000200", "trace_exact_accuracy": 0.0, "value_accuracy": 0.017543859649122806, "step_exact_accuracy": 0.0},
  {"checkpoint": "step_000300", "trace_exact_accuracy": 0.0, "value_accuracy": 0.0, "step_exact_accuracy": 0.0},
  {"checkpoint": "last", "trace_exact_accuracy": 0.0, "value_accuracy": 0.0, "step_exact_accuracy": 0.0}
]
```

Best-candidate primitive-off ablation:

```json
{
  "full_trace_exact_accuracy": 0.0,
  "full_value_accuracy": 0.017543859649122806,
  "ablation_value_accuracy": 0.0,
  "value_drop": 0.017543859649122806,
  "decision": "rejected"
}
```

Conclusion: source-position pointer state is still not a valid L2 promotion on
the corrected combination split. The earlier heldout72 source-pointer result
was too narrow. The current bottleneck is not the answer renderer. It is
prompt-position binding plus recurrent pointer update generalization.

Next architecture direction:

```text
prompt token stream
-> explicit source-slot binder from actual prompt tokens
-> recurrent pointer/filter/update core
-> pointer-state probe must pass held-out combination split
-> only then copy/edit or LM renderer work
```

## Runner Result 2026-05-09T10:04:48

```text
gate: qtrm_source_pointer_state
target_level: L2 local gate
profile: smoke
decision: rejected
accepted: False
next_action: do not add renderer complexity; fix prompt-position binding or recurrent pointer updates before claiming L2 state progress
```

Decisive metrics:

```json
{
  "full_trace_exact_accuracy": 0.0,
  "full_value_accuracy": 0.2,
  "full_step_exact_accuracy": 0.0,
  "ablation_trace_exact_accuracy": 0.0,
  "ablation_value_accuracy": 0.0,
  "ablation_step_exact_accuracy": 0.0,
  "trace_drop": 0.0,
  "value_drop": 0.2
}
```

Report: `local_eval/research_gate_runner/qtrm_source_pointer_state_smoke/report.json`

## Runner Result 2026-05-09T10:05:55

```text
gate: qtrm_source_pointer_state
target_level: L2 local gate
profile: standard
decision: rejected
accepted: False
next_action: do not add renderer complexity; fix prompt-position binding or recurrent pointer updates before claiming L2 state progress
```

Decisive metrics:

```json
{
  "full_trace_exact_accuracy": 0.0,
  "full_value_accuracy": 0.0,
  "full_step_exact_accuracy": 0.0,
  "ablation_trace_exact_accuracy": 0.0,
  "ablation_value_accuracy": 0.0,
  "ablation_step_exact_accuracy": 0.0,
  "trace_drop": 0.0,
  "value_drop": 0.0
}
```

Report: `/mnt/nvme1n1p2/qtrm-runs/research_gate_runner/qtrm_source_pointer_state_standard/report.json`

## Numeric Source Feature And Role-Attention Check 2026-05-09

After the numeric-aware prompt binder passed L1, the feature was ported into
the QTRM source-pointer gate as projected source-slot tokens. The gate runner
now evaluates both primitive-off and numeric-source-off ablations.

Smoke and 100-step diagnostics executed successfully, but the numeric feature
was not causal:

```text
run: /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/qtrm_numeric_source_pointer_state_s100_diag
full value accuracy: 0.0730
primitive-off value accuracy: 0.0000
numeric-source-off value accuracy: 0.0730
numeric value drop: 0.0000
decision: rejected
```

This means the primitive executor still affects the answer-state probe, but
the numeric source feature is not being used by the recurrent state path.

A role-conditioned prompt/source token attention module was then added as an
optional architecture candidate so each role/value state can query the prompt
tokens independently instead of sharing one global prompt-context vector.
That candidate also failed the 100-step diagnostic:

```text
run: /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/qtrm_numeric_source_pointer_state_role_attn_s100_diag
full value accuracy: 0.0309
primitive-off value accuracy: 0.0000
numeric-source-off value accuracy: 0.0309
numeric value drop: 0.0000
decision: rejected
```

Decision:

```text
numeric visual/source feature path: rejected as L2 promotion
role-conditioned primitive prompt attention: keep optional, not promoted
```

Next direction: do not add more side-channel numeric features. The next
candidate should make value-aware numeric representation part of the canonical
prompt/token path, then re-run the prompt source-position binder before
retrying QTRM L2.

## Token-Path Numeric QTRM L2 Check 2026-05-09

The canonical token-path value-aware binder passed L1:

```text
run: /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/prompt_source_position_binder_token_plus_numeric_standard
input source: token_plus_numeric_value
best exact accuracy: 0.9453125
decision: accepted_l1
```

This improves over the token-only binder while keeping numeric information in
the prompt/token representation rather than a separate visual/source side
channel.

The same idea was then ported into QTRM as `token_numeric_value_ids` added
directly to `text_seq = token_embedding(input_ids)`. The 100-step QTRM L2
diagnostic did not pass:

```text
run: /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/qtrm_token_numeric_source_pointer_state_s100_diag
full value accuracy: 0.0140
primitive-off value accuracy: 0.0000
token-numeric-off value accuracy: 0.0197
token-numeric value drop: -0.0056
decision: rejected
```

Decision:

```text
canonical token-path numeric binder: accepted_l1
QTRM token-numeric recurrent state: rejected_l2
```

Interpretation: the numeric representation itself is valid in a small binder,
but QTRM does not yet transfer that representation into its recurrent
source-pointer state. The next bottleneck is not number encoding alone; it is
the interface between token-context binding and the recursive role/value state
update.

## Internal Binder Smoke 2026-05-09

The L1 binder was promoted into the QTRM causal path as an optional internal
token-context source-position binder:

```text
prompt tokens
-> token embeddings + gated token-numeric value residual
-> prelude token context
-> gated source-position binder logits
-> core_role_value_state_prompt_logits
-> primitive recurrent role/value state
```

This keeps the candidate inside the universal LLM path. It is not a separate
solver and it is ablatable through `--disable-core-source-position-binder`.

Two failure checks were useful:

```text
ungated internal binder, 20 steps:
  full value accuracy: 0.0284
  token-numeric-off:   0.0284
  binder-off:          0.1932
  decision: rejected

gated binder + gated token-numeric residual, 20 steps:
  full value accuracy: 0.0000
  primitive-off:       0.0000
  token-numeric-off:   0.0000
  binder-off:          0.0000
  decision: rejected
```

Decision:

```text
internal binder wiring: implemented
unguarded overwrite: rejected
gated residual protection: implemented
L2 source-pointer promotion: still rejected
```

Interpretation: the L1 binder architecture is valid in isolation, but simply
adding a randomly initialized binder to a partially trained QTRM checkpoint is
not enough. The next candidate should pretrain or import the L1
token+numeric binder weights into the internal QTRM binder before recurrent
primitive-state training. Longer training alone is not a clean next step unless
the binder initialization is controlled and a binder-off ablation is required.

## Runner Result 2026-05-09T11:02:23

```text
gate: qtrm_numeric_source_pointer_state
target_level: L2 local gate
profile: smoke
decision: rejected
accepted: False
next_action: numeric-aware L1 does not yet route causally through QTRM; inspect projector/core binding and recurrent pointer update
```

Decisive metrics:

```json
{
  "full_trace_exact_accuracy": 0.0,
  "full_value_accuracy": 0.2,
  "full_step_exact_accuracy": 0.0,
  "ablation_trace_exact_accuracy": 0.0,
  "ablation_value_accuracy": 0.0,
  "ablation_step_exact_accuracy": 0.0,
  "trace_drop": 0.0,
  "value_drop": 0.2,
  "numeric_ablation_value_accuracy": 0.2,
  "numeric_value_drop": 0.0
}
```

Report: `/mnt/nvme1n1p2/qtrm-runs/research_gate_runner/qtrm_numeric_source_pointer_state_smoke/report.json`

## Runner Result 2026-05-09T11:15:44

```text
gate: qtrm_numeric_source_pointer_state
target_level: L2 local gate
profile: smoke
decision: rejected
accepted: False
next_action: numeric-aware L1 does not yet route causally through QTRM; inspect projector/core binding and recurrent pointer update
```

Decisive metrics:

```json
{
  "full_trace_exact_accuracy": 0.0,
  "full_value_accuracy": 0.2,
  "full_step_exact_accuracy": 0.0,
  "ablation_trace_exact_accuracy": 0.0,
  "ablation_value_accuracy": 0.0,
  "ablation_step_exact_accuracy": 0.0,
  "trace_drop": 0.0,
  "value_drop": 0.2,
  "numeric_ablation_value_accuracy": 0.2,
  "numeric_value_drop": 0.0
}
```

Report: `/mnt/nvme1n1p2/qtrm-runs/research_gate_runner/qtrm_numeric_source_pointer_state_role_attn_smoke/report.json`

## Runner Result 2026-05-09T11:40:16

```text
gate: qtrm_token_numeric_source_pointer_state
target_level: L2 local gate
profile: smoke
decision: rejected
accepted: False
next_action: token-path L1 binding has not yet become QTRM recurrent L2; inspect token numeric embedding load/training and pointer update
```

Decisive metrics:

```json
{
  "full_trace_exact_accuracy": 0.0,
  "full_value_accuracy": 0.2,
  "full_step_exact_accuracy": 0.0,
  "ablation_trace_exact_accuracy": 0.0,
  "ablation_value_accuracy": 0.0,
  "ablation_step_exact_accuracy": 0.0,
  "trace_drop": 0.0,
  "value_drop": 0.2,
  "token_numeric_ablation_value_accuracy": 0.2,
  "token_numeric_value_drop": 0.0
}
```

Report: `/mnt/nvme1n1p2/qtrm-runs/research_gate_runner/qtrm_token_numeric_source_pointer_state_smoke/report.json`

## Accepted L2 After Role-Capacity And Target Fix 2026-05-09

The previous L2 retries mixed three problems:

```text
1. The data rows said role_value_list_class_mode=absolute, while the gate
   intended source_position. Training/eval now force the selected mode into
   every row before target construction.
2. Prompt-state eval compared the prompt-initial binder against transition
   targets. Eval now supports role_value_target_mode=initial.
3. A 10-role state gives only (10 - 2) / 2 = 4 list fields, but the gate data
   uses length-5 lists. The source-position gate now uses 12 roles so it can
   represent five raw slots, five doubled slots, and two scalar slots.
```

A second target bug also mattered: in `list_transform`, a single-number state
such as `"8"` is a one-element list, not a scalar arithmetic state. The target
builder now treats scalar-looking states as one-element lists only for
`task_family=list_transform`.

Validated tests:

```text
tests.test_qtrm_algorithmic_value_state_eval
tests.test_qtrm_source_pointer_state_gate
```

L1 prompt/source binder rerun:

```text
run:
  /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/
  qtrm_internal_forced_gate_binder_prompt_l1_roles12_s300_sourcefix
accepted checkpoint:
  accepted_l1_prompt_binder_roles12_step_000050.pt
held-out initial full:
  trace_exact_accuracy = 1.0
  value_accuracy       = 1.0
source-binder-off:
  trace_exact_accuracy = 0.0
  value_accuracy       = 0.0
```

L2 recurrent source-pointer rerun:

```text
run:
  /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/
  qtrm_l2_source_pointer_roles12_targetfix_s120
accepted checkpoint:
  accepted_l2_source_pointer_roles12_step_000040.pt
manual decision:
  manual_l2_decision_step_000040.json
```

Metrics:

```json
{
  "decision": "accepted_l2",
  "full_trace_exact_accuracy": 0.1796875,
  "full_value_accuracy": 0.5093123209169055,
  "primitive_off_value_accuracy": 0.0,
  "primitive_value_drop": 0.5093123209169055,
  "source_binder_off_value_accuracy": 0.03939828080229226,
  "source_binder_value_drop": 0.46991404011461324,
  "token_numeric_off_value_accuracy": 0.5093123209169055,
  "token_numeric_value_drop": 0.0
}
```

Decision:

```text
accepted as L2 local gate
not L3/L4
```

Why this is a real L2 and not just a renderer/probe artifact:

```text
full exceeds the local trace/value thresholds
primitive executor off collapses value accuracy to 0
source-position binder off collapses trace exact to 0 and value accuracy near 0
the accepted checkpoint is preserved by hardlink
the final answer renderer is still outside this claim
```

Remaining caveat:

```text
token-numeric-off has no drop.
```

So this L2 proves that the internal prompt/source binder plus primitive
recurrent state is causal. It does not prove that the explicit token-numeric
embedding path is causal. For the next major gate, either the prompt/donor
hidden state is already carrying enough numeric/parity information, or we need
a stricter token-numeric causal split that forces unseen numeric patterns where
plain text/donor shortcuts are insufficient.

LeWM is not the immediate fix for this gate. LeWM-style next-latent prediction
can be useful only after it is semantically anchored to these corrected
role/value transition targets. A self-latent LeWM objective alone was already
demoted because it improved transition MSE without improving answer/state
correctness.

## L3 Hard Gate: Primitive Core Accepted, Strict Binder Rejected 2026-05-09

Hard perturbation split:

```text
script: scripts/321_run_source_pointer_l3_hard_gate.py
train: data/filtered/qtrm_source_pointer_l3_hard_train512_s1321.jsonl
eval: data/eval/qtrm_source_pointer_l3_hard_eval128.jsonl
variants:
  range_shift_v32to63
  fifth_position_single_even
  duplicate_even_binding
  surface_paraphrase
```

The first hard eval rejected L3 because `fifth_position_single_even` was 0%
value accuracy. The fix was a state-codec correction: hard rows now set
`role_value_supervise_null_slots=true`, so padded list roles are supervised as
class `0` instead of being ignored as `-100`.

Accepted primitive-core checkpoint:

```text
run: /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/qtrm_l3_source_pointer_roles12_null_tune_s200
checkpoint: train/accepted_l3_primitive_core_null_step_000050.pt
manual report: manual_l3_null_decision_step_000050.json
audit report:
  /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/
  qtrm_l3_source_pointer_roles12_null_step50_l3_audit/report.json
```

Null-slot hard eval metrics:

```json
{
  "gate": "primitive_core_l3_gate",
  "decision": "accepted_l3",
  "full_trace_exact_accuracy": 0.2890625,
  "full_value_accuracy": 0.63125,
  "primitive_off_value_accuracy": 0.0,
  "primitive_value_drop": 0.63125,
  "token_numeric_off_value_accuracy": 0.63125,
  "source_binder_off_value_accuracy": 0.515625,
  "source_binder_value_drop": 0.115625,
  "variant_value_accuracy": {
    "duplicate_even_binding": 1.0,
    "fifth_position_single_even": 0.4,
    "range_shift_v32to63": 0.58125,
    "surface_paraphrase": 0.54375
  }
}
```

Decision:

```text
primitive recurrent core: accepted_l3
null-slot state codec: promoted for hard list-transform gates
source binder strict causal drop: rejected as a required L3 condition
token numeric path: diagnostic only; still no causal drop
```

The accepted L3 claim is narrow: recurrent primitive state is causal on a
held-out perturbation split, and primitive-off collapses value accuracy to 0.
The source-position binder remains auxiliary, not the canonical intelligence
claim. L4 still requires routing this state through the canonical LM answer
path without damaging donor language behavior.

## Strict Source-Binder Repair Reject 2026-05-09

The narrow L3 primitive-core checkpoint was not sufficient for strict source
binder promotion. A short repair run started from the accepted primitive-core
checkpoint and required source-binder-off to drop strongly:

```text
run:
  /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/
  qtrm_l3_source_pointer_strict_repair_s040
init:
  qtrm_l3_source_pointer_roles12_null_tune_s200/train/
  accepted_l3_primitive_core_null_step_000050.pt
target level: L2 local gate
major bottleneck: QTRM source-position ordered recurrent state refresh
decision: rejected
```

Decisive metrics:

```json
{
  "full_trace_exact_accuracy": 0.0625,
  "full_value_accuracy": 0.59375,
  "primitive_off_value_accuracy": 0.0,
  "token_numeric_off_value_accuracy": 0.59375,
  "source_binder_off_value_accuracy": 0.4765625,
  "source_binder_value_drop": 0.1171875
}
```

Reject reasons:

```text
full trace exact below minimum
full value accuracy below minimum
source-position-binder-off ablation does not drop enough
```

Error inspection showed the model often emits a near-fixed source-position
pattern like `[1, 5, 0, 0, 0]` instead of reading the actual prompt/source
positions. Therefore the current bottleneck is not extra training steps or LM
rendering. It is source binding becoming the causal recurrent state.

Follow-up prompt-binder probes on the same hard split show:

```text
numeric fixed source-slot input: accepted_l1, best_exact_acc 0.9609375
token_plus_numeric prompt path: rejected, best_exact_acc 0.8046875
donor_hidden prompt path: rejected, best_exact_acc 0.671875
token-derived compact source slots: accepted_l1, best_exact_acc 0.9921875
```

Decision:

```text
Do not promote the narrow primitive-core L3 into a general L3/L4 claim.
Do not retry L4 renderer until source binding is causal on the hard split.
Next candidate: token-derived compact source-slot latent, followed by recurrent
core update and source-slot-off/core-off ablations.
```

## Token-Derived Source-Slot QTRM Port Reject 2026-05-09

The L1 prompt-binder result was ported into the QTRM forward path as compact
token-derived source-slot embeddings. The source slots are derived from
tokenizer offsets over the visible prompt, then prepended to the canonical model
context before the recursive core. This preserves the universal LLM path more
cleanly than a hidden answer channel, but it still must pass ablation.

Run:

```text
/mnt/nvme1n1p2/qtrm-runs/research_gate_runner/
qtrm_l2_source_pointer_token_source_slots_s040_smoke
```

Gate:

```text
target level: L2 local gate
major bottleneck: QTRM source-position ordered recurrent state refresh
baseline to beat: primitive-off, source-slot-off, source-binder-off
minimum: trace_exact >= 0.30, value_accuracy >= 0.65
decision: rejected
```

Decisive metrics:

```json
{
  "full_trace_exact_accuracy": 0.0,
  "full_value_accuracy": 0.50625,
  "primitive_off_value_accuracy": 0.0,
  "source_slot_off_value_accuracy": 0.50625,
  "source_binder_off_value_accuracy": 0.503125,
  "source_slot_value_drop": 0.0,
  "source_binder_value_drop": 0.003125
}
```

Decision:

```text
token-derived source-slot input: rejected as an L2 QTRM promotion
primitive core path: still causal, but insufficient for source binding
source-slot module: not causal yet; source-slot-off ties full
source-binder module: not causal yet; binder-off nearly ties full
```

Interpretation:

The compact source-slot representation is learnable in an isolated L1 probe,
but the integrated QTRM model bypasses it or fails to bind it into recurrent
state. This is a root-path problem, not a data-size or renderer problem. The
next candidate should force the recurrent core to own source binding directly,
for example by replacing prepended source-slot tokens with a mandatory
recurrent state update target and rejecting any model whose source-slot-off
ablation does not drop.

## L4 Blocker Repair Contract: Source Binder To Core State 2026-05-09

This is not an L4 promotion claim. It is a blocker repair for L4 source binding:
the normal LM path cannot be promoted while source binding is non-causal.

Prior principle:

```text
Pointer/copy-style state should bind positions or spans from the visible prompt
and then update that state recurrently. This follows the general pointer
network / retrieval-conditioned transformer / recurrent-state principle, but
the exact QTRM source-binder state injection is an original hypothesis.
Therefore it is capped at L2/L3 until ablation proves it.
```

QTRM tensor path:

```text
prompt text -> tokenizer offsets -> token_numeric_source_slot_ids
prompt token embeddings -> prelude text_context_seq
text_context_seq -> core_source_position_binder_logits
source_position_logits -> value embedding -> core role-state token delta
core role-state tokens -> QTRM recursive core trajectory
trajectory -> core_primitive_role_value_state_logits
later L4 only: role/value state -> LM answer path -> autoregressive text
```

Causal ablation:

```text
full must beat:
  --disable-core-primitive-role-value-executor
  --disable-token-numeric-source-slots
  --disable-core-source-position-binder

Minimum local repair target:
  trace_exact >= 0.30
  value_accuracy >= 0.65
  primitive/source-slot/source-binder value drops >= 0.20-0.25
```

Shortcut risk:

```text
The source binder must not become a hidden answer solver. It may initialize
source-position state only. The recurrent primitive core must still update
state causally, and source-slot-off/source-binder-off must drop.
```

Kill criterion:

```text
If source-binder-off ties full, the binder is still non-causal.
If full value accuracy collapses while binder-off drops, injection is too
strong and must be bounded rather than promoted.
If the LM answer path later ties donor-only/core-off, L4 is still rejected.
```

Initial unbounded state-injection smoke:

```text
run:
  /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/
  qtrm_l4_blocker_source_binder_state_init_s080
decision: rejected
full_trace_exact_accuracy: 0.0625
full_value_accuracy: 0.19375
source_binder_value_drop: 0.1296875
source_slot_value_drop: 0.0
```

Interpretation:

The patch made the source binder more causal than before
(`source_binder_value_drop` rose from about `0.003` to `0.130`), but the full
path collapsed. Therefore the next candidate is bounded state injection with a
separate `core_source_position_binder_state_gate_min`, not a stronger binder.

Bounded state-injection smoke:

```text
run:
  /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/
  qtrm_l4_blocker_source_binder_state_init_gate025_s080
checkpoint:
  train/step_000040.pt
state gate:
  core_source_position_binder_state_gate_min=0.25
decision:
  rejected / partial signal only
```

Metrics:

```json
{
  "full_trace_exact_accuracy": 0.25,
  "full_value_accuracy": 0.5625,
  "primitive_off_value_accuracy": 0.0,
  "source_binder_off_trace_exact_accuracy": 0.0,
  "source_binder_off_value_accuracy": 0.5125,
  "source_slot_off_trace_exact_accuracy": 0.25,
  "source_slot_off_value_accuracy": 0.5625
}
```

Decision:

```text
primitive recurrent core remains strongly causal.
source binder now affects trace exactness but not value accuracy enough.
token-derived compact source slots are redundant in this configuration.
do not promote to L4.
```

Next hypothesis:

The canonical path should prefer visible prompt-token binding over compact
source-slot side features. The source-slot module should be demoted unless a
future gate explicitly needs it. The next repair should improve prompt-token
source binding into recurrent state without adding another side channel.

Prompt-only binder smoke:

```text
run:
  /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/
  qtrm_l4_blocker_prompt_binder_state_gate025_s080
checkpoint:
  train/step_000040.pt
source slots:
  disabled / not used
decision:
  rejected / partial signal only
```

Metrics:

```json
{
  "full_trace_exact_accuracy": 0.25,
  "full_value_accuracy": 0.5625,
  "primitive_off_value_accuracy": 0.0,
  "source_binder_off_trace_exact_accuracy": 0.0,
  "source_binder_off_value_accuracy": 0.5125,
  "prompt_role_off_trace_exact_accuracy": 0.0,
  "prompt_role_off_value_accuracy": 0.5125
}
```

Decision:

```text
source slots are not needed for the current partial signal.
generic prompt_extract was not the hidden source of the score; prompt_role_off
ties source_binder_off.
the main remaining score comes from the primitive checkpoint/fallback state.
source binder improves trace exactness but its value effect is too weak.
```

Next hypothesis:

The source binder is too soft: position logits are converted to a soft value
embedding delta, which may blur discrete pointer state. A stricter pointer-style
state codec, such as straight-through hard source-position binding, is the next
candidate. It must still be derived from visible prompt tokens and rejected if
prompt/source-binder ablations tie full.

Straight-through source-state smoke:

```text
run:
  /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/
  qtrm_l4_blocker_prompt_binder_state_st_gate025_s080
checkpoint:
  train/step_000080.pt
state codec:
  core_source_position_binder_state_straight_through=true
decision:
  rejected
```

Metrics:

```json
{
  "full_trace_exact_accuracy": 0.25,
  "full_value_accuracy": 0.5625,
  "primitive_off_value_accuracy": 0.0,
  "source_binder_off_trace_exact_accuracy": 0.0,
  "source_binder_off_value_accuracy": 0.5125,
  "source_binder_value_drop": 0.05
}
```

Decision:

```text
straight-through hard source-position state does not improve over soft bounded
state on the hard split.
primitive core remains necessary, but source binding is still too weak for L4.
do not tune renderer or LM answer path from this checkpoint.
```

Updated root-cause hypothesis:

The current source-position state codec mostly tells the core which source
slots exist, but it does not provide a robust value/parity/readout contract for
the transformed numeric answer. The model can preserve some trace structure,
but value accuracy saturates near `0.56` and binder-off remains `0.5125`.
The next architecture candidate must separate:

```text
1. source position binding: which prompt spans are relevant;
2. source value reading: what numeric value or token span is bound there;
3. primitive transition: what operation updates the latent state;
4. answer rendering: how the final state maps back to text.
```

This should be treated as an L4 blocker, not as a renderer problem.

Token-numeric plus straight-through source-state smoke:

```text
run:
  /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/
  qtrm_l4_blocker_token_numeric_prompt_binder_state_st_gate025_s080
checkpoint:
  train/step_000040.pt
decision:
  rejected
```

Metrics:

```json
{
  "full_trace_exact_accuracy": 0.2890625,
  "full_value_accuracy": 0.63125,
  "primitive_off_value_accuracy": 0.0,
  "token_numeric_off_value_accuracy": 0.63125,
  "source_binder_off_value_accuracy": 0.515625,
  "token_numeric_value_drop": 0.0,
  "source_binder_value_drop": 0.115625
}
```

Decision:

```text
the token-numeric residual is not causal in the integrated QTRM path.
full score is near the local threshold but fails strict value/trace targets.
the source binder still contributes too little value information.
do not add more numeric residual tuning before changing the binder-state path.
```

## L4 Blocker Repair Contract: Query-State Source Reader 2026-05-09

This is a blocker repair, not an L4 promotion claim.

Prior principle:

```text
Latent query readers such as Perceiver-style resamplers and Q-Former-style
query tokens let trainable slots attend to a token sequence and carry the
attended hidden state forward. The QTRM adaptation is to use the source-binder
query hidden states as recurrent role-state initialization, while keeping the
visible prompt/token context as the only input source.
```

QTRM tensor path:

```text
prompt text -> tokenizer -> token embeddings / donor text states
-> prelude text_context_seq
-> source-binder slot queries attend to text_context_seq
-> query hidden states + source-position logits
-> gated core role-state token delta
-> QTRM recursive core
-> primitive role/value state logits
later L4 only: role/value state -> canonical LM logits -> generated text
```

Causal ablation:

```text
full must beat:
  --disable-core-primitive-role-value-executor
  --disable-core-source-position-binder
  --disable-token-numeric-value-features if token numeric features are enabled

Minimum blocker repair target:
  trace_exact >= 0.30
  value_accuracy >= 0.65
  source-binder value drop >= 0.20
  primitive value drop >= 0.25
```

Shortcut risk:

```text
The query-state reader must not compute the answer or bypass the recursive
core. It can only initialize role-state tokens from prompt-token hidden states;
the primitive recurrent core must still perform the state transition.
```

Kill criterion:

```text
If source-binder-off ties full, the query-state reader is non-causal.
If primitive-off does not collapse value accuracy, the result is not recurrent
reasoning.
If token-numeric-off ties full, token numeric remains diagnostic only.
If the later LM answer path ties donor-only/core-off, L4 is still rejected.
```

Query-state reader smoke:

```text
run:
  /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/
  qtrm_l4_blocker_query_state_reader_gate025_s040
checkpoint:
  train/step_000040.pt
state reader:
  core_source_position_binder_query_state_enabled=true
  core_source_position_binder_query_state_gate_min=0.25
decision:
  rejected
```

Metrics:

```json
{
  "full_trace_exact_accuracy": 0.0,
  "full_value_accuracy": 0.5890625,
  "primitive_off_value_accuracy": 0.0,
  "source_binder_off_value_accuracy": 0.515625,
  "source_binder_value_drop": 0.0734375
}
```

Decision:

```text
query hidden-state injection is implemented and causal-testable, but the first
smoke rejected it.
the injected query state disrupted the existing source-position codec: trace
exact collapsed from the previous partial 0.25-0.289 range to 0.0.
source-binder value drop weakened, so this is not the missing value reader.
keep the implementation as a diagnostic switch, but do not promote it.
```

Updated blocker:

```text
The problem is no longer "can the prompt query carry value information?".
The problem is that value information and position-pointer information need
separate state slots or losses. Blending query hidden states into the same
position roles destroys the pointer codec. The next candidate should separate
source-position state and source-value state, then require both ablations to
drop before any renderer or LM-path promotion.
```

## L4 Blocker Repair Contract: Factorized Source-Value Reader 2026-05-09

This is a blocker repair, not an L4 promotion claim.

Prior principle:

```text
Structured state should separate entity identity / pointer location from
entity attributes. For this source-pointer gate, source position and source
numeric value are different factors. A factorized reader is closer to key/value
binding than a single blended hidden-state delta.
```

QTRM tensor path:

```text
prompt text -> tokenizer -> prelude text_context_seq
-> source-binder slot queries
-> position head predicts source slot ids
-> value head predicts numeric value classes for those source slots
-> position embedding delta + value embedding delta initialize role-state tokens
-> recursive primitive core updates role/value state
-> primitive role/value logits
```

Causal ablation:

```text
full must beat:
  --disable-core-primitive-role-value-executor
  --disable-core-source-position-binder

The source-value reader is not accepted if it only improves an auxiliary CE.
It must improve held-out trace/value and source-binder-off must drop.
```

Shortcut risk:

```text
The source-value reader must not produce the final answer. It can only encode
the prompt source values into initial role-state tokens; recurrent primitive
state still has to choose/filter/transform the list.
```

Kill criterion:

```text
If full trace/value do not improve over the previous partial baseline, reject.
If source-binder-off ties full, reject.
If primitive-off does not collapse, reject.
If later LM answer generation ties donor-only/core-off, L4 is still rejected.
```

Factorized source-value reader smoke:

```text
run:
  /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/
  qtrm_l4_blocker_factorized_source_value_gate025_s040
checkpoint:
  train/step_000040.pt
reader:
  core_source_value_binder_enabled=true
  core_source_value_binder_state_gate_min=0.25
  core_source_value_binder_state_straight_through=true
  core_source_value_prompt_ce_weight=1.0
decision:
  rejected
```

Metrics:

```json
{
  "full_trace_exact_accuracy": 0.2890625,
  "full_value_accuracy": 0.63125,
  "primitive_off_value_accuracy": 0.0,
  "source_binder_off_value_accuracy": 0.515625,
  "source_binder_value_drop": 0.115625
}
```

Decision:

```text
the factorized value reader does not hurt the partial baseline, unlike raw
query-state injection.
however, it does not improve over the previous partial baseline and the
source-binder-off drop remains too weak.
the new value factor is implemented and supervised, but the recurrent primitive
path is not yet using it causally.
do not promote this to L4 or renderer work.
```

Updated blocker:

```text
Adding source-value state to the prompt initialization is insufficient unless
the primitive recurrent executor is explicitly conditioned to consume the
factorized value state. The next repair should target the primitive update
itself: value-conditioned primitive transition or source-value feedback into
the primitive role/value executor, with primitive-off and value-reader-off
ablations.
```

## L4 Blocker Repair Contract: Value-Conditioned Primitive Update 2026-05-09

Prior principle:

```text
If state has separate factors, the transition function must consume the
relevant factors. A source-value reader that only initializes hidden tokens is
not enough when the primitive executor's recurrence starts from position
logits. The transition MLP needs an explicit value-conditioning path.
```

QTRM tensor path:

```text
prompt tokens -> source-position/value reader logits
-> source-value class probabilities -> value embeddings
-> gated additive conditioning into primitive role/value update hidden states
-> recurrent primitive state logits
```

Causal ablation:

```text
full must beat primitive-off and source-binder-off.
If a dedicated source-value-off ablation is added, it must also drop.
```

Shortcut risk:

```text
The conditioning path must not compute the final answer. It only provides the
primitive transition with source-value features for the same roles whose
position pointers are already represented.
```

Kill criterion:

```text
If value accuracy ties the factorized-reader baseline or source-binder-off
still ties full, reject.
```

Value-conditioned primitive update smoke:

```text
run:
  /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/
  qtrm_l4_blocker_value_conditioned_primitive_gate025_s040
checkpoint:
  train/step_000040.pt
decision:
  rejected
```

Metrics:

```json
{
  "full_trace_exact_accuracy": 0.046875,
  "full_value_accuracy": 0.61875,
  "primitive_off_value_accuracy": 0.0,
  "source_binder_off_value_accuracy": 0.515625,
  "source_binder_value_drop": 0.103125
}
```

Decision:

```text
conditioning the primitive update on source-value state hurts trace exactness
and does not create a strong source-binder dependency.
do not promote numeric source-value conditioning for this source-position gate.
```

Metric correction:

```text
In this gate, `value_accuracy` means role/value class accuracy, not numeric
answer value accuracy. Under role_value_list_class_mode=source_position, those
classes are source-position ids. Therefore numeric source-value readers do not
directly optimize the measured bottleneck unless the target representation is
changed.
```

Updated root cause:

```text
The immediate blocker is source-position transition and anti-shortcut binding,
not numeric value reading. The model is still able to score around 0.51 value
accuracy with the source binder off because the primitive/fallback path has
learned dataset-level source-position priors. The next repair should create
hard negatives where the same surface/operator family has different source
positions, and require source-binder-off to collapse. Do not spend more local
work on numeric value readers for this gate.
```

## L3 Prerequisite Repair Contract: Paired Source-Position Hard Negatives 2026-05-09

This is a prerequisite repair for L4, not an L4 promotion.

Prior principle:

```text
Counterfactual paired examples are the canonical way to test whether a model
uses the intended causal feature instead of a shortcut. For this gate, the
intended feature is prompt source-position binding. The paired split keeps the
same task surface and same value multiset while changing only the input order,
so source-position targets must change.
```

QTRM tensor path:

```text
prompt text -> tokenizer -> donor/token hidden states
-> source-position binder / prompt-binding path
-> recurrent primitive role/value state
-> source-position class logits
```

Causal ablation:

```text
full must beat:
  --disable-core-primitive-role-value-executor
  --disable-core-source-position-binder

On this paired split, source_binder_off must drop much more than the previous
0.51 shortcut baseline. If it does not, the model is still not causally bound
to prompt source positions.
```

Shortcut risk:

```text
If rows are sampled independently, the model can learn dataset-level priors
about which positions usually contain evens. The paired split blocks that by
placing the same values into multiple different source-position patterns.
```

Kill criterion:

```text
If source_binder_off remains near full, reject the architecture/gate claim.
If full improves only because primitive roles memorize pair ids or answer
surface, reject and add a stricter prompt-binding ablation.
```

Implementation:

```text
script:
  scripts/323_build_source_position_pair_hard_negatives.py
test:
  PYTHONPATH=src .venv/bin/python -m unittest \
    tests.test_source_position_pair_hard_negative_builder
result:
  2 tests passed
smoke:
  train_groups=4 eval_groups=2 permutations_per_group=4
  rows=24
  each pair group shares a value_multiset_signature but has multiple
  source_even_position_signature patterns
```

40-step paired split smoke:

```text
run:
  /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/
  qtrm_pair_source_position_s040
init:
  qtrm_l3_source_pointer_roles12_null_tune_s200/
  train/accepted_l3_primitive_core_null_step_000050.pt
data:
  data/filtered/qtrm_source_position_pair_hard_train512.jsonl
  data/eval/qtrm_source_position_pair_hard_eval128.jsonl
decision:
  rejected
```

Metrics:

```json
{
  "full_trace_exact_accuracy": 0.046875,
  "full_value_accuracy": 0.57958984375,
  "primitive_off_value_accuracy": 0.0,
  "source_binder_off_value_accuracy": 0.57763671875,
  "source_binder_value_drop": 0.001953125
}
```

Interpretation:

```text
primitive-off collapse confirms that the recurrent primitive path is causal.
source-position-binder-off does not collapse, so that ablation was too narrow:
the primitive prompt-context path can still read the prompt and carry source
position information.
```

Strict prompt-binding ablation:

```text
added:
  model forward flag: disable_core_primitive_prompt_context
  eval flag: --disable-core-primitive-prompt-context
  runner option: --strict-prompt-binding-ablation
strict ablation command disables:
  --disable-core-source-position-binder
  --disable-core-role-value-prompt-extract
  --disable-core-primitive-prompt-context
```

Strict result on the same checkpoint:

```json
{
  "strict_prompt_binding_off_trace_exact_accuracy": 0.0,
  "strict_prompt_binding_off_value_accuracy": 0.1806640625,
  "strict_prompt_binding_value_drop": 0.39892578125
}
```

Updated root cause:

```text
The broad prompt-binding path is causal, but the explicit source-position
binder is not yet uniquely responsible. L4 remains blocked because full trace
exact is too low and the architecture still has multiple prompt-reading
routes. The next gate should require strict prompt-binding drop while improving
full trace/value on the paired split; source_binder_off alone is no longer a
sufficient diagnostic.
```

80-step strict gate:

```text
run:
  /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/
  qtrm_pair_source_position_strict_s080
runner:
  --strict-prompt-binding-ablation
  --min-source-binder-value-drop 0.0
  --min-strict-prompt-binding-value-drop 0.25
decision:
  rejected
```

Metrics:

```json
{
  "full_trace_exact_accuracy": 0.0,
  "full_value_accuracy": 0.41259765625,
  "primitive_off_value_accuracy": 0.0,
  "source_binder_off_value_accuracy": 0.412109375,
  "source_binder_value_drop": 0.00048828125,
  "strict_prompt_binding_off_value_accuracy": 0.1484375,
  "strict_prompt_binding_value_drop": 0.26416015625
}
```

Decision:

```text
strict prompt-binding is causal, but longer tuning on the paired split damaged
the full recurrent transition. The 40-step checkpoint is a better partial
baseline than 80-step. Do not solve this by simply increasing steps. The next
repair should inspect role/depth errors and use a conservative schedule or
loss mix that preserves the accepted L3 primitive transition while adding
paired prompt-binding pressure.
```

Forced source-position binder logit gate:

```text
run:
  /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/
  qtrm_pair_source_position_forced_logit_gate_s040
change:
  --core-source-position-binder-gate-min 1.0
  --core-source-position-binder-state-gate-min 0.25
  --strict-prompt-binding-ablation
decision:
  rejected
```

Metrics:

```json
{
  "full_trace_exact_accuracy": 0.109375,
  "full_value_accuracy": 0.587890625,
  "primitive_off_value_accuracy": 0.0,
  "source_binder_off_value_accuracy": 0.25,
  "source_binder_value_drop": 0.337890625,
  "strict_prompt_binding_off_value_accuracy": 0.255859375,
  "strict_prompt_binding_value_drop": 0.33203125,
  "reject_reasons": ["full trace exact below minimum"]
}
```

Interpretation:

```text
Forcing the source-position binder logit gate fixed the previous narrow
ablation problem: the explicit binder is now causally necessary. The run still
does not pass L2 because full trace exact is only 0.109375. This is a
prerequisite repair, not L4/general-LM promotion.

The next repair should improve source-position binding accuracy without
destroying the accepted primitive transition. Candidate directions:
  1. shorter validation-gated schedules around 20-40 steps,
  2. mixed original + paired source-position rows,
  3. depth/role error analysis before adding new heads.
```

ID split and source-slot causal repair sequence:

```text
target:
  L2 source-position ordered recurrent state repair
status:
  prerequisite repair, not L4/general-LM promotion
```

Runs:

```text
qtrm_pair_source_position_id_forced_logit_s060:
  train/eval values both 0-31, disjoint paired hard-negative multisets
  decision: rejected
  full_trace_exact_accuracy: 0.109375
  full_value_accuracy: 0.587890625
  source_binder_value_drop: 0.337890625
  strict_prompt_binding_value_drop: 0.33203125

qtrm_pair_source_position_id_source_slots_s040:
  adds token-derived compact source slots
  decision: rejected
  full_trace_exact_accuracy: 0.109375
  full_value_accuracy: 0.587890625
  source_slot_value_drop: 0.0
  conclusion: source slots are ignored in the normal prompt+binder path

qtrm_pair_source_position_id_source_slots_only_s040:
  restricts source-position binder context to source slots only
  decision: rejected
  full_trace_exact_accuracy: 0.109375
  full_value_accuracy: 0.587890625
  source_slot_value_drop: 0.322265625
  source_binder_value_drop: 0.322265625
  strict_prompt_binding_value_drop: 0.33203125
  conclusion: source slots become causal, but full exact is unchanged

qtrm_pair_source_position_id_numeric_features_s040:
  adds numeric source features
  decision: rejected
  full_trace_exact_accuracy: 0.109375
  full_value_accuracy: 0.587890625
  numeric_value_drop: 0.0
  conclusion: numeric feature path is ignored under this policy/config

qtrm_pair_source_position_id_source_slots_parity_s040:
  adds source-slot-only binder plus token source-slot parity CE
  decision: rejected
  full_trace_exact_accuracy: 0.109375
  full_value_accuracy: 0.587890625
  primitive_off_value_accuracy: 0.0
  source_slot_value_drop: 0.322265625
  source_binder_value_drop: 0.322265625
  strict_prompt_binding_value_drop: 0.33203125
  reject_reasons: ["full trace exact below minimum"]
  rejected_checkpoints_deleted: true

qtrm_pair_source_position_id_pair_trace_contrast_s040:
  adds source-slot-only binder, token source-slot parity CE, and paired
  hard-negative trace contrast:
    --core-primitive-role-value-pair-trace-contrast-weight 1.0
    --core-primitive-role-value-pair-trace-contrast-margin 0.25
  decision: rejected
  full_trace_exact_accuracy: 0.109375
  full_value_accuracy: 0.587890625
  primitive_off_value_accuracy: 0.0
  source_slot_value_drop: 0.322265625
  source_binder_value_drop: 0.322265625
  strict_prompt_binding_value_drop: 0.33203125
  reject_reasons: ["full trace exact below minimum"]
  rejected_checkpoints_deleted: true

qtrm_pair_source_position_id_source_value_conditioned_s040:
  adds factorized source-value binder and conditions the primitive recurrent
  role-value update on source-value state:
    --core-source-value-binder
    --core-source-value-prompt-ce-weight 1.0
    --core-primitive-role-value-source-value-conditioning
  decision: rejected
  full_trace_exact_accuracy: 0.109375
  full_value_accuracy: 0.587890625
  primitive_off_value_accuracy: 0.0
  source_slot_value_drop: 0.322265625
  source_binder_value_drop: 0.322265625
  strict_prompt_binding_value_drop: 0.33203125
  reject_reasons: ["full trace exact below minimum"]
  rejected_checkpoints_deleted: true
```

Key diagnostic:

```text
The parity run learned the auxiliary slot-parity probe on train slices
(token_numeric_source_slot_parity_acc reached 1.0), and ablations now prove
that primitive/core, source slots, source binder, and prompt binding are
causally used. However, eval predictions are still a fixed template:

  128/128 predicted first-step source positions: 1,3,4,0

Per-role accuracy shows template bias rather than input-specific binding:

  role 0: 68/128
  role 1: 60/128
  role 2: 45/128
  role 3: 128/128

This rejects the hypothesis that the main blocker was only OOD value range,
missing source-slot causality, or missing parity perception. The current
blocker is source-position hard-negative binding: the model can use the path,
but it has not learned to select/order source slots differently for paired
inputs with the same multiset.

The pair-trace contrast run preserved the same causal drops but did not change
the eval template collapse. Training logs showed the pair-trace contrast was
already satisfied on the seen train slices, so this loss by itself is not a
strong held-out generalization pressure. Train/eval source-position signatures
are reasonably balanced; the failure is not explained by a missing signature in
the data distribution.

The source-value conditioned run also preserved causal drops but did not change
the fixed template. This rejects a role-query value-injection-only repair: the
model can learn source-value supervision on train slices, but role-indexed
source-value conditioning still does not force source-slot-specific selection
on held-out paired permutations.
```

Promotion lock:

```text
L4/general-LM promotion remains locked. The orthodox next step is not MemoryOS,
larger donor, broader data, or renderer tuning. The next L2/L3 repair must
force input-specific source-position selection on paired hard negatives while
preserving the accepted primitive transition.
```

Next candidate directions:

```text
1. Pairwise anti-template contrastive objective:
   group rows by pair_group_id and penalize identical source-position traces
   when the paired hard negatives require different traces.
   Status: implemented as a scaffold loss, rejected at s040 because it did not
   improve held-out full trace exact.

2. Slot-predicate recurrent scan:
   make a shared learned predicate over source slots, then recurrently update
   the selected position state instead of predicting a fixed role template.
   Status: next root-structure candidate. The next minimal form is learned
   source-slot predicate feedback: predict selected/even per source slot,
   feed that predicate embedding back into the source-slot token path before
   the source-position binder, and reject if held-out traces still collapse to
   one role template.

Orthodox-method audit for source-slot predicate feedback:

```text
Target level:
  L2/L3 prerequisite repair, not L4/general-LM promotion.

Official/prior reference:
  Attention/pointer binding and recurrent state update are the prior family:
  source tokens must carry a learned predicate state before the pointer/binder
  reads them. This is a QTRM-specific minimal scaffold, not an official model
  reproduction.

Minimum faithful reproduction:
  A source-slot token path predicts a per-slot predicate, injects the learned
  predicate embedding back into the same token states, and the downstream
  binder/recurrent core must use those states causally.

QTRM tensor path:
  token_numeric_source_slot_ids -> source_slot_seq -> predicate_logits ->
  predicate embedding feedback -> text_context_seq source-slot prefix ->
  core_source_position_binder -> core primitive role-value state -> eval trace.

Shortcut exclusion:
  The predicate head does not compute the final answer. It only changes token
  states that the existing binder/core path must use. The source-slot-off,
  source-binder-off, primitive-off, and strict prompt-binding ablations remain
  mandatory.

Canonical LLM compatibility:
  This is still a scaffold because it starts from structured source-slot IDs,
  but the intended canonical mapping is prompt tokens -> learned predicate
  states -> pointer/core state -> LM/copy renderer.

Kill criterion:
  Reject if full_trace_exact_accuracy remains below the L2 threshold or if
  predictions stay collapsed to one fixed source-position template on paired
  hard negatives.
```

qtrm_pair_source_position_id_predicate_feedback_s040:
  adds learned source-slot predicate feedback before the binder.
  decision: rejected
  full_trace_exact_accuracy: 0.109375
  full_value_accuracy: 0.587890625
  predicted first-step template: 128/128 -> 1,3,4,0
  conclusion: predicate feedback did not break the held-out source-position
  template collapse.

prompt_source_position_binder_token_source_slots_pair_id_s300:
  standalone source-slot binder probe using token_numeric_source_slots.
  decision: accepted_l1
  best_exact_acc: 1.0
  reaches exact_acc 1.0 by step 100
  conclusion: source-slot IDs and the attention/pointer binder family can
  solve the paired hard-negative split outside integrated QTRM.

qtrm_pair_source_position_id_predicate_feedback_s120:
  longer integrated QTRM run after the accepted standalone probe.
  decision: rejected
  full_trace_exact_accuracy: 0.1015625
  full_value_accuracy: 0.478515625
  predicted first-step template: 128/128 -> 3,4,5,0
  source_slot_value_drop: 0.228515625
  source_binder_value_drop: 0.228515625
  conclusion: more single-row QTRM steps moved the fixed template but did not
  learn input-specific source-slot binding.

Orthodox-method audit for raw source-slot binder context:

```text
Target level:
  L2/L3 prerequisite repair.

Prior/reference:
  The accepted standalone probe is the local faithful reproduction: source-slot
  embeddings feed directly into an attention/pointer binder.

Observed QTRM mismatch:
  Integrated QTRM reads source slots after prelude transformation, while the
  successful L1 probe reads raw source-slot embeddings.

QTRM tensor path:
  token_numeric_source_slot_ids -> raw source_slot_seq -> source-position
  binder -> core role-value state -> primitive recurrent state -> eval trace.

Shortcut exclusion:
  This does not compute answers outside the model. It changes only the states
  the existing binder sees, and the same source-slot/source-binder/primitive
  ablations remain mandatory.

Kill criterion:
  Reject if raw-source-slot binder context still collapses to one template or
  if source-slot/source-binder ablation drops disappear.
```

qtrm_pair_source_position_id_raw_source_slots_s120:
  changes integrated QTRM source-position binder to read raw prepended
  source-slot states before prelude, matching the accepted standalone probe
  tensor path more closely.
  decision: rejected
  full_trace_exact_accuracy: 0.1015625
  full_value_accuracy: 0.478515625
  source_slot_value_drop: 0.228515625
  source_binder_value_drop: 0.228515625
  conclusion: raw source-slot context alone does not fix integrated QTRM.

qtrm_pair_source_position_id_raw_source_slots_lr3e4_s300:
  same raw source-slot context, higher lr=3e-4, 300 integrated single-row
  steps.
  decision: rejected
  full_trace_exact_accuracy: 0.125
  full_value_accuracy: 0.599609375
  source_slot_value_drop: 0.349609375
  source_binder_value_drop: 0.349609375
  strict_prompt_binding_value_drop: 0.349609375
  predicted first-step template: 128/128 -> 2,3,5,0
  conclusion: higher LR restores value/drop pressure but still moves to a new
  fixed source-position template instead of learning input-specific binding.

Current bottleneck after raw-source-slot tests:

```text
The input representation and pointer/binder family are not the blocker:
standalone source-slot binder reaches exact_acc=1.0 on the paired held-out
split. The integrated QTRM blocker is the training/coupling path from
source-position binder logits into recurrent primitive role-value state. With
single-row QTRM updates, the model learns a global template faster than an
input-conditioned source-position rule.

Next orthodox candidates:
  1. Batch the integrated source-position training path so each optimizer step
     sees multiple paired permutations, matching the standalone L1 probe.
  2. Pretrain or transplant the source-position binder from the accepted L1
     probe into QTRM, then fine-tune only the recurrent coupling.
  3. Add an integrated pair-consistency gate that rejects checkpoints whose
     paired hard-negative predictions are identical templates, rather than
     relying only on aggregate trace exact.
```

Orthodox-method audit for integrated batch source-position training:

```text
Claim level:
  L2 prerequisite repair. This is not L4/general-LM promotion.

Canonical path:
  prompt text -> tokenizer/donor hidden states + prompt-derived numeric source
  slots -> QTRM source-position binder -> core role-value prompt state ->
  primitive recurrent role-value state -> existing role-value eval trace.

Prior mapping:
  Same attention/pointer binding family as the accepted standalone L1 source
  slot binder probe; the difference is that the binder remains inside the QTRM
  causal path instead of being a separate probe.

Failure evidence:
  Standalone source-slot binder accepted_l1 with exact_acc=1.0, but integrated
  QTRM single-row runs still collapsed to one fixed first-step template:
  1,3,4,0 then 3,4,5,0 then 2,3,5,0.

Next causal hypothesis:
  Current integrated training gives one row per optimizer step, so gradients
  can reward a global source-position template before paired-permutation
  contrasts are seen together. Batch training should expose multiple hard
  permutations per step and make input-conditioned binder/state coupling
  easier than a global template.

Shortcut exclusion:
  The trainer does not add a solver, answer renderer, or hidden answer path.
  It only changes optimization granularity for the existing QTRM tensor path
  and keeps source-slot, source-binder, primitive-off, and strict prompt-binding
  ablations as the acceptance checks.

Kill criterion:
  Reject if the batch-trained integrated checkpoint still has low trace exact
  or if predicted first-step traces remain one dominant template on held-out
  paired hard negatives.
```

qtrm_pair_source_position_batch_integrated_s300_b16:
  integrated QTRM batch source-position trainer on paired hard negatives.
  trainer: scripts/324_train_qtrm_source_pointer_batch.py
  runner: scripts/319_run_qtrm_source_pointer_state_gate.py
  row_batch_size: 16
  train steps: 300
  best checkpoint: step_000200
  decision: accepted_l2
  full_trace_exact_accuracy: 1.0
  full_value_accuracy: 1.0
  full_step_exact_accuracy: 1.0
  primitive_off_value_accuracy: 0.0
  source_slot_off_value_accuracy: 0.25
  source_binder_off_value_accuracy: 0.25
  strict_prompt_binding_off_value_accuracy: 0.25
  value_drop: 1.0
  source_slot_value_drop: 0.75
  source_binder_value_drop: 0.75
  strict_prompt_binding_value_drop: 0.75
  accepted_checkpoint:
    /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/qtrm_pair_source_position_batch_integrated_s300_b16/train/accepted_l2_source_pointer_refresh.pt

Interpretation:
  The standalone L1 diagnosis was correct: the pointer family and source-slot
  representation were not the blocker. The blocker was integrated single-row
  optimization/coupling. Batch exposure of multiple paired permutations per
  optimizer step removed the fixed-template collapse and made the source-slot,
  source-binder, and primitive recurrent paths causally necessary under
  ablation.

Claim boundary:
  This is a real L2 prerequisite repair for source-position recurrent state
  binding. It is not yet L4/general-LM promotion because the accepted gate is
  still a narrow synthetic list-transform split and it does not verify natural
  language autoregressive answer preservation.

qtrm_pair_source_position_batch_l3_hard_eval:
  evaluated the accepted L2 batch source-position checkpoint on the existing
  L3 hard perturbation split with source-slot/source-binder flags enabled.
  decision: rejected_l3
  full_trace_exact_accuracy: 0.3671875
  full_value_accuracy: 0.591796875
  primitive_off_value_accuracy: 0.0
  token_numeric_off_value_accuracy: 0.40625
  source_binder_off_value_accuracy: 0.40625
  primitive_value_drop: 0.591796875
  token_numeric_value_drop: 0.185546875
  source_binder_value_drop: 0.185546875
  reject_reasons:
    - token-numeric-off ablation does not drop enough
    - source-binder-off ablation does not drop enough
    - at least one hard variant is below minimum value accuracy

Variant values:
  duplicate_even_binding: 1.0
  fifth_position_single_even: 0.25
  range_shift_v32to63: 0.5
  surface_paraphrase: 0.6171875

Interpretation:
  Batch L2 fixed paired-permutation template collapse but did not generalize
  to L3 hard perturbations. The largest concrete failure is the fifth-position
  singleton-even case. Source-slot/source-binder ablations still hurt globally,
  but the drop is too small for strict L3 causality because some hard variants
  fall back to partial template behavior when the source path is disabled.

Next causal hypothesis:
  Train the same integrated batch path on a mixed curriculum that includes L2
  paired hard negatives plus L3 perturbation variants, especially
  fifth_position_single_even and range_shift_v32to63, then rerun the strict
  source-slot/source-binder L3 gate.

qtrm_source_position_l3_hard_batch_s240_b8:
  trained the accepted L2 batch source-position checkpoint on the L3 hard
  train512 split with batch size 8 after batch 16 OOM.
  init:
    qtrm_pair_source_position_batch_integrated_s300_b16/train/accepted_l2_source_pointer_refresh.pt
  train:
    data/filtered/qtrm_source_pointer_l3_hard_train512_s1321.jsonl
  eval:
    data/eval/qtrm_source_pointer_l3_hard_eval128.jsonl
  decision: accepted_l3
  full_trace_exact_accuracy: 0.8203125
  full_value_accuracy: 0.95849609375
  primitive_off_value_accuracy: 0.0
  token_numeric_off_value_accuracy: 0.40625
  source_binder_off_value_accuracy: 0.40625
  primitive_value_drop: 0.95849609375
  token_numeric_value_drop: 0.55224609375
  source_binder_value_drop: 0.55224609375
  min_variant_value_accuracy: 0.916015625
  accepted_checkpoint:
    /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/qtrm_source_position_l3_hard_batch_s240_b8_eval/accepted_l3_last.pt

Variant values:
  duplicate_even_binding: 1.0
  fifth_position_single_even: 1.0
  range_shift_v32to63: 0.91796875
  surface_paraphrase: 0.916015625

Interpretation:
  The source-position recurrent state now passes the strict L3 perturbation
  gate with source-slot/source-binder causality. The earlier fifth-position
  failure was not an architecture impossibility; it was missing L3
  perturbation exposure under the same integrated batch source-pointer path.

Claim boundary:
  This is L3 major-bottleneck progress for source-position recurrent state
  generalization. It is still not L4/general LM promotion because it validates
  internal role-value traces, not natural autoregressive answer generation.

2026-05-14 L4 source-position logits probe preflight:
  command:
    scripts/328_probe_qtrm_source_position_logits.py
  report:
    local_eval/qtrm_l3_source_position_logits_probe_20260514/source_copy_probe.json
  decision:
    checkpoint_chain_missing
  missing base checkpoint:
    local_eval/research_gate_runner/primitive_field_heads_delta_codec_s90_lr5e4_seed11/last.pt

Interpretation:
  The accepted L3 checkpoint is a trainable-delta chain, and one older base
  checkpoint in that chain is no longer present in the repo/local_eval tree.
  This blocks a fair L4 source-position logits probe before model execution.
  Treat this as experiment-operation hygiene, not architecture evidence.

Operational fix:
  The L4 LM path runner and source-position logits probe now preflight the
  checkpoint/base_checkpoint chain and write a structured
  `checkpoint_chain_missing` report instead of crashing with a late
  FileNotFoundError. This is the Autoresearch-style operating rule applied to
  this repo: fail before expensive work, preserve the decisive failure artifact,
  and do not silently continue with a broken baseline.

2026-05-14 checkpoint integrity extension:
  The same preflight now validates known checkpoint sha256 values, not only
  path existence. A regenerated file at the missing-base path is rejected as
  `checkpoint_chain_sha256_mismatch` unless it matches the recorded original
  digest. This prevents a non-equivalent recipe replay from falsely satisfying
  the accepted L3 chain.

  Verified commands:
    .venv/bin/python -m py_compile
      scripts/322_run_source_pointer_l4_lm_path_gate.py
      scripts/328_probe_qtrm_source_position_logits.py
      src/qtrm_mm/qtrm_model.py
    .venv/bin/python -m unittest
      tests.test_source_position_logits_probe
      tests.test_source_pointer_l4_lm_path_gate -v

  Current preflight result:
    source-position probe:
      checkpoint_chain_missing with checkpoint_chain_issues
    L4 LM path runner:
      checkpoint_chain_missing with checkpoint_chain_issues

  Added materialization tool:
    scripts/329_materialize_qtrm_checkpoint_stack.py

  Use after any newly accepted L2/L3 delta checkpoint:
    .venv/bin/python scripts/329_materialize_qtrm_checkpoint_stack.py \
      --config <gate-config.yaml> \
      --checkpoint <accepted-delta.pt> \
      --out <accepted-self-contained.pt>

  Rule:
    Do not promote a new accepted checkpoint to the next level unless either
    the exact base chain is archived by sha256 or a self-contained materialized
    checkpoint has been written and replayed through the acceptance gate.

Next action:
  Restore the missing base checkpoint or materialize a self-contained L3
  checkpoint before running the L4 lexicalization/source-copy probe. Do not
  spend GPU time on L4 tuning from this chain until that preflight passes.

2026-05-14 base regeneration audit:
  expected original base sha256 from the 2026-05-09 source-pointer run:
    9a9204a9b01001713772294afcf30ae5753b0e3cd3877adabb83918caf52747d

  attempted reproduction:
    config:
      configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_primitive_field_heads_delta_codec_s160.yaml
    out:
      local_eval/research_gate_runner/primitive_field_heads_delta_codec_s90_lr5e4_seed11
    steps:
      90
    lr:
      5.0e-5
    seed:
      11
    allow_random_init:
      true

  reproduced last.pt sha256:
    b7ef8923f44ae6e4159157e16535f1011c27a67e7f4bb8bc5935a3f1a7554601

  L3 full replay with reproduced base:
    report:
      local_eval/qtrm_l3_regenerated_base_validation_20260514/full.json
    trace_exact_accuracy:
      0.0
    value_accuracy:
      0.57958984375

  expected accepted L3:
    trace_exact_accuracy:
      0.8203125
    value_accuracy:
      0.95849609375

Decision:
  Rejected. The recreated base is not bit-equivalent and does not preserve the
  accepted L3 chain. It was moved out of the canonical missing-base path and
  deleted after preserving this audit. Do not satisfy the checkpoint preflight
  by recreating this file from recipe alone.

Next valid recovery paths:
  1. Find the exact original file by sha256.
  2. Rebuild the source-pointer L2/L3 stack from a new self-contained base and
     re-accept it under the same L3 full/source-slot/source-binder/primitive
     ablation gates.
  3. Materialize a self-contained checkpoint only after a replayed L3 gate
     matches or exceeds the accepted metrics.

3. Validation-gated mixed curriculum:
   mix original accepted L3 rows with paired hard negatives and keep only
   checkpoints that improve full trace exact while maintaining source-slot,
   source-binder, strict-prompt, and primitive-off drops.
```

## Autoresearch-Style Operations Applied 2026-05-14

Karpathy `autoresearch` is useful here as an experiment-operations pattern, not
as a QTRM architecture prior. The applicable rule is:

```text
fixed budget -> one changed mechanism -> one decisive metric/gate ->
keep/discard/probe/crash ledger -> preserve accepted checkpoint chain
```

Mapping for this source-pointer bottleneck:

```text
autoresearch fixed 5-minute val_bpb
  -> QTRM smoke/standard profiles with fixed eval cases and fixed gates

autoresearch keep/discard loop
  -> QTRM accepted/rejected gate decision plus results.tsv ledger

autoresearch small editable research scope
  -> QTRM one wrapper/recipe per gate, not ad hoc shell history

autoresearch reset bad experiments
  -> QTRM never promote rejected checkpoints or broken delta chains
```

Implemented runner:

```text
scripts/350_rebuild_source_pointer_selfcontained_stack.py
```

Purpose:

```text
self-contained base train
-> L2 source-pointer gate
-> materialize accepted L2
-> L3 hard tune
-> L3 hard audit
-> materialize accepted L3
-> only then allow L4 preflight
```

The runner now writes an Autoresearch-style operation ledger row to:

```text
local_eval/research_gate_runner/results.tsv
```

Smoke execution:

```text
run:
  local_eval/source_pointer_selfcontained_rebuild_smoke_exec_20260514
profile:
  smoke
decision:
  l2_rejected
accepted:
  false
```

Decisive L2 smoke metrics:

```json
{
  "full_trace_exact_accuracy": 0.0,
  "full_value_accuracy": 0.3,
  "value_drop": 0.3,
  "token_numeric_value_drop": -0.10000000000000003,
  "source_binder_value_drop": 0.09999999999999998,
  "full_step_exact_accuracy": 0.0
}
```

Interpretation:

```text
The self-contained rebuild pipeline is operationally valid, but the 5-step
smoke is rejected at L2. That is expected for a wiring smoke; it proves the
runner stops before spending L3/L4 compute on a failed prerequisite.
```

Verified:

```text
.venv/bin/python -m py_compile \
  scripts/329_materialize_qtrm_checkpoint_stack.py \
  scripts/350_rebuild_source_pointer_selfcontained_stack.py \
  scripts/322_run_source_pointer_l4_lm_path_gate.py \
  scripts/328_probe_qtrm_source_position_logits.py \
  tests/test_source_pointer_selfcontained_rebuild_runner.py

.venv/bin/python -m unittest \
  tests.test_source_pointer_selfcontained_rebuild_runner \
  tests.test_training_checkpoint_init \
  tests.test_source_position_logits_probe \
  tests.test_source_pointer_l4_lm_path_gate -v
```

Result:

```text
187 tests passed.
```

2026-05-14 accepted recipe recovery:

```text
Old self-contained rebuild runner mistake:
  It rebuilt L2 through the single-row token_numeric_value_features path.

Accepted historical L2/L3 recipe:
  batch-integrated source-position training
  token_numeric_source_slots
  source-slot predicate feedback
  source-position binder state gate
  source-slots-only + raw-source-slots binder context
  paired hard negative train/eval rows

Operational correction:
  scripts/350_rebuild_source_pointer_selfcontained_stack.py now uses the same
  batch/source-slot recipe for L2 and L3 instead of the earlier single-row
  diagnostic path.
```

Triage after the recipe correction:

```text
run:
  local_eval/source_pointer_selfcontained_rebuild_batch_triage_retry_20260514
decision:
  l2_rejected
primary failed metric:
  full_trace_exact_accuracy = 0.0
causal signal:
  full_value_accuracy = 0.681640625
  source_slot_value_drop = 0.51953125
  source_binder_value_drop = 0.509765625
  strict_prompt_binding_value_drop = 0.431640625
```

Interpretation:

```text
The corrected batch/source-slot path is causal again, but a short 60-step
triage run is not enough to recover full trace exactness from a new
self-contained base. Do not promote it. Use it as evidence that the source-slot
recipe is the right recovery path, then run the standard profile.
```

Current standard recovery run:

```text
run:
  local_eval/source_pointer_selfcontained_rebuild_batch_standard_20260514
profile:
  standard
L2 decision:
  accepted_l2
L2 selected checkpoint:
  01_l2_gate/train/step_000200.pt
L2 full metrics:
  trace_exact_accuracy = 1.0
  value_accuracy = 1.0
  step_exact_accuracy = 1.0
L2 causal ablations:
  primitive_off value_accuracy = 0.0
  source_slot_off value_accuracy = 0.25
  source_binder_off value_accuracy = 0.25
  strict_prompt_binding_off value_accuracy = 0.25
materialized L2:
  02_l2_self_contained/accepted_l2_self_contained.pt
L3 tune selected checkpoint:
  03_l3_tune/train/step_000240.pt
L3 audit decision:
  accepted_l3
L3 audit metrics:
  trace_exact_accuracy = 0.7421875
  value_accuracy = 0.97314453125
  step_exact_accuracy = 0.908203125
L3 audit causal drops:
  primitive_value_drop = 0.97314453125
  token_numeric_value_drop = 0.56689453125
  source_binder_value_drop = 0.56689453125
materialized L3:
  05_l3_self_contained/accepted_l3_self_contained.pt
self-contained L3 replay:
  trace_exact_accuracy = 0.7421875
  value_accuracy = 0.97314453125
root decision:
  accepted_l3_self_contained_stack
```

Interpretation:

```text
The operation loop found the real failure: the first self-contained runner used
the wrong single-row diagnostic path. After switching to the accepted
batch/source-slot recipe, L2 reproduced cleanly from a new self-contained base
and preserved causal ablation drops. This validates the Autoresearch-style
experiment operation rule for this repo: do not argue from memory or shell
history; replay the accepted recipe, gate it, ledger it, and only then move on.
```

Operational bugs caught by this run:

```text
1. L3 audit checkpoint selection bug:
   scripts/350_rebuild_source_pointer_selfcontained_stack.py originally built
   the L3 audit command from l3_save_every, so standard profile audited
   step_000120 even though L3 tune selected step_000240. Fixed: audit now uses
   l3_steps/final selected candidate.

2. Materialization config bug:
   scripts/329_materialize_qtrm_checkpoint_stack.py originally constructed the
   model from YAML only, so source-slot/binder modules enabled by runner flags
   were treated as unexpected keys and silently dropped. The first materialized
   L3 replay failed with trace_exact_accuracy = 0.0. Fixed: materialize accepts
   the same source-slot/binder flags and the runner passes
   --fail-on-unmatched-keys.
```

Current boundary:

```text
This is a valid self-contained L3 recovery, but it is not the historical best:
the archived accepted L3 replay reached trace_exact_accuracy = 0.8203125,
whereas the rebuilt self-contained L3 reaches 0.7421875. Promote it to L4
preflight as a reproducible baseline, not as a new SOTA checkpoint.
```
