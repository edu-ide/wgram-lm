# Fixed Operation To General Recursive Core Roadmap

Status: architecture roadmap, 2026-05-05.

## Decision

The current primitive-runtime result is accepted only as a causal probe:

```text
prompt -> recurrent QTRM core -> fixed operation id -> symbolic executor -> answer
```

It is not yet a general LLM reasoning architecture. The operation set is
hand-defined, and the executor performs the state transition outside the neural
model. This is useful because it isolates whether the recurrent core can read a
prompt and causally select a procedure, but it cannot be the final QTRM claim.

The final direction must remove the hand-coded bottlenecks in stages:

```text
fixed operation policy
-> learned latent operation codebook
-> neural state-transition model
-> open-ended recursive reasoning core
-> memory/metacognition composition gates
```

## Current Accepted Scope

Current accepted claim:

```text
QTRM can use its mandatory recurrent core to select supported primitive
operations from prompt wording, and disabling the core destroys the runtime.
```

Evidence:

```text
canonical heldout7000:
  QTRM primitive runtime: 128/128

OOD surface heldout8000 after augmentation:
  QTRM primitive runtime: 128/128
  QTRM core-off:            0/128

OOD paraphrase stress heldout10000 after heldout-separated recovery:
  baseline from surface-aug checkpoint: 191/256
  recovered QTRM raw primitive runtime: 252/256
  state-constrained primitive runtime:  256/256
  recovered QTRM core-off:                0/256
  state-constrained core-off:             0/256
  recovered canonical heldout7000:      128/128
```

Boundary:

```text
This is a neuro-symbolic scaffold.
The model does not invent new operations.
The model does not yet perform neural state updates by itself.
The result does not prove broad open-domain reasoning or ASI capability.
```

## Stage Roadmap

### Stage 0: Fixed Operation Causal Probe

Purpose:

```text
Prove the recurrent core is causally used.
```

Current status: accepted for the synthetic primitive-runtime gate.

Required before leaving this stage:

- [x] Donor-only baseline measured.
- [x] Core-off ablation measured.
- [x] Held-out canonical gate accepted.
- [x] OOD surface-form gate accepted after augmentation.
- [x] Larger paraphrase stress gate recovered to 252/256 with no canonical regression.
- [x] Operation-family holdout feasibility diagnosed as a fixed-label reject gate.
- [ ] Harder arithmetic chains accepted.
- [ ] Mixed-family multi-step tasks accepted.

Kill criterion:

```text
If fixed operations only work on narrow templates and fail broad paraphrases,
do not promote to learned latent operations. Expand data/gates first.
```

### Stage 0.1: Larger OOD Paraphrase Stress Result

Artifacts:

```text
builder:
  scripts/227_build_pure_recursive_ood_paraphrase_stress_cases.py

heldout:
  data/eval/pure_recursive_primitive_transition_ood_paraphrase_stress_heldout10000_preferences.jsonl

heldout-separated recovery train:
  data/filtered/pure_recursive_primitive_transition_ood_paraphrase_stress_train11000_preferences.jsonl

checkpoint:
  local_eval/qwen35_2b_pure_recursive_primitive_transition_oodstress_s1024_from_surface_aug/last.pt
```

Baseline before recovery:

```text
checkpoint:
  local_eval/qwen35_2b_pure_recursive_primitive_transition_surface_aug_s640_from_promptattn/last.pt

stress heldout10000:
  total:            191/256 = 0.7461
  arithmetic_chain:  58/64
  symbolic_binding:  50/64
  boolean_logic:     64/64
  list_transform:    19/64

core-off:
  total:              0/256
```

After recovery:

```text
stress heldout10000 raw argmax:
  total:            252/256 = 0.9844
  arithmetic_chain:  64/64
  symbolic_binding:  64/64
  boolean_logic:     64/64
  list_transform:    60/64

stress heldout10000 with state-constrained operation decoding:
  total:            256/256 = 1.0000

canonical heldout7000:
  total:            128/128 = 1.0000

core-off on stress heldout10000:
  total:              0/256 = 0.0000

state-constrained core-off on stress heldout10000:
  total:              0/256 = 0.0000
```

Remaining failures:

```text
list_transform variant 7:
  4/8

failure pattern:
  predicted operations:
    filter_even, multiply_sum, hold_final, hold_final

  desired operation at step 2:
    double_filtered
```

Interpretation:

```text
The larger paraphrase failure was mostly recoverable through the causal
primitive-transition path, and the canonical gate did not regress. This is a
valid Stage 0 causal-core improvement, because disabling the core still drops
the stress result to zero.

The remaining raw bottleneck is not answer rendering. It is operation routing
under one list-transform paraphrase surface. A state-constrained operation
decoder can safely mask the invalid `multiply_sum` choice after the core has
already produced the correct `filter_even` first step, raising runtime answer
accuracy to 256/256 while keeping core-off at 0/256. This is a runtime scaffold
improvement, not a stronger raw-core claim.

Do not move to open-ended reasoning claims from this alone. The next proof
should close raw list variant 7 or test operation-family transfer to see
whether the learned operation policy is still template-bound.
```

### Stage 0.2: Operation-Family Holdout Feasibility

Artifacts:

```text
builder:
  scripts/228_build_pure_recursive_operation_family_holdout.py

train split:
  data/filtered/pure_recursive_primitive_transition_family_holdout_list_train.jsonl

eval split:
  data/eval/pure_recursive_primitive_transition_family_holdout_list_eval.jsonl

summary:
  local_eval/qwen35_2b_pure_recursive_primitive_transition_oodstress_s1024_from_surface_aug/operation_family_holdout_list_summary.json
```

Diagnostic split:

```text
holdout family:
  list_transform

train rows:
  192

eval rows:
  64

train families:
  arithmetic_chain, boolean_logic, symbolic_binding

eval family:
  list_transform

train operations:
  add_operands, and_with_p, first_mapping, hold_final, multiply_sum,
  not_q, or_with_r, second_mapping, subtract_offset

eval operations:
  double_filtered, filter_even, hold_final

unseen eval operations:
  double_filtered, filter_even
```

Interpretation:

```text
Full operation-family holdout is rejected as an acceptance gate for the
current fixed-label primitive scaffold. It is not a fair "should pass" test,
because the held-out list family requires operation ids that are absent from
the training split. A classifier over fixed operation labels cannot learn to
emit labels it never sees.

This is useful as a big-structure doubt result: the current Stage 0 system
proves causal recurrent operation routing over supported primitives, but it
does not prove operation invention, latent algorithm discovery, or broad
open-ended reasoning.

The next architectural promotion should not be "train the same fixed labels
harder." It should replace the string-label bottleneck with a learned latent
operation codebook or neural transition model, then rerun family/operation
transfer with action-code and transition-state ablations.
```

### Stage 1: Learned Latent Operation Codebook

Replace hand-named operation labels with learned latent action codes:

```text
prompt -> recurrent core -> latent operation code -> transition/readout
```

The codebook may start with supervision from the fixed operation labels, but
promotion requires it to generalize beyond exact labels.

Acceptance checklist:

- [x] Family-agnostic latent action targets exist without operation strings.
- [ ] Codebook actions are trainable vectors, not only string labels.
- [ ] Fixed operation labels are used only as teacher scaffolding.
- [ ] Held-out tasks can use latent actions without exposing operation names.
- [ ] Codebook collapse is measured.
- [ ] Action diversity increases on harder tasks without accuracy collapse.
- [ ] Core-off and action-code-shuffle ablations fail.

Kill criterion:

```text
If learned codes collapse to one action, or if label supervision is required at
runtime, the codebook is not yet a general reasoning mechanism.
```

### Stage 1.0: Latent Action Codebook Feasibility

Artifacts:

```text
builder:
  scripts/229_build_pure_recursive_latent_action_codebook_cases.py

train split:
  data/filtered/pure_recursive_latent_action_codebook_family_holdout_list_train.jsonl

eval split:
  data/eval/pure_recursive_latent_action_codebook_family_holdout_list_eval.jsonl

summary:
  local_eval/qwen35_2b_pure_recursive_primitive_transition_oodstress_s1024_from_surface_aug/latent_action_codebook_family_holdout_list_summary.json
```

Latent action codebook:

```text
0: extract_or_unary_transform
1: compose_from_previous
2: final_compose_from_previous
3: hold_final
```

Diagnostic split:

```text
holdout family:
  list_transform

train rows:
  192

eval rows:
  64

fixed operation view:
  unseen eval operations:
    double_filtered, filter_even

latent action view:
  train latent action codes:
    0, 1, 2, 3

  eval latent action codes:
    0, 1, 3

  unseen eval latent action codes:
    none
```

Interpretation:

```text
The fixed-label family holdout is structurally impossible because list eval
requires operation strings absent from train. The latent-action version removes
that immediate impossibility: list eval still has unseen fixed operations, but
it does not require unseen latent action codes.

This is only a feasibility gate. It does not prove the model can execute the
transitions neurally. It proves the next Stage 1 training target is no longer
blocked by impossible fixed-label invention.

Promotion now requires training the transition_state_code path on this split
and checking:

1. full QTRM predicts latent action codes on held-out list cases;
2. core-off or transition-state-off drops;
3. action-code shuffle/dropout drops;
4. final answer quality improves without symbolic operation execution.
```

### Stage 1.1: Latent Action Codebook S120 Rejection

Artifacts:

```text
v1 config:
  configs/qwen35_2b_4090_pure_recursive_latent_action_codebook_s120.yaml

v1 checkpoint:
  local_eval/qwen35_2b_pure_recursive_latent_action_codebook_s120_from_oodstress/last.pt

v1 eval:
  local_eval/qwen35_2b_pure_recursive_latent_action_codebook_s120_from_oodstress/eval_latent_action_codebook_list_holdout_full.json
  local_eval/qwen35_2b_pure_recursive_latent_action_codebook_s120_from_oodstress/eval_latent_action_codebook_list_holdout_transition_off.json

v2 builder/config:
  scripts/229_build_pure_recursive_latent_action_codebook_cases.py --codebook-version terminal_v2
  configs/qwen35_2b_4090_pure_recursive_latent_action_codebook_v2_s120.yaml

v2 checkpoint:
  local_eval/qwen35_2b_pure_recursive_latent_action_codebook_v2_s120_from_oodstress/last.pt

v2 eval:
  local_eval/qwen35_2b_pure_recursive_latent_action_codebook_v2_s120_from_oodstress/eval_latent_action_codebook_v2_list_holdout_full.json
  local_eval/qwen35_2b_pure_recursive_latent_action_codebook_v2_s120_from_oodstress/eval_latent_action_codebook_v2_list_holdout_transition_off.json
```

Results:

```text
role_v1:
  full:
    trace exact: 0/64
    step acc:    0.7500

  transition-state-off:
    trace exact: 0/64
    step acc:    0.2500

  depth pattern:
    depth 1: 64/64 correct
    depth 2: 64/64 correct
    depth 4:  0/64 correct, predicted final_compose, target hold
    depth 8: 64/64 correct

terminal_v2:
  full:
    trace exact: 0/64
    step acc:    0.5000

  transition-state-off:
    trace exact: 0/64
    step acc:    0.2500

  depth pattern:
    depth 1: 64/64 correct
    depth 2:  0/64 correct, predicted nonterminal compose, target terminal compose
    depth 4:  0/64 correct, predicted final compose, target hold
    depth 8: 64/64 correct
```

Interpretation:

```text
The code path is causal: full beats transition-state-off on step accuracy.
However, neither codebook passes the held-out list-family trace gate because
trace exact remains 0/64.

role_v1 failed because terminality was not represented. terminal_v2 added
terminal/nonterminal separation but still failed: the model mapped held-out
list prompts to the nonterminal arithmetic/boolean chain rather than the
terminal symbolic/list-like chain.

Therefore Stage 1 is not accepted. The next fix should not be another minor
code-name split. The missing capability is prompt-grounded termination and
semantic transition grounding for unseen families.
```

Next candidates:

```text
1. Add a separate halt/finality head supervised at every recurrent depth.
2. Train neural transition-state prediction against intermediate state text,
   then evaluate whether predicted state semantics, not only action codes,
   transfer to held-out families.
3. Add mixed-family terminality data where terminal and nonterminal compose
   roles appear under more diverse surfaces before retrying family holdout.
```

### Stage 1.2: Dense Joint-State S120 Rejection

Artifacts:

```text
builder:
  scripts/232_build_dense_transition_targets.py

decision:
  docs/wiki/decisions/transition-joint-dense-s120.md

role_v1 checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dense_s120_from_oodstress/last.pt

terminal_v2 checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dense_terminal_v2_s120_from_oodstress/last.pt
```

Results:

```text
role_v1 dense:
  full step_acc:      0.8750
  off step_acc:       0.1250
  halted exact:       0/64

terminal_v2 dense:
  full step_acc:      0.7500
  off step_acc:       0.1250
  halted exact:       0/64
```

Interpretation:

```text
Dense 1..N transition supervision fixes the previous sparse-label ambiguity and
proves the compact transition-state path is causal. However, Stage 1 still
fails because held-out list prompts choose the arithmetic/nonterminal compose
state at depth 2 and only halt one step late.

The next bottleneck is prompt-grounded terminal routing and semantic family
transfer. More scalar heads are unlikely to solve it unless the training
distribution contains counterfactual terminal/nonterminal examples that force
the recurrent state to condition on prompt semantics.
```

Next candidates:

```text
1. Mixed terminality augmentation with similar surfaces requiring different
   halt/compose depths.
2. Counterfactual terminality pairs where prompt semantics, not family name,
   decide whether depth 2 should halt.
3. If this fails, replace action-code supervision with direct neural
   transition-state content prediction.
```

### Stage 1.3: Terminality Fix And All-Families Sanity Control

Artifacts:

```text
decision:
  docs/wiki/decisions/transition-joint-terminality-and-sanity-s120.md

terminality counterfactual builder:
  scripts/233_build_terminality_counterfactual_targets.py

terminality-aug checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dense_terminality_aug_s120_from_oodstress/last.pt

all-families sanity checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dense_terminal_v2_allfamilies_s120_from_oodstress/last.pt
```

Result:

```text
terminality augmentation, list fully held out:
  full step_acc:      0.7500
  off step_acc:       0.1250
  halted exact:       0/64

all-families sanity, list present in train and held out by index:
  full step_acc:      1.0000
  off step_acc:       0.1250
  halted exact:       64/64
```

Interpretation:

```text
The dense joint-state path is capable of learning list_transform recurrent
traces when the family is present in training, and the transition path is
causally necessary. The full list-family holdout failure is therefore not proof
that the architecture cannot learn list transitions. It is proof that the
current compact codebook does not zero-shot transfer terminal list semantics
from arithmetic/symbolic/boolean examples alone.
```

Decision:

```text
Accept as Stage 1 mechanism sanity control.
Reject as broad Stage 1 family-transfer promotion.
```

Next candidates:

```text
1. Hold out list value ranges and paraphrase clusters rather than the entire
   list family.
2. Run action-code shuffle/dropout ablations on the accepted sanity checkpoint.
3. If held-out list-surface transfer passes, move to neural state-content
   transition targets instead of stronger action-code classifiers.
```

### Stage 1.4: List Paraphrase Transfer Accepted

Artifacts:

```text
decision:
  docs/wiki/decisions/transition-joint-list-transfer-s120.md

builder:
  scripts/234_build_list_transfer_gate.py

checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dense_terminal_v2_list_transfer_s120_from_oodstress/last.pt
```

Split:

```text
train:
  all four primitive families
  list_transform variants: 0, 1, 2, 3, 4, 5

eval:
  list_transform only
  list_transform variants: 6, 7
```

Result:

```text
full:
  exact:         16/16
  halted exact:  16/16

transition-state-off:
  exact:         0/16
  halted exact:  0/16

code shuffle:
  exact:         0/16
  halted exact:  0/16

code dropout:
  exact:         0/16
  halted exact:  0/16
```

Interpretation:

```text
The dense terminal_v2 joint-state path can generalize within the list family to
held-out paraphrase clusters, and the result depends on the transition-state
path plus action-code semantics. This is accepted as within-family surface
transfer, not full operation-family zero-shot reasoning.
```

Next candidates:

```text
1. Increase list transfer eval size and include multiple held-out variant
   clusters.
2. Add held-out list value ranges and longer list lengths.
3. Add mixed-family multi-step tasks.
4. Move to neural transition-state content prediction after the action-code
   transfer gate remains stable.
```

### Stage 1.5: Long List Value/Length Transfer Accepted

Artifacts:

```text
decision:
  docs/wiki/decisions/transition-joint-list-transfer-long-s120.md

builder:
  scripts/234_build_list_transfer_gate.py

checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dense_terminal_v2_list_transfer_long_s120_from_oodstress/last.pt
```

Split:

```text
train:
  all four primitive families
  list_transform variants: 0, 1, 2, 3, 4, 5
  list length: 5

eval:
  list_transform only
  list_transform variants: 6, 7
  list lengths: 7, 9
  disjoint value range starting at 30001+
```

Result:

```text
full:
  exact:         32/32
  halted exact:  32/32

transition-state-off:
  exact:         0/32
  halted exact:  0/32

code shuffle:
  exact:         0/32
  halted exact:  0/32

code dropout:
  exact:         0/32
  halted exact:  0/32
```

Interpretation:

```text
The dense terminal_v2 joint-state path is not only memorizing exact list
length, exact values, or seen list paraphrase variants. It survives train
length 5 -> eval lengths 7 and 9 under a disjoint value range, and the result
collapses when the transition-state path or action-code semantics are removed.

This remains a within-family result. It should be promoted only as a stronger
Stage 1 transfer gate, not as unseen operation invention or open-ended
reasoning.
```

Next candidates:

```text
1. Add mixed-family multi-step composition tasks.
2. Add held-out chain/list lengths with donor-only/core-off/state-off/code
   shuffle baselines.
3. Move to neural transition-state content prediction after composition
   remains stable.
```

### Stage 1.6: Mixed-Family Composition Accepted

Artifacts:

```text
decision:
  docs/wiki/decisions/transition-joint-mixed-composition-s720.md

builder:
  scripts/235_build_mixed_family_composition_gate.py

checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_balanced_interleave_s720_from_s240/last.pt
```

Architecture correction:

```text
terminal_v2:
  action code partly encoded terminality

dynamic_halt_v3:
  action code and halt/finality are separated
  finality is trained by answer_match
```

Accepted trace:

```text
codes:    0, 1, 2, 3, 4, 4, 4, 4
finality: 0, 0, 0, 1, 1, 1, 1, 1
```

Result:

```text
held-out mixed list->arithmetic, variants 6/7, lengths 7/9:
  full exact:             32/32
  transition-state-off:    0/32
  code shuffle:            0/32
  code dropout:            0/32

train mixed only:
  full exact:            240/240
```

Interpretation:

```text
The recurrent transition-state path can now compose a list intermediate state
into an arithmetic aggregation/subtraction trace under held-out surfaces and
lengths. This is a real Stage 1 progression beyond isolated list transfer.

It remains a supervised synthetic latent-action gate, not open-ended neural
state transition or natural-language reasoning.
```

### Stage 1.7: Mixed Composition Length 11/13 Accepted

Artifacts:

```text
decision:
  docs/wiki/decisions/transition-joint-mixed-composition-len1113-s080.md

checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_len1113_jointonly_s080_from_s720/last.pt
```

Baseline S720 on held-out list lengths 11/13:

```text
trace exact:    32/32
halted exact:  30/32
```

The two misses were finality/halt errors, not action-code errors. A short
joint-only recovery run fixed that without answer-token CE:

```text
full:
  trace exact:    32/32
  finality acc:   1.0000
  halted exact:  32/32

transition-state-off:
  trace exact:     0/32
  halted exact:   0/32

code shuffle:
  trace exact:     0/32
  halted exact:   0/32

code dropout:
  trace exact:     0/32
  halted exact:   0/32

canonical len7/9 regression:
  full exact:     32/32
```

Interpretation:

```text
The dynamic-halt transition-state path survives a longer held-out list-length
transfer probe. This strengthens Stage 1, but does not remove the Stage 2
requirement: the model still predicts latent action/finality codes rather than
neural state contents.
```

### Stage 2: Neural Transition Model

Replace the symbolic executor with a learned state transition:

```text
state_t, latent_action_t, prompt_context -> state_t+1
```

The executor remains only as a teacher/verifier during training and evaluation.

Acceptance checklist:

- [ ] Neural transition predicts intermediate states on held-out cases.
- [ ] Executor is not used at runtime for accepted metrics.
- [ ] Transition error correlates with answer error.
- [ ] Counterfactual state/action swaps change the final answer predictably.
- [ ] Longer chain lengths degrade gracefully, not catastrophically.
- [ ] Core depth improves transition quality or answer quality.
- [x] Reject answer-renderer-only final CE because it preserves action-code
      32/32 but leaves LM causal forced-choice at 0/8.
- [x] Reject role-value answer bridge because S80 preserves action-code 32/32
      but LM causal forced-choice remains 0/8 and bridge-off matches core8.

Kill criterion:

```text
If transition MSE improves but answer accuracy does not move, the target is
not semantic enough. Redesign the transition target before scaling.
```

Stage-2 clarification from `final-answer-bridge-s120`:

```text
The blocker is not only answer formatting. The final answer renderer cannot
generalize held-out arithmetic values unless a neural value-state transition
computes the state that should be rendered. Final-answer CE without that state
is rejected as a local patch.
```

Rejected Stage-2 candidate from `role-value-answer-bridge-s120`:

```text
Convert core role-value logits into learned internal tokens and feed them to
the same answer-state loop that produces LM logits. S80 preserved action-code
but did not move held-out LM forced-choice and had no bridge-off causal drop.
Role-value states therefore remain probe telemetry rather than reasoning state.
```

Typed-field Stage-2 probe from `typed-algorithmic-value-state-len1113-s080`:

```text
field-specific heads:
  raw list offsets
  doubled list offsets
  scalar coefficient
  scalar residual
  final residual

held-out len11/13 mixed-only:
  content-field accuracy: 352/1024 = 0.34375
  head-off content acc:     0/1024 = 0.0
  trace exact:              0/32

action-code preservation:
  len11/13 exact: 32/32
  len7/9 exact:   32/32
```

Decision:

```text
Accept only as a causal probe improvement. The typed head learns some value
content without damaging the action controller, but trace exact remains 0/32.
Do not add another readout-only value head as the next canonical candidate.
Typed CE is not a universal LLM objective. Keep it as probe-only unless the
typed state causally improves LM logits/text answers.
```

Next Stage-2 candidate:

```text
Universal LM-path recurrent answer/transition candidate:
prompt/donor hidden -> recurrent QTRM answer/core state -> LM logits/text
```

Auxiliary typed/process targets may be used during training, but promotion
requires donor/core-off/depth/module-off ablations to show that the universal
LM answer path improved. A typed register or field head that computes the
answer outside LM logits remains a rejected architecture claim.

### Stage 3: Open-Ended Recursive Reasoning Core

Move from fixed primitive families to open-ended recursive state updates:

```text
prompt -> recurrent latent state -> halt/continue -> answer
```

This stage is where QTRM becomes closer to a looped latent reasoning model
rather than a neuro-symbolic operation router.

Acceptance checklist:

- [ ] No fixed operation ids are required at runtime.
- [ ] No symbolic executor is required at runtime.
- [ ] Core depth sweep shows positive scaling on held-out reasoning.
- [ ] Early exit works without large inference cliffs.
- [ ] Donor-only and core-off baselines are beaten.
- [ ] Depth 1/2/4/8 outputs are not trivially identical.
- [ ] Hard negatives expose reasoning failure instead of formatting failure.
- [ ] The answer path stays fluent and donor language quality is preserved.

Kill criterion:

```text
If deeper core steps do not improve held-out reasoning, QTRM is not yet a
recursive reasoning core. Do not hide the failure with retrieval or answer
governors.
```

### Stage 4: Memory And Metacognition Composition

Only after Stage 3 shows raw recursive gain should memory and metacognition be
promoted as final architecture components.

Acceptance checklist:

- [ ] Trainable memory on/off improves recall/use over memory-off.
- [ ] Core + memory composition beats core-only and memory-only.
- [ ] Known/unknown calibration improves through model state, not thresholds.
- [ ] Contradiction and weak-evidence cases trigger search/defer/abstain.
- [ ] Retrieval is compiled into the canonical token stream or explicitly
      tested as a non-canonical probe.
- [ ] MemoryOS remains runtime infrastructure unless the trainable memory path
      is inside the model forward pass and passes ablations.

Kill criterion:

```text
If MemoryOS/RAG improves score but core/memory ablations do not show a causal
gain, the result is tooling progress, not QTRM raw-intelligence progress.
```

## Workflow

Every architecture iteration must follow this loop:

```text
1. Define the claim.
2. Define the simpler baseline that must be beaten.
3. Define the causal tensor/path that must matter.
4. Build the smallest held-out gate.
5. Add or update tests before implementation when code behavior changes.
6. Train or run the smallest falsifying experiment.
7. Run core-off/component-off/counterfactual ablations.
8. Preserve artifacts: config, command, checkpoint, eval JSON, examples.
9. Promote, reject, or demote the component.
10. Write the wiki entry before moving to the next architecture idea.
```

Promotion requires all of these:

- [ ] Full model beats donor-only or simpler baseline.
- [ ] Full model beats component-off ablation.
- [ ] The improvement survives held-out cases.
- [ ] The improvement survives at least one surface or distribution shift.
- [ ] The claimed path is causal, not a post-hoc script.
- [ ] The result has a documented boundary.

Rejection requires a failure ledger with:

- [ ] Observed failure.
- [ ] Evidence artifact.
- [ ] Current information path.
- [ ] Missing information path.
- [ ] Alternative explanations.
- [ ] Replacement candidates.
- [ ] Smallest next experiment.
- [ ] Kill criterion.

## Immediate Next Checklist

The next work should not jump straight to ASI claims. It should close the gap
between Stage 0 and Stage 1:

- [x] Build and run a larger OOD paraphrase primitive gate.
- [x] Add a scaffold-safe state-constrained runtime decoder for invalid
      operation choices after a core-produced intermediate state.
- [ ] Close the remaining raw list_transform variant 7 operation-routing
      failure without relying on the state-constrained decoder.
- [ ] Add longer arithmetic chains and held-out chain lengths.
- [ ] Add operation-family holdout: train without one family, evaluate transfer.
- [ ] Add mixed-family tasks requiring two different primitive families.
- [x] Build a family-agnostic latent action codebook dataset.
- [x] Train/evaluate transition_state_code S120; rejected on list-family trace exact.
- [x] Accept within-family list paraphrase transfer with held-out variants.
- [x] Accept within-family list transfer with held-out lengths and value range.
- [x] Accept mixed list-to-arithmetic composition with dynamic halt/finality.
- [x] Reject role-value transition auxiliary training because it regresses
      held-out value-state accuracy and does not prove monotonic core-step
      improvement.
- [x] Train/evaluate role-value answer bridge S80 with bridge-off ablation;
      rejected on held-out LM causal forced-choice.
- [x] Design/train/evaluate an Ouro/LoopLM-style recurrent answer-state
      candidate with recurrent-answer-state-off and depth-sweep ablations.
      S80 accepted as a smoke result; S240/choice-margin continuations
      rejected.
- [x] Promote the answer-halt-head variant as the current canonical
      forced-choice baseline: smoke8 full 8/8 vs donor/core-off/depth1/2
      0/8, smoke16 full 10/16 vs halt-gate-off 0/16.
- [x] Run a partial scale32 depth4 gate: donor/core-off 0/32, depth1/2 4/32,
      depth4 16/32. Treat as partial only because depth8/halt-off scale32 was
      stopped incomplete.
- [x] Demote typed CE to probe-only because it is not a universal LLM
      objective and trace exact remains 0/32.
- [ ] Improve greedy autoregressive rendering while preserving the accepted
      answer-halt forced-choice gate.
- [x] Train/evaluate transition_state_finality S120; accepted as a narrow causal
      finality signal but rejected as a Stage 1 reasoning promotion.
- [x] Train/evaluate transition_state_text S120; accepted as a narrow causal
      semantic-token signal at high LR but rejected because it collapses to the
      depth-1 token and trace exact remains 0/64.
- [x] Train/evaluate transition_state_text depth-contrast S120; rejected
      because the held-out prediction still collapses to the same depth-1 token
      and trace exact remains 0/64.
- [x] Train/evaluate compact code+finality and joint code/finality heads;
      rejected because held-out strict trace exact and halted exact remain 0/64.
- [ ] Evaluate action-code shuffle and action-code dropout.
- [x] Build dense 1..N transition targets from solver traces, then retry the
      joint state head with strict halted exact and transition-state-off gates.
- [x] Document whether the next bottleneck is prompt routing, action collapse,
      transition semantics, or answer rendering.
- [ ] Build mixed terminality augmentation and counterfactual terminality pairs.
- [ ] Evaluate action-code shuffle and action-code dropout on the dense joint
      head before any Stage 1 promotion claim.
- [x] Add action-terminal finality mode to remove answer-match process ambiguity.
- [x] Run terminality augmentation; rejected because list-family zero-shot still
      has halted exact 0/64.
- [x] Run all-families list sanity control; accepted as a mechanism sanity
      result because full reaches 64/64 and transition-state-off reaches 0/64.
- [x] Build a fair list transfer gate: hold out value ranges or paraphrase
      clusters, not the entire list family.
- [x] Run transition-state-off, code-shuffle, and code-dropout ablations on the
      list transfer gate.
- [ ] Scale list transfer to longer lists, held-out value ranges, and larger
      eval sets.

## Claim Boundary

Allowed next claim if Stage 1 passes:

```text
QTRM can learn reusable latent action codes that generalize beyond fixed
operation labels on held-out primitive reasoning tasks.
```

Not allowed until later stages:

```text
QTRM is a broad general-purpose reasoner.
QTRM no longer needs the donor.
QTRM performs open-ended neural reasoning.
QTRM is an ASI architecture.
```
