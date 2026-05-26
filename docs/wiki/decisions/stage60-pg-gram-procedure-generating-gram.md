# Stage60 PG-GRAM: Procedure-Generating GRAM

Date: 2026-05-22

Status: v45 generic PG-GRAM rejected; LBEC-P successor scaffold implemented.

## Plain-Language Thesis

Stage59 v44 proves that the model can answer through a typed working desk when
each answer family owns its own causal path.

The remaining weakness is also clear:

```text
numeric/list rows still use BENAPE, a hard-coded calculation routine.
```

So the next big jump is not another loss weight, readout variant, curriculum
change, or larger run. The next big jump is to make the model generate and run
its own procedure.

Human story:

```text
Before:
  The student reads the problem, then a built-in calculation routine performs
  numeric/list moves.

After:
  The student reads the problem, writes a candidate procedure, runs that
  procedure on the working desk, checks/commits the desk state, and speaks only
  from that committed state.
```

This is the local successor to BENAPE:

```text
Base-Equivariant Neural Arithmetic Primitive Executor
-> Procedure-Generating GRAM
```

## Latest-First Research Map

| date | source | mechanism | local failure explained | candidate implication |
| --- | --- | --- | --- | --- |
| 2026-05-20 | HRM-Text, arXiv:2605.20613 | One-body recurrent text model: reader, recurrent thought, speaker are trained together | A side calculator can solve a toy slice without becoming a general thinking path | PG-GRAM must feed the normal answer path, not a probe or detached executor |
| 2026-05-19/20 | GRAM, arXiv:2605.19376 | Stochastic recursive latent trajectories and inference-time multi-trajectory computation | Search is useful only if the latent state space contains valid procedures | Use GRAM to sample/score procedures, not to sample arbitrary drifting z states |
| 2026-03-23 | Depth-Recurrent Transformers, arXiv:2603.21676 | Silent final-output pressure, LayerScale, identity-biased recurrence | Deep recurrence fails when the state drifts or learns shallow step shortcuts | Keep procedure execution stable and final-path owned |
| 2026-03-02 | Recursive Models for Long-Horizon Reasoning, arXiv:2603.02112 | Recursive decomposition into smaller active contexts/subtasks | Long tasks need explicit decomposition, not one compressed hidden blob | Procedure generation should decompose a row into small executable state updates |

## Current Evidence

Stage59 v44:

```text
run:
  /tmp/stage59_v44_learned_tpto_lane_owned_train256_eval128

eval = 128/128 = 1.0
typed_register_off = 0/128 = 0.0
digit_transition_executor_off = 64/128 = 0.5
ledger_pact_renderer_off = 64/128 = 0.5
learned_primitive_lane_renderer_off = 64/128 = 0.5
boolean_primitive_lane_off = 96/128 = 0.75
symbolic_primitive_lane_off = 96/128 = 0.75
```

Interpretation:

```text
The desk and lane-ownership contract works.
The learned boolean/symbolic routines work.
The numeric/list lane still depends on BENAPE.
```

Stage60 v45 generic PG-GRAM:

```text
run:
  /tmp/stage60_v45_pggram_train256_eval128

eval = 64/128 = 0.50
arithmetic_chain = 0/32
list_transform = 0/32
boolean_logic = 32/32
symbolic_binding = 32/32

learned_numeric_procedure_trace_digit_accuracy improved to 0.4662, but final
numeric/list answers stayed at zero.
```

Interpretation:

```text
The generic procedure routine learned digit-shaped patterns, not arithmetic.
The missing structure is not "more GRAM noise"; it is a repeated column/carry
procedure interface.
```

Implemented successor scaffold:

```text
LBEC-P: Learned Base-Equivariant Column Procedure

This replaces the fuzzy vector routine with a learned cell shared across digit
columns:

  op/action + operand digit + current digit slot + carry state
  -> next digit logits + presence logits + next carry state

The cell is still learned. The architecture only forces the same place-value
procedure to be reused across columns, which is the part humans use to
generalize arithmetic beyond memorized examples.
```

Stage60 v46 LBEC-P gate:

```text
run:
  /tmp/stage60_v46_lbecp_column_gate_train256_eval128

eval = 64/128 = 0.50
arithmetic_chain = 0/32
list_transform = 0/32
boolean_logic = 32/32
symbolic_binding = 32/32

digit_transition_executor_trace_digit_accuracy improved from 0.2607 to 0.6393.
```

Interpretation:

```text
Reject as final. The column/carry finger exists, but the visible source digits
were not reliably placed on the worksheet before the procedure started. The
model practiced digit-shaped moves without closing the answer path.
```

Stage60 v47 correction:

```text
Source-Seeded LBEC-P:
  visible source digit columns -> initial digit ledger state
  initial digit ledger state + learned column/carry cell -> next states
  final digit logits -> ledger pact renderer

This fixes the humanistic preflight Reader axis: the thinker now starts from
the actual visible numbers, not from a fuzzy latent guess.
```

## Seven-Axis Humanistic Preflight

1. Architecture:
   The reader is Qwen/source slots. The thinker is the typed working desk. The
   new procedure generator must create the update policy for that desk. The
   speaker must read from the committed desk only.

2. Curriculum:
   Do not ask the model to learn all families again. Start by replacing only
   the BENAPE numeric/list hard primitive while preserving the accepted
   boolean/symbolic learned lanes from v44.

3. Reward/loss:
   Reward correct next desk states and final answer ownership. Do not train the
   free mouth on numeric/list answers while claiming the procedure lane owns
   those answers.

4. Evaluation:
   The first exam is not broad LLM generality. The first exam is whether the
   learned numeric/list procedure replaces BENAPE on the Stage59 local gate.

5. Exploration:
   If GRAM is claimed, sampled procedures must differ under K-sampling and
   K-scaling must be logged. If all samples collapse to the same procedure, call
   it deterministic procedure learning, not GRAM.

6. Data contract:
   The prompt and solver trace must contain enough information to reconstruct
   the numeric/list operation sequence. Labels may supervise procedure state,
   but labels must not become a direct answer oracle.

7. Causality/ablation:
   Turning off procedure generation, procedure execution, or committed
   writeback must specifically destroy arithmetic/list while preserving
   boolean/symbolic lanes.

## Accepted-Likelihood Score

```text
Direct bottleneck replacement: 3/3
  Replaces the exact remaining hard primitive, BENAPE.

Normal answer-path enforcement: 3/3
  Numeric/list answer ownership can be masked away from the free mouth and
  routed through the learned procedure lane.

Ablation clarity: 2/2
  procedure_generator_off, procedure_executor_off, and writeback_off should
  destroy numeric/list only.

Different from last rejected family: 2/2
  This is not another schedule, readout, LSCR, or scalar-tuning run.

Total: 10/10
```

## Architecture Contract

```text
Qwen/source reader
-> typed source slots
-> procedure generator
-> learned numeric/list procedure executor
-> committed typed register writeback
-> ledger/answer renderer
-> evaluated answer
```

Required modules:

```text
ProcedureGenerator
  input:
    source numeric/list slots
    current typed digit register state
    operation trace embeddings
    step index or recurrent state
  output:
    procedure/action logits for update, hold, add, multiply, subtract,
    filter, map/list-transform, carry/borrow/read/write choices.

LearnedNumericProcedureExecutor
  input:
    current digit/list register state
    procedure/action distribution
    source slots
  output:
    next digit/list register logits and presence logits.

CommittedProcedureWriteback
  input:
    executor state logits
  output:
    committed typed register trajectory used by the answer renderer.
```

## No-Bypass Contract

PG-GRAM is invalid if any of these are true:

```text
numeric/list final answer CE still trains the free answerer mouth;
answer rendering can read Python-computed BENAPE values;
procedure logits are logged but not used by the evaluated answer path;
procedure_off leaves arithmetic/list accuracy high;
K-sampling is claimed but sampled procedures are identical.
```

## Minimum Local Gate

Run only after RED tests prove BENAPE is bypassable and the new switches can
destroy the path.

Promote if:

```text
full eval >= 0.90 on the Stage59 128-row local eval;
arithmetic_chain >= 30/32;
list_transform >= 30/32;
boolean_logic and symbolic_binding stay >= 30/32;
typed_register_off <= 0.10;
learned_numeric_procedure_off destroys arithmetic/list to <= 4/64;
procedure_writeback_off destroys arithmetic/list to <= 4/64;
boolean/symbolic lane-off behavior remains v44-clean.
```

Reject if:

```text
full accuracy improves but numeric/list survives procedure_off;
numeric/list accuracy depends on BENAPE or oracle operation execution;
boolean/symbolic regress because the new path destabilizes accepted lanes;
procedure samples collapse while the run is claimed as GRAM;
train loss improves but ablation evidence is weak.
```

## First Implementation Rule

The first code step must be test-driven:

```text
RED:
  test that numeric/list answer targets can be owned by the learned procedure
  lane and masked from the free mouth.

RED:
  test that procedure_off removes only arithmetic/list rendered candidates.

RED:
  test that the learned numeric executor output, not BENAPE, is what the
  renderer uses.

GREEN:
  add the smallest ProcedureGenerator and LearnedNumericProcedureExecutor that
  can pass these ownership and ablation tests.
```

## Decision

PG-GRAM is the highest-probability next big-jump candidate after v44.

It should be implemented before any further schedule/readout/curriculum sweeps.
The goal is not to make another toy solver; the goal is to remove the last
hard-coded calculation routine while preserving the accepted one-body answer path.

Update after v45:

```text
Generic PG-GRAM is rejected as too unconstrained.
The active next architecture is PG-GRAM with LBEC-P execution, launched through:

  --digit-transition-executor
  --digit-transition-executor-mode lbecp_column

Do not spend more epochs on the v45 generic vector executor before testing the
column/carry executor.
```

Update after v46:

```text
Plain LBEC-P is rejected as incomplete because it does not seed the visible
source numbers into the worksheet before rollout.

The active next architecture is Source-Seeded LBEC-P. This is the minimum
humanistic correction: first put the numbers on the internal worksheet, then
run the learned column/carry manipulation routine.
```

Update after v47:

```text
Source-Seeded LBEC-P with digit-transition pretraining is also rejected as a
final architecture:

  pretrain digit_accuracy = 0.8861
  integrated final arithmetic/list = 0/64

The isolated manipulation skill exists, but it does not survive integrated
rollout.
The next required correction is Digit-Commit feedback: every predicted digit
must be written back as a discrete digit-state before the next operation reads
it.
```

Update after v48 smoke:

```text
Digit-Commit LBEC-P is implemented and smoke-tested.

Remaining architectural gap:
  list_transform needs value-slot scan/compaction. A pure digit-column routine can
  add/carry across places, but it cannot decide "keep this item and write it to
  the next output slot" without a second scan over list positions.

Next big-jump candidate:
  Dual-Axis Procedure Executor:
    source-seeded digit columns
    + digit commit feedback
    + value-slot pointer/compaction scan
    + ledger pact renderer
```

Update after v49 smoke:

```text
Dual-Axis LBEC-P is implemented and test-covered:
  digit axis = column/carry scan
  value axis = list-slot filter/compaction scan

Local smoke:
  /tmp/stage60_v49_dual_axis_lbecp_smoke_train64_eval32
  eval = 16/32 = 0.50
  arithmetic_chain = 0/8
  list_transform = 0/8
  boolean_logic = 8/8
  symbolic_binding = 8/8

Decision:
  Reject as final. The missing ingredient is no longer "does the numeric
  routine have a second axis?" The missing ingredient is one-body commitment
  and verification: numeric/list answers must be impossible to speak unless
  they pass through a committed procedure worksheet and a learned checker.

Next accepted-likelihood bar:
  Any new big-jump candidate must change the main answer path, not only add a
  helper routine or trace loss.
```

Update after v50 smoke:

```text
Choice-Verifier Speaker is implemented and tested.

Local result:
  /tmp/stage60_v50_choice_verifier_smoke_train64_eval32
  best_accuracy = 0.4375
  arithmetic_chain = 6/8
  list_transform = 1/8

5-epoch follow-up:
  /tmp/stage60_v50_choice_verifier_gate_train64_eval32_e5
  best_accuracy = 0.375
  final_accuracy = 0.34375

Decision:
  Reject as final. However, this is a useful causal signal: arithmetic moves
  from 0/8 to 5-6/8 when the verifier owns the final choice, and drops to 0/8
  under choice_verifier_off. The verification path is real, but pointwise
  verification is too weak and unstable.

Next candidate:
  Pairwise/Tournament Procedure Verifier.
  The model should compare answer choices against each other under the same
  committed worksheet, not score each choice independently.
```

Update after v51 smoke:

```text
Pairwise/Tournament Procedure Verifier is implemented and tested.

Local result:
  /tmp/stage60_v51_pairwise_verifier_smoke_train64_eval32
  best_accuracy = 0.4375
  arithmetic_chain = 5/8
  list_transform = 3/8
  boolean_logic = 5/8
  symbolic_binding = 1/8

choice_verifier_off:
  arithmetic_chain = 0/8
  list_transform = 0/8
  boolean_logic = 8/8
  symbolic_binding = 8/8

Decision:
  Reject as final.
  Accept as diagnosis: the verifier path is causal for numeric/list, but it
  only chooses among supplied candidates. It does not yet put the correct answer
  on the table.

Next big-jump candidate:
  CE-GRAM/PTRM: Candidate-Exposing GRAM with Procedure Tournament Reward Model.

Human story:
  The model must stop trying to speak one answer immediately. It should first
  make a table of candidate answers/procedures, then let PTRM compare those
  candidates pairwise, then copy the winner.

Preflight bar:
  The first metric is oracle_candidate_coverage for generated candidates,
  especially arithmetic/list. Do not spend another long run on a verifier until
  the proposer can expose the correct answer often enough for the verifier to
  choose it.
```

Update after v52B:

```text
CE-GRAM/PTRM generated-candidate path is implemented at the answerer/verifier
interface.

Local result:
  /tmp/stage60_v52b_cegram_ptrm_train_eval_candidate_matched_e5
  best_accuracy = 0.5
  boolean_logic = 8/8
  symbolic_binding = 8/8
  arithmetic_chain = 0/8
  list_transform = 0/8

What changed:
  The verifier can now judge generated candidates instead of supplied choices.
  Training now applies the same ledger/primitive renderers before verifier
  supervision that eval applies before verifier selection.

Decision:
  This is an accepted architecture correction, not the final breakthrough.

Human diagnosis:
  The checker now sees the same worksheet in school and on the exam. That fixes
  boolean/symbolic. Numeric/list still fails because the correct numeric/list
  candidate never appears on the table.

Next rule:
  Do not run another verifier-only experiment. The next high-probability move is
  ledger-native numeric/list candidate exposure: produce K candidates from
  committed digit/presence logits, then let PTRM choose.
```

Update after v53B:

```text
Ledger-native candidate exposure is implemented and tested.

Local results:
  /tmp/stage60_v53_ledger_candidate_exposure_train64_eval32_e5
    best_accuracy = 0.5
    arithmetic_chain oracle coverage = 0/8
    list_transform oracle coverage = 0/8

  /tmp/stage60_v53b_ledger_candidate_allslots_train64_eval32_e5
    best_accuracy = 0.5
    best_epoch = 3
    arithmetic_chain = 0/8
    list_transform = 0/8
    boolean_logic = 8/8
    symbolic_binding = 8/8
    choice_verifier_accuracy reached 1.0 after epoch 3

Representative failures:
  8017  -> [600, 600, ...]
  12032 -> [100, 100, ...]
  8008,8004 -> [EMPTY, 600000, ...]

Decision:
  Accept v53 as a diagnostic correction.
  Reject it as the final big jump.
```

Human diagnosis:

```text
The checker is no longer the bottleneck.
The model can judge candidates when a valid candidate is on the desk.

The failure is earlier: numeric/list source values do not stay pinned to the
working desk. The manipulation routine keeps producing training-like habits
such as 100, 600, or EMPTY before the verifier ever gets a meaningful choice.

This means GRAM/PTRM was attached to the right high-level loop but not yet to a
brain-like working-memory manipulation routine. A human does not solve 8017 by
storing it as a vague vector and hoping the mouth reconstructs it. A human pins
the source digits to working memory, applies the same column-by-column rule,
tracks carry/borrow state, then checks the resulting candidate.
```

Mandatory next architecture rule:

```text
Do not spend the next run on another selector, verifier, or candidate expander.

The next accepted-likelihood direction is v54 Source-Anchored Procedure GRAM:

  reader/parser
  -> source-anchored numeric/list working memory
  -> GRAM procedure policy over source slot, column, op, carry/update
  -> typed belief register write
  -> PTRM/verifier candidate tournament
  -> final answer

Plain-language invariant:
  정답 후보가 책상 위에 태어나기 전에는 검산자를 키워도 의미가 없다.
  다음 실험은 먼저 source 숫자가 작업기억에 붙잡혀 있고, 내적 조작 루틴이
  그 숫자를 안정적으로 갱신한다는 것을 보여야 한다.

Promote preflight:
  arithmetic/list generated-candidate oracle coverage must rise above 0/16
  before selected accuracy is used as the main gate.
```

Update after v54 source-anchor pretraining alignment:

```text
Implementation:
  DigitTransitionPretrainingExamples now carries source_numeric_features and
  source_numeric_feature_mask.
  pretrain_digit_transition_executor now passes those source anchors into the
  executor during digit-transition pretraining.

Why this matters:
  Before this correction, the internal manipulation routine practiced on one
  working-memory layout but was evaluated on another. Pretraining saw seeded
  digit state, while integrated rollout also injected source-number slots. That
  mismatch can make source anchors behave like unfamiliar noise at the exact
  point where v54 needs them to be stable.

Decision:
  Accept as required architecture hygiene for v54.
  This is not yet the final big jump.

Next gate:
  Rerun the v53B generated-candidate path with source-anchor-aligned pretraining.
  Promote only if arithmetic/list generated-candidate oracle coverage rises
  above 0/16. Selected accuracy remains secondary until valid numeric/list
  candidates appear on the table.
```

Update: HRM-Text one-body surgery direction

```text
Question:
  Can a Qwen+GRAM/PTRM system that was not born as one body be surgically made
  to behave like one body?

Answer:
  Yes, but not by adding more external renderers or verifiers.
  The surgery must graft the typed/GRAM thought back into a Qwen-compatible
  hidden representation, then force the evaluated answer to come through the
  normal LM-compatible mouth.

Research anchors:
  HRM-Text (arXiv:2605.20613):
    one-body recurrent language model with slow strategic and fast execution
    layers, stabilized by MagicNorm and warmup deep credit assignment.

  ReFT / LoReFT (arXiv:2404.03592):
    frozen language model plus learned interventions on hidden
    representations; this is the closest published analogue to surgical
    hidden-state grafting.

  Prefix-Tuning / P-Tuning v2 (arXiv:2101.00190, arXiv:2110.07602):
    learned continuous prompts/virtual tokens that a frozen model consumes as
    part of its own input stream.

  LayerSkip (arXiv:2404.16710):
    draft and verification share the same model body; useful as a warning that
    self-verification should not remain a detached judge forever.

  Model Grafting / Skill Localization (arXiv:2302.06600):
    learned task skills can be localized and grafted into a pretrained model,
    supporting the idea that a small surgical region can carry a new skill.

Plain-language rule:
  지금 구조는 "Qwen 옆 계산 작업대"다.
  다음 구조는 "계산 작업대가 Qwen의 생각공간 안으로 들어간 상태"여야 한다.

Minimal v55 preflight:
  Residual Thought Graft
    typed_digit/working register summary
    -> zero-initialized low-rank residual intervention
    -> Qwen-compatible readout/workspace state

  It must start as exact identity, be ablatable, and later be evaluated through
  an LM-compatible answer path rather than a new side renderer.
```

v55 preflight implementation:

```text
Implemented:
  ResidualThoughtGraft
  apply_residual_thought_graft_context(...)

Tests:
  test_residual_thought_graft_starts_identity_and_is_ablatable
  test_residual_thought_graft_requires_qwen_compatible_readout_shape

Decision:
  Accept as the first code-level one-body surgery scaffold.
  This is not yet wired into train/eval and is not yet a promoted model result.

Next:
  Wire the graft behind flags and require graft_off causality before any
  one-body claim is accepted.
```

## 2026-05-22 Implementation Entry: Ownership and Procedure Scaffolding

First RED/GREEN entry implemented:

```text
mask_primitive_lane_owned_answer_targets(..., numeric_procedure_owns_answer=True)
ProcedureGenerator
ProcedureGeneratorOutput
LearnedNumericProcedureExecutor
render_learned_numeric_procedure_texts(...)
```

Tests added:

```text
test_numeric_procedure_owned_targets_mask_numeric_and_list_free_mouth_supervision
test_procedure_generator_outputs_action_logits_and_has_off_ablation
test_learned_numeric_procedure_executor_consumes_generated_actions_and_is_ablatable
test_learned_numeric_procedure_renderer_reads_executor_logits_not_benape_values
```

What this proves:

```text
The Stage60 path can now express the three necessary contracts:
  numeric/list answers can be removed from the free mouth;
  a procedure generator can produce ablatable action logits;
  a learned executor can consume generated procedure actions and update digit
  registers with destructive executor-off behavior;
  numeric rendering can read learned executor logits rather than BENAPE values.
```

What this does not prove yet:

```text
The learned procedure executor is not yet wired into train/eval as the evaluated
BENAPE replacement, and no Stage60 training gate has been run.
```

Next code step:

```text
Wire ProcedureGenerator -> learned numeric executor -> committed register
writeback -> renderer inside the evaluated path, then add
learned_numeric_procedure_off and procedure_writeback_off eval summaries.
```

## 2026-05-22 Implementation Entry: Train/Eval Wiring

Stage60 now has the first evaluated-path wiring:

```text
--numeric-procedure-generator
--numeric-procedure-actions
--numeric-procedure-hidden-dim
--learned-numeric-procedure-executor
--learned-numeric-procedure-hidden-dim
--learned-numeric-procedure-trace-weight
--answerer-learned-numeric-procedure-renderer
--numeric-procedure-own-answer-supervision
--eval-learned-numeric-procedure-renderer-off
--eval-numeric-procedure-generator-off
--eval-numeric-procedure-executor-off
```

New shared train/eval path:

```text
apply_learned_numeric_procedure_context(...)

typed_digit_register_trajectory
-> ProcedureGenerator action logits
-> LearnedNumericProcedureExecutor digit/presence logits
-> typed_digit_register_trajectory replacement
-> optional learned numeric procedure renderer
```

The training loop now supports:

```text
learned_numeric_procedure_trace_loss
learned_numeric_procedure_trace_digit_accuracy
learned_numeric_procedure_trace_presence_accuracy
```

The eval loop now supports:

```text
numeric_procedure_generator_off
numeric_procedure_executor_off
learned_numeric_procedure_renderer_off
```

What this proves:

```text
The PG-GRAM numeric/list routine is no longer only a unit-test object. It can be
constructed, trained with trace supervision, ablated, checkpointed, and used by
the evaluated candidate renderer.
```

What this still does not prove:

```text
No local Stage60 training gate has been run yet. The current implementation is
architecture wiring, not an accepted replacement for BENAPE.
```

Next experiment:

```text
Run local Stage60 v45:
  v44 flags
  replace BENAPE answer ownership with learned numeric procedure ownership
  keep learned boolean/symbolic lanes
  require full >= 0.90 and arithmetic/list collapse under
  numeric_procedure_generator_off / numeric_procedure_executor_off /
  learned_numeric_procedure_renderer_off.
```

## 2026-05-22 Local Gate: v45 Generic PG-GRAM Reject

Run:

```text
/tmp/stage60_v45_pggram_train256_eval128
```

Result:

```text
best_epoch = 3
best_accuracy = 0.50

final eval:
  eval = 64/128 = 0.50
  arithmetic_chain = 0/32
  list_transform = 0/32
  boolean_logic = 32/32
  symbolic_binding = 32/32

final ablations:
  typed_register_off = 0/128 = 0.0
  numeric_procedure_generator_off = 64/128 = 0.50
  numeric_procedure_executor_off = 64/128 = 0.50
  learned_numeric_procedure_renderer_off = 64/128 = 0.50
  learned_primitive_lane_renderer_off = 0/128 = 0.0
  boolean_primitive_lane_off = 32/128 = 0.25
  symbolic_primitive_lane_off = 32/128 = 0.25

trace:
  learned_numeric_procedure_trace_loss: 2.9269 -> 1.5210
  learned_numeric_procedure_trace_digit_accuracy: 0.2140 -> 0.4662
  learned_numeric_procedure_trace_presence_accuracy: 0.8076 -> 0.9207
```

Plain-language diagnosis:

```text
The generic procedure routine learned to make digit-shaped patterns, but not to
carry out exact place-value calculation. It produced nearby-looking answers
such as 8017 -> 6111 or list rows like 8008,8004 -> 6088,4.

This is not a free-mouth bypass: numeric/list stay 0 and the non-numeric lanes
remain clean. It is a precision failure inside the learned numeric routine.
```

Decision:

```text
Reject v45 as a BENAPE replacement.
Do not run the same generic MLP procedure executor longer as the next step.
```

Root cause:

```text
The action-generator + MLP executor is too unconstrained for exact arithmetic.
It has no built-in repeated column move, no explicit carry/borrow state, and no
base-equivariant digit transition table. It is therefore learning a fuzzy
numeric pattern habit instead of a reusable digit procedure.
```

Next high-probability successor:

```text
Learned Base-Equivariant Column Procedure (LBEC-P)

Keep the PG-GRAM ownership contract, but replace the generic vector executor
with a learned finite-state column procedure:

  op/action + current digit + source/arg digit + carry/borrow + presence
  -> next digit + next carry/borrow + next presence

The table/cell is learned, not Python hard-coded, but its interface forces the
same repeated place-value move that made BENAPE work.
```

## 2026-05-22 Update: Residual Thought Graft wired as one-body preflight

Decision:

```text
Accept the v55 Residual Thought Graft wiring as the next architecture-clean
preflight, not as a solved accuracy result.
```

Why:

```text
HRM-Text works as a one-body model: the state that thinks is trained to become
the state that speaks. Our earlier Stage60 components could still behave like a
side workspace, side executor, or side verifier.

The new wiring forces the typed/GRAM working-memory summary to enter the
Qwen-compatible readout before the learned answerer emits candidates. The flag
validator rejects --residual-thought-graft unless --answerer-use-qtrm-readout is
also enabled.
```

Contract:

```text
reader:
  Qwen hidden states and source-number slots.

memory/state:
  working registers plus typed digit/value registers.

thinker/transition:
  GRAM/PTRM and typed transition modules.

graft:
  ResidualThoughtGraft translates working-memory summaries into the readout
  residual stream. It starts as identity.

speaker:
  TypedRegisterAnswerer using the QTRM readout path.

ablation:
  residual_thought_graft_off must remove any real arithmetic/list gain.
```

Verification:

```text
test_residual_thought_graft_requires_normal_readout_answer_path
test_residual_thought_graft_active_respects_ablation_and_typed_register_off
focused residual thought graft tests passed
stage59 typed value trace tests passed
py_compile passed
git diff --check passed
```

## 2026-05-22 Update: v55-v56 one-body graft smoke result

Decision:

```text
Reject the current ResidualThoughtGraft + small Stage530 answerer path as a
complete HRM-Text-like one-body solution.
```

Evidence:

```text
v55 qtrm_readout graft:
  best eval = 0/32
  final eval = 0/32
  residual_thought_graft_off = 0/32

v55B qtrm_readout graft + speaker alignment:
  best eval = 0/32
  final eval = 0/32
  residual_thought_graft_off = 1/32

v56 register_mean graft + speaker alignment:
  best eval = 0/32
  final eval = 0/32
  residual_thought_graft_off = 0/32
```

Humanistic diagnosis:

```text
This is still not one body in the HRM-Text sense.

The reader is Qwen.
The thinker is typed/GRAM/PTRM working memory.
But the speaker is still a small newly trained character answerer.

That speaker learns answer-shaped habits instead of the actual answer language:
  8017 -> 1022
  TRUE -> TALSE/FALS
  green -> siee/seee/blle
  8008,8004 -> 6022,666666666

So the system is no longer just a side calculator, but it is also not yet a
single trained person. It is closer to a reader plus a working desk plus a weak
new mouth.
```

Corrected one-body target:

```text
The next architecture-clean target must make Qwen's own token path speak:

  Qwen hidden states
  -> recurrent typed/GRAM/PTRM thought
  -> residual-stream intervention or soft-prefix tokens
  -> Qwen LM head / PrefixLM token logits

This is the closest Qwen-based analogue of HRM-Text:
the thought state must become part of the normal language-model speaking path.
```

Research anchors:

```text
HRM-Text:
  recurrent thought is part of the language model's own token-producing body.

ReFT / LoReFT:
  low-rank representation interventions can edit frozen model hidden states.

Prefix/P-Tuning:
  learned continuous tokens can steer a frozen LM while keeping the native LM
  mouth intact.

Model stitching:
  affine maps between residual streams can transfer represented features,
  supporting the idea that a learned graft can translate one latent dialect into
  another.

Activation State Machines / activation steering:
  dynamic stateful interventions can steer reasoning without fully replacing the
  base model.
```

Rule:

```text
Do not call future Stage60 variants "one-body" unless the evaluated answer path
uses Qwen-compatible token logits and thought_graft_off removes the gain.
```

## 2026-05-23 Update: Qwen LM-mouth graft becomes the required speaker path

Decision:

```text
Accept QwenLmAnswerMouth as the current required speaker path for one-body
Stage60 experiments.

Reject the idea that QwenLmAnswerMouth alone solves numeric/list reasoning.
```

Why:

```text
v55-v56 showed that a small character answerer is not enough. It produced
answer-shaped strings but no correct answers.

v57-v58 replaced that small mouth with Qwen-compatible token logits:

  typed/GRAM/PTRM thought -> graft -> Qwen LM-head logits

This immediately moved eval from 0/32 to 6/32 and then 7/32, and the gain
vanished under residual_thought_graft_off. That is a real causal signal.
```

Evidence:

```text
v57 readout-context Qwen LM-mouth:
  best eval = 6/32 = 0.1875
  residual_thought_graft_off = 0/32
  typed_register_off = 0/32
  arithmetic/list = 0/16

v58 register-prefix Qwen LM-mouth:
  best eval = 7/32 = 0.21875
  residual_thought_graft_off = 0/32
  typed_register_off = 0/32
  arithmetic/list = 0/16
```

Humanistic diagnosis:

```text
The system now has a more believable body:

reader:
  Qwen hidden states.

working memory:
  typed working registers and digit registers.

thinker:
  GRAM/PTRM transition plus digit-transition executor.

speaker:
  Qwen LM head, reached through QwenLmAnswerMouth.

But the calculation desk itself is still dirty. The Qwen mouth can read the
desk, yet for numeric/list rows the desk mostly says "00000000" or "1".
Therefore the next change must improve exact ledger writing, not add another
speaker or verifier.
```

Rule update:

```text
All next Stage60 runs that claim HRM-Text-like one-body reasoning must use:

  --qwen-lm-mouth-answerer
  --qwen-lm-mouth-context-mode register_prefix
  --residual-thought-graft
  --eval-residual-thought-graft-off

Runs using only the small Stage530 character candidate answerer are now
diagnostic baselines, not final architecture candidates.
```

Next successor:

```text
Stage60 v59 should keep QwenLmAnswerMouth fixed and repair the numeric/list
working ledger:

  enable typed-digit trace supervision;
  enable committed digit writeback;
  keep register_prefix LM mouth;
  require arithmetic/list > 0 as the promote gate.
```

## 2026-05-23 Update: v59 trace/writeback overcorrection rejected

Decision:

```text
Reject Stage60 v59 as a curriculum/order failure, not as proof that exact ledger
supervision is useless.
```

Evidence:

```text
run:
  /tmp/stage60_v59_lm_mouth_digit_trace_writeback_train64_eval32_e5

best eval:
  0/32 = 0.0

families:
  arithmetic_chain = 0/8
  boolean_logic = 0/8
  list_transform = 0/8
  symbolic_binding = 0/8

ablation:
  residual_thought_graft_off = 0/32
  typed_register_off = 0/32
  digit_transition_writeback_off = 3/32 at epoch 5

training signals:
  char_token_accuracy rose to 0.3269
  typed_digit_register_trace_digit_accuracy reached 0.5786
  digit_transition_executor_trace_digit_accuracy reached 0.5976
```

Plain-language diagnosis:

```text
v58 taught the thought to reach the Qwen mouth.

v59 immediately forced the numeric/list ledger to match traces and then forced
that unstable ledger to overwrite the main thought. That is like correcting a
student's scratchpad in red ink while also making the scratchpad take over the
student's voice. The scratchpad scores improved, but the answer mouth collapsed
into blank strings and repeated digits.
```

New rule:

```text
Do not combine strong trace supervision and committed writeback from epoch 1 on
the Qwen LM-mouth path. The next local-only run must add ledger pressure in
one step:

  --qwen-lm-mouth-answerer
  --qwen-lm-mouth-context-mode register_prefix
  --residual-thought-graft
  --eval-residual-thought-graft-off
  --typed-digit-register-trace-weight 0.05

and must not enable:

  --digit-transition-committed-writeback

Promote only if:
  1. total eval >= v58's 7/32 or arithmetic/list becomes nonzero;
  2. the gain still disappears under residual_thought_graft_off;
  3. outputs stop collapsing to blank/repeated digit strings.
```

## 2026-05-23 Update: one-body embodied/surgery analogy

Use the user's surgery analogy as a plain-language validity test, not as a
metaphor-only argument.

```text
A surgical expert generalizes because perception, working memory, internal
procedure, checking, and action have been trained as one coordinated policy.
They are not a fluent observer plus a detached calculator plus a separate
speaker.
```

Current research map:

```text
HRM-Text:
  one-body recurrent language model; the recurrent thought path is the language
  model's own answer path.

Ouro / Looped Language Models:
  latent iterative reasoning is built into language pretraining and normal LM
  output, rather than added as a side module.

LoopFormer:
  recurrent depth must be trained as stable representation trajectories under
  varying budgets.

Coconut:
  continuous thought can be fed back directly as the next reasoning state.

VLA line, including pi0/pi0.5 and OpenVLA:
  embodied "one body" means perception/context/action are trained end-to-end as
  one causal policy, often with broad cross-embodiment data.

Surgical VLM/copilot line:
  relevant to surgical scene understanding, but not sufficient evidence for
  one-body surgical action unless the model directly controls actions through
  the same learned path.
```

Stage60 implication:

```text
The project should not add another side organ. The accepted architecture must
make the evaluated answer route itself become:

  Qwen reader
  -> recurrent semantic/typed thought
  -> GRAM/PTRM transition/search
  -> Qwen-compatible token state
  -> Qwen LM head

If a typed register, executor, checker, or verifier does not feed this route, it
is diagnostic only.
```

## 2026-05-23 Update: v60-v61 weak trace and target-length guard

v60 result:

```text
run:
  /tmp/stage60_v60_lm_mouth_weak_digit_trace_no_writeback_train64_eval32_e5

best eval:
  9/32 = 0.28125

families:
  arithmetic_chain = 0/8
  boolean_logic = 5/8
  list_transform = 0/8
  symbolic_binding = 4/8

causality:
  residual_thought_graft_off = 0/32
  typed_register_off = 0/32
```

v61 target-length check:

```text
Using --qwen-lm-mouth-max-answer-tokens 8 was invalid for list rows:

  8008,8004 tokenizes to 9 tokens before EOS.
  8004,8016,8008 tokenizes to 14 tokens before EOS.
  eval rows over 8 tokens including EOS = 8/32.
  eval rows over 16 tokens including EOS = 0/32.

v61 with max_answer_tokens=16:
  best eval = 7/32 = 0.21875
  arithmetic/list = 0/16
```

Decision:

```text
1. Keep v60 as the current best Qwen LM-mouth one-body scratch result.
2. Do not interpret old list 0/8 results from max_answer_tokens=8 as clean
   architecture failures; the answer field was too short.
3. Do interpret v61 as evidence that length alone is not enough. The numeric
   ledger still lacks a token-position reading contract.
4. All future --qwen-lm-mouth-answerer runs must validate answer target length
   before training.
```

Implemented guard:

```text
validate_qwen_lm_answer_target_lengths(...)

If any train/eval answer needs more than --qwen-lm-mouth-max-answer-tokens after
adding EOS, the script exits instead of silently truncating labels.
```

Next candidate contract:

```text
QwenLmAnswerMouth needs a learned ledger-token pointer/alignment path:

  position-aware answer query
  -> selects ledger digit/comma/EOS evidence
  -> adds token-compatible evidence before or at Qwen LM logits

This is not a detached renderer if it remains inside the evaluated Qwen
LM-mouth path and if ledger_pointer_off removes numeric/list gains.
```

## 2026-05-23 Update: v62 ledger-token reader inserted into Qwen LM-mouth

Implementation:

```text
QwenLmAnswerMouth now has an optional ledger_token_reader:

  answer token state
  -> attends to final typed digit ledger
  -> predicts digit/comma/EOS evidence
  -> adds that evidence to Qwen vocabulary logits

Flags:
  --qwen-lm-mouth-ledger-token-reader
  --qwen-lm-mouth-ledger-token-reader-scale
  --eval-qwen-lm-mouth-ledger-token-reader-off
```

This is still inside the one-body answer path:

```text
Qwen reader
-> recurrent typed/GRAM thought
-> residual thought graft
-> QwenLmAnswerMouth
-> Qwen vocab logits
```

Local v62 result:

```text
run:
  /tmp/stage60_v62_lm_mouth_ledger_token_reader_train64_eval32_e5

best eval:
  8/32 = 0.25

families:
  arithmetic_chain = 0/8
  boolean_logic = 6/8
  list_transform = 0/8
  symbolic_binding = 2/8

causal ablations:
  residual_thought_graft_off = 0/32
  typed_register_off = 0/32
  qwen_lm_mouth_ledger_token_reader_off = 8/32 at best epoch
  digit_transition_executor_off = 8/32 at best epoch
```

Failure pattern:

```text
Numeric/list answers collapsed into repeated digit strings:

  8017 -> 22222
  12032 -> 22222222222222
  8008,8004 -> 2222222222222222
```

Decision:

```text
Reject v62 as a numeric/list generalization fix.

Keep the code because it adds a clean ablatable reader inside Qwen LM-mouth.
But do not scale this recipe until a diagnostic proves the digit ledger actually
contains a renderable answer sequence.
```

Next pre-training-run diagnostic:

```text
Ask:
  if we render directly from digit_transition_executor_digit_logits, can the
  numeric/list answer be recovered?

If no:
  the ledger/executor is wrong, so more mouth work is waste.

If yes:
  the ledger is good and the remaining failure is token-position alignment from
  ledger to Qwen LM-mouth.
```
