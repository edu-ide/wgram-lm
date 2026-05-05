# Pure Recursive Solver State-Machine Probe S240

Date: 2026-05-04

## Decision

The next raw-intelligence architecture should not ask the recurrent core to
emit arbitrary numeric/list state strings directly. The cleaner path is:

```text
canonical prompt/question
-> recurrent core state
-> operation / primitive transition selection
-> explicit state update
-> final answer renderer
```

This keeps the causal recursive loop, but moves brittle arithmetic/list
execution out of free-form string generation.

## Implemented Probe

```text
module:
  src/qtrm_mm/agentic/solver_state_machine.py

train script:
  scripts/216_train_pure_recursive_solver_state_machine.py

tests:
  tests/test_solver_state_machine.py
  tests/test_pure_recursive_solver_state_machine_train_script.py
```

The probe is intentionally donor-free and MemoryOS-free. It trains a small
character-level recurrent state machine on rows shaped as:

```text
question + operation + previous_state -> target_state
```

## Compact Input Fix

Initial training exposed an input-design bug: full prompt text was placed before
`operation` and `previous_state`, so long list cases could truncate the causal
state at `max_input_len=256`.

Fixed input contract:

```text
Operation
Depth
Previous state
Task/question
Target state
```

The generated train trace max input length dropped:

```text
before: max 325 chars
after:  max 246 chars
```

## Results

Small train set:

```text
train:
  cases: 32
  rows: 128
  loss: 0.02497
  teacher_forced_state_exact: 0.9453
  rollout_state_exact: 0.8906
  rollout_final_exact: 0.9063

heldout:
  cases: 8
  rows: 32
  teacher_forced_state_exact: 0.0
  rollout_state_exact: 0.0
  rollout_final_exact: 0.0
```

Large train set:

```text
train:
  cases: 512
  rows: 2048
  loss: 0.1479
  teacher_forced_state_exact: 0.7207
  rollout_state_exact: 0.5181
  rollout_final_exact: 0.4648

heldout:
  cases: 64
  rows: 256
  teacher_forced_state_exact: 0.0
  rollout_state_exact: 0.0
  rollout_final_exact: 0.0
```

Representative failure:

```text
arith-chain-2000 add_operands:
  target: 2010
  pred:   1010

list-transform-2000 filter_even:
  target: 2004,2002
  pred:   1004,1002
```

Interpretation: the char recurrent core learned training-range patterns, not
out-of-range arithmetic/list transition rules.

## Primitive Ceiling

The same trace rows are solved perfectly when the operation is executed as a
causal primitive transition:

```text
heldout200:
  state: 32/32
  final: 8/8

heldout_large:
  state: 256/256
  final: 64/64
```

This proves the trace schema is sufficient. The failure is the architecture
that maps the recurrent state directly to free-form state text.

## Architecture Consequence

Do not continue loss-weight tuning on donor-residual/free-form state readouts.
The next candidate should train the core to:

```text
1. maintain recurrent latent state
2. select or compose a small transition operation
3. apply the transition to the current state
4. render only after the causal state has been updated
```

This is closer to a state-machine/TRM-style loop than the previous readout-only
approach, while remaining KISS/SSOT: one prompt stream, one recurrent state,
one transition update, one answer renderer.

## Operation-Policy Follow-Up

Two operation selector variants were tested after the primitive ceiling.

### Char Operation Policy

```text
script:
  scripts/217_train_pure_recursive_operation_policy.py

input:
  question text + previous_state + task_family/trace_index/depth as text

large heldout:
  operation_exact: 0.625
  rollout_state_exact: 0.5
  rollout_final_exact: 0.5
```

Failure:
the char selector learned the list schedule but collapsed arithmetic rows to
`hold_final`. This means trace metadata should not be reparsed through a text
GRU when the metadata already exists structurally.

### Structured Operation Policy

```text
script:
  scripts/218_train_pure_recursive_structured_operation_policy.py

input:
  task_family id
  trace_index id
  depth id

large train:
  operation_exact: 1.0
  rollout_state_exact: 1.0
  rollout_final_exact: 1.0

large heldout:
  operation_exact: 1.0
  rollout_state_exact: 1.0
  rollout_final_exact: 1.0
```

All-family heldout:

```text
data:
  data/filtered/pure_recursive_solver_trace_all_family_train.jsonl
  data/eval/pure_recursive_solver_trace_all_family_heldout.jsonl

families:
  arithmetic_chain
  symbolic_binding
  boolean_logic
  list_transform

primitive ceiling:
  state: 512/512
  final: 128/128

learned structured operation policy:
  operation_exact: 1.0
  rollout_state_exact: 1.0
  rollout_final_exact: 1.0
```

Updated architecture:

```text
prompt/question
-> recurrent latent core
-> structured transition metadata head
-> primitive transition executor
-> explicit state
-> answer renderer
```

The important result is not that the synthetic task is solved by metadata. The
important result is that the causal loop works when the model is not forced to
learn arithmetic/list execution as free-form character generation.

## Remaining Risk

The structured policy uses explicit `task_family`, `trace_index`, and `depth`.
That is acceptable as a falsification probe, but it is not yet a general LLM
architecture. The next QTRM integration must make those fields predicted by the
latent core:

```text
prompt tokens
-> QTRM recurrent latent state
-> transition metadata logits
-> primitive transition selection
-> explicit state update
-> answer renderer
```

Acceptance requires ablations:

```text
metadata_head_off drops operation/state/final accuracy
primitive_executor_off drops state/final accuracy
core_depth changes transition metadata or state
heldout all-family stays above donor/core-off baselines
```

## QTRM Integration Surface

Implemented the first in-model hook:

```text
config:
  QTRMConfig.primitive_transition_enabled
  QTRMConfig.primitive_transition_num_operations
  QTRMConfig.primitive_transition_hidden_dim
  QTRMConfig.primitive_transition_prompt_context_enabled
  QTRMConfig.primitive_transition_prompt_token_attention_enabled

model output:
  primitive_transition_operation_logits

path:
  core_depth_states
  -> primitive_transition_operation_head
  -> primitive operation logits
```

Training hook:

```text
script:
  scripts/196_train_pure_recursive_depth_supervised.py

arg:
  --primitive-transition-operation-ce-weight

target:
  row.solver_trace[*].operation in trace order
```

This does not yet execute primitives inside QTRM generation. It makes the next
experiment trainable and ablatable: the QTRM recurrent core can now be directly
supervised to predict transition operations from its own depth states.

## QTRM Prompt-Conditioned Operation Head

Initial in-model training exposed a root information-path failure:

```text
core_depth_states only -> primitive operation logits
```

The 240-step operation-only checkpoint reduced loss but collapsed to a
family-insensitive schedule:

```text
checkpoint:
  local_eval/qwen35_2b_pure_recursive_primitive_transition_s240_oponly/last.pt

heldout first64:
  operation accuracy: 128/256 = 0.5000

typical prediction for every family:
  not_q -> and_with_p -> hold_final -> hold_final
```

This falsifies the idea that the existing core depth state alone was routing
task-family information strongly enough for primitive selection under this
training recipe.

Implemented fix:

```text
canonical prompt tokens
-> QTRM prelude text context
-> masked prompt pooled context
-> concat(core_depth_state, prompt_context)
-> primitive operation logits
```

Config field:

```text
primitive_transition_prompt_context_enabled: true
```

The path is still SSOT: evidence and task text remain in the canonical prompt
token stream. The change does not add a hidden MemoryOS/RAG path; it only makes
the primitive transition head causally see the same prompt context that the
model already processed.

Results:

```text
s240 prompt-conditioned checkpoint:
  local_eval/qwen35_2b_pure_recursive_primitive_transition_promptctx_s240_oponly/last.pt

train all:
  operation accuracy: 417/512 = 0.8145

heldout existing:
  operation accuracy: 210/256 = 0.8203

remaining errors:
  arithmetic first two operations sometimes mapped to boolean operations
  list double_filtered sometimes mapped to second_mapping
```

Continuing the same architecture from s240 to s720 fixed almost all routing:

```text
s720 prompt-conditioned checkpoint:
  local_eval/qwen35_2b_pure_recursive_primitive_transition_promptctx_s720_oponly/last.pt

train all:
  operation accuracy: 512/512 = 1.0000

heldout existing:
  operation accuracy: 250/256 = 0.9766

heldout7000:
  operation accuracy: 507/512 = 0.9902
```

Heldout7000 by family:

```text
arithmetic_chain: 128/128
symbolic_binding: 128/128
boolean_logic:    128/128
list_transform:   123/128
```

Remaining failure class:

```text
double_filtered -> second_mapping
```

Interpretation: the corrected causal path is a real step forward. The model is
no longer only learning trace position; it routes operation families from the
prompt context. The next architecture improvement should focus on token-level
operation grounding for list transforms, not on more free-form answer heads.

## Token-Level Prompt Grounding

The s720 prompt-conditioned head still used a masked mean pool over prompt
tokens. That made the primitive operation head partially blind to token order
and local task evidence. The remaining heldout7000 failures were all list
transform confusions:

```text
double_filtered -> second_mapping
```

Implemented fix:

```text
canonical prompt tokens
-> QTRM prelude text context
-> core_depth_state as query
-> cross-attend over prompt token context
-> concat(core_depth_state, attended_prompt_context)
-> primitive operation logits
```

Config field:

```text
primitive_transition_prompt_token_attention_enabled: true
```

TDD gate:

```text
test:
  tests.test_core_halting.CoreHaltingTests.test_prompt_token_attention_can_distinguish_same_mean_prompt_contexts

red:
  two prompt contexts with identical mean produced identical operation logits

green:
  token-level prompt attention distinguishes the contexts
```

Training:

```text
checkpoint:
  local_eval/qwen35_2b_pure_recursive_primitive_transition_promptattn_s480_from_s720/last.pt

init:
  local_eval/qwen35_2b_pure_recursive_primitive_transition_promptctx_s720_oponly/last.pt

steps:
  480
```

Evaluation:

```text
heldout existing:
  operation accuracy: 256/256 = 1.0000

heldout7000:
  operation accuracy: 512/512 = 1.0000

heldout7000 by family:
  arithmetic_chain: 128/128
  symbolic_binding: 128/128
  boolean_logic:    128/128
  list_transform:   128/128
```

Decision: token-level prompt grounding replaces mean-pooled prompt context for
the primitive transition head. This is still a single-source token path, not a
separate evidence channel. It is the current canonical primitive operation
selector checkpoint for this synthetic raw-core gate.

## Rollout Causality Gate

Operation accuracy alone is not enough. The next gate executes the QTRM
predicted primitive operations against the explicit solver transition function:

```text
QTRM primitive operation logits
-> argmax operation per core depth
-> explicit primitive transition executor
-> state sequence
-> final answer
```

Implemented evaluator:

```text
script:
  scripts/221_eval_qtrm_primitive_transition_rollout.py

shared primitive:
  qtrm_mm.agentic.solver_state_machine.rollout_solver_trace_from_operations
```

Heldout7000 rollout:

```text
checkpoint:
  local_eval/qwen35_2b_pure_recursive_primitive_transition_promptattn_s480_from_s720/last.pt

report:
  local_eval/qwen35_2b_pure_recursive_primitive_transition_promptattn_s480_from_s720/eval_rollout_heldout7000.json

operation exact: 512/512
state exact:     512/512
final exact:     128/128
```

Core-off ablation:

```text
report:
  local_eval/qwen35_2b_pure_recursive_primitive_transition_promptattn_s480_from_s720/eval_rollout_heldout7000_coreoff.json

operation exact: 0/512
state exact:     0/512
final exact:     0/128
```

Interpretation: this is now a causal primitive-reasoning gate, not only an
operation-label classifier. The recurrent core path produces the primitive
operation sequence; that sequence drives explicit state updates; the final
state equals the answer on the heldout7000 synthetic gate.

## Runtime Primitive Answer Path

The rollout gate still used gold rows for scoring. The next step exposes the
same primitive path as a label-free runtime:

```text
prompt
-> QTRM recurrent core operation logits
-> predicted operation sequence
-> primitive transition executor
-> state sequence
-> answer
```

Implemented:

```text
script:
  scripts/222_infer_qtrm_primitive_transition_answer.py

runtime primitive:
  qtrm_mm.agentic.solver_state_machine.answer_from_primitive_operations
```

Arithmetic smoke:

```text
prompt:
  Compute ((7007 + 3) * 2) - 3.

predicted operations:
  add_operands -> multiply_sum -> subtract_offset -> hold_final

states:
  7010 -> 14020 -> 14017

answer:
  14017
```

List-transform smoke:

```text
prompt:
  From [7001, 7004, 7002, 7007, 7003], keep even numbers and double them.

predicted operations:
  filter_even -> double_filtered -> hold_final -> hold_final

states:
  7004,7002 -> 14008,14004

answer:
  14008,14004
```

Artifact examples:

```text
local_eval/qwen35_2b_pure_recursive_primitive_transition_promptattn_s480_from_s720/runtime_answer_arith_7000.json
local_eval/qwen35_2b_pure_recursive_primitive_transition_promptattn_s480_from_s720/runtime_answer_list_7000.json
```

Boundary: this is still a synthetic primitive-reasoning runtime for the
supported operation family set. It is not yet a general autoregressive answer
channel. The important improvement is that final answers can now be produced
from the recurrent core's primitive operation sequence without using
`solver_trace` labels at runtime.

## Answer-Only Runtime Gate

The previous rollout gate scored operation/state traces. The stricter runtime
gate removes `solver_trace` from the runtime row and scores only final answers:

```text
runtime input:
  prompt

runtime does not receive:
  solver_trace
  target intermediate states
  chosen answer

score:
  predicted answer == chosen/answer
```

Implemented:

```text
script:
  scripts/223_eval_qtrm_primitive_answer_runtime.py
```

Heldout7000 answer-only runtime:

```text
report:
  local_eval/qwen35_2b_pure_recursive_primitive_transition_promptattn_s480_from_s720/eval_answer_runtime_heldout7000.json

answer exact:
  128/128

by family:
  arithmetic_chain: 32/32
  symbolic_binding: 32/32
  boolean_logic:    32/32
  list_transform:   32/32
```

Core-off answer-only ablation:

```text
report:
  local_eval/qwen35_2b_pure_recursive_primitive_transition_promptattn_s480_from_s720/eval_answer_runtime_heldout7000_coreoff.json

answer exact:
  0/128
```

Interpretation: this is the current strongest synthetic primitive-reasoning
evidence. It proves the supported-family runtime can answer from prompt-only
primitive operations selected by the recurrent core, and that disabling the
core destroys the path.

## Donor-Only Baseline Comparison

Added a donor-only baseline evaluator:

```text
script:
  scripts/224_eval_donor_only_baseline.py
```

Modes:

```text
forced_choice:
  choose the candidate answer with highest mean donor token logprob

greedy:
  donor.generate from the same prompt, then exact-match the first generated line
```

Heldout7000 comparison:

```text
QTRM primitive answer runtime:
  answer exact: 128/128 = 1.0000

donor-only forced_choice:
  answer exact: 61/128 = 0.4766

donor-only greedy:
  answer exact: 29/128 = 0.2266
```

Donor-only by family:

```text
forced_choice:
  arithmetic_chain: 15/32
  symbolic_binding: 17/32
  boolean_logic:    29/32
  list_transform:    0/32

greedy:
  arithmetic_chain:  0/32
  symbolic_binding:  0/32
  boolean_logic:    29/32
  list_transform:    0/32
```

Artifact:

```text
local_eval/qwen35_2b_pure_recursive_primitive_transition_promptattn_s480_from_s720/eval_donor_only_heldout7000.json
```

Conclusion for this gate: QTRM primitive runtime is better than donor-only on
the synthetic primitive reasoning heldout7000 benchmark. This is not a general
claim that QTRM beats the donor as a broad LLM; it is a scoped claim for the
supported primitive-reasoning families and this heldout distribution.

## Benchmark Definition And Examples

The current `synthetic primitive reasoning` benchmark is not just math. It is a
small algorithmic reasoning benchmark with four supported families:

```text
arithmetic_chain:
  chained arithmetic

list_transform:
  list filtering and transformation

symbolic_binding:
  symbolic mapping / state propagation

boolean_logic:
  boolean expression evaluation
```

The model receives only the prompt at runtime. It must choose a primitive
operation sequence, execute the state transitions, and return the final state
as the answer.

Arithmetic example:

```text
prompt:
  Compute ((7007 + 3) * 2) - 3.

target answer:
  14017

primitive sequence:
  add_operands -> multiply_sum -> subtract_offset -> hold_final

state tape:
  "" -> 7010 -> 14020 -> 14017
```

List transform example:

```text
prompt:
  From [7001, 7004, 7002, 7007, 7003], keep only even numbers
  and double each kept number.

target answer:
  14008,14004

primitive sequence:
  filter_even -> double_filtered -> hold_final

state tape:
  "" -> 7004,7002 -> 14008,14004
```

Symbolic binding example:

```text
prompt:
  If A maps to green, green maps to violet, and violet maps to D,
  what does A map to after two mappings?

target answer:
  violet

primitive sequence:
  first_mapping -> second_mapping -> hold_final

state tape:
  "" -> green -> violet
```

Boolean logic example:

```text
prompt:
  Let P=TRUE, Q=FALSE, R=FALSE.
  Evaluate (P AND NOT Q) OR R.

target answer:
  TRUE

primitive sequence:
  not_q -> and_with_p -> or_with_r -> hold_final

state tape:
  "" -> TRUE -> TRUE -> TRUE
```

Interpretation: this benchmark measures whether the QTRM recurrent core can
select a correct procedure from the prompt and drive explicit state updates. It
does not yet measure open-domain knowledge, broad natural-language instruction
following, long-context retrieval, or general LLM quality.

## OOD Surface-Form Robustness Gate

To test whether the result is only exact-template memorization, added a
surface-form OOD builder:

```text
script:
  scripts/225_build_pure_recursive_ood_surface_cases.py

data:
  data/eval/pure_recursive_primitive_transition_ood_surface_heldout8000_preferences.jsonl

cases:
  128 total
  32 per family
```

The OOD gate preserves the same primitive families and answers, but rewrites
the prompt wording. Runtime still receives only the prompt.

OOD surface comparison:

```text
QTRM primitive answer runtime:
  107/128 = 0.8359

QTRM core-off:
    0/128 = 0.0000

donor-only forced_choice:
   71/128 = 0.5547

donor-only greedy:
   32/128 = 0.2500
```

QTRM by family:

```text
arithmetic_chain: 11/32
symbolic_binding: 32/32
boolean_logic:    32/32
list_transform:   32/32
```

Arithmetic surface variants:

```text
variant 0:
  "Solve this arithmetic expression exactly: ..."
  11/11

variant 1:
  "What integer do you get after evaluating ...?"
  0/11

variant 2:
  "Evaluate the expression ... and report only the result."
  0/10
```

Dominant arithmetic failure:

```text
target operation family:
  add_operands -> multiply_sum -> subtract_offset -> hold_final

predicted operation families:
  not_q -> double_filtered -> subtract_offset -> hold_final
  filter_even -> double_filtered -> subtract_offset -> hold_final
```

Interpretation: QTRM still beats donor-only on the OOD surface gate, and the
core path remains causal. However, the model is not robust enough to claim
general arithmetic-language understanding. The bottleneck is prompt wording
sensitivity in arithmetic operation-family routing. The next training gate
should add surface-form augmentation for arithmetic and require OOD arithmetic
to recover before moving to new operation families.

## Surface-Form Augmentation Recovery Gate

Built a small canonical plus OOD-surface training mix and resumed from the
prompt-attention primitive checkpoint:

```text
builder:
  scripts/226_build_pure_recursive_surface_aug_mix.py

train data:
  data/filtered/pure_recursive_primitive_transition_surface_aug_mix_train.jsonl

checkpoint:
  local_eval/qwen35_2b_pure_recursive_primitive_transition_surface_aug_s640_from_promptattn/last.pt
```

Training command used only the primitive transition operation CE path. Answer
CE stayed off, so this tests whether the recurrent core can repair procedure
routing without directly memorizing final answer strings.

Recovery result on the same OOD surface gate:

```text
QTRM primitive answer runtime:
  128/128 = 1.0000

QTRM core-off:
    0/128 = 0.0000

arithmetic_chain: 32/32
symbolic_binding: 32/32
boolean_logic:    32/32
list_transform:   32/32
```

Canonical heldout regression check:

```text
QTRM primitive answer runtime:
  128/128 = 1.0000

arithmetic_chain: 32/32
symbolic_binding: 32/32
boolean_logic:    32/32
list_transform:   32/32
```

Interpretation: the earlier arithmetic surface-form failure was trainable
through the causal primitive-operation path. The recovery does not prove broad
open-domain arithmetic or general LLM ability, but it does strengthen the
primitive-runtime claim: prompt wording changes can be absorbed by the
mandatory recurrent core without breaking the canonical gate.
