# Stage59 Final-Only Typed Register Answer Path

Date: 2026-05-22

## HRM-Text Reference Standard

HRM-Text is now the mandatory reference point for QTRM/GRAM/PTRM language
reasoning experiments.

This does not mean "copy the HRM-Text implementation." It means every next
architecture must pass the same plain-language standard HRM-Text makes obvious:

```text
the state that reads must feed the state that thinks,
and the state that thinks must be the state that speaks.
```

Plain-language comparison:

```text
HRM-Text:
  one body
  tokens -> hierarchical recurrent thought -> token logits

Our old failure mode:
  fluent Qwen reader
  + external calculation desk
  + separate mouth that often ignores or mistranslates the desk
```

Stage59 consequence:

```text
Typed registers, GRAM, PTRM, executors, attractors, or ledgers are valid only
when they become the normal answer-causal path. A trace-accurate side organ is
not progress unless the final emitted answer depends on it and the off ablation
destroys the gain.
```

Required preflight before any new run:

```text
1. Reader:
   Does Qwen preserve the information the recurrent thought needs?

2. Thinker:
   Is the recurrent/GRAM/PTRM state the main state used for the final answer?

3. Speaker:
   Can the final token/character speaker understand the thought state?

4. No bypass:
   Can a pretrained shortcut, side head, candidate pool, or trace-only module
   answer without the thought path?

5. Ablation:
   Does disabling the HRM-like thought-to-speech path remove the measured gain?
```

Current implication:

```text
The next highest-probability fix is not another standalone thought probe.
It is a stricter thought-to-mouth path, such as a ledger-forced renderer or
Qwen-compatible LM-token rendering, where the typed ledger becomes the alphabet
the answerer actually speaks.
```

## Accepted Next Local Candidate: LFCR

`Ledger-Forced Copy Renderer` is the current local candidate that most directly
implements the HRM-Text one-body rule for numeric/list answers.

Plain-language contract:

```text
The desk does not merely whisper to the mouth.
The final digit ledger becomes the mouth's digit alphabet.
```

Technical contract:

```text
typed_digit_register_trajectory[-1]
-> learned ledger copy attention
-> digit/presence logits
-> mapped into actual answer character vocab digit IDs
-> normal char_logits used by candidate decoding
```

This is still learned deep-learning machinery, not a Python calculator:

```text
No task_family branch.
No formula execution in eval.
No label-filled oracle register.
No heuristic candidate pool.
```

Required ablation:

```text
--eval-ledger-forced-copy-renderer-off
```

Promotion gate:

```text
arithmetic_chain or list_transform must rise above 0/N, and the exact gain must
disappear when ledger_forced_copy_renderer_off is enabled.
```

## Next Big-Jump Candidate: BECTO

`Base-Equivariant Column Thought Organ` is the next architecture candidate after
LFCR rejected.

Plain-language diagnosis:

```text
LFCR gave the model a better mouth for reading the ledger, but the ledger still
contains the wrong large-number world. The failure examples are not random:

  target 8017 -> selected 4077
  target 8008,8004 -> selected 4024,4024

The model speaks number-shaped strings, but it has not learned the human-like
place-value procedure that makes "4000 input base" turn into "8000 output base."
```

Plain-language invention:

```text
Give the model a hand procedure, not another mouth.

Numbers must be split into digit columns. The same learned column cell must run
on ones, tens, hundreds, and thousands. A carry/keep/procedure state must move
between neighboring columns. The final typed ledger then speaks through LFCR.
```

Technical contract:

```text
Qwen/source number slots
-> typed digit-column workspace
-> shared column procedure cell
   inputs: previous digit column, source digit column, operation id,
           operation argument, carry/procedure state
   outputs: next digit logits, presence logits, carry/procedure state
-> committed typed digit ledger
-> LFCR/token output
```

Why it is a big-jump candidate:

```text
GRAM/PTRM changed the reasoning/search path.
BECTO changes the numeric world's physics: the model cannot treat "4007" as a
new memorized blob. It must reuse the same column rule that worked for 7, 1007,
2007, and 3007.
```

Required ablation:

```text
--eval-column-procedure-off
```

Promotion gate:

```text
arithmetic_chain or list_transform must rise above 0/N, and the exact gain must
disappear when the column procedure is disabled.
```

Sources:

- HRM-Text, 2026-05 arXiv: `https://arxiv.org/abs/2605.20613`
- Hugging Face Transformers HRM-Text docs, added 2026-05:
  `https://huggingface.co/docs/transformers/main/model_doc/hrm_text`
- Original HRM, 2025: `https://arxiv.org/abs/2506.21734`

## Decision

Stage59 work is now final-path only.

Deprecated diagnostic scaffolds:

- hand-built `typed_heuristic_candidates`
- noisy typed-choice materialization
- typed-pool selector over a heuristic pool
- standalone char proposer as the main answer writer
- standalone extractor/probe results presented as progress

These artifacts may be used only to audit old claims. They are not valid next
experiments and cannot be promoted.

## Plain-Language Rule

The model needs one body:

```text
reader -> working desk -> thought loop -> candidate table -> checker -> answer
```

It is no longer acceptable to place a hand-made answer table next to the model
and then celebrate that the checker can pick from it. That proved the failure
mode, but it is not the machine we are trying to build.

The final architecture must make the working desk itself learned and must route
the final answer through that desk.

## What Typed Register Means

`typed register` does not mean a hand-coded calculator, parser, or family
executor.

Plain-language definition:

```text
Typed registers are learned working-memory slots. We give the desk several
named places to put things; the model must learn what to write there, how to
update it, and how to use it to answer.
```

Deep-learning definition:

```text
Qwen hidden states
-> learned slot/type embeddings
-> learned attention/gated recurrent writes
-> learned register states
-> learned candidate and verifier heads
```

Allowed:

- learned slot embeddings;
- learned cross-attention from registers to Qwen token states;
- learned gated recurrent updates;
- learned heads that decode/register-score candidates;
- auxiliary supervision on registers when the same registers are used by the
  evaluated answer path.

Not allowed:

- `if task_family == arithmetic_chain: compute formula`;
- Python arithmetic/list-transform functions in the normal answer path;
- a hand-built candidate pool presented as model output;
- an oracle register filled from labels at eval time.

Human analogy:

```text
The register is graph paper, not a calculator.
The grid helps the student organize work, but the student still has to learn
what to write, how to revise it, and how to check the answer.
```

## Fresh Memory-Slot Literature Check

Do not justify typed registers only from classic memory papers. In 2025-2026,
the same idea appears under newer names:

```text
working memory
auxiliary memory
latent memory tokens
learnable memory banks
test-time memory
feed-forward memory
```

Current sources:

- Mixture of Chapters, 2026: learnable sparse memory banks queried by
  cross-attention and routed by chapters.
  `https://arxiv.org/abs/2603.21096`
- MemoryLLM, 2026: feed-forward layers treated as interpretable token-wise
  neural retrieval memory.
  `https://arxiv.org/abs/2602.00398`
- ATLAS, 2025: test-time long-term memory module that learns what to memorize
  from current and past tokens.
  `https://arxiv.org/abs/2505.23735`
- LM2, 2025: decoder Transformer with an auxiliary memory module, cross
  attention, and gated updates.
  `https://arxiv.org/abs/2502.06049`
- Memory-Augmented Transformers survey, 2025: frames the field around reading,
  writing, forgetting, capacity management, and adaptive test-time memory.
  `https://arxiv.org/abs/2508.10824`

Plain-language interpretation:

```text
The modern literature has not abandoned memory slots.
It renamed and generalized them into trainable working-memory lanes.
```

Stage59 consequence:

```text
Typed registers remain a current, defensible direction only if they are used
inside the evaluated answer path and pass register-off/candidate-off ablations.
They are not enough by themselves; the speaker that turns register states into
answer strings must also be trained as part of the same path.
```

## Canonical Answer Path

```text
Qwen token reader
-> learned typed working registers
   (register slots may read the preserved Qwen token workspace while writing)
-> recurrent / GRAM thought transition over those registers
-> learned typed candidate table
-> learned verifier / selector
-> selected answer copy or Qwen-compatible token output
```

## Accepted Register Write Fix

Plain-language finding:

```text
The desk was real, but each drawer was reading a tiny summary note instead of
the original page. Similar arithmetic/list problems could therefore overwrite
each other inside the desk.
```

Accepted local fix:

```text
working-register source attention
```

Meaning:

```text
During register initialization and recurrent register writes, each learned
register slot may cross-attend to the preserved Qwen token workspace. The
answer speaker still reads the typed registers by default, not the workspace.
```

Why this is not a bypass:

```text
reader -> register write -> thought/register trajectory -> AR register speaker
```

The source tokens are used to write the working desk. They are not exposed to
the final answerer as a separate shortcut unless `--answerer-use-workspace` is
explicitly enabled, which is off for the accepted Stage59 path.

Local causality gate:

```text
v6 trainable QTRM/GRAM typed-register thought core:
  best 8-row overfit exact = 0.75

v7 + mean register identity anchor:
  best 8-row overfit exact = 0.75

v8 + register source attention:
  best 8-row overfit exact = 1.0 at epoch 78
  typed_register_off exact = 0.0
```

Interpretation:

```text
The high-probability direction is source-written working memory, not a
post-hoc selector or heuristic candidate table. The next generalization run
should preserve this causal path.
```

## Numeric/List Generalization Finding

Plain-language finding from train64/eval32:

```text
The desk can now read the page, and it can generalize simple names/booleans.
But for arithmetic and lists it still speaks old training-range numbers.
```

Observed gate:

```text
train64 -> eval32, source-written register path:
  best selected exact = 0.46875
  typed_register_off exact = 0.0

family split:
  boolean_logic = 7/8
  symbolic_binding = 8/8
  arithmetic_chain = 0/8
  list_transform = 0/8
```

Rejected as insufficient:

```text
Qwen-pretrained character/LM-head initialization:
  same best exact = 0.46875

per-character register attention in the AR speaker:
  same best exact = 0.46875

per-character value lanes in the AR speaker:
  same best exact = 0.46875
  arithmetic_chain = 0/8
  list_transform = 0/8

register trace supervision on meaningful solver_trace states:
  same best exact = 0.46875
  register_trace_char_accuracy rose to 0.7374
  arithmetic_chain = 0/8
  list_transform = 0/8

scalar numeric/list value fields:
  same best exact = 0.46875
  numeric_presence_accuracy = 0.9463
  numeric_value_loss = 0.1072
  arithmetic_chain = 0/8
  list_transform = 0/8

answerer-side digit-place fields:
  same best exact = 0.46875
  digit_place_digit_accuracy = 0.3307
  digit_place_presence_accuracy = 0.8914
  arithmetic_chain = 0/8
  list_transform = 0/8
```

Interpretation:

```text
The remaining bottleneck is not just "the mouth does not know digits" or "the
mouth needs to re-read the desk per character." It is also not solved merely by
asking the desk to speak intermediate numeric strings. The model needs a
structured numeric/list value representation inside typed registers, so
arithmetic/list answers are carried as quantities and list elements before
being spoken as characters.

The scalar value-field gate showed that a rough quantity side signal is still
not enough. It nudged outputs toward large training-range numbers but did not
learn digit-place arithmetic or list element updates.

The answerer-side digit-place gate also failed. It learned "this position is
digit-like" better than the actual digit identity, and outputs still collapsed
to training-range patterns such as 6022 or 6022,6000.
```

Guardrail:

```text
Do not reopen heuristic calculators or answerer-workspace bypasses to fix this.
The next accepted direction must keep:

Qwen token reader
-> source-written typed working registers
-> QTRM/GRAM typed-register thought core
-> register-only answer speaker
```

## Numeric Value Field Requirement

Plain-language rule:

```text
For numbers and lists, the desk must stop treating values as decorative text.
It needs typed value fields: a learned place where "8017" lives as a quantity
or element value before the mouth renders the characters 8-0-1-7.
```

Allowed next direction:

```text
operation-conditioned typed value fields inside the same register path
```

This means:

```text
source-written typed registers
-> recurrent/GRAM transition
-> learned value-field heads for numeric/list state
-> AR typed-register speaker
```

It does not mean:

```text
if arithmetic_chain: run Python arithmetic
if list_transform: run a hand-coded list transform
```

Falsifiable gate:

```text
On the existing train64 -> eval32 split, arithmetic_chain or list_transform
must rise above 0/8 while typed_register_off remains 0.0. If value-field losses
improve but both families remain 0/8, the value field is still text-shaped and
not a portable quantity representation.
```

Current status:

```text
This gate failed for scalar numeric value fields.
```

Next narrower requirement:

```text
Use digit-place/list-element typed value registers in the recurrent thought
core, not only in the speaker and not as a single scalar smell:

source-written typed registers
-> digit/list element slots
-> learned operation/carry/list-position update state
-> AR typed-register speaker
```

Plain-language rule:

```text
For arithmetic, "8017" is not one smell. It is four places with rules.
For lists, "8008,8004" is not one string. It is ordered element slots.
The register path must represent that structure before speaking.
```

## Stage59 v15-v17 Narrowing Evidence

Plain-language read:

```text
We tried giving the model a bigger desk, telling it which action verb to use,
and then placing the visible source numbers on the desk. The simple
boolean/name tasks still worked, but arithmetic and list updates stayed at
zero. So the missing organ is not just more room, a verb label, or source
number visibility. The missing organ is the hand movement that updates typed
value/list registers step by step.
```

Observed local gates:

```text
v15 core 16 slots:
  best selected exact = 0.46875
  arithmetic_chain = 0/8
  list_transform = 0/8
  boolean_logic = 7/8
  symbolic_binding = 8/8
  typed_register_off = 0.0

v16 trace operations:
  best selected exact = 0.46875
  arithmetic_chain = 0/8
  list_transform = 0/8
  boolean_logic = 7/8
  symbolic_binding = 8/8
  typed_register_off = 0.0

v17 source number slots:
  best selected exact = 0.46875 at epoch 23
  arithmetic_chain = 0/8
  list_transform = 0/8
  boolean_logic = 7/8
  symbolic_binding = 8/8
  typed_register_off = 0.0
  register_trace_char_accuracy = 0.6797

v18 typed value registers inside recurrent thought core:
  best selected exact = 0.46875 at epoch 23
  arithmetic_chain = 0/8
  list_transform = 0/8
  boolean_logic = 7/8
  symbolic_binding = 8/8
  typed_register_off = 0.0
  register_trace_char_accuracy = 0.7673

v19 typed value registers + digit-place speaker:
  best selected exact = 0.46875 at epoch 26
  arithmetic_chain = 0/8
  list_transform = 0/8
  boolean_logic = 7/8
  symbolic_binding = 8/8
  typed_register_off = 0.0

v20 typed value registers + core-side trace value loss:
  best selected exact = 0.46875 at epoch 18
  arithmetic_chain = 0/8
  list_transform = 0/8
  boolean_logic = 7/8
  symbolic_binding = 8/8
  typed_register_off = 0.0
  typed_value_trace_loss = 0.1198
  typed_value_trace_value_mae = 3820.08
```

Representative v17 failures:

```text
8017 -> 6007
12032 -> 6007
16051 -> 6007
20074 -> 6007
8008,8004 -> 6004,6000
```

Decision:

```text
Do not run another answerer-only or source-visibility-only variant as the main
experiment. The v18/v19 gates also show that merely appending learned value
slots to the recurrent trajectory is not enough; they start touching digit
fragments but do not learn reliable arithmetic/list state evolution.

Required story:
  reader places source values into typed slots
  recurrent thought core selects an operation
  core updates digit/place/list-element slots
  speaker renders from those updated slots

Local gate:
  train64 -> eval32
  arithmetic_chain or list_transform must exceed 0/8
  typed_register_off must remain 0.0
```

Updated failure read:

```text
v18 changed numeric errors from pure training-range copying toward partial
digit fragments, e.g. 16064 -> 12064 and 20090 -> 15090. That is useful
evidence that value slots are being touched.

But exact arithmetic/list remained 0/8, and v19's digit-place speaker did not
repair it. The missing part is not only "a place to store numbers" or "a mouth
that knows digit positions." The core needs a directly supervised internal
value-state objective at trace steps, so the recurrent hand learns what each
typed value/list slot should hold after each operation.

v20 added that scalar value-state objective. It learned presence and reduced
the loss, but exact arithmetic/list still stayed 0/8. The errors got closer in
some cases, e.g. 8017 -> 6017 and 12032 -> 9032, but the thousands/place
structure still collapses. So a single scalar value trace is still too blunt.

Next narrower requirement:
  teach digit-place/list-element state inside the typed value registers
  directly, not only scalar magnitude.
```

## Brain-Inspired Literature Rule

Plain-language rule:

```text
Use brain-inspired papers as architectural pressure, not decoration. A paper is
relevant only if it changes one of these organs:

reader:
  what information is preserved from the prompt;

working memory:
  how values are stored in typed slots;

executive update:
  how a step chooses preserve/update/overwrite;

speaker:
  how the internal state becomes the evaluated answer;

ablation:
  how we prove the answer depends on the thought path.
```

Current relevant threads:

```text
On the Failure of Latent State Persistence in Large Language Models
  https://arxiv.org/abs/2505.10571
  Use for the failure claim: ordinary LLMs lack persistent hidden working
  state for variable binding and state evolution.

G-MemLLM: Gated Latent Memory Augmentation for Long-Context Reasoning
  https://arxiv.org/abs/2602.00015
  Use for the mechanism: gated latent memory slots that preserve/update/
  overwrite rather than compressing everything into one vector.

LLM Reasoning Is Latent, Not the Chain of Thought
  https://arxiv.org/abs/2604.15726
  Use for the evaluation frame: reasoning should be measured as latent-state
  trajectory, not only surface text traces.

BMAM: Brain-inspired Multi-Agent Memory Framework
  https://arxiv.org/abs/2601.20465
  Use only as a high-level memory-system analogy. It is agent-memory work, not
  direct evidence for the neural core.
```

Freshness guard:

```text
Classic memory papers are roots, not enough. If a proposed change is justified
mainly by old NTM/DNC/Memory Network/Slot Attention style citations, search
for 2025-2026 follow-ups before launching a run.
```

## v21-v22 Computational Organ Narrowing

Plain-language read:

```text
The working desk now has drawers, can read the source page, can remember typed
values, and can even learn a better erase/write motion. But it is still not a
calculator. It writes in a foggy continuous language where "8017" is a blurry
quantity, not four visible columns with carry rules.

So the next accepted architecture must add the missing organ: digit columns,
carry state, and ordered list element positions inside the recurrent thought
state.
```

Latest mechanisms being used:

```text
GatedDeltaNet-2 / FG^2-GDN:
  Separate erasing old content from writing new content. Use finer per-channel
  control instead of one scalar handbrake.

Preconditioned DeltaNet / OSDN:
  Treat recurrent memory update as an online correction problem and scale the
  correction by feature-wise confidence/curvature.

LoopFormer / recurrent-depth work:
  Keep recurrence depth elastic, but do not confuse loop stability with actual
  algorithmic execution.
```

Current anchors:

```text
Gated DeltaNet-2:
  https://huggingface.co/papers/2605.22791
  https://github.com/NVlabs/GatedDeltaNet-2
  Use the erase/write separation principle, not a wholesale repo transplant.

FG^2-GDN:
  https://arxiv.org/abs/2604.19021
Preconditioned DeltaNet:
  https://arxiv.org/abs/2604.21100
OSDN:
  https://arxiv.org/abs/2605.13473
LoopFormer:
  https://arxiv.org/abs/2602.11451
Loop, Think, & Generalize:
  https://arxiv.org/abs/2604.07822
```

Local evidence:

```text
v21 core-side digit-place trace:
  best selected exact = 0.46875 at epoch 22
  arithmetic_chain = 0/8
  list_transform = 0/8
  boolean_logic = 7/8
  symbolic_binding = 8/8
  typed_register_off = 0.0
  register_trace_char_accuracy = 0.7495
  typed_value_trace_loss = 0.0542
  typed_value_trace_value_mae = 2379.99
  typed_value_digit_trace_loss = 1.4602
  typed_value_digit_trace_digit_accuracy = 0.5170

v22 GatedDeltaNet-2-style typed value executor:
  best selected exact = 0.46875 at epoch 22
  arithmetic_chain = 0/8
  list_transform = 0/8
  boolean_logic = 7/8
  symbolic_binding = 8/8
  typed_register_off = 0.0
  register_trace_char_accuracy = 0.7395
  typed_value_trace_loss = 0.0177
  typed_value_trace_value_mae = 1089.29
  typed_value_digit_trace_loss = 1.3348
  typed_value_digit_trace_digit_accuracy = 0.5089

v22 later training signal:
  best value MAE epoch = 27
  best value MAE = 632.48
  best digit trace accuracy epoch = 28
  best digit trace accuracy = 0.6633
  exact arithmetic/list remained 0/8 at the selected gate.
```

Interpretation:

```text
GatedDeltaNet-2-style erase/write separation helped the memory-editing motion:
the typed value state became easier to inspect and the value MAE improved.

But the model still did not perform arithmetic/list execution. That means the
missing part is not merely "better memory update." The missing part is a
first-class computational state: digit columns, carry, operation phase, and
list element cursor/position.
```

Rejected as the next main move:

```text
Another answerer-only digit head.
Another trace reader that decodes blurry values after the fact.
Another scalar numeric smell loss.
Another pure memory-gating variant without explicit digit/carry/list-position
state.
```

Accepted next architecture constraint:

```text
Qwen token reader
-> source numeric/list slots
-> explicit digit/list element registers
-> operation-conditioned carry/list-position executor
-> typed-register speaker
-> typed_register_off ablation must collapse the result
```

Plain-language rule:

```text
Do not ask the model to "feel" the number 8017 anymore.
Give it visible columns: 8 | 0 | 1 | 7.
Give it a carry pocket.
Give lists visible element boxes.
Then let the learned recurrent executor decide what to erase, retain, write,
and advance at each operation step.
```

Falsifiable local gate:

```text
On the same train64 -> eval32 split:
  arithmetic_chain > 0/8 or list_transform > 0/8
  typed_register_off exact = 0.0
  digit/carry/list-position trace improves without using a Python executor
  boolean/symbolic must not regress below the current 15/16 combined hits
```

Implemented v23 skeleton:

```text
StateTransitionCore now supports:
  --typed-digit-registers
  --typed-digit-register-digits
  --typed-digit-update-scale

The recurrent answer trajectory now can include:
  learned working registers
  typed value registers
  explicit digit-column registers
  one carry pocket per source-number slot

Stage59 training now supports:
  --typed-digit-register-trace-weight

This trace loss reads the explicit digit-column registers directly. It is not
another post-hoc value-slot reader.
```

Smoke evidence:

```text
/tmp/stage59_v23_typed_digit_smoke
  train_limit = 4
  eval_limit = 4
  epochs = 1
  typed_digit_register_trace_loss = 3.7658
  typed_digit_register_trace_digit_accuracy = 0.25
  typed_digit_register_trace_presence_accuracy = 0.4375
  answerer bypasses off
  typed_register_off accuracy = 0.0

This is only a wiring smoke, not a promotion run. It proves the new
digit/carry organ is in the forward/backward path and is visible to the
register-only speaker.
```

v23 local gate result:

```text
/tmp/stage59_final_typed_register_answerer_train64_eval32_v23_typed_digit_carry_executor
  train64 -> eval32
  best selected exact = 0.46875 at epoch 17
  typed_register_off exact = 0.0
  arithmetic_chain = 0/8
  list_transform = 0/8
  boolean_logic = 7/8
  symbolic_binding = 8/8

best epoch trace signals:
  register_trace_char_accuracy = 0.7165
  typed_value_trace_value_mae = 965.58
  typed_value_digit_trace_digit_accuracy = 0.5833
  typed_digit_register_trace_loss = 1.3249
  typed_digit_register_trace_digit_accuracy = 0.5802
  typed_digit_register_trace_presence_accuracy = 0.9429

strongest digit-register trace signal:
  epoch 30 typed_digit_register_trace_digit_accuracy = 0.8268
  eval exact still = 0.46875
  arithmetic_chain/list_transform still = 0/8
```

v23 interpretation:

```text
The digit/carry worksheet is real and trainable, but the speaker still mostly
renders old training-range templates. The next missing bridge is not another
hidden memory slot. It is an execution-to-speech bridge: the answerer must read
the digit-column registers as output columns, not only as extra context in a
large register soup.
```

## Naming Rule

Do not call the current trainable thought module only "QTRM core" in Stage59
notes. That phrase is too broad now.

Use this short name:

```text
QTRM/GRAM typed-register thought core
```

Use this precise name when reporting experiments:

```text
QTRM StateTransition + true-GRAM + typed working-register core
```

Plain-language map:

```text
QTRM:
  the overall research body / wrapper family.

StateTransitionCore:
  the repeated thought loop.

true-GRAM:
  the stochastic transition mode inside the thought loop.

typed working register:
  the learned working desk used by the answer path.

PTRM / VTE:
  the candidate-exposure / runtime search lineage used by Stage58-style
  evaluation. Do not describe the Stage59 final typed-register answerer as a
  PTRM run unless that runtime search path is actually enabled.

AR typed-register speaker:
  the mouth that turns register states into answer strings.
```

Therefore, the current Stage59 final-path smoke should be described as:

```text
QTRM/GRAM typed-register thought core -> AR typed-register speaker
```

Required properties:

- The typed register is in the normal evaluated answer path.
- The candidate table is generated by learned modules, not by a hand-coded
  family solver or heuristic pool.
- `typed` means role-biased latent slots, not task-specific symbolic code.
- Auxiliary losses are allowed only if they train modules used by the evaluated
  answer path.
- The final metric is selected-answer exact accuracy by family.

## Seven-Axis Preflight

Before any local or DGX run, the architecture must pass:

1. Architecture: who reads, thinks, remembers, checks, and speaks?
2. Curriculum: is the model learning the final path, not a disconnected probe?
3. Reward/loss: does the loss reward the same answer path used at eval?
4. Evaluation: is the reported score the final selected answer, not oracle
   coverage?
5. Exploration: if GRAM/search is claimed, do sampled paths differ and help?
6. Data contract: does the prompt contain the needed information?
7. Causality/ablation: does accuracy fall when typed registers, candidate
   generation, verifier, or recurrent thought are disabled?

If any axis fails, do not launch the run.

## Accepted Prior Evidence

The earlier heuristic typed-pool experiments are retained only as causal
evidence:

```text
free char candidate generation:
  arithmetic_chain/list_transform coverage = 0.0

Qwen-generated candidates:
  oracle coverage around 0.31-0.38, arithmetic/list effectively missing

heuristic typed pool + selector:
  exposed oracle coverage around 0.945

selector-exposed verifier:
  selected accuracy around 0.628 on usable rows
```

Interpretation:

```text
The checker is not the main missing organ.
The missing organ is a learned typed working table that can preserve and expose
full integer/list/symbol/boolean values.
```

## Implementation Lock

The following scripts now require `--allow-diagnostic-scaffold` before running
their hand-built candidate path:

```text
scripts/525_eval_qwen_candidate_exposure.py --candidate-source typed_heuristic
scripts/526_materialize_noisy_candidate_choices.py
scripts/528_train_candidate_pool_selector.py
scripts/529_materialize_pool_selector_choices.py
```

This is intentional. Future Stage59 runs must use the learned final path, not a
middle scaffold.

Current final-path implementation:

```text
scripts/530_train_final_typed_register_answerer.py
```

This script trains the evaluated path directly:

```text
Qwen/QTRM context
-> qtrm_working_register_trajectory
-> learned TypedRegisterAnswerer
-> generated candidate strings
-> learned selector logits
-> selected answer exact score
```

Default answerer bypass policy:

```text
--no-answerer-use-qtrm-readout
--no-answerer-use-state-trajectory
--no-answerer-use-workspace
```

Meaning:

```text
The answerer should speak from the learned working-register trajectory.
Direct Qwen readout, raw state trajectory, and raw workspace attention are
optional diagnostic bypasses, not the default final path.
```

Important boundary:

```text
The row choices are training targets for the learned candidate table, not
runtime candidate inputs. At eval, candidates are generated from the learned
register path.
```

Training-target boundary:

```text
Default candidate supervision is answer_only with a fixed answer slot.
Do not train the model to reproduce shuffled row-choice distractor order unless
explicitly running a diagnostic. The prompt does not contain that hidden order.
```

## Promotion Gate

Promote only if all are true:

- held-out exact accuracy improves over the 0.6484 supplied-choice verifier
  baseline or clearly improves the arithmetic/list families first;
- arithmetic_chain and list_transform are both nonzero and rising;
- oracle coverage is reported separately and is not counted as model accuracy;
- typed-register-off, candidate-generator-off, verifier-off, and recurrent-off
  ablations remove the gain.

## 2026-05-22 Stage59 v24-v27 Digit Bridge and Multibase Gate

Plain-language read:

```text
The student can now read the page, write a digit worksheet, and speak from that
worksheet. But the student is still mostly copying old worksheet shapes. The
missing skill is the hand movement from one line of work to the next:

previous written state + operation + operand -> next written state
```

Local runs:

```text
v24 final digit-register output bridge:
  run = /tmp/stage59_final_typed_register_answerer_train64_eval32_v24_digit_output_bridge
  best selected exact = 0.46875 at epoch 27
  typed_register_off exact = 0.0
  arithmetic_chain = 0/8
  list_transform = 0/8
  boolean_logic = 7/8
  symbolic_binding = 8/8
  max typed_digit_register_trace_digit_accuracy = 0.8556 at epoch 29

v25 trace-step digit bridge:
  run = /tmp/stage59_final_typed_register_answerer_train64_eval32_v25_trace_digit_bridge
  best selected exact = 0.46875 at epoch 19
  typed_register_off exact = 0.0
  arithmetic_chain = 0/8
  list_transform = 0/8
  boolean_logic = 7/8
  symbolic_binding = 8/8

v26 multibase train256 -> eval4000:
  run = /tmp/stage59_final_typed_register_answerer_v26_multibase_train256_eval4000
  best selected exact = 0.5 at epoch 11
  last selected exact = 0.46875
  typed_register_off exact = 0.0
  arithmetic_chain = 0/8
  list_transform = 0/8
  boolean_logic = 8/8 at best, 7/8 last
  symbolic_binding = 8/8
  max typed_digit_register_trace_digit_accuracy = 0.9995 at epoch 40

v27 multibase + operation-argument conditioning:
  run = /tmp/stage59_final_typed_register_answerer_v27_multibase_oparg_train256_eval4000
  best selected exact = 0.46875 at epoch 7
  last selected exact = 0.46875
  typed_register_off exact = 0.0
  arithmetic_chain = 0/8
  list_transform = 0/8
  boolean_logic = 7/8
  symbolic_binding = 8/8
  max typed_digit_register_trace_digit_accuracy = 0.9992 at epoch 34
```

Representative v27 failures:

```text
8017 -> 6017
12032 -> 9032
16051 -> 8051
20074 -> 15074
8008,8004 -> 4008,4004
8016,8028,8020 -> 4016,4028,4020
```

Seven-axis humanistic check:

```text
1. Architecture:
   reader, worksheet, and speaker exist; the transition executor is weak.

2. Curriculum:
   multibase data fixed the most obvious "3000s train -> 4000s eval" shortcut,
   but the model still learned trace reconstruction, not reusable execution.

3. Reward/loss:
   digit trace losses reward copying the teacher's written lines. They do not
   force a shared operation-conditioned next-state rule.

4. Evaluation:
   family split is decisive: boolean/symbolic pass, arithmetic/list stay 0/8.

5. Exploration:
   not the active bottleneck; this path is deterministic typed execution, not
   a GRAM K-sampling claim.

6. Data contract:
   source values, operations, digit slots, and op arguments are now present.
   The remaining failure is not missing information.

7. Causality/ablation:
   typed_register_off = 0.0, so the answer depends on the register path. The
   path itself still does not execute arithmetic/list transitions.
```

Decision:

```text
Do not spend the next local run on a bigger memory bank, another mouth bridge,
or another operation label. v24-v27 show those organs are no longer the highest
probability bottleneck.

The next accepted experiment must add a shared operation-conditioned
next-state executor:

  previous digit/list register state
  + operation id
  + operation argument
  + source slots
  -> predicted next digit/list register state

The executor must be learned, must live inside the normal typed-register answer
path, and must have an ablation that removes the transition contribution.
```

Promote/reject gate:

```text
Local only, before DGX:

Promote:
  arithmetic_chain > 0/8 or list_transform > 0/8 on eval4000,
  selected exact beats 0.5,
  typed_register_off remains 0.0,
  transition_executor_off drops arithmetic/list hits back toward 0/8.

Reject:
  digit trace accuracy improves while arithmetic_chain and list_transform stay
  0/8. That would prove we built a better copier, not a calculator.
```

## 2026-05-22 Stage59 v28 Weak Digit Transition Executor

Implemented and tested the first learned operation-conditioned next-state
executor:

```text
TypedDigitNextStateExecutor:
  previous typed digit/carry trajectory
  + operation id
  + operation argument
  + source-number feature summary
  -> rolled digit/carry trajectory for the normal register speaker

ablation:
  --eval-digit-transition-executor-off gives the speaker the original digit
  trajectory instead of the executor-rolled one.
```

Run:

```text
/tmp/stage59_final_typed_register_answerer_v28_digit_transition_executor_train256_eval4000
```

Result:

```text
best selected exact = 0.5 at epoch 28
best digit_transition_executor_off exact = 0.5
typed_register_off exact = 0.0

best family split:
  arithmetic_chain = 0/8
  list_transform = 0/8
  boolean_logic = 8/8
  symbolic_binding = 8/8

last epoch:
  selected exact = 0.46875
  digit_transition_executor_off exact = 0.46875
  register_trace_char_accuracy = 0.9937
  typed_digit_register_trace_digit_accuracy = 0.7257
```

Representative failures:

```text
8017 -> 4017
12032 -> 9032
16064 -> 12064
20090 -> 15090
8008,8004 -> 6028,6024
8016,8028,8020 -> 6016,6028,6020
```

Decision:

```text
Rejected as a causal arithmetic/list generalization fix.
```

Plain-language diagnosis:

```text
The new hand touches the worksheet, but it still writes in a soft latent ink.
The answer speaker can learn the teacher's written trace shape very well, but
turning the executor off does not change the exam score. That means the hand is
not yet doing the arithmetic. It is another trace-shaping filter.
```

Next requirement:

```text
The executor must stop producing only continuous register vectors. It must
produce explicit digit/list-state logits and feed the chosen or soft digit
embedding back into the next worksheet state:

  previous digit columns
  + operation/argument
  -> next digit logits / carry logits / list-position logits
  -> digit-state embedding used by the next recurrent step and by the speaker

Required ablation:
  transition_executor_off must remove arithmetic/list gains.

Reject condition:
  trace text accuracy rises while arithmetic_chain/list_transform remain 0/8
  or executor_off ties full.
```

## 2026-05-22 Stage59 v29/v30 Discrete and Column-Scan Executors

Terminology correction:

```text
"hand movement" is only a plain-language metaphor.

Formal term:
  procedural executor

Concrete Stage59 implementation term:
  carry-propagating column-scan executor
```

Implemented:

```text
v29:
  TypedDigitNextStateExecutor with direct digit/presence logits,
  straight-through discrete digit feedback, and direct executor trace loss.

v30:
  --digit-transition-executor-mode column_scan
  Right-to-left column scan over digit/carry slots using a shared recurrent
  scan state before writing the next typed digit trajectory.
```

Runs:

```text
v29:
  /tmp/stage59_final_typed_register_answerer_v29_discrete_digit_transition_train256_eval64

v30:
  /tmp/stage59_final_typed_register_answerer_v30_column_scan_train256_eval64
```

Best checkpoint evidence:

```text
v29 best epoch = 11
  selected exact = 0.46875
  digit_transition_executor_off exact = 0.46875
  typed_register_off exact = 0.0
  arithmetic_chain = 0/8
  list_transform = 0/8
  boolean_logic = 7/8
  symbolic_binding = 8/8

v30 best epoch = 9
  selected exact = 0.46875
  digit_transition_executor_off exact = 0.46875
  typed_register_off exact = 0.0
  arithmetic_chain = 0/8
  list_transform = 0/8
  boolean_logic = 7/8
  symbolic_binding = 8/8
```

Decision:

```text
Rejected as the current big-jump fix.
```

Plain-language diagnosis:

```text
We added a more plausible procedure for moving across digit columns, and the
model did learn better digit-trace behavior. But the final answer still does
not care whether that executor is present. Turning the executor off gives the
same score and the same arithmetic/list failure.

So the missing organ is not merely "a hand that scans columns." The deeper
missing piece is committed writeback: the procedural executor's result must
become the authoritative next thought/register state that the speaker cannot
ignore.
```

Seven-axis update:

```text
1. Architecture:
   reader, worksheet, speaker, and procedural scan now exist. The weak point is
   committed writeback into the main thought path.

2. Curriculum:
   trace loss teaches visible worksheet lines, but not yet "this predicted
   next line replaces the old internal state."

3. Reward/loss:
   direct executor trace loss improves digit logits, but the final answer loss
   can still be satisfied through non-executor register features.

4. Evaluation:
   boolean/symbolic success remains misleading; arithmetic/list are the actual
   gate and remain 0/8.

5. Exploration:
   not active here; this is deterministic procedural execution, not GRAM
   search.

6. Data contract:
   operation IDs, arguments, digit columns, source slots, and scan order are
   present. The failure is not missing input information.

7. Causality/ablation:
   typed_register_off = 0.0 proves the register body is used. But
   executor_off ties full, proving the procedural executor is not yet the
   causal source of arithmetic/list answers.
```

Next accepted direction:

```text
Do not add another side bridge from executor logits to answer tokens.

Instead add committed procedural writeback:

  executor predicted next digit/list state
  -> replace or strongly gate the next typed register state
  -> next recurrent thought step reads that committed state
  -> speaker reads only the committed register trajectory

Required local gate:
  arithmetic_chain > 0/8 or list_transform > 0/8,
  selected exact > 0.5,
  typed_register_off = 0.0,
  executor/writeback_off removes the arithmetic/list gain.
```

## 2026-05-22 Stage59 v31 Committed Procedural Writeback

CoT distinction:

```text
CoT:
  verbal trace; the model says or imitates a solution explanation.

Procedural state update:
  latent/register transition; the model turns one internal written state into
  the next internal written state.

Stage59 target:
  procedural state update, not more CoT text.
```

Implemented:

```text
TypedDigitCommittedWriteback:
  executor-rolled typed digit/list trajectory
  -> cross-attend into the main working-register trajectory
  -> gated writeback into steps 1..T
  -> normal register-only speaker reads the committed register trajectory

flags:
  --digit-transition-committed-writeback
  --digit-transition-writeback-gate-init-bias
  --eval-digit-transition-writeback-off
```

Run:

```text
/tmp/stage59_final_typed_register_answerer_v31_committed_writeback_train256_eval64
```

Best checkpoint evidence:

```text
best epoch = 12
selected exact = 0.46875
typed_register_off exact = 0.0
digit_transition_executor_off exact = 0.0
digit_transition_writeback_off exact = 0.25

family split:
  arithmetic_chain = 0/8
  list_transform = 0/8
  boolean_logic = 7/8
  symbolic_binding = 8/8
```

Decision:

```text
Partial causal fix, not a promoted generalization fix.
```

Plain-language diagnosis:

```text
This is the first version where the procedural path clearly matters. When the
executor is removed, the score collapses from 0.46875 to 0.0. When writeback is
removed, it drops to 0.25. So the executor is finally inside the main thought
body, not merely a side note.

But the recovered behavior is still boolean/symbolic, not arithmetic/list.
The student is now writing into the real notebook, but still has not learned
the arithmetic rule that should change the numbers.
```

Seven-axis update:

```text
1. Architecture:
   The procedural executor now affects the evaluated register path. This axis
   improved.

2. Curriculum:
   Still too broad: arithmetic/list rule learning is mixed with speaker,
   selector, boolean, symbolic, and register alignment learning.

3. Reward/loss:
   Trace and final losses make the path causal but do not isolate arithmetic
   transition correctness.

4. Evaluation:
   Overall exact is misleading. The real gate is arithmetic_chain/list_transform,
   and both remain 0/8.

5. Exploration:
   Not a GRAM/search issue in this stage.

6. Data contract:
   Inputs are present. The issue is learning the transition rule from them.

7. Causality/ablation:
   Causality improved: executor_off and writeback_off now hurt. But the causal
   path carries the wrong skill.
```

Next accepted direction:

```text
Do not add more CoT or another answer bridge.

Add an arithmetic/list transition pretraining curriculum for the same committed
writeback path:

  source numbers + operation + previous typed digit/list state
  -> next typed digit/list state

Only after local arithmetic/list transition accuracy is non-random should the
same weights be used in the final answerer run.

Promote gate:
  arithmetic_chain > 0/8 or list_transform > 0/8,
  selected exact > 0.5,
  executor_off/writeback_off removes that arithmetic/list gain.
```

## 2026-05-22 Latest-Paper Sweep and New Architecture Direction

Reason for sweep:

```text
Repeated local rejects mean the next step cannot be another small flag tweak.
The decision must be anchored in the newest looped-reasoning literature and in
our actual ablation evidence.
```

Recent papers, date ordered:

```text
2026-05-19
  Memory-Efficient Looped Transformer (MELT)
  https://arxiv.org/abs/2605.07721
  Mechanism: recurrent reasoning depth with in-place gated memory/KV updates.
  Stage59 implication: side memories are not enough; updates must enter the
  authoritative recurrent memory.

2026-05-12
  Solve the Loop: Attractor Models for Language and Reasoning
  https://arxiv.org/abs/2605.12466
  Mechanism: iterative refinement solved as convergence toward fixed points.
  Stage59 implication: a next-state candidate should be refined/settled before
  the speaker reads it.

2026-05-10
  LoopUS: Recasting Pretrained LLMs into Looped Latent Refinement Models
  https://arxiv.org/abs/2605.11011
  Mechanism: selective gates, random deep supervision, confidence head, and
  drift control for recasting pretrained models into looped latent refiners.
  Stage59 implication: Qwen-attached recurrent organs need gate-controlled
  refinement and trajectory supervision, not only final answer CE.

2026-04-13
  A Mechanistic Analysis of Looped Reasoning Language Models
  https://arxiv.org/abs/2604.11791
  Mechanism: looped blocks form stable cyclic fixed-point trajectories.
  Stage59 implication: arithmetic/list execution should become a stable
  state trajectory, not a sequence of unrelated slot predictions.

2026-02-12
  Prioritize the Process, Not Just the Outcome / RLTT
  https://arxiv.org/abs/2602.10520
  Mechanism: distribute credit over latent reasoning trajectories rather than
  only the final state.
  Stage59 implication: arithmetic/list transition correctness needs direct
  process credit at each ledger transition.

2026-01-31
  Reasoning as State Transition
  https://arxiv.org/abs/2602.00770
  Mechanism: reasoning is visible as internal representational transitions.
  Stage59 implication: the target object is the transition itself, not verbal
  CoT text.
```

Synthesis:

```text
The current field is converging on four ideas:

1. latent looped reasoning,
2. gated in-place memory update,
3. trajectory-level process credit,
4. attractor/fixed-point stabilization.

Stage59 v31 already has (1) and a first version of (2). It lacks a strong
version of (3) for arithmetic/list transitions and lacks (4) entirely.
```

New architecture idea:

```text
Executable Ledger Attractor (ELA)
```

Plain-language model:

```text
The model should not merely write a scratch note and then speak.
It should maintain a real ledger.

For each operation:
  read current ledger line
  draft next ledger line
  check/refine the draft until stable
  commit the stable line as the next ledger state
  speak only from the committed ledger
```

Technical contract:

```text
Qwen reader
-> typed source slots
-> typed digit/list ledger state
-> procedural executor proposes next ledger state
-> attractor checker/refiner runs 1..K settlement iterations
-> committed writeback replaces/gates the main working-register trajectory
-> register-only speaker
```

Non-negotiable ablations:

```text
executor_off:
  removes the proposed next-state rule.

attractor_off:
  uses the raw draft without settlement.

writeback_off:
  prevents the settled ledger from becoming the main thought state.

typed_register_off:
  removes the entire ledger body.
```

Local promote gate:

```text
Before DGX:
  arithmetic_chain > 0/8 or list_transform > 0/8,
  selected exact > 0.5,
  executor_off / attractor_off / writeback_off removes the arithmetic/list gain,
  transition-level digit/list accuracy is non-random on heldout rows.
```

Humanistic prediction before local training:

```text
Why it should help:
  v31 proved that the learned worksheet can become causal when committed back
  into the main thought state, but the worksheet still writes raw, unchecked
  drafts. ELA adds a small internal "review desk": draft the next line, settle
  it for a few iterations, then commit it. 문과적으로는 계산 실수를 바로 입으로
  말하지 않고, 장부에 한 번 검산한 뒤 말하게 만드는 구조다.

Why it might still fail:
  If the executor has not learned the arithmetic/list transition rule at all,
  the attractor can only polish a bad draft. It cannot invent the missing
  operation from nothing. In that case transition-level digit/list accuracy
  remains random and arithmetic/list stay 0/8.

Expected first signal:
  arithmetic_chain or list_transform must move above 0/8 on the small local
  gate, and attractor_off should remove that family-specific gain while
  executor_off and writeback_off remain destructive.

Falsifier:
  total exact accuracy improves only through boolean/symbolic rows, or the
  arithmetic/list gain survives attractor_off. That would mean ELA is not the
  reason for the improvement.
```

Implemented local code path:

```text
TypedDigitNextStateExecutor:
  drafts the next digit/list ledger state.

TypedDigitLedgerAttractor:
  refines the drafted ledger state for K settlement steps.

TypedDigitCommittedWriteback:
  commits the refined ledger into the main working-register trajectory.

Clean ablation:
  executor_off also disables the attractor so the checker cannot fake an
  executor-off gain.
```

Rejected next moves:

```text
Do not:
  add longer CoT,
  add another answer-token bridge,
  scale to DGX while arithmetic/list stay 0/8,
  tune generic LR/epoch knobs as the main idea,
  cite old memory-slot papers without checking newer looped-memory work.
```

## 2026-05-22 Stage59 v32 ELA Local Gate

Run:

```text
/tmp/stage59_final_typed_register_answerer_v32_ela_train256_eval64
train_limit=256
eval effective total=32
epochs=12
digit-ledger-attractor=true
digit-ledger-attractor-steps=2
committed-writeback=true
```

Result:

```text
best_epoch = 12
best_exact = 0.3125

by_family:
  arithmetic_chain = 0/8
  list_transform = 0/8
  boolean_logic = 5/8
  symbolic_binding = 5/8

ablations at best:
  typed_register_off = 0.0
  executor_off = 0.0625
  attractor_off = 0.3125
  writeback_off = 0.0

train digit_transition_executor_trace_digit_accuracy = 0.70698
```

Gate decision:

```text
Reject / do not promote.

The local promote gate required arithmetic_chain > 0/8 or list_transform > 0/8
and an attractor_off drop on that family. Neither happened.
```

Humanistic diagnosis:

```text
The model learned to write a more legible internal worksheet, but the mouth
still does not reliably speak the worksheet digits.

Example failures:
  target 8017 -> selected 6017
  target 8008,8004 -> selected 6008,6008

This means the next bottleneck is not simply "add a checker." The missing
contract is ledger-to-mouth rendering: the explicit digit/list ledger must
become the speaker's alphabet, not just another hint inside a free AR decoder.
```

Next highest-probability candidate:

```text
Ledger-Causal Renderer (LCR)

Qwen reader
-> typed source slots
-> procedural digit/list ledger
-> optional ELA checker
-> committed writeback
-> learned ledger-causal renderer that emits answer characters from ledger
   digit/presence slots
-> selector

Prediction:
  If this is correct, arithmetic/list should move above 0/8 before boolean or
  symbolic improve, and disabling the renderer bridge should remove exactly
  that arithmetic/list gain.

Falsifier:
  renderer bridge improves only boolean/symbolic, or arithmetic/list remains
  0/8 while transition digit trace accuracy stays high.
```

## 2026-05-22 Update: BECTO Did Not Create a One-Body Path

Stage59 v35 tested a Base-Equivariant Column Thought Organ (BECTO):

```text
/tmp/stage59_final_typed_register_answerer_v35_becto_train256_eval64
best_exact = 0.40625 at epoch 11
arithmetic_chain = 0/8
list_transform = 0/8
digit_transition_executor_trace_digit_accuracy = 0.8031 at epoch 12
column_procedure_off at final = full final accuracy
```

Decision:

```text
Reject as a promoted architecture.
Keep the diagnosis.
```

Plain-language read:

```text
BECTO is a better calculating hand, but the model still does not have a binding
promise that this hand writes the final answer. The fluent answerer can keep
speaking from its old template habits.
```

The next accepted-probability idea is therefore not a larger side calculator.
It is a stricter one-body answer contract:

```text
Thought-to-Speech Register Pact (TSRP)

Qwen reader
-> typed source slots
-> recurrent GRAM/PTRM typed ledger thought
-> mandatory ledger-token renderer for numeric/list spans
-> evaluated answer
```

Rule:

```text
For numeric/list answers, digits must come from committed ledger columns. The
pretrained/free speaker can format separators and symbolic text, but it cannot
invent numeric digits independently.
```

This is the current big-jump target because it fixes the exact HRM-Text
one-body failure:

```text
old shape:
  reader -> side calculator
  reader/free speaker -> answer

required shape:
  reader -> recurrent thought ledger -> speaker -> answer
```

Promote only if arithmetic/list improve and the improvement disappears under
both `column_procedure_off` and renderer-off ablations.

## 2026-05-22 Update: TSRP Mouth Contract Narrows the Bottleneck

Stage59 v36 implemented the Thought-to-Speech Register Pact:

```text
/tmp/stage59_final_typed_register_answerer_v36_tsrp_train256_eval64
best_exact = 0.1875
final digit_transition_executor_trace_digit_accuracy = 0.8167
arithmetic_chain = 0/8
list_transform = 0/8
ledger_pact_renderer_off = full accuracy
column_procedure_off = full accuracy
```

Decision:

```text
Reject as a promoted architecture.
Keep the narrowed diagnosis.
```

Plain-language read:

```text
The mouth problem is no longer the main excuse. Numeric/list rows are now
spoken directly from the ledger, and the ledger still says the wrong number.

Examples:
  8017 -> 4017
  8008,8004 -> 4028,4024

The hand is editing nearby digits but not applying a base-complete column
procedure across all places.
```

Next fixed direction:

```text
Teacher-forced Column Procedure Curriculum (TCPC)
```

Rule:

```text
Before final-answer training, teach the digit executor exact one-line ledger
moves:

  source ledger -> first trace state
  trace state t -> trace state t+1

Then test through TSRP, not through a free answer mouth.
```

Why this is the highest-probability next move:

```text
Current training asks the model to learn the desk, the hand, and the mouth at
once. v36 proved the mouth can be forced. What remains is the hand procedure.
TCPC gives that hand clean handwriting drills before the final exam.
```

## 2026-05-22 Update: TCPC Warm-Start Still Does Not Generalize Numeric Bases

Stage59 v37 tested TCPC + TSRP:

```text
/tmp/stage59_final_typed_register_answerer_v37_tcpc_tsrp_train256_eval64
TCPC pretrain examples = 1024
TCPC pretrain digit_accuracy = 0.9780
best_exact = 0.46875
arithmetic_chain = 0/8
list_transform = 0/8
boolean_logic = 7/8
symbolic_binding = 8/8
```

Decision:

```text
Reject for arithmetic/list promotion.
Keep TCPC as useful machinery.
```

Plain-language read:

```text
The hand can copy the training notebook, but it still does not understand the
column law well enough to move from 0/1000/2000/3000-style training bands to
4000-style heldout rows.
```

Updated next architecture direction:

```text
Base-Equivariant Neural Arithmetic Primitive Executor (BENAPE)
```

Rule:

```text
Do not ask an unconstrained GRU cell to rediscover place-value arithmetic from
small synthetic traces. Give the thought organ a shared column primitive:

  current digit + source digit + operation + argument + carry
  -> next digit + next carry

The learned model still decides routing, operation use, and integration into
the recurrent ledger. But the place-value physics is no longer a memorized
semantic blob.
```

This is the next highest-probability "big jump" because every prior local gate
now points to the same plain-language failure:

```text
the model edits digits;
it does not perform the repeated vertical calculation move.
```

## 2026-05-22 Update: BENAPE Is the First Numeric/List Big-Jump Signal

Stage59 v38b tested BENAPE + TSRP after fixing the typed-register-off ablation
loophole:

```text
/tmp/stage59_final_typed_register_answerer_v38b_benape_tsrp_ablationfix_train256_eval64
best_epoch = 2
best_exact = 0.65625

by_family:
  arithmetic_chain = 8/8
  list_transform = 8/8
  boolean_logic = 5/8
  symbolic_binding = 0/8

ablations:
  typed_register_off = 0.0
  digit_transition_executor_off = 0.15625
  column_procedure_off = 0.15625
  ledger_pact_renderer_off = 0.15625
```

Decision:

```text
Promote as numeric/list bottleneck proof.
Do not call it final general reasoning.
```

Plain-language read:

```text
The big jump did not come from asking the GRU hand to practice harder. It came
from giving the system a real calculation hand: a shared place-value procedure
that works the same way on every digit column.

When that hand is present, arithmetic/list rows go 16/16.
When that hand or its speech pact is removed, numeric/list rows fall back to
0/16.
```

Research principle:

```text
GRAM/PTRM-style discoveries are valuable only when they create a missing organ,
not when they decorate the same broken causal path.

Future "surprising ideas" must pass the seven-axis humanistic preflight before
training:

  reader, thinker, speaker, curriculum, reward, evaluation, ablation.

If the idea cannot be explained as a missing human-like role in the answer
factory, do not spend GPU time on it.
```

Current limitation:

```text
BENAPE is a calculation organ proof, not a complete HRM-Text-like one-body
thinker. It still relies on task-family typed primitives, and symbolic_binding
is 0/8 in the short local gate.

Next direction:
  turn the hard primitive into a learned/routed thought organ;
  preserve the ablation-clean numeric/list gain;
  repair symbolic/text binding without reintroducing a side-mouth shortcut.
```

## 2026-05-22 Update: Longer BENAPE Gate Reaches 0.75 but Shows the Next Missing Organ

Stage59 v39 extended BENAPE + TSRP to 8 epochs:

```text
/tmp/stage59_final_typed_register_answerer_v39_benape_tsrp_longer_train256_eval32
best_epoch = 7
best_exact = 0.75

by_family at best/final:
  arithmetic_chain = 8/8
  list_transform = 8/8
  symbolic_binding = 5/8
  boolean_logic = 3/8

typed_register_off = 0.0

digit_transition_executor_off / column_procedure_off / ledger_pact_renderer_off:
  arithmetic_chain = 0/8
  list_transform = 0/8
  symbolic_binding = 5/8
  boolean_logic = 3/8
```

Decision:

```text
Promote the numeric/list organ.
Do not spend the next step on more numeric calculation tricks.
```

Plain-language read:

```text
The model now has a real numeric hand. It can do arithmetic/list work in the
4000-band, and that gain disappears when the hand or its speech pact is turned
off.

The remaining failures are not the same failure. Symbolic rows recover slowly
through the free character mouth. Boolean rows collapse toward a TRUE-shaped
habit. Those are "pointer hand" and "logic hand" failures, not arithmetic-hand
failures.
```

Next invention target:

```text
Typed Primitive Thought Organ (TPTO)

Generalize BENAPE from one numeric calculator into a small typed desk:

  numeric lane:   place-value column primitive
  boolean lane:   NOT / AND / OR primitive with typed TRUE/FALSE state
  symbolic lane:  follow first_mapping / second_mapping as pointer transitions
  speaker pact:   every typed lane must have an explicit way to say its final
                  value through the evaluated answer path

This is closer to HRM-Text in plain language: one body reads, writes typed
thoughts on its desk, applies the right primitive, and speaks from that same
desk.
```

Gate for the next local-only experiment:

```text
Promote if:
  total >= 0.90 on the 32-row eval,
  each family >= 7/8,
  typed_register_off collapses the gain,
  lane-specific off switches remove only their family.

Reject if:
  boolean remains TRUE-biased,
  symbolic remains word-imitation rather than pointer-following,
  or any family works only through a side mouth.
```

## 2026-05-22 Update: Oracle TPTO Pact Closes the 32-Row Gate

Stage59 v40 added an oracle Typed Primitive Thought Organ pact:

```text
/tmp/stage59_final_typed_register_answerer_v40_oracle_tpto_pact_train256_eval32
best_epoch = 1
best_exact = 1.0

by_family:
  arithmetic_chain = 8/8
  list_transform = 8/8
  boolean_logic = 8/8
  symbolic_binding = 8/8

typed_register_off = 0.0
typed_primitive_pact_renderer_off = 0.0

digit_transition_executor_off / column_procedure_off:
  arithmetic_chain = 0/8
  list_transform = 0/8
  boolean_logic = 8/8
  symbolic_binding = 8/8
```

Decision:

```text
Promote as an upper-bound target contract.
Do not promote as the final learned architecture.
```

Plain-language read:

```text
The full task closes when the system has all three hands:

  numeric hand:   place-value arithmetic/list procedure
  boolean hand:   TRUE/FALSE logic procedure
  symbolic hand:  pointer-following mapping procedure

v40 proves this is the right decomposition. It also proves v39's remaining
boolean/symbolic failures were not mysterious scale failures; they were missing
typed primitive roles.
```

Important caveat:

```text
The boolean/symbolic v40 path is an oracle primitive pact parsed from prompt
text and solver operations. It is not yet a learned one-body HRM-like thinker.

The next real architecture step is to move those oracle boolean/symbolic hands
into learned typed lanes, then require lane-specific ablations:

  numeric lane off   -> only arithmetic/list fall
  boolean lane off   -> only boolean falls
  symbolic lane off  -> only symbolic falls
  typed body off     -> all gains fall
```

Next implementation target:

```text
Learned TPTO lanes:
  Boolean lane stores P, Q, R and applies NOT / AND / OR.
  Symbolic lane stores current symbol/color and applies mapping pointer moves.
  Numeric lane keeps BENAPE's place-value primitive.
  One speaker pact reads from the typed desk, not from separate family-specific
  side solvers.
```

## 2026-05-22 Update: TPTO Lane-Specific Ablations Are Causal

Stage59 v41 added lane-specific oracle TPTO ablations:

```text
/tmp/stage59_final_typed_register_answerer_v41_oracle_tpto_lane_ablation_train256_eval32
best_epoch = 1
best_exact = 1.0

full:
  arithmetic_chain = 8/8
  list_transform = 8/8
  boolean_logic = 8/8
  symbolic_binding = 8/8

typed_register_off:
  all families = 0/8

digit_transition_executor_off:
  arithmetic_chain = 0/8
  list_transform = 0/8
  boolean_logic = 8/8
  symbolic_binding = 8/8

boolean_primitive_lane_off:
  boolean_logic = 0/8
  all other families = 8/8

symbolic_primitive_lane_off:
  symbolic_binding = 0/8
  all other families = 8/8

typed_primitive_pact_renderer_off:
  all families = 0/8
```

Decision:

```text
Promote the TPTO role decomposition.
Do not promote the oracle implementation as learned reasoning.
```

Plain-language read:

```text
This is the cleanest story so far:

  take away the whole desk -> nobody can answer;
  take away the numeric hand -> only number/list tasks fail;
  take away the boolean hand -> only TRUE/FALSE tasks fail;
  take away the symbolic hand -> only mapping tasks fail.

That means the next learned model should not chase another global loss. It
should learn these three typed hands inside one answer body.
```

Next learned-lane gate:

```text
Replace oracle boolean/symbolic renderers with learned lane states while keeping
the same ablation contract:

  full >= 0.90
  typed_register_off <= 0.10
  numeric lane off drops only arithmetic/list
  boolean lane off drops only boolean
  symbolic lane off drops only symbolic

If the lane-off pattern is lost, the architecture has fallen back into a side
mouth or memorized shortcut.
```

## 2026-05-22 Update: Learned-Lane Source/Trace Contract

Before building the learned boolean/symbolic TPTO lanes, Stage59 now fixes the
reader contract those lanes must consume:

```text
boolean_lane_source_and_trace_tensors(...)
symbolic_lane_source_and_trace_tensors(...)
build_symbolic_lane_vocab(...)
```

Decision:

```text
Use these contracts as the first learned-lane target.
Do not train a boolean/symbolic answer classifier directly from answer labels.
```

Plain-language read:

```text
The student should not memorize that this row says "green" or "FALSE".
It should first put the usable objects on the desk:

  boolean desk:
    P, Q, R
    step targets: NOT Q -> P AND previous -> previous OR R

  symbolic desk:
    current pointer
    mapping table
    step targets: map[current] -> map[current] -> hold
```

Critical invariant:

```text
These targets are derived from prompt text and solver_trace operations, not
answer_aliases. If answer_aliases is replaced with "WRONG", the source and trace
targets remain correct.
```

Next implementation target:

```text
TypedPrimitiveLaneExecutor

Inputs:
  boolean source tensor, symbolic source index, symbolic mapping table,
  operation ids.

Outputs:
  boolean step logits and symbolic step logits that can be trace-supervised.

Promote only if the learned lanes preserve the v41 lane-off pattern.
```

## 2026-05-22 Update: First Learned TPTO Lane Module Exists

Stage59 now has the first learned-lane building block:

```text
TypedPrimitiveLaneExecutor
TypedPrimitiveLaneOutput
compute_typed_primitive_lane_trace_loss(...)
```

What it does:

```text
boolean lane:
  reads P/Q/R source tensor plus operation ids;
  emits step-wise TRUE/FALSE logits.

symbolic lane:
  reads current symbol index, mapping table, and operation ids;
  emits step-wise symbol logits.
```

Tests:

```text
test_typed_primitive_lane_executor_returns_boolean_and_symbolic_trace_logits
test_compute_typed_primitive_lane_trace_loss_prefers_correct_logits
```

Decision:

```text
Promote as the first learned TPTO component.
Do not call it solved until it is wired into train/eval and preserves v41's
lane-off behavior under learned logits.
```

Plain-language read:

```text
The student now has trainable hands for boolean and symbolic work, but those
hands are still practicing on a worksheet. They are not yet the hands that the
final answer mouth uses during the exam.
```

Next gate:

```text
Wire TypedPrimitiveLaneExecutor into Stage59 training:
  add trace loss;
  add renderer path that can speak from learned lane logits;
  add boolean/symbolic learned-lane off switches;
  require full >= 0.90 and the same v41 lane-specific collapse pattern.
```

## 2026-05-22 Update: Learned TPTO Lane Renderer Wired

Implemented the first learned primitive-lane answer path:

```text
typed_primitive_lane_batch_inputs(...)
render_learned_primitive_lane_texts(...)
learned_primitive_lane_renderer_active(...)
--typed-primitive-lane-executor
--typed-primitive-lane-trace-weight
--answerer-learned-primitive-lane-renderer
--eval-learned-primitive-lane-renderer-off
```

The learned lane renderer reads `TypedPrimitiveLaneOutput` logits, not oracle
prompt parsing and not `answer_aliases`.

Tests added:

```text
test_learned_primitive_lane_renderer_uses_logits_not_answer_label
test_learned_primitive_lane_renderer_ablation_masks_one_lane_only
test_learned_primitive_lane_renderer_is_inactive_when_typed_register_body_is_off
test_typed_primitive_lane_batch_inputs_bundle_sources_trace_targets_and_operations
```

First local gate:

```text
run:
  /tmp/stage59_v42_learned_tpto_lane_train256_eval32

result:
  eval = 32/32 = 1.0
  typed_register_off = 0/32 = 0.0
  ledger_pact_renderer_off = 16/32 = 0.5
  learned_primitive_lane_renderer_off = 29/32 = 0.90625 at epoch 8
  symbolic_primitive_lane_off = 32/32 = 1.0 at epoch 8
```

Diagnosis:

```text
This was a performance success but a causal-contract failure.

The free answerer was still CE-trained to speak boolean/symbolic answers. With
enough epochs it learned a bypass mouth for symbolic rows, so turning off the
symbolic lane no longer removed the gain.
```

Plain-language read:

```text
We told the hand to write the answer, but we also kept training the mouth to say
that same answer without looking at the hand. The mouth eventually learned to
cheat.
```

Fix:

```text
mask_primitive_lane_owned_answer_targets(...)
cross_entropy_ignore_or_zero(...)
--primitive-lane-own-answer-supervision
```

When learned primitive-lane ownership is enabled, boolean/symbolic rows are not
used for free answerer CE/select supervision. The boolean/symbolic answer must
come through the learned primitive lane renderer.

Tests added:

```text
test_primitive_lane_owned_targets_mask_boolean_and_symbolic_free_mouth_supervision
test_cross_entropy_ignore_or_zero_returns_zero_when_every_target_is_ignored
```

Owned local gate:

```text
run:
  /tmp/stage59_v43_learned_tpto_lane_owned_train256_eval32

final epoch 8:
  eval = 32/32 = 1.0
  typed_register_off = 0/32 = 0.0
  digit_transition_executor_off = 16/32 = 0.5
    arithmetic_chain = 0/8
    list_transform = 0/8
    boolean_logic = 8/8
    symbolic_binding = 8/8
  ledger_pact_renderer_off = 16/32 = 0.5
    arithmetic_chain = 0/8
    list_transform = 0/8
  learned_primitive_lane_renderer_off = 16/32 = 0.5
    boolean_logic = 0/8
    symbolic_binding = 0/8
  boolean_primitive_lane_off = 24/32 = 0.75
    boolean_logic = 0/8
  symbolic_primitive_lane_off = 24/32 = 0.75
    symbolic_binding = 0/8
  typed_primitive_lane_trace_loss: 1.3454 -> 0.0022
```

Decision:

```text
Promote the learned TPTO lane renderer with primitive-lane-owned supervision as
the current accepted local Stage59 causal gate.

Do not promote the earlier v42 run despite 1.0 accuracy, because the lane-off
pattern degraded under longer training.
```

Boundary:

```text
This is still not a fully universal HRM-Text-like one-body LM. Numeric/list rows
use the BENAPE digit primitive plus ledger pact, while boolean/symbolic rows now
use learned primitive lanes. The next architecture step is to replace the
remaining hard numeric/list primitive with a learned procedure compiler/executor
while preserving the same no-bypass ownership contract.
```

Full-eval validation:

```text
run:
  /tmp/stage59_v44_learned_tpto_lane_owned_train256_eval128

epoch 8:
  eval = 128/128 = 1.0
    arithmetic_chain = 32/32
    boolean_logic = 32/32
    list_transform = 32/32
    symbolic_binding = 32/32
  typed_register_off = 0/128 = 0.0
  digit_transition_executor_off = 64/128 = 0.5
    arithmetic_chain = 0/32
    list_transform = 0/32
  ledger_pact_renderer_off = 64/128 = 0.5
    arithmetic_chain = 0/32
    list_transform = 0/32
  learned_primitive_lane_renderer_off = 64/128 = 0.5
    boolean_logic = 0/32
    symbolic_binding = 0/32
  boolean_primitive_lane_off = 96/128 = 0.75
    boolean_logic = 0/32
  symbolic_primitive_lane_off = 96/128 = 0.75
    symbolic_binding = 0/32
  typed_primitive_lane_trace_loss: 1.3454 -> 0.0022
```

Decision update:

```text
Promote v44 over v43 as the stronger local Stage59 causal gate because it keeps
the same lane-specific ablation pattern on the full 128-row eval file.

This is a real causal-path improvement, but not the final architecture. The
remaining big-jump target is the numeric/list hard primitive: replace BENAPE
with a learned procedure compiler/executor that still owns the normal answer
path and still fails under executor-off/writeback-off ablation.
```
