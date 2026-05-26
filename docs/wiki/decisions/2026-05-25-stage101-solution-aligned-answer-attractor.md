# 2026 Solution-Aligned Answer Attractor

## Decision

The next promoted local architecture target is not "more thinking" and not
"more stable latent state." It is:

```text
more thinking
-> same LM head moves toward the intelligence answer
-> same LM head moves away from the parrot answer
-> held-out language quality does not regress
```

Call this the Solution-Aligned Answer Attractor requirement.

## Plain-Language Idea

The model is now closer to one body, but one body is not enough.

```text
Bad:
  The student thinks longer and becomes calmer, but settles on the wrong answer.

Good:
  The student thinks longer, notices the tempting shortcut, corrects course,
  and the final mouth becomes more likely to say the real answer.
```

This distinction matters because Stage99I already showed residual convergence
without answer improvement.

## Paper Sources

- EqR, 2026: <https://arxiv.org/abs/2605.21488>

  Mechanism:

  ```text
  Generalizable iterative reasoning should converge to task-conditioned
  attractors whose fixed points correspond to valid solutions.
  ```

- Attractor Models, 2026: <https://arxiv.org/abs/2605.12466>

  Mechanism:

  ```text
  A backbone proposes output embeddings; an attractor module refines them by
  solving toward a fixed point. Convergence is chosen adaptively.
  ```

- Looped Reasoning LM Mechanistic Analysis, 2026:
  <https://arxiv.org/abs/2604.11791>

  Mechanism:

  ```text
  Looped language models can converge to fixed points or cyclic trajectories.
  This proves convergence can happen, but not that the convergence is correct.
  ```

- Generalization Dynamics, 2026:
  <https://jiaxin-wen.github.io/blog/generalization-dynamics>

  Mechanism:

  ```text
  Falling loss can hide mode-hops between parrot-like and intelligence-like
  algorithms. Therefore answer-attractor training must monitor shortcut traps.
  ```

- Nested Learning, 2025: <https://arxiv.org/abs/2512.24695>

  Mechanism:

  ```text
  A model should be read as nested learning systems with different context
  flows and update frequencies, not as one flat pile of layers and losses.
  Source learning therefore needs a fast source-adaptation path, a slower
  old-skill consolidation path, and explicit replay between them.
  ```

## Current Evidence

Run:

```text
local_eval/20260525_STAGE100_LOCAL_STAGE99I_GD_DEPTH_SWEEP/summary.json
```

Gate:

```text
local_eval/20260525_STAGE100_LOCAL_STAGE99I_GD_DEPTH_SWEEP/solution_aligned_answer_attractor_gate.json
```

Result:

```text
accepted = false

baseline depth: 2
candidate depth: 8

passed:
  gd_accuracy_improves
  residual_decreases
  elapsed_not_exploded

failed:
  gd_mean_margin_improves
  critical_tasks_pass
  heldout_loss_not_regressed
```

Plain-language read:

```text
Depth 8 is not useless: it passes one extra GD-lite item and residual is lower.
But it is not a solution attractor. It is more settled, not more reliably
correct.
```

## Stage101A Smoke Update

Stage101A added a direct answer-facing contrastive smoke:

```text
script:
  scripts/570_train_solution_aligned_answer_attractor.py

summary:
  local_eval/20260525_STAGE101_LOCAL_SOLUTION_ATTRACTOR_SMOKE12/stage101_summary.json
```

Result:

```text
Stage99I depth4:
  GD-lite accuracy = 0.3333
  mean_margin = -0.0049

Stage101A depth4:
  GD-lite accuracy = 0.6667
  mean_margin = 0.1519
```

Interpretation:

```text
The answer-facing route is trainable: the same LM head can be pulled toward
intelligence answers. But the checkpoint is still rejected because
successive-answer and truthy-answer remain failed, and tiny language heldout
loss slightly regresses.
```

Next:

```text
Stage101B must add more successive/truthy counterexamples and a
language-preserving regularizer or gate.
```

## Stage101B CONT14 Update

Run:

```text
summary:
  local_eval/20260525_STAGE101B_LOCAL_SOLUTION_ATTRACTOR_KL_CONT14/stage101b_cont14_summary.json
```

Result:

```text
Original GD-lite depth4:
  accuracy = 1.0000
  mean_margin = 0.3450
  accepted = true

Stage101B heldout depth4:
  accuracy = 0.9000
  mean_margin = 0.2760
  accepted = false
  remaining failed axis = truthy_answer_icl

Tiny language heldout:
  CONT14 best loss = 4.532797
  Stage99I best loss = 4.530996
  delta = +0.001801
```

Interpretation:

```text
The core-mind-to-mouth route is now real on the original gate: the same LM
head prefers the intelligence answer over shortcut answers after recurrent
thinking.

But the attractor is not complete. The model still mistakes plausible-looking
truth claims on heldout cases, and language quality is preserved only nearly,
not strictly better than the Stage99I baseline.
```

Do not regress to bridge/readback/selector ideas from this result. The next
move stays on the same one-body answer path:

```text
reader
-> recurrent thought state
-> same LM head
-> truth/claim contrast repair
-> language-preserving gate
```

## Stage101C/101D Update

Summary:

```text
local_eval/20260525_STAGE101D_LOCAL_OVERTHINK_LOCK_KL_SMOKE20/stage101cd_summary.json
```

Result:

```text
Stage101C:
  original GD-lite depth4 accepted
  Stage101B heldout depth4 accepted
  broader Stage101C truth-claim heldout = 12/14
  tiny language loss delta vs Stage99I = +0.000820

Stage101D:
  original GD-lite depth8 accepted
  Stage101B heldout depth8 accepted
  tiny language loss delta vs Stage99I = +0.001433
```

Interpretation:

```text
The project now has two separate positive signals:

1. answer attractor:
   the normal LM head can be pulled toward intelligence answers.

2. overthinking lock:
   depth8 can be trained not to undo the depth4 answer on the old shortcut
   gate.
```

This means the next bottleneck is not "thought cannot speak." It is:

```text
truth/claim world-model coverage:
  the model still lacks a broad enough internal truth sense for unfamiliar
  commonsense and physics claims.
```

Do not solve this with a side verifier. The canonical next path remains:

```text
same reader
-> same recurrent thought
-> same LM head
-> broader truth-claim contrast curriculum
-> depth4/depth8 consistency
-> language preservation
```

## Stage101E/101F Data-Contract Update

Summary:

```text
local_eval/20260525_STAGE101F_LOCAL_SOURCE_GROUNDED_TRUTH_SMOKE48/stage101ef_summary.json
```

Stage101E showed that unsupported world-truth rows are not a clean architecture
gate. A small checkpoint cannot reliably answer facts about sound, mass, and
physics if the prompt never supplies the relevant fact.

Stage101F therefore added source-grounded rows:

```text
context gives a fact
-> question asks whether a claim follows from that context
-> same LM head must answer True/False
```

Result:

```text
source-grounded heldout depth8:
  before Stage101F = 0.2500
  after Stage101F  = 0.5000

preserved:
  original GD-lite depth8 = 1.0000
  Stage101B heldout depth8 = 1.0000
  tiny language delta vs Stage99I = +0.000804
```

Interpretation:

```text
The old attractor/overthinking path survives. The source-to-truth skill starts
to move, but generalization is still weak. The remaining bottleneck is now a
data-contract/curriculum problem: teach the model to read supplied facts and
turn them into truth judgments across paraphrases, not to memorize isolated
world facts through tiny contrastive rows.
```

Next required local gate:

```text
Build a larger source-grounded truth curriculum with:
  paraphrased facts
  claim negations
  fact before/after question position variants
  old GD-lite and Stage101B anchors
  depth4->8 consistency

Promote only if:
  source-grounded truth heldout passes
  original GD-lite depth8 remains accepted
  Stage101B heldout depth8 remains accepted
  tiny language loss does not materially regress
```

## Required Gate

Executable:

```bash
PYTHONPATH=src python scripts/569_eval_solution_aligned_answer_attractor_gate.py \
  --depth-sweep-summary local_eval/.../summary.json \
  --out local_eval/.../solution_aligned_answer_attractor_gate.json \
  --baseline-depth 2 \
  --min-candidate-depth 4
```

Accept only if:

```text
1. Deeper recurrence improves GD-lite mean margin.
2. Critical shortcut axes pass:
   - flipped_answer_icl
   - successive_answer_icl
   - truthy_answer_icl
3. Held-out loss does not regress.
4. Residual decreases.
5. Runtime does not explode.
```

Reject if:

```text
Residual decreases but GD-lite margin does not improve.
More tasks pass but the average intelligence-vs-parrot margin worsens.
Held-out loss worsens after depth 2.
The failed axes are exactly the shortcut traps the architecture claims to fix.
```

## Architecture Implication

The next local architecture must add answer-facing correction pressure inside
the one-body route.

Allowed direction:

```text
reader
-> recurrent thought state
-> same LM head estimates intelligence-vs-parrot margin
-> recurrent update is trained/conditioned to increase that margin
-> final decoder speaks from the corrected state
```

Rejected direction:

```text
reader
-> recurrent state becomes stable
-> final decoder still prefers shortcut answers
```

## One-Sentence Lock

```text
Do not optimize for a quiet mind; optimize for a mind that makes the mouth more
likely to say the right answer.
```

## Stage101G Overthinking-Noise Update

New required diagnostic:

```text
scripts/576_eval_overthinking_noise_probe.py
```

Why it exists:

```text
Depth failure has two different meanings.

1. Stable wrong:
   the model is wrong at every depth. This is usually reading, data contract,
   missing concept, or curriculum.

2. Overthinking noise:
   the model is correct at shallow depth, then deeper recurrence erodes the
   margin or flips the answer.

These must not be merged into one "reasoning failed" bucket.
```

Stage101G result:

```text
summary:
  local_eval/20260525_STAGE101G_LOCAL_SOURCE_PARAPHRASE_LOCK_KL_SMOKE120/stage101g_summary.json

source paraphrase:
  depth8 accuracy = 0.2500
  flip_to_wrong_count = 0
  wrong_at_all_depths_count = 6/8
  read = mostly stable wrong, not main overthinking noise

source truth:
  depth8 accuracy = 0.5000
  flip_to_wrong_count = 0
  wrong_at_all_depths_count = 2/4
  read = source-to-truth skill still incomplete

Stage101B anchor:
  depth2 mean margin = 0.4904
  depth16 mean margin = 0.4275
  accuracy stays 1.0000
  read = real overthinking-noise margin erosion
```

Policy:

```text
Before any longer Stage101-style run, run the overthinking-noise probe on:
  source/paraphrase heldout
  source-truth heldout
  Stage101B anchor

Promote only if:
  no shallow-correct answer is lost at deeper depth
  old anchors remain accepted
  source heldouts improve by quality, not just by stable wrong confidence
  language heldout does not materially regress
```

Stage101H addendum:

```text
summary:
  local_eval/20260525_STAGE101H_LOCAL_OVERTHINK_MARGIN_LOCK_SMOKE80/stage101h_summary.json

effect:
  Stage101B depth16 heldout mean_margin improved 0.4275 -> 0.5419.
  Original GD-lite depth8 mean_margin improved 0.4767 -> 0.5764.

limit:
  stability still rejected because depth16 remains below depth2 by -0.0532.
  Source paraphrase depth8 mean_margin worsened -0.2115 -> -0.2846.

rule:
  Do not run anchor-only overthinking locks as the main solution. They are useful
  to prove that margin erosion is trainable, but source-reading rows need their
  own curriculum in the same one-body path.
```

## Stage101I/J/K Policy Update

Evidence:

```text
Stage101I:
  more row-level source training did not fix source/paraphrase.

Stage101J:
  adding same-mouth template consistency improved source paraphrase heldout
  depth8 from 0.2500 to 0.5000.

Stage101K:
  adding polarity-balanced source rows improved source paraphrase heldout depth8
  to 0.6250.
```

Failure:

```text
Stage101J/K damaged the old Stage101B depth16 anchor:
  Stage101H depth16 = 1.0000
  Stage101J depth16 = 0.9000
  Stage101K depth16 = 0.8000
```

Plain-language rule:

```text
The student is learning the source-reading class, but the new class is crowding
out the old shortcut-resistance class. This is not solved by adding a verifier.
It is a curriculum balance problem inside the same one-body mouth.
```

Implementation policy:

```text
Keep template consistency.
Keep polarity-balanced source rows.
Do not promote unless old anchors stay accepted at depth16.

Next run must either:
  reduce template/source weight, or
  warm up source-True recall separately, then
  replay Stage101B/truthy anchors strongly during the mixed phase.
```

## Nested Learning Interpretation For H/K

Stage101H and Stage101K should not be read as unrelated rejects. They show the
same failure from opposite sides.

```text
Stage101H:
  preserved the old solution attractor
  failed to absorb source-grounded reading

Stage101K:
  absorbed source-grounded reading better
  damaged the old depth16 shortcut/truthy attractor
```

Plain-language read:

```text
H is a student who still remembers the old workbook but ignores the new source
sheet.

K is a student who starts reading the new source sheet, but the new lesson is
so loud that old exam discipline gets overwritten.
```

Nested Learning makes this failure legible:

```text
source reading = fast adaptation to the current evidence context
old attractor  = slower consolidated skill that must survive new evidence
language       = still slower mouth/fluency prior that must not be damaged
```

Therefore the next accepted-probability path is not "more source rows" and not
"more anchor rows" alone. The next path must be multi-timescale:

```text
1. Fast source phase:
   teach source-True/source-False without template or polarity shortcut.

2. Slow replay phase:
   replay Stage101B/GD-lite/truthy anchors hard enough that the old attractor
   remains intact at depth16.

3. Language preservation phase:
   keep the same LM head fluent with teacher KL and free-generation gates.

4. Promotion gate:
   promote only if source heldout rises and old Stage101B depth16 does not
   regress.
```

Stage101L is the first small local test of this interpretation:

```text
checkpoint:
  local_eval/20260525_STAGE101H_LOCAL_OVERTHINK_MARGIN_LOCK_SMOKE80/last_model.pt

data:
  data/eval/stage101l_balanced_replay_train_probe.jsonl

intent:
  lower source/template pressure
  increase old-anchor replay pressure
  preserve language with the same LM head
```

## Stage101L/M Update

Stage101L tested the first multi-timescale replay idea from Stage101H:

```text
H checkpoint
-> stronger old-anchor replay
-> weaker template consistency
-> language preserve KL
```

Result:

```text
Stage101L old Stage101B anchor:
  depth16 accuracy = 1.0000
  depth16 mean_margin = 0.7360
  accepted = true

Stage101L source paraphrase:
  depth16 accuracy = 0.2500
  depth16 mean_margin = -0.3025
  accepted = false

Stage101L source balanced:
  depth16 accuracy = 0.2500
  depth16 mean_margin = -0.2929
  accepted = false
```

Plain-language read:

```text
L remembered the old workbook perfectly, but the source lesson faded again.
Starting from H plus heavy replay is too anchor-heavy.
```

Stage101M then treated Stage101K as the fast source-learning phase and applied
a slower replay phase from there:

```text
K checkpoint
-> old-anchor slow replay
-> lower LR
-> lower template consistency
-> K as language teacher
```

Result:

```text
Stage101M old Stage101B anchor:
  depth16 accuracy = 1.0000
  depth16 mean_margin = 0.5531
  accepted = true

Stage101M source paraphrase:
  depth16 accuracy = 0.2500
  depth16 mean_margin = -0.1074
  accepted = false

Stage101M source balanced:
  depth16 accuracy = 0.2500
  depth16 mean_margin = -0.1029
  accepted = false
```

Plain-language read:

```text
M is better than L but still not promoted. The old exam discipline is restored,
and the source margins are much closer to zero than L, but the model still
does not cross the source-reading decision boundary on heldout rows.
```

Current causal diagnosis:

```text
K proves source reading can be taught, but it is not consolidated.
L proves old-anchor replay can protect the old attractor, but it suppresses
source reading.
M proves a K->slow-replay phase can recover anchors without fully erasing the
source basin, but source needs a stronger fast-adaptation path than row-level
answer-margin contrast alone.
```

Next accepted-probability experiment:

```text
Stage101N:
  start from Stage101M or K
  add source-only micro-bursts that update only source rows/templates
  immediately follow each burst with short anchor replay
  reject unless both gates pass:
    source paraphrase/balanced depth16 accuracy improves above 0.25
    Stage101B anchor depth16 remains accepted
```

Stage101N result:

```text
checkpoint:
  local_eval/20260525_STAGE101N_LOCAL_NESTED_SOURCE_MICROBURST_ANCHOR_LOCK_SMOKE144/last_model.pt

train smoke:
  accepted = false
  final_depth_accuracy_gain = -0.0139
  final_depth_margin_gain = +0.0641

old Stage101B anchor:
  depth16 accuracy = 1.0000
  depth16 mean_margin = 0.6448
  accepted = true

source paraphrase:
  depth16 accuracy = 0.2500
  depth16 mean_margin = -0.1576
  accepted = false

source balanced:
  depth16 accuracy = 0.2500
  depth16 mean_margin = -0.1440
  accepted = false
```

Plain-language read:

```text
N preserved the old workbook but did not teach new source reading. It is worse
than M on source margin, so source microburst followed by anchor lock should
not be promoted as the current recipe.

The live bottleneck is not overthinking noise. It is source-binding: the model
still prefers its old answer prior over the evidence in the prompt for many
heldout source rows.
```

Updated next move:

```text
Do not add side verifiers or revive old bridge-style modules.
The next accepted-probability move must make source evidence part of the same
one-body answer path more directly:
  source evidence token/span
  -> recurrent thought state
  -> same LM head answer

Reject any experiment that improves old anchors while source heldout remains
at 0.25.
```

## Stage101O/P Source-Obedience Update

Stage101O tested direct counterfactual source binding:

```text
same claim
-> source says support or contradict
-> same LM head must flip True/False with the source
```

Result:

```text
Stage101O counterfactual source-binding heldout:
  depth16 accuracy = 0.7500
  depth16 mean_margin = 0.1602
  accepted = false

old Stage101B anchor:
  depth16 accuracy = 1.0000
  depth16 mean_margin = 0.6371
  accepted = true

older source-balance gate:
  depth16 accuracy = 0.2500
  depth16 mean_margin = -0.1991
  accepted = false
```

Plain-language read:

```text
O taught the student to look at a source-shaped sentence in its own artificial
exam, but it did not create real evidence judgment. The old workbook survived,
yet the broader source-balance exam still failed.
```

This reveals a sharper risk:

```text
source binding alone can become source obedience.
```

That is not "reasoning from evidence." It is closer to:

```text
If a sentence is written in the authority voice, follow it.
```

Stage101P therefore adds a belief-update/source-reliability gate:

```text
reliable evidence can revise a belief
untrusted evidence must be ignored
conflicting evidence must be resolved by reliability
insufficient evidence must produce Unknown
```

Current Stage101O checkpoint on Stage101P:

```text
eval:
  local_eval/20260525_STAGE101P_O_CHECKPOINT_BELIEF_UPDATE_SOURCE_RELIABILITY/eval_overthinking.json

depth16:
  accuracy = 0.4286
  mean_margin = -0.1722
  accepted = false

breakdown at depth16:
  claim_first_belief_revision = 6/8 correct
  untrusted_source_override   = 0/2 correct
  trusted_source_conflict     = 0/2 correct
  insufficient_source_unknown = 0/2 correct
```

Plain-language read:

```text
The model is not yet an evidence judge. It can sometimes revise a belief when
the prompt directly says the source supports or contradicts the claim, but it
does not know when not to trust a source, when evidence is insufficient, or
when to say Unknown.
```

Policy:

```text
Do not promote Stage101O.
Do not scale source-binding-only training.
The next accepted-probability gate is Stage101P:
  source evidence
  -> reliability/sufficiency judgment
  -> belief update
  -> same LM head answer

Promote only if:
  Stage101P heldout improves,
  untrusted/conflict/Unknown cases move above chance,
  Stage101O heldout does not regress,
  old Stage101B anchor remains accepted.
```

## Stage101Q Numeric Belief Update

User-level correction:

```text
Bare True/False/Unknown is too word-like.
The model needs a graded belief state:
  support direction
  source reliability
  evidence sufficiency
```

Stage101Q implements this as a same-mouth numeric ledger, not a side head:

```text
prompt/source
-> recurrent thought
-> same LM head outputs:
   support=...
   reliability=...
   sufficiency=...
   final=...
```

Artifacts:

```text
builder:
  scripts/583_build_stage101q_numeric_belief_probe.py

tests:
  tests/test_stage101q_numeric_belief_probe.py

data:
  data/eval/stage101q_numeric_belief_update_train_probe.jsonl
  data/eval/stage101q_numeric_belief_update_heldout_probe.jsonl

build report:
  local_eval/20260525_STAGE101Q_NUMERIC_BELIEF_PROBE_BUILD/report.json
```

Stage101O baseline on Q:

```text
eval:
  local_eval/20260525_STAGE101Q_O_BASELINE_NUMERIC_BELIEF/eval_overthinking.json

depth16:
  accuracy = 0.5714
  mean_margin = -0.0195
  accepted = false

breakdown:
  direct_reliable_numeric_belief = 4/8 correct
  untrusted_override_numeric_belief = 2/2 correct
  trusted_conflict_numeric_belief = 2/2 correct
  insufficient_numeric_belief = 0/2 correct
```

Plain-language read:

```text
The numeric form is a better diagnostic than bare True/False/Unknown. It shows
that the model can choose some negative-belief ledgers and can resist explicit
untrusted True in these rows, but it still cannot produce positive support for
some True evidence and still refuses the low-sufficiency Unknown ledger.
```

Stage101Q 96-step local treatment:

```text
checkpoint:
  local_eval/20260525_STAGE101Q_LOCAL_NUMERIC_BELIEF_SMOKE96/last_model.pt

train report:
  local_eval/20260525_STAGE101Q_LOCAL_NUMERIC_BELIEF_SMOKE96/report.json

Q heldout depth16:
  accuracy = 0.5714
  mean_margin = -0.0157
  accepted = false

old Stage101B anchor depth16:
  accuracy = 1.0000
  mean_margin = 0.6699
  accepted = true
```

Interpretation:

```text
Q96 preserved the old answer attractor but did not solve numeric belief update.
The failure is not overthinking noise; the wrong belief ledger is stable across
depth. The model keeps preferring the negative/False-shaped ledger on positive
support rows and insufficient-evidence rows.
```

Policy update:

```text
Numeric belief is the right semantic direction, but the full ledger string is
too entangled as the first training target.

Next accepted-probability move:
  factorize the numeric belief lesson:
    support score only
    reliability score only
    sufficiency score only
    then final answer from the same LM head

Reject:
  scaling long full-ledger Q rows without first proving that each scalar axis
  can move on heldout rows.
```

## Stage101R Factorized Numeric Belief

User-level correction:

```text
True/False/Unknown is too coarse, but one long numeric ledger is too tangled.
The model should first learn one scalar belief axis at a time:
  support
  reliability
  sufficiency
```

Stage101R implements this through the same LM mouth, not through a side scalar
head:

```text
prompt/source
-> recurrent thought
-> same LM head outputs exactly one numeric scalar choice
```

Artifacts:

```text
builder:
  scripts/584_build_stage101r_factorized_numeric_belief_probe.py

tests:
  tests/test_stage101r_factorized_numeric_belief_probe.py

data:
  data/eval/stage101r_factorized_numeric_belief_train_probe.jsonl
  data/eval/stage101r_factorized_numeric_belief_heldout_probe.jsonl

build report:
  local_eval/20260525_STAGE101R_FACTORIZED_NUMERIC_BELIEF_PROBE_BUILD/report.json
```

Stage101O baseline on R:

```text
eval:
  local_eval/20260525_STAGE101R_O_BASELINE_FACTORIZED_NUMERIC_BELIEF/eval_overthinking.json

depth16:
  accuracy = 0.1667
  mean_margin = -0.0956
  accepted = false

axis breakdown:
  reliability = 0/16 correct
  sufficiency = 0/16 correct
  support = 8/16 correct
```

Stage101R 160-step local treatment:

```text
checkpoint:
  local_eval/20260525_STAGE101R_LOCAL_FACTORIZED_NUMERIC_BELIEF_SMOKE160/last_model.pt

train report:
  local_eval/20260525_STAGE101R_LOCAL_FACTORIZED_NUMERIC_BELIEF_SMOKE160/report.json

heldout R depth16:
  accuracy = 0.7083
  mean_margin = 0.0248
  accepted = false

old Stage101B anchor depth16:
  accuracy = 1.0000
  mean_margin = 0.7064
  accepted = true
```

Heldout axis/case read at depth16:

```text
by axis:
  reliability = 14/16 correct, mean_margin = 0.0655
  sufficiency = 12/16 correct, mean_margin = 0.0175
  support = 8/16 correct, mean_margin = -0.0086

by case:
  direct_reliable = 20/24 correct
  trusted_conflict = 6/6 correct
  untrusted_override = 6/6 correct
  insufficient = 2/6 correct
  untrusted_only = 0/6 correct
```

Plain-language read:

```text
Factorizing the belief state worked much better than the full ledger. The model
can now often say "this source is reliable" and "this evidence is sufficient"
without destroying the older answer-attractor ability.

The remaining failure is restraint, not raw recognition. When evidence is
missing or untrusted, the model should keep support near zero and reliability or
sufficiency low. Instead, it often collapses to the stronger old answer prior:
  support +0.00 -> predicted -0.80
  reliability 0.10 -> predicted 0.90
  sufficiency 0.10 -> predicted 0.90
```

Policy update:

```text
Numeric belief should stay factorized. Do not return to long ledger-only Q.

Next accepted-probability move:
  scalar prior calibration before another broad belief run:
    balance support +0.80 / -0.80 / +0.00
    oversample low reliability 0.10
    oversample low sufficiency 0.10
    include source-free number-prior rows before source semantics

Promote only if:
  R heldout depth16 passes,
  untrusted_only and insufficient cases move above 4/6,
  old Stage101B anchor remains accepted.
```

## Stage101S Scalar Prior Calibration

Reason for S:

```text
Stage101R did not mainly fail because it could not read every source.
It failed because the same mouth had a strong numeric habit:
  -0.80 and 0.90 are easy to say,
  +0.00, +0.80, 0.10, and 0.50 are hard to say.
```

Stage101S therefore tried a direct scalar-prior lesson:

```text
source-free scalar cards first
then source-semantics scalar cards
then old Stage101R factorized rows for preservation
```

Artifacts:

```text
builder:
  scripts/585_build_stage101s_scalar_prior_calibration_probe.py

tests:
  tests/test_stage101s_scalar_prior_calibration_probe.py

data:
  data/eval/stage101s_scalar_prior_calibration_train_probe.jsonl
  data/eval/stage101s_scalar_prior_calibration_heldout_probe.jsonl

build report:
  local_eval/20260525_STAGE101S_SCALAR_PRIOR_CALIBRATION_PROBE_BUILD/report.json
```

Stage101R checkpoint baseline on S:

```text
eval:
  local_eval/20260525_STAGE101S_R_BASELINE_SCALAR_PRIOR_CALIBRATION/eval_overthinking.json

depth16:
  accuracy = 0.3621
  mean_margin = -0.1000
  accepted = false

by target:
  +0.00 = 0/7 correct
  +0.80 = 0/3 correct
  -0.80 = 5/5 correct
  0.10 = 0/25 correct
  0.50 = 0/2 correct
  0.90 = 16/16 correct
```

Stage101S 240-step local treatment:

```text
checkpoint:
  local_eval/20260525_STAGE101S_LOCAL_SCALAR_PRIOR_CALIBRATION_SMOKE240/last_model.pt

S heldout depth16:
  accuracy = 0.3621
  mean_margin = -0.0662
  accepted = false

R heldout depth16:
  accuracy = 0.7083
  mean_margin = 0.1194
  accepted = false

old Stage101B anchor depth16:
  accuracy = 1.0000
  mean_margin = 0.7648
  accepted = true
```

S heldout after treatment:

```text
by target:
  +0.00 = 0/7 correct, mean_margin = -0.4472
  +0.80 = 0/3 correct, mean_margin = -0.5637
  -0.80 = 5/5 correct, mean_margin = 0.7806
  0.10 = 0/25 correct, mean_margin = -0.2899
  0.50 = 0/2 correct, mean_margin = -0.2805
  0.90 = 16/16 correct, mean_margin = 0.3054
```

Plain-language read:

```text
S is not promoted.

The treatment improved average margin but did not flip any low/neutral targets.
Worse, it made the already-easy 0.90 habit stronger. The model is not yet
learning "low trust" as a concept; it is still speaking from a numeric answer
prior.

This is not a reason to scale S. More of the same rows would likely strengthen
the dominant labels again.
```

Policy update:

```text
Do not run a longer S as the next move.

Next accepted-probability move:
  Stage101T bucket-to-number disentanglement:
    first learn semantic buckets:
      support = contradicts / neutral / supports
      reliability = low / unknown / high
      sufficiency = insufficient / partial / sufficient
    then learn numeric readback:
      neutral -> +0.00
      supports -> +0.80
      low -> 0.10
      high -> 0.90

Promote only if:
  bucket rows pass,
  numeric readback rows pass,
  S/R heldout low and neutral targets finally flip,
  old Stage101B anchor remains accepted.
```

## Stage101T Bucket-To-Number Disentanglement

Reason for T:

```text
Stage101S showed that directly drilling numeric strings mostly strengthened
the easy 0.90 prior. Stage101T therefore tried to separate two lessons:
  semantic bucket:
    low / unknown / high
    contradicts / neutral / supports
    insufficient / partial / sufficient
  numeric readback:
    bucket -> scalar string
```

Artifacts:

```text
builder:
  scripts/586_build_stage101t_bucket_to_number_probe.py

tests:
  tests/test_stage101t_bucket_to_number_probe.py

data:
  data/eval/stage101t_bucket_to_number_train_probe.jsonl
  data/eval/stage101t_bucket_to_number_heldout_probe.jsonl

build report:
  local_eval/20260525_STAGE101T_BUCKET_TO_NUMBER_PROBE_BUILD/report.json

local treatment:
  local_eval/20260525_STAGE101T_LOCAL_BUCKET_TO_NUMBER_SMOKE320/report.json
```

Stage101T 320-step local treatment:

```text
accepted = false
final_depth_accuracy_gain = 0.0307
final_depth_margin_gain = 0.0936
```

Plain-language read:

```text
T is not promoted.

It made the dictionary problem clearer, but it still did not make the evidence
judgment causal enough. It asks the model to name or translate buckets, while
the real intelligence problem is earlier:

  Which kind of source is this?
  Should this source be trusted?
  Is the evidence actually about the claim?
  Does it support, contradict, or fail to decide the claim?
  Is the evidence sufficient for a conclusion?

Without that chain, "low", "neutral", and "insufficient" are still answer
labels, not inferred states.
```

Policy update:

```text
Do not run longer T as the next move.

The next move must test data causality before another model run. Evidence
reliability must be produced through an explicit causal chain, not by direct
bucket readback.
```

## Stage101U Causal Evidence Chain

User correction:

```text
Evidence should be causally reasoned through before the model evaluates
reliability, support, and sufficiency.
```

Stage101U implements that correction:

```text
source role
  -> source reliability

evidence relevance and polarity
  -> claim support
  -> evidence sufficiency

parent chain
  -> numeric support / reliability / sufficiency
```

Artifacts:

```text
builder:
  scripts/587_build_stage101u_causal_evidence_chain_probe.py

tests:
  tests/test_stage101u_causal_evidence_chain_probe.py

data:
  data/eval/stage101u_causal_evidence_chain_train_probe.jsonl
  data/eval/stage101u_causal_evidence_chain_heldout_probe.jsonl

build report:
  local_eval/20260525_STAGE101U_CAUSAL_EVIDENCE_CHAIN_PROBE_BUILD/report.json
```

Build result:

```text
train_rows = 56
eval_rows = 32
chain steps:
  source_role
  source_reliability
  evidence_relevance
  claim_support
  evidence_sufficiency
  numeric_belief_support
  numeric_belief_reliability
  numeric_belief_sufficiency
source_quality_counterfactual_pairs = 2
```

Humanistic read:

```text
This is the first data contract in this thread that matches the story we
actually want.

The model is no longer merely asked to say a belief number. It must learn the
ordinary judgment chain a person would use:

  "Who said this?"
  "Can I trust that source?"
  "Is the evidence about the claim?"
  "Does it support or contradict the claim?"
  "Is it enough to decide?"
  "Only then, how confident should I be?"

This is still not an accepted model result. It is the corrected next experiment
contract.
```

Stage101U 160-step local treatment:

```text
checkpoint:
  local_eval/20260525_STAGE101U_LOCAL_CAUSAL_EVIDENCE_CHAIN_SMOKE160/last_model.pt

U train depth16:
  accuracy 0.4286 -> 0.7143
  mean_margin -0.1987 -> 0.7624
  accepted = false

U heldout depth16:
  baseline from R: accuracy = 0.3750, mean_margin = -0.2239
  after U:         accuracy = 0.6250, mean_margin =  0.4816
  accepted = false

R heldout after U:
  accuracy = 0.6250
  mean_margin = 0.4053
  accepted = false

old Stage101B anchor after U:
  accuracy = 1.0000
  mean_margin = 0.6888
  accepted = true
```

Read:

```text
U is useful but not enough.

It teaches more of the evidence chain, but the model still fails exactly where
the user pointed: when evidence is untrusted or irrelevant, the model needs a
policy that says "do not answer yet; ask for the missing evidence."
```

## Stage101V Evidence-Seeking Curiosity

User correction:

```text
Can we put curiosity into the AI? It should ask for the extra material needed
to evaluate the evidence.
```

Stage101V defines project-local curiosity:

```text
curiosity = metacognitive evidence acquisition

If evidence is trusted and sufficient:
  answer_now
  no_more_evidence

If evidence is untrusted, irrelevant, partial, or conflicted:
  ask_more
  ask_reliable_source / ask_relevant_evidence / ask_exact_detail /
  ask_conflict_resolution
```

Artifacts:

```text
builder:
  scripts/588_build_stage101v_evidence_seeking_curiosity_probe.py

tests:
  tests/test_stage101v_evidence_seeking_curiosity_probe.py

data:
  data/eval/stage101v_evidence_seeking_curiosity_train_probe.jsonl
  data/eval/stage101v_evidence_seeking_curiosity_heldout_probe.jsonl

build report:
  local_eval/20260525_STAGE101V_EVIDENCE_SEEKING_CURIOSITY_PROBE_BUILD/report.json
```

Build result:

```text
train_rows = 15
eval_rows = 9
answer_now_rows = 2
ask_more_rows = 6
request_types:
  no_more_evidence
  ask_reliable_source
  ask_relevant_evidence
  ask_exact_detail
  ask_conflict_resolution
```

Baseline from U checkpoint on V heldout:

```text
eval:
  local_eval/20260525_STAGE101V_U_BASELINE_EVIDENCE_SEEKING_CURIOSITY/eval_overthinking.json

depth16:
  accuracy = 0.3333
  mean_margin = -0.1925
  accepted = false
```

Stage101V 120-step local treatment:

```text
checkpoint:
  local_eval/20260525_STAGE101V_LOCAL_EVIDENCE_SEEKING_CURIOSITY_SMOKE120/last_model.pt

train depth16:
  accuracy 0.2000 -> 0.6667
  mean_margin -0.3476 -> 1.3127
  accepted = false

V heldout depth16:
  accuracy = 0.4444
  mean_margin = -0.3060
  accepted = false

old Stage101B anchor:
  accuracy = 1.0000
  mean_margin = 0.7307
  accepted = true
```

Plain-language read:

```text
V proves the mechanism is learnable, but the first version overcorrects.

Before V:
  The model answered too eagerly.

After V:
  The model learned to ask_more on some untrusted/irrelevant rows, but it also
  asks_more on a trusted-and-sufficient row. That is not intelligence; that is
  over-curiosity.

The next move should not be "more V." It should add a curiosity brake:
  when evidence is sufficient and trusted, answer_now must beat ask_more.
  when evidence is insufficient, ask_more must beat answer_now.
  request type should be a second decision after the ask_more gate, not fused
  into the same brittle long-string competition.
```
