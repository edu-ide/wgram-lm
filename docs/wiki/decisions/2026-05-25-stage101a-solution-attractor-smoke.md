# 2026 Stage101 Solution-Attractor Smoke

## Decision

Stage101A is a useful signal, but not a promoted architecture.

```text
accepted: false
useful signal: true
```

The new mechanism can move the same LM head toward intelligence answers on
GD-lite shortcut pairs. It still fails the full solution-attractor gate because
successive/truthy shortcuts remain negative and tiny language heldout loss
slightly regresses.

## Plain-Language Read

Before Stage101A, the student could sit quietly and think longer, but the mouth
did not reliably move toward the right answer.

After a 12-step answer-facing contrastive smoke, the mouth does move:

```text
The student starts preferring the real answer over several tempting fake
answers. But the student still fails two kinds of temptation:

1. answer-sequence temptation:
   "the previous answers were 1, 2, 3, so say 4"

2. plausible-truth temptation:
   "this sounds true, so say True"
```

So the core idea is alive, but incomplete.

## Implemented Mechanism

Executable:

```text
scripts/570_train_solution_aligned_answer_attractor.py
```

Training signal:

```text
same prompt
same recurrent one-body model
same LM head

increase:
  logprob(intelligence_answer)

decrease:
  logprob(parrot_answer)

preserve:
  deeper depth should not lose the margin
```

This is not a bridge, selector, readback, or auxiliary side speaker. The normal
LM head is the scored and trained path.

## Run

```text
input checkpoint:
  local_eval/20260524_STAGE99I_LOCAL_ONE_BODY_GATE400/last_model.pt

output checkpoint:
  local_eval/20260525_STAGE101_LOCAL_SOLUTION_ATTRACTOR_SMOKE12/last_model.pt

summary:
  local_eval/20260525_STAGE101_LOCAL_SOLUTION_ATTRACTOR_SMOKE12/stage101_summary.json

steps:
  12

depths trained:
  2, 4
```

## GD-Lite Result

```text
depth | acc    | mean margin | failed
2     | 0.3333 | 0.1557      | flipped, persona, successive, truthy
4     | 0.6667 | 0.1519      | successive, truthy
8     | 0.6667 | 0.1032      | successive, truthy
```

Before Stage101A, Stage99I depth4 had:

```text
accuracy = 0.3333
mean margin = -0.0049
```

After Stage101A, depth4 has:

```text
accuracy = 0.6667
mean margin = 0.1519
```

This proves the answer-facing route is trainable.

## Tiny Language Heldout

Permanent tiny heldout sample:

```text
local_eval/20260525_STAGE101_TINY_LANGUAGE_HELDOUT_SAMPLE/sampled
```

Same heldout, baseline Stage99I vs Stage101A:

```text
depth | Stage99I loss | Stage101A loss | delta
2     | 4.81997       | 4.82358        | +0.00361
4     | 4.73560       | 4.74177        | +0.00617
8     | 4.53100       | 4.53320        | +0.00220
```

This is a tiny sample, so do not overinterpret the absolute values. The
direction is still a warning: answer-margin training must be paired with a
language-preserving term before promotion.

## Verdict

```text
Promote the idea:
  yes, as the next local research direction.

Promote the checkpoint:
  no.

Launch long DGX from this:
  no.
```

Reason:

```text
The model learned to resist some parrot shortcuts, especially flipped-answer
and persona. It still fails successive-answer and truthy-answer, and language
heldout loss is not preserved.
```

## Stage101B Requirement

The next experiment must add two things:

```text
1. More counterexamples for successive/truthy shortcuts.
2. Language-preserving regularization or heldout gate during answer-margin
   training.
```

Promotion gate:

```text
GD-lite:
  flipped, successive, truthy all pass
  mean margin improves at depth 4 and stays positive at depth 8

Language:
  tiny heldout loss does not regress
  free generation does not become repetitive
```

## One-Sentence Lock

```text
Stage101A proves the mouth can be pulled toward the right answer; Stage101B
must prove it can do that without forgetting how to speak.
```

## Stage101B CONT14 Update

Stage101B adds:

```text
1. more successive/truthy counterexamples
2. language-preserving KL against the previous checkpoint
```

Run:

```text
output:
  local_eval/20260525_STAGE101B_LOCAL_SOLUTION_ATTRACTOR_KL_CONT14/last_model.pt

summary:
  local_eval/20260525_STAGE101B_LOCAL_SOLUTION_ATTRACTOR_KL_CONT14/stage101b_cont14_summary.json
```

Original GD-lite depth4:

```text
accuracy = 1.0000
mean_margin = 0.3450
min_margin = 0.0453
accepted = true
```

Stage101B heldout depth4:

```text
accuracy = 0.9000
mean_margin = 0.2760
min_margin = -0.0404
accepted = false

passed:
  flipped, repetitive, successive, intuitive, persona

failed:
  truthy
```

Tiny language heldout best loss:

```text
Stage99I        = 4.530996
Stage101A       = 4.533199
Stage101B-28    = 4.533719
Stage101B-CONT14= 4.532797
```

Plain-language read:

```text
One-body plus answer-attractor training now makes the normal LM head choose
the intelligence answer on the original shortcut gate. This is no longer a
"the thought cannot reach the mouth" failure.

The remaining failure is narrower: the model still has a weak truth/claim
judgment attractor on heldout cases, and language preservation is close but not
yet better than the Stage99I baseline.
```

Verdict:

```text
Promote the mechanism:
  yes, as the current local path.

Promote the checkpoint to long DGX:
  no.

Next local gate:
  repair truthy/successive with language-preserving KL, then rerun original
  GD-lite, Stage101B heldout, and tiny language heldout.
```

## Stage101C/101D Update

Summary:

```text
local_eval/20260525_STAGE101D_LOCAL_OVERTHINK_LOCK_KL_SMOKE20/stage101cd_summary.json
```

Stage101C added truth-claim repair rows:

```text
train:
  data/eval/stage101c_truth_claim_train_probe.jsonl

heldout:
  data/eval/stage101c_truth_claim_heldout_probe.jsonl
```

Stage101C result:

```text
original GD-lite depth4:
  accuracy = 1.0000
  mean_margin = 0.4064
  accepted = true

Stage101B heldout depth4:
  accuracy = 1.0000
  mean_margin = 0.3389
  accepted = true

Stage101C broader truth-claim heldout depth4:
  accuracy = 0.8571
  accepted = false

tiny language heldout:
  delta vs Stage99I = +0.000820
```

Stage101D added overthinking lock training:

```text
depths:
  4 -> 8

plain-language objective:
  Once the model has found the answer at depth4, depth8 must not drift back
  into a shortcut answer.
```

Stage101D result:

```text
original GD-lite depth8:
  accuracy = 1.0000
  mean_margin = 0.3911
  accepted = true

Stage101B heldout depth8:
  accuracy = 1.0000
  mean_margin = 0.3210
  accepted = true

tiny language heldout:
  delta vs Stage99I = +0.001433
```

Plain-language read:

```text
The old one-body failure is no longer the main blocker. The thought now reaches
the same LM mouth, and the overthinking lock can keep depth8 from undoing the
depth4 answer on the original and Stage101B heldout gates.

The remaining problem is broader truth/commonsense coverage: when a claim
sounds physically intuitive but is false, or sounds odd but is true, the model
still needs a wider truth-claim curriculum.
```

Verdict:

```text
Promote Stage101D to DGX long run:
  not yet.

Reason:
  old gates pass, but Stage101C broader truth-claim heldout is still 12/14 and
  language loss is slightly above Stage99I.

Next local move:
  broaden truth-claim heldout/train curriculum, keep depth4->8 overthinking
  consistency, and require both depth4 and depth8 heldout truth gates to pass.
```

## Stage101E/101F Update

Summary:

```text
local_eval/20260525_STAGE101F_LOCAL_SOURCE_GROUNDED_TRUTH_SMOKE48/stage101ef_summary.json
```

Stage101E tried broader unsupported world-truth repair:

```text
script:
  scripts/573_build_stage101e_world_truth_probe.py

train:
  data/eval/stage101e_world_truth_train_probe.jsonl
  data/eval/stage101e_world_truth_hard_anchor_train_probe.jsonl

result:
  useful diagnostic, not accepted
```

Plain-language read:

```text
The model is being asked whether claims about sound, mass, density, and
everyday physics are true. If the prompt does not provide the needed fact, this
is partly a world-knowledge exam, not a clean reasoning exam.
```

Stage101F separated the contract:

```text
script:
  scripts/574_build_stage101f_source_grounded_truth_probe.py

train:
  data/eval/stage101f_source_grounded_truth_train_probe.jsonl

heldout:
  data/eval/stage101f_source_grounded_truth_heldout_probe.jsonl
```

Result:

```text
Stage101D source-grounded heldout depth8:
  accuracy = 0.2500

Stage101F source-grounded heldout depth8:
  accuracy = 0.5000

Stage101F original GD-lite depth8:
  accuracy = 1.0000
  accepted = true

Stage101F Stage101B heldout depth8:
  accuracy = 1.0000
  accepted = true

Stage101F tiny language heldout:
  delta vs Stage99I = +0.000804
```

Verdict:

```text
Stage101F proves the same LM path can begin learning source-to-truth mapping
without breaking the old answer-attractor and overthinking gates.

It is not enough for promotion: source-grounded heldout is still 2/4.
```

Architecture implication:

```text
Do not add a side verifier.

The next fix is data-contract aligned:
  broader source-grounded truth curriculum
  paraphrase/source-position variation
  old-gate anchors
  depth4->8 consistency
  language preservation
```

## Stage101G Update: Source Paraphrase and Overthinking Noise

Summary:

```text
local_eval/20260525_STAGE101G_LOCAL_SOURCE_PARAPHRASE_LOCK_KL_SMOKE120/stage101g_summary.json
```

New diagnostic:

```text
script:
  scripts/576_eval_overthinking_noise_probe.py

purpose:
  separate "the model does not know/read the answer" from "extra thought erased
  a shallow correct answer"
```

Result:

```text
source paraphrase heldout, depth8:
  accuracy = 0.2500
  mean_margin = -0.2115
  accepted = false

source paraphrase overthinking stability:
  stability_accepted = true
  flip_to_wrong_count = 0
  wrong_at_all_depths_count = 6/8

source truth heldout after Stage101G, depth8:
  accuracy = 0.5000
  mean_margin = -0.0202
  accepted = false

Stage101B anchor overthinking:
  depth8 accuracy = 1.0000
  depth16 accuracy = 1.0000
  mean_margin depth2 -> depth16 = 0.4904 -> 0.4275
  stability_accepted = false
  reason = mean_deep_margin_degraded

original GD-lite depth8:
  accuracy = 1.0000
  mean_margin = 0.4767
  accepted = true

tiny language heldout:
  Stage99I best loss = 4.5310 @ depth8
  Stage101F best loss = 4.5318 @ depth8
  Stage101G best loss = 4.4837 @ depth16
```

Plain-language read:

```text
There are two different problems now.

For source/paraphrase rows, the model is not mainly losing a correct answer by
thinking too long. It is usually wrong at every depth. That means the bottleneck
is source reading, unit/equivalence semantics, and prompt-template robustness.

For the old Stage101B anchor, the answer stays correct, but extra depth thins
the margin. That is real overthinking noise: the mind keeps talking after it
already found the answer, and confidence leaks even though the final choice has
not flipped yet.
```

Verdict:

```text
Reject Stage101G as promotion.

Keep:
  overthinking-noise probe as a required diagnostic
  original GD-lite and Stage101B anchors
  language-preserving KL

Next local move:
  fix source/paraphrase through curriculum/data contract
  separately add a stop-or-stabilize pressure so depth16 does not erode an
  already-correct Stage101B margin
```

## Stage101H Update: Overthinking Margin Lock

Summary:

```text
local_eval/20260525_STAGE101H_LOCAL_OVERTHINK_MARGIN_LOCK_SMOKE80/stage101h_summary.json
```

Training:

```text
checkpoint in:
  Stage101G last_model.pt

probe:
  data/eval/stage101b_solution_attractor_train_probe.jsonl

depths:
  2 4 8 16

purpose:
  make already-correct shortcut/answer-attractor anchors stop losing margin at
  deeper thought depths
```

Heldout result:

```text
Stage101B heldout:
  depth16 mean_margin Stage101G -> Stage101H = 0.4275 -> 0.5419
  depth8 mean_margin = 0.5623
  depth16 mean_margin = 0.5419
  accuracy remains 1.0000
  stability_accepted = false
  reason = depth16 still below depth2 by -0.0532 mean margin

original GD-lite depth8:
  mean_margin Stage101G -> Stage101H = 0.4767 -> 0.5764
  accepted = true

source paraphrase depth8:
  mean_margin Stage101G -> Stage101H = -0.2115 -> -0.2846
  accuracy remains 0.2500

language:
  best heldout loss = 4.4853 @ depth16
```

Plain-language read:

```text
The overthinking lock can strengthen old anchors, but it does not solve source
reading. In fact, optimizing only old anchors can pull the model farther away
from source/paraphrase truth rows.

So the next run must not be a single scalar "think harder" fix. It needs two
lanes in the same one-body path:
  1. source-reading/equivalence curriculum so the model has the right answer
     state at all
  2. stop-or-stabilize pressure so deeper thought preserves that state once it
     is found
```

Verdict:

```text
Do not promote Stage101H.

Keep as evidence:
  overthinking noise is real and trainable
  source/paraphrase failure is separate and must be repaired by curriculum/data
  contract, not by more anchor-only margin locking
```

## Stage101I/J/K Update: Template and Polarity Shortcuts

Stage101I:

```text
checkpoint in:
  Stage101H

data:
  Stage101G source/paraphrase train rows

change:
  train source + anchors at depths 2/4/8/16 for 360 steps

result:
  rejected
```

Plain-language read:

```text
More of the same source training was not enough. The model kept treating prompt
template as a hidden answer key: context_first/claim_first often chose one
polarity while answer_from_note/after_question chose the opposite polarity for
the same source and claim.
```

Stage101J added same-mouth template consistency:

```text
code:
  scripts/570_train_solution_aligned_answer_attractor.py

new option:
  --template-consistency-weight

summary:
  local_eval/20260525_STAGE101J_LOCAL_TEMPLATE_CONSISTENCY_SMOKE240/stage101ij_summary.json
```

Result:

```text
source paraphrase heldout depth8:
  Stage101H = 0.2500
  Stage101J = 0.5000

original GD-lite depth8:
  accepted = true

Stage101B heldout depth16:
  accuracy = 0.9000
  accepted = false
```

Plain-language read:

```text
Template consistency hit the right causal bug. It stopped the model from
splitting the same source fact into opposite answers based only on prompt
format.

But it exposed the next shortcut: whole source concepts became stable-wrong in
one True/False direction. The model learned "this group should be False" more
readily than "read the source and decide."
```

Stage101K added polarity-balanced source rows:

```text
script:
  scripts/577_build_stage101k_polarity_balanced_source_probe.py

summary:
  local_eval/20260525_STAGE101K_LOCAL_POLARITY_BALANCED_TEMPLATE_CONSISTENCY_SMOKE240/stage101k_summary.json
```

Result:

```text
source/paraphrase heldout depth8:
  Stage101H = 0.2500
  Stage101J = 0.5000
  Stage101K = 0.6250

Stage101K polarity-balanced heldout depth8:
  accuracy = 0.5833
  accepted = false

Stage101B anchor depth16:
  Stage101J = 0.9000
  Stage101K = 0.8000
  accepted = false

language heldout:
  accepted = true
  best loss = 4.4848 @ depth16
```

Verdict:

```text
Do not promote Stage101K.

Keep:
  template consistency is causal and useful
  polarity-balanced source curriculum improves source heldout

Reject:
  running source/template pressure without enough old-anchor and truthy replay

Next local move:
  Stage101L should lower source/template pressure or stage it as source-True
  warmup first, then mix with stronger Stage101B/truthy anchor replay. Promotion
  requires source heldout improvement without depth16 anchor damage.
```
