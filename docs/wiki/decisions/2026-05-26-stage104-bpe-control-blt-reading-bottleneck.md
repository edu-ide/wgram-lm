# Stage104 BPE Control: BLT Reading Bottleneck

Date: 2026-05-26

Status: active diagnostic anchor

## Plain-Language Read

The current question is not "BPE or tokenizer-free forever?"

The question is:

```text
Can the student think if we give it a stable way to read?
```

Stage104 says yes, at least on the small Stage103 reasoning microscope. With
the same native recurrent PrefixLM body and the same reasoning rows, the BPE
reader trains stably and benefits from recurrent depth up to depth 4.

That means the latest failure should not be read as:

```text
the recurrent thought core is useless
```

It should be read as:

```text
the BLT/H-Net reading-compression path is still unstable enough to hide or
damage a usable thought signal.
```

## Evidence

### BLT Stage103D, Corrected Finite-Row Depth Read

File:

```text
local_eval/20260526_STAGE103D_LOCAL_82M_REASONING_ANTI_OVERCOMPRESSION_CONT360/depth_residual_eval128_finite_summary.json
```

Finite rows show depth gain:

```text
depth1 loss 1.7889
depth2 loss 1.7589
depth4 loss 1.7120
depth8 loss 1.7039
```

But the run is still not promoted because non-finite rows remain:

```text
nonfinite_loss_rows: 8
nonfinite_residual_rows: 17
failed_checks: nonfinite_depth_rows_present
```

Plain read:

```text
The student thinks better when thinking longer, but some reading/compression
pages still tear during the exam.
```

### BPE Stage104B Control

Files:

```text
local_eval/20260526_STAGE104_LOCAL_BPE_CONTROL/
local_eval/20260526_STAGE104B_LOCAL_BPE_RECURRENT_CONTROL_FAST240/
```

Data contract:

```text
same Stage103 reasoning rows
HRM-Text/Data-IO official BPE tokenizer
native recurrent PrefixLM trainer
same LM-head answer path
official_gated_delta2 strict backend
```

Training signal:

```text
eval loss:
  step 1    11.0748
  step 60    7.0763
  step 120   2.5512
  step 180   2.1627
  step 240   2.0018

eval_nonfinite_batches: 0
```

Depth probe:

```text
depth1 loss 2.0807, residual 0.5656
depth2 loss 2.0179, residual 0.2267
depth4 loss 2.0018, residual 0.1312
depth8 loss 2.0083, residual 0.0776
```

Plain read:

```text
BPE gives the model stable eyes. The thought core then helps until depth 4.
Depth 8 is calmer but slightly overthinks on this small checkpoint.
```

Generation gate:

```text
file:
  local_eval/20260526_STAGE104B_LOCAL_BPE_RECURRENT_CONTROL_FAST240/generation_gate_eval16.json

first-response token accuracy: 0.18125
EOA-as-first-token rate:       0.0
generation exact:              1/16
starts_with_eoa:               0/16
ended_with_eoa:                16/16
repeated_token_loops:          0/16
```

Plain read:

```text
The model is not generation-ready after 240 steps. It often falls into easy
numeric answers such as "522", but it does not instantly close, does not loop,
and can stop cleanly. This is a stable early student, not a solved reasoner.
```

### BPE Stage104C Continuation To 1200 Steps

File:

```text
local_eval/20260526_STAGE104C_LOCAL_BPE_RECURRENT_CONTROL_CONT1200/report.json
```

Training completed to the configured 1200 steps and stopped normally; no
training process remains for this local run.

Signal:

```text
train loss:
  step 420   0.8057
  step 600   0.0789
  step 840   0.0226
  step 1080  0.0022
  step 1200  0.0036

eval loss:
  step 240   2.0018
  step 480   1.8093
  step 840   1.7939
  step 1200  2.1871

eval_nonfinite_batches:
  0 at every logged eval
```

Plain read:

```text
The BPE reader is stable, but the tiny microscope can be memorized. The student
keeps getting better at the exact worksheet while the heldout exam stops
improving and then gets worse. So the low free-generation accuracy is not
explained only by "not enough steps"; after a point the bottleneck becomes
data diversity / curriculum / checkpoint selection.
```

Consequence:

```text
Do not answer the current failure by simply running the same tiny Stage104C
sample longer. For local reasoning microscopes, use Stage104C as evidence that
BPE can read stably and overfit cleanly. The next meaningful local question is
whether the best available checkpoint improves generation/depth gates before
overfit, not whether train loss can go lower.
```

Generation comparison:

```text
Stage104B @ step240:
  first-response token accuracy: 0.18125
  gold first-token probability:  0.10937
  generation exact:              1 / 16 = 0.0625
  prefix token accuracy:         0.4355
  repeated loops:                0 / 16

Stage104C @ step1200 last:
  file:
    local_eval/20260526_STAGE104C_LOCAL_BPE_RECURRENT_CONTROL_CONT1200/generation_gate_eval16_last.json
  first-response token accuracy: 0.44375
  gold first-token probability:  0.42480
  generation exact:              6 / 16 = 0.375
  prefix token accuracy:         0.6613
  repeated loops:                0 / 16
```

Plain read:

```text
Longer BPE training did improve speaking. The model moved from "mostly guesses
522" to producing several correct full answers and clean <|box_end|> stops.
So the mouth is learnable through the one-body path.

But it is not solved: after several exact answers, many later arithmetic rows
fall into wrong repeated answer basins such as 290, and eval loss worsened by
step1200. This is not random collapse; it is a student who learned the small
worksheet style but still lacks robust rule generalization.
```

Accepted interpretation:

```text
The BPE one-body path is a viable local reading/speaking control.
More training helps generation early, but the tiny microscope overfits.
Next improvement should come from broader/counterfactual curriculum and
checkpoint selection, not from simply lowering train loss.
```

Operational correction:

```text
Stage104C did not preserve the step840 model even though step840 had the best
logged eval loss. The run therefore teaches a checkpointing lesson too:

  last_model.pt answers "what happened after all scheduled training?"
  best_eval_model.pt answers "where did the student generalize best?"

Future native/BPE and BLT PrefixLM trainers now save:
  best_eval_model.pt
  copy_best_eval_model.pt

whenever eval loss improves. Long microscope runs should judge both the last
checkpoint and the best-eval checkpoint before declaring overfit or failure.
```

Plain-language guardrail:

```text
Do not let the final tired student be the only student we grade. If the student
was best at step840 and over-memorized by step1200, keep the step840 notebook.
```

## Decision

BPE is now the local reading-stability control for short reasoning microscopes.

Do not promote BLT/semantic-BLT to large from-scratch pretraining until it
beats or matches the BPE control on all three axes:

```text
1. no non-finite eval/depth rows;
2. depth 2/4/8 improves or preserves heldout loss against depth 1;
3. free generation is non-degenerate, not only teacher-forced CE.
```

This does not demote tokenizer-free research. It demotes premature BLT scale-up.

## OPUS/GD Boundary For This Decision

Stage104B/C are not OPUS/GD pretraining-efficiency runs.

```text
Stage104B/C:
  local BPE reading-stability control
  same small Stage103 reasoning rows
  no OPUS data-window selection
  no GD-lite row-selection proxy

Stage95I on DGX:
  tokenizer-free byte/BLT full pretraining
  OPUS-style utility-selected data window
  Generalization Dynamics rows included in OPUS proxy
  GD-lite required as a post-training gate before acceptance
```

Plain-language separation:

```text
Stage104 asks: can the student read the worksheet stably if we give it BPE eyes?
Stage95I asks: can the student read a much larger tokenizer-free textbook
chosen by an OPUS audition without becoming a parrot?
```

## Next Gate

Run the same small microscope with:

```text
BPE anchor:
  depth 1/2/4/8
  response-token CE
  first-token accuracy
  generation sample

BLT candidate:
  same rows
  same model width if possible
  finite-row and nonfinite-row depth report
  boundary/latent length telemetry
```

Promotion rule:

```text
BLT can return to from-scratch scale only when it stops tearing pages and
matches the BPE anchor's stable reading signal.
```

Reject rule:

```text
If BLT still has non-finite rows or depth gain appears only after filtering
broken batches, treat BLT as a reading-compression research thread, not the
main pretraining tokenizer.
```

Anti-overfit next run:

```text
Keep the same one-body answer path.
Change the data contract, not the architecture label:
  broaden the worksheet with counterfactual arithmetic/source rows;
  keep train/eval split row-fixed;
  save best-eval and last checkpoints;
  evaluate first-token, free generation, depth 1/2/4/8, and repetition.

Expected causal effect:
  if overfit was mostly narrow-data memorization, broader/counterfactual rows
  should keep eval loss flatter while generation continues improving.

Fast reject:
  if train loss falls but best-eval/free-generation does not improve over
  Stage104C, the bottleneck is not just data breadth; inspect answer-attractor
  and recurrent-depth dynamics next.
```

## Stage105A Broad HRM-Text Split Probe

File:

```text
local_eval/20260526_STAGE105A_LOCAL_BPE_BROAD_HRM_TEXT_120/report.json
```

Data contract:

```text
Same BPE one-body native PrefixLM architecture as Stage104.
Train/eval are now a deterministic 90/10 split over HRM-Text cleaned sources:
  gsm8k_train
  math_train
  omnimath
  Platypus/openbookqa

train rows after split: 33088
eval rows after split:   3671
trainer max rows:        4096
eval max rows:            128
```

Signal:

```text
train loss:
  step 1    11.1632
  step 60    7.5018
  step 120   5.7894

heldout eval loss:
  step 1    11.0939
  step 60    7.7943
  step 120   4.3923

eval_nonfinite_batches:
  0 at every eval

best_eval_checkpoint:
  local_eval/20260526_STAGE105A_LOCAL_BPE_BROAD_HRM_TEXT_120/best_eval_model.pt
```

Plain read:

```text
This is the first corrective evidence after Stage104C. With the same BPE
one-body mouth but a broader HRM-Text-style textbook, heldout eval loss falls
with training instead of immediately separating from it.

So the strongest current diagnosis is not:
  "the architecture always overfits."

It is:
  "the tiny arithmetic microscope was too narrow, and the strong recurrent
   answer path learned that worksheet too well."
```

Boundary:

```text
Stage105A is not a solved language model. It is only a 120-step causal probe.
It proves that broad data can remove the early overfit smell; it does not yet
prove free generation quality, depth generalization, or full pretraining
efficiency.
```

## Stage105B Broad HRM-Text Continuation

Files:

```text
local_eval/20260526_STAGE105B_LOCAL_BPE_BROAD_HRM_TEXT_CONT600_ALLROWS/report.json
local_eval/20260526_STAGE105B_LOCAL_BPE_BROAD_HRM_TEXT_CONT600_ALLROWS/generation_gate_eval12_best.json
```

Data/training contract:

```text
resume:
  local_eval/20260526_STAGE105A_LOCAL_BPE_BROAD_HRM_TEXT_120/last.pt

train rows:
  all rows from the broad train split

eval rows:
  512 heldout rows from the broad eval split

steps:
  continue to 600
```

Loss signal:

```text
heldout eval loss:
  step 120   4.3923
  step 240   3.5739
  step 360   3.2487
  step 480   3.0587
  step 600   2.9360

eval_nonfinite_batches:
  0 at every eval

best_eval_step:
  600
```

Plain read:

```text
The broad-data diagnosis strengthened. Unlike Stage104C, the model did not
memorize a tiny worksheet and then drift upward on heldout loss. Eval kept
falling through the final logged step.

So the current root cause is not "BPE one-body architecture inevitably
overfits." The narrower read is stronger:
  small arithmetic microscope + strong recurrent answer path = fast worksheet
  memorization.
```

Generation gate:

```text
checkpoint:
  best_eval_model.pt at step600

first-response accuracy:
  0.2109 over 256 positions

generation exact:
  3 / 12 = 0.25

ended_with_eoa:
  12 / 12

repeated_token_loops:
  0 / 12
```

Plain read:

```text
The mouth is alive but young. It stops cleanly and does not loop, but on hard
math it still collapses to easy answer basins such as "1", "1/2", or a multiple
choice letter. This is normal early broad pretraining behavior, not a solved
reasoner.
```

Operational follow-up:

```text
Stage105C was launched from Stage105B/last.pt and completed the same broad
split to 2000 steps:

heldout eval loss:
  step 600   2.9360
  step 720   2.8965
  step 840   2.8665
  step 1800  2.5300
  step 1920  2.5017
  step 2000  2.4680

out:
  local_eval/20260526_STAGE105C_LOCAL_BPE_BROAD_HRM_TEXT_CONT2000_ALLROWS

log:
  /tmp/20260526_STAGE105C_LOCAL_BPE_BROAD_HRM_TEXT_CONT2000_ALLROWS.log
```

Plain read:

```text
The broad BPE one-body run is still learning. This is no longer the Stage104C
tiny-worksheet overfit shape. The heldout curve kept improving through 2000.
```

## Stage105D Batch-Increased Continuation

Stage105C used batch 8. After completion, local batch capacity was tested from
the Stage105C/last.pt checkpoint.

Capacity result:

```text
1-step smoke:
  batch 16 pass
  batch 24 pass
  batch 32 pass
  batch 48 pass
  batch 64 pass

20-step smoke:
  batch 64 + loss_chunk 64 reject: CUDA OOM in cross_entropy
  batch 64 + loss_chunk 32 reject: CUDA OOM in cross_entropy
  batch 48 + loss_chunk 64 pass
```

Root cause:

```text
The batch-64 failure is not a recurrent-core failure. It is a logits/loss
materialization limit in prefixlm_loss_from_hidden -> F.cross_entropy when a
long response-heavy batch arrives. Reducing loss_chunk_size helps but does not
make batch 64 stable on the local 24GB RTX 4090.
```

Promoted local continuation:

```text
tmux session:
  stage105d_local_bpe_broad_bs48_5000

out:
  local_eval/20260526_STAGE105D_LOCAL_BPE_BROAD_HRM_TEXT_BS48_CHUNK64_CONT5000_ALLROWS

log:
  /tmp/20260526_STAGE105D_LOCAL_BPE_BROAD_HRM_TEXT_BS48_CHUNK64_CONT5000_ALLROWS.log

resume:
  local_eval/20260526_STAGE105C_LOCAL_BPE_BROAD_HRM_TEXT_CONT2000_ALLROWS/last.pt

batch:
  48

loss_chunk_size:
  64

target step:
  5000
```

First live signal:

```text
Stage105C final/best eval loss:
  2.4680 at step2000

Stage105D live eval:
  2.4012 at step2040
  2.3736 at step2160
  2.2555 at step2280
  2.2072 at step2400
  2.1619 at step2520
```

Plain read:

```text
The student is now reading more pages per update without immediately breaking.
Batch 48 is the current local maximum that survived a real multi-step smoke.
Batch 64 is tempting but too close to the memory cliff; do not promote it until
the loss path is made more memory efficient.
```

## Why Old 0.x Loss / High Accuracy Does Not Contradict This

Old Stage59-style results are important diagnostic evidence, but they are not
the same exam as Stage103/104.

Example:

```text
local_eval/stage59_local_choice_verifier_allfamily_t256_e64_ep30_s1604/summary.json

epoch 24 train_eval accuracy: 0.8008
epoch 30 train loss:          0.4329
epoch 30 heldout eval:        0.4375
best heldout accuracy:        0.5156
```

Plain-language difference:

```text
Stage59:
  The right answer is already one of the choices on the table.
  The model/verifier asks: "Can I recognize the right card?"

Stage103/104:
  The model must read the problem, think in one body, and write the answer
  through the normal LM head.
  The exam asks: "Can I make the right answer myself?"
```

So the old high number mostly proved:

```text
candidate recognition / verifier signal / train-set fit can work.
```

It did not prove:

```text
free generation,
same-LM-head speaking,
tokenizer/reader stability,
or general one-body reasoning.
```

This is why a newer architecture can look numerically worse while being a more
honest architecture test. The old path gave the student answer cards. The new
path asks the student to read, calculate, and speak.

## Qwen3.6-27B Comparison Boundary

The Stage58 Qwen3.6 comparison is also narrow evidence, not a broad 27B-beating
claim.

Relevant local files:

```text
local_eval/stage58_qwen36_mtp_proxy_baseline_128_20260522/report.json
local_eval/stage58_qwen36_mtp_proxy_baseline_128_answeronly_20260522/report.json
local_eval/stage58_qwen36_mtp_proxy_baseline_full_answeronly_20260522/report.json
```

Observed Qwen3.6-27B baseline:

```text
same-prompt modulo-10, first standalone two-digit exact:
  0 / 128 = 0.0

answer-only modulo-10, final standalone single-digit exact:
  81 / 128 = 0.6328

answer-only modulo-10 full suite:
  324 / 768 = 0.4219
```

Plain-language difference:

```text
Qwen3.6 Stage58 comparison:
  a narrow modulo-10 quiz where the answer is one digit.

Stage103/104 one-body LM gate:
  read a PrefixLM row, think through the recurrent body, and write response
  tokens through the same LM head.

Public benchmark claim:
  official or matched public benchmark harness, same prompt/scorer/decoding,
  Qwen baseline saved, QTRM-native score saved, and ablations saved.
```

Allowed claim:

```text
Under a narrow modulo-10 synthetic OOD answer-only protocol, QTRM-style
specialized reasoning showed a signal that can beat the saved Qwen3.6-27B
baseline on that private suite.
```

Forbidden claim:

```text
0.8B has broadly beaten Qwen3.6-27B as a general LLM, public benchmark model,
or agentic coding/tool model.
```
