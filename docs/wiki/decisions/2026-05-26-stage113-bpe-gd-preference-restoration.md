# 2026-05-26 Stage113 BPE GD Preference Restoration

## Decision

Stage113 is the first strong local positive signal after the OPUS/GD and BLT
rejects, but it is not fully accepted yet.

The causal route changed:

```text
Before:
  train on ordinary answer CE
  -> hope the model naturally prefers the intelligence answer

After:
  same BPE PrefixLM path scores:
    prompt + intelligence_answer
    prompt + parrot_answer
  and directly trains:
    logp(intelligence_answer) > logp(parrot_answer)
```

No side verifier, oracle executor, or external calculator was added. The normal
LM head carries the preference.

## Main Result

Heldout GD smoke rows were excluded from preference training.

```text
Baseline Stage105D BPE broad best:
  checkpoint: local_eval/20260526_STAGE105D_LOCAL_BPE_BROAD_HRM_TEXT_BS48_CHUNK64_CONT5000_ALLROWS/best_eval_model.pt
  valid rows: 20 / 44
  accuracy: 0.200000
  mean margin: -0.037395
  min margin: -2.210893

Stage113A step40:
  checkpoint: local_eval/20260526_STAGE113A_LOCAL_BPE_GD_PREFERENCE_80STEP/checkpoint_pref_step000040.pt
  valid rows: 20 / 44
  accuracy: 0.650000
  mean margin: 0.897934
  min margin: -0.817070

Stage113B continued model-only:
  checkpoint: /mnt/sdc1/tripleyoung/qtrm_eval/20260526_STAGE113B_LOCAL_BPE_GD_PREFERENCE_CONT40_MODELONLY/last.pt
  valid rows: 20 / 44
  accuracy: 0.700000
  mean margin: 1.232448
  min margin: -0.744675
```

Delta versus Stage105D:

```text
accuracy:    +0.500000
mean margin: +1.269843
```

## Plain-Language Read

This finally teaches the student the missing habit:

```text
When two answers are plausible, lean toward the answer that follows the rule,
not the answer that merely looks familiar.
```

The good news is that the habit moves quickly. The model does not need a new
organ for this first effect; the same LM mouth can be trained to prefer the
right answer.

The bad news is that the habit is uneven. Code tracing, letter counting,
successive letters, number words, and surprising truth flipped positive. The
algebra variants and one CRT row remain wrong.

## Remaining Failures

Stage113B still rejects because not every valid row has positive margin.

Weak heldout families:

```text
intuitive_answer/crt:
  accuracy: 0.500000
  mean margin: -0.077943

repetitive_answer/algebra/instruction:
  accuracy: 0.500000
  mean margin: -0.293425

repetitive_answer/algebra/numbered:
  accuracy: 0.500000
  mean margin: -0.171532

repetitive_answer/algebra/original:
  accuracy: 0.000000
  mean margin: -0.206020

repetitive_answer/algebra/v2fmt:
  accuracy: 0.500000
  mean margin: -0.218851
```

This means Stage113 is not a generalization pass. It is a causal proof that the
missing answer-preference route is real and trainable.

## Language Preservation Check

Small language heldout, 8 cases:

```text
Stage105D:
  loss: 11.087320
  token_accuracy: 0.105263

Stage113B:
  loss: 11.106356
  token_accuracy: 0.105263
```

Short direct generation gate:

```text
Stage105D:
  first_response_accuracy: 0.265625
  generation exact: 1 / 12
  prefix_token_accuracy: 0.386364
  repeated_token_loops: 0

Stage113B:
  first_response_accuracy: 0.234375
  generation exact: 1 / 12
  prefix_token_accuracy: 0.317073
  repeated_token_loops: 0
```

Read:

```text
Preference training did not catastrophically damage language, but it did
slightly hurt generation. The next accepted run must include language
preservation or mixed CE while keeping the GD margin gain.
```

## Implementation

Added:

```text
scripts/625_train_bpe_gd_preference.py
tests/test_bpe_gd_preference_train.py
```

The trainer:

```text
loads a BPE PrefixLM checkpoint,
filters GD rows that fit the checkpoint seq_len,
excludes heldout smoke IDs,
optimizes pairwise preference loss on the same LM logits,
optionally adds a small CE term on the intelligence answer,
saves model-only checkpoints by default to avoid local disk exhaustion,
records selected row IDs for reproducibility.
```

Operational note:

```text
The first Stage113A 80-step run trained successfully, but final checkpoint
write failed because the local root partition was full. The valid step40
checkpoint was preserved and evaluated. Stage113B continued from it and saved
model-only output under /mnt/sdc1.
```

## Next Move

Do not go back to OPUS scalar tuning as the main move.

The next high-probability local run is:

```text
Stage114:
  BPE same-LM preference training
  + hard-family replay for algebra/CRT
  + language-preservation CE/KL mix
  + same heldout GD smoke gate
  + language/generation preservation gate
```

Promotion gate:

```text
GD valid accuracy >= 0.90 on smoke,
mean margin stays positive,
min margin becomes positive or near-zero on algebra/CRT,
language heldout does not regress materially,
generation prefix accuracy does not regress versus Stage105D,
and the gain disappears if preference loss is disabled.
```

Only after that should this route be considered for larger local or DGX
training.
