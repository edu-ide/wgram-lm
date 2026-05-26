# 2026-05-26 Stage112 BPE GD Restoration Gate

## Decision

Stage112 rejects the hypothesis that the latest GD failure is only a
tokenizer-free/BLT reading problem.

Stable BPE one-body checkpoints were evaluated with the same local
Generalization Dynamics choice gate. Both failed.

```text
Stage104C BPE recurrent control:
  checkpoint: local_eval/20260526_STAGE104C_LOCAL_BPE_RECURRENT_CONTROL_CONT1200/last.pt
  valid rows: 20 / 44
  accuracy: 0.400000
  mean margin: -0.159194
  min margin: -1.743019
  accepted: false

Stage105D BPE broad HRM-text best:
  checkpoint: local_eval/20260526_STAGE105D_LOCAL_BPE_BROAD_HRM_TEXT_BS48_CHUNK64_CONT5000_ALLROWS/best_eval_model.pt
  valid rows: 20 / 44
  accuracy: 0.200000
  mean margin: -0.037395
  min margin: -2.210893
  accepted: false
```

## Plain-Language Read

BPE gives the model steadier eyes. It can read the worksheet more reliably than
the current rough BLT path.

But the model still does not consistently prefer the true answer over the
parrot answer. That means the main failure is not only "the tokenizer is
wrong." The deeper failure is:

```text
The student can read some rows, but has not learned the habit of choosing the
causal answer when a tempting surface answer is nearby.
```

## What Improved

Stage104C still preserved small local pockets:

```text
repetitive_answer/algebra/original: accuracy 1.000000
repetitive_answer/letter_counting: accuracy 1.000000
```

Stage105D learned some successive-pattern rows strongly:

```text
successive_answer/letters: accuracy 1.000000, mean margin 2.942383
successive_answer/number_words: accuracy 1.000000, mean margin 5.516602
```

This is useful evidence: broader BPE pretraining can create real family-level
preferences.

## What Failed

The failures are not random. Stage105D became worse on many anti-parrot rows:

```text
intuitive_answer/crt: accuracy 0.000000
repetitive_answer/algebra/*: mostly 0.000000
repetitive_answer/code_tracing: accuracy 0.000000
truthy_answer/surprising_truth: accuracy 0.000000
```

So the current broad pretraining contract teaches some surface sequences, but
does not reliably teach:

```text
read the prompt
resist the obvious wrong answer
prefer the causally correct answer
```

## Implementation

Added a BPE counterpart to the BLT GD choice gate:

```text
scripts/624_eval_bpe_generalization_dynamics_probe.py
```

The evaluator loads native PrefixLM checkpoints, scores the log probability of
`intelligence_answer` versus `parrot_answer`, writes per-task margins, and logs
TensorBoard scalars when requested.

Smoke reports:

```text
local_eval/20260526_STAGE112_LOCAL_BPE_GD_RESTORATION_GATE/stage104c_bpe_gdsuite_smoke44.json
local_eval/20260526_STAGE112_LOCAL_BPE_GD_RESTORATION_GATE/stage105d_best_bpe_gdsuite_smoke44.json
```

TensorBoard:

```text
local_eval/20260526_STAGE112_LOCAL_BPE_GD_RESTORATION_GATE/tensorboard
```

## Consequence

Do not explain Stage110/111 rejection as "BLT only."

The better diagnosis is:

```text
BLT reading is one bottleneck.
Answer-preference curriculum is another bottleneck.
The second bottleneck remains even under stable BPE.
```

Therefore do not promote OPUS/GD full selection to DGX as a generalization
claim yet.

## Next High-Probability Move

The next local experiment should restore the worked-before answer preference
mechanism before scaling:

```text
reader -> recurrent thought -> candidate exposure/search -> same LM answer path
```

Fast gate:

```text
Use long-context BPE or a stable BLT reader.
Train/evaluate explicit anti-parrot answer preference.
Require improvement on GD accuracy, mean margin, and generation samples.
Require core/search/candidate ablations to remove the gain.
```

If this gate fails, the project should not tune OPUS weights or BLT boundaries
again as the main move. It needs a causal answer-path change, not another data
selection scalar.
