# Universal LM Path After Typed CE Scope Correction

Date: 2026-05-07

## Decision

Typed algorithmic CE remains useful, but only as a probe. It is not the
canonical QTRM objective because a task-specific typed field loss can drift
toward a hidden neuro-symbolic executor.

Canonical QTRM progress must be measured on the universal LLM path:

```text
prompt/chat template
-> tokenizer
-> donor hidden states or token embeddings
-> mandatory recurrent QTRM core / answer loop
-> LM logits
-> autoregressive text or causal forced-choice answer
```

## Current Canonical Baseline

The current canonical raw-recursive answer-path baseline is:

```text
docs/wiki/decisions/ouro-answer-halt-head-s080.md
docs/wiki/decisions/ouro-answer-halt-head-s080-raw-gate.md
```

Accepted smoke8 gate:

```text
donor_only:   0/8
core_off:     0/8
depth1:       0/8
depth2:       0/8
depth4:       8/8
depth8:       8/8
halt_off:     0/8
```

Partial scale32 depth gate:

```text
donor_only:   0/32
core_off:     0/32
depth1:       4/32
depth2:       4/32
depth4:      16/32
```

Artifact:

```text
docs/wiki/decisions/ouro-answer-halt-head-s080-depth4-scale32-partial-gate.md
```

Boundary:

```text
This is accepted only as a depth4 partial scale-up. The attempted full
depth8/halt-off scale32 sweep was stopped after 170/256 rows because it was
too slow for the current loop. It did not produce a complete halt-off gate.
```

Expanded smoke16 gate:

```text
depth4:       10/16
depth8:       10/16
halt_off:      0/16
bridge_off:   10/16
```

Interpretation:

```text
accepted:
  recurrent answer path + in-loop halt gate causally improves LM
  forced-choice reasoning over donor/core-off.

not accepted:
  broad general reasoning
  greedy autoregressive answer rendering
  transition bridge causality, because bridge_off ties full.
```

## Failure Ledger

```text
Failure:
  typed CE is task-specific and can leave the normal LM answer path.

Evidence:
  typed value-state content-field accuracy improves over head-off, but trace
  exact remains 0/32 and the final text path is not the promoted metric.

Known limitation class:
  process-probe overfitting / hidden structured solver risk.

Root architecture hypothesis:
  The core must update answer hidden states that directly feed LM logits.
  Side fields are useful only if they causally improve that path.

Information path needed:
  prompt -> recurrent answer hidden -> halt/finality -> LM logits.

Current best information path:
  Ouro answer halt head S080.

Alternative explanations:
  smoke8 may be too small; forced-choice may hide renderer failure; depth4 may
  be a memorized terminal point rather than adaptive recursion.

Smallest next experiment:
  keep halt-head S080 as baseline and train/evaluate only changes that
  preserve or improve forced-choice while improving greedy generation.

Acceptance gate:
  full > donor/core_off and module_off, no retrieval/MemoryOS, generation hit
  improves without forced-choice regression.

Kill criterion:
  if generation remains 0 while forced-choice is preserved, reject renderer
  patch and redesign the answer-token training distribution.
```

## Next Candidate Ranking

1. Autoregressive answer-path renderer that preserves the halt-head gate.

```text
target:
  make the accepted forced-choice answer state produce stable tokens.

constraints:
  no donor-logit shortcut
  no typed answer channel
  validation-gated checkpoints
  forced-choice must not regress
```

2. Adaptive depth generalization for mixed terminal depths.

```text
target:
  show the halt head chooses different useful depths across tasks, not one
  fixed depth-4 pattern.
```

3. General latent transition auxiliary.

```text
target:
  use text/latent process targets only as auxiliary signals, and promote only
  if they improve LM logits/text under ablation.
```

## Workflow Contract

```text
1. Run donor/core-off/depth sweep.
2. Run recurrent/halt/module-off ablations.
3. Run forced-choice and greedy generation.
4. Reject lower train CE unless held-out LM metrics improve.
5. Keep typed/register metrics as diagnostics, not the main claim.
```
