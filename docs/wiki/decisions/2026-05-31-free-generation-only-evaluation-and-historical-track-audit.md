# 2026-05-31 Free-Generation-Only Evaluation And Historical Track Audit

## Decision

Promotion evaluation for language/reasoning checkpoints is now
free-generation-only.

```text
prompt
-> model autoregressively decodes visible answer tokens
-> exact/normalized answer check + decoded sample audit
```

Do not promote with:

- forced-choice logprob ranking,
- selected-vs-oracle accuracy,
- oracle candidate coverage / pass@K,
- candidate reranking,
- teacher-forced first-token or target-rank scores.

Those metrics confused the difference between "the model can recognize the
right option" and "the model can actually answer." They may remain in old logs
only as historical diagnostics.

## Executable Policy

Allowed active gate:

```text
scripts/565_eval_blt_generation_gate.py
```

Default behavior is now free-generation-only. Teacher-forced first-token and
continuation diagnostics require explicit debugging opt-in and are written
under `diagnostic_not_promotion`.

Disabled active gate:

```text
scripts/566_eval_blt_candidate_rerank_gate.py
```

This script is kept for old report readability, but its `main()` exits before
running. It must not be used in active promotion.

## Historical Track Audit

### Stage56/58 PTRM / VTE Track

Primary files inspected:

- `scripts/517_train_qwen_register_extractor.py`
- `scripts/513_train_true_gram_smoke.py`
- `docs/wiki/decisions/qwen35-hrmtext-attention-pooling-diagnostics.md`

Historical high numbers:

```text
Stage56:
  K4 selected  = 0.3229
  K64 selected = 0.7031
  K128 selected = 0.7682

Stage58B:
  K64 top3 selected = 0.9336
  oracle coverage   = 0.9401
  register accuracy = 0.9935
```

Updated read:

```text
Do not count these as language ability.
They are selected/oracle/candidate-exposure scores.
```

Useful mechanisms to recover:

- stochastic recurrent trajectories,
- K-scaling breadth,
- trajectory diversity telemetry,
- an internal selector that is close to the answer path,
- typed/register-style state stabilization.

Do not recover:

- external answer tables,
- top-k visible candidate exposure as the final answer path,
- oracle-selected metrics,
- hand-built typed verifier as a general language evaluator.

### Stage53 Register Extractor

Primary file:

- `scripts/517_train_qwen_register_extractor.py`

Good idea:

```text
frozen reader hidden states
-> cross-attention typed registers
-> deterministic executor/checker
```

Updated read:

The typed-register design was useful because it made latent computation
inspectable and stable. It is not acceptable as a final answer evaluator. For
BLT/IMTA, the transferable idea is to give internal trajectories structured
state slots or selector telemetry, then force the normal LM head to speak.

### Stage59 Candidate Proposer / Pool Selector

Primary files:

- `scripts/527_train_state_candidate_proposer.py`
- `scripts/528_train_candidate_pool_selector.py`

Good idea:

```text
candidate speaker reads recurrent thought trajectory and workspace
```

Updated read:

The workspace-aware proposer is a useful architecture hint, but the pool
selector is explicitly a deprecated scaffold using hand-built typed heuristics.
Do not promote pool-selector numbers. Only reuse the idea that a speaker should
read the thought trajectory, not a detached answer table.

### Stage104 BPE PrefixLM Control

Primary evidence:

- `docs/wiki/decisions/0001-active-decision-index.md`
- `scripts/539_eval_prefixlm_generation_gate.py`

Historical free-generation result:

```text
Stage104B step240 exact generation = 1/16
Stage104C step1200 exact generation = 6/16
```

Updated read:

This is one of the more relevant historical tracks because it measured actual
autoregressive generation. It suggests the one-body PrefixLM mouth can become
non-degenerate, but the narrow worksheet overfit means it is not enough by
itself.

Useful mechanisms to recover:

- PrefixLM DataIO contract,
- response-only labels,
- best-eval checkpointing,
- generation gate with decoded samples,
- stable BPE/byte reading controls.

### Ouro / Halt-Head Track

Primary evidence:

- `docs/wiki/decisions/ouro-answer-halt-head-s080.md`

Historical result:

```text
forced-choice smoke8/16 improved under halt gate
greedy generation remained 0/8
```

Updated read:

This track is the clearest warning against forced-choice promotion. It found a
real internal ranking signal, but the model still could not speak. Use it only
as negative evidence: a causal hidden answer basin is not enough without a
working autoregressive mouth.

## Active BLT/IMTA Implication

The next BLT architecture may borrow:

```text
Stage56/58:
  K stochastic internal trajectories and trajectory diversity

Stage53:
  typed/structured latent-state stabilization

Stage104:
  DataIO + actual decoded generation gates

2605.27734:
  same-body own-latent prediction for hidden grammar learning
```

But the final claim must always be:

```text
same body
same LM head
free generated answer
decoded sample audit
```
