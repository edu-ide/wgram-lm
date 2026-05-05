# Temporal-Spatial Context S240 Failure Ledger

Status: rejected, root-causality failure.

## Failure

The temporal/spatial context prefix is wired into the QTRM forward path, but
S240 training did not make it causally useful.

```text
context on:
  qtrm_core_steps_8_no_evidence = 4/24

context off:
  qtrm_core_steps_8_temporal_spatial_off_no_evidence = 4/24

changed completions:
  0/24

mean chosen-logprob delta:
  +0.00048
```

## Evidence

```text
checkpoint:
  /mnt/nvme1n1p2/qtrm-local-checkpoints/temporal_spatial_context_probe/s240/last.pt

eval:
  local_eval/temporal_spatial_context_gate.jsonl

gate:
  docs/wiki/decisions/temporal-spatial-context-gate.md
  docs/wiki/decisions/temporal-spatial-context-gate-summary.json

delta:
  docs/wiki/decisions/temporal-spatial-context-delta-summary.json
```

## Root Hypothesis

The context path is too weakly supervised. The prompt already contains the
source facts, while the structured context tokens are only auxiliary derived
features. Cross-entropy on the final answer can minimize loss through the text
path and ignore the new temporal/spatial prefix.

Spatial also exposes a binding problem: the vector says positions, but the
answer is a text label. There is no explicit pressure forcing the core to bind
structured coordinates back to answer choices.

## Could The Big Structure Be Wrong?

Not yet. The forward path is plausible, but the current training objective does
not create enough causal pressure. If context-swap or context-dropout training
still produces zero delta, then the prefix-token design should be replaced by a
stronger prompt-conditioned context reader inside the core.

## Candidates

Candidate A: context dropout and swap contrast.

```text
Train pairs:
  full context -> chosen answer
  context disabled or swapped -> lower chosen-answer logprob

Gate:
  context-on hit/logprob must beat context-off and swapped-context.
```

This is the smallest next experiment and directly attacks the zero-delta
failure.

Candidate B: choice-bound structured context.

```text
For each answer choice, emit a context token with:
  choice_slot, temporal_validity, spatial_relation, confidence

The model must score choices using prompt labels plus structured features.
```

This fixes label-binding, especially for spatial tasks.

Candidate C: core cross-attention reader.

```text
Instead of only prepending context tokens before the prelude, let each recursive
core step query context tokens through an ablatable gated cross-attention path.
```

This is a larger architecture change and should follow only if A/B fail.

## Recommended Next Step

Implement Candidate A as a diagnostic/probe:

```text
1. Extend training data with context_off and context_swap rejected variants.
2. Add a context-contrastive margin loss over chosen-answer logprob.
3. Add a context-swap eval mode.
4. Accept only if:
   context_on > context_off
   context_on > context_swap
   changed completions or chosen-logprob deltas are non-trivial
```

Do not claim temporal/spatial intelligence from the current S240 result.

## Candidate A Implementation Status

Implemented after this failure:

```text
scripts/196_train_pure_recursive_depth_supervised.py
  --temporal-spatial-context-contrast-weight
  --temporal-spatial-context-contrast-margin

scripts/208_run_temporal_spatial_context_gate.sh
  TEMPORAL_SPATIAL_CONTEXT_CONTRAST_WEIGHT
  TEMPORAL_SPATIAL_CONTEXT_CONTRAST_MARGIN
```

The contrast loss is:

```text
max(0, margin - (logp_context_on(answer) - logp_context_off(answer)))
```

A 1-step runtime smoke passed. A new full run is still required before any
claim:

```text
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm \
PYTHON_BIN=.venv/bin/python \
PYTHONPATH=src \
OUT_DIR=/mnt/nvme1n1p2/qtrm-local-checkpoints/temporal_spatial_context_probe/contrast_s240 \
EVAL_OUT=local_eval/temporal_spatial_context_contrast_gate.jsonl \
GATE_MD=docs/wiki/decisions/temporal-spatial-context-contrast-gate.md \
GATE_JSON=docs/wiki/decisions/temporal-spatial-context-contrast-gate-summary.json \
bash scripts/208_run_temporal_spatial_context_gate.sh
```
