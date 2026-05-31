# Token Superposition Training

## Source

```text
paper:
  Efficient Pre-Training with Token Superposition

arxiv:
  https://arxiv.org/abs/2605.06546

submitted:
  2026-05-07

authors:
  Bowen Peng, Théo Gigant, Jeffrey Quesnelle

local pdf:
  references/papers/2605.06546-efficient-pre-training-with-token-superposition.pdf
```

## Core Idea

Token-Superposition Training (TST) is not a new inference architecture. It is a
two-phase pretraining recipe:

```text
phase 1:
  group contiguous tokens into fixed-size bags
  average embeddings inside each input bag
  predict the next non-overlapping bag of tokens
  train with equal-weight multi-hot cross entropy

phase 2:
  return to normal next-token training
  keep the same model architecture, tokenizer, optimizer, and data
```

The paper reports that this can improve data throughput per FLOP and reach equal
loss faster, with a reported up-to-2.5x total pretraining-time reduction at the
10B A1B scale under their setting.

## QTRM Interpretation

TST fits QTRM-native as a language-pretraining efficiency tool, not as a
replacement for the recursive core.

Useful for:

```text
QTRM-native language bootstrap
larger English/Korean corpus training
faster low-cost pretraining before reasoning gates
multi-token future supervision without visible CoT
```

Not sufficient for:

```text
claiming raw reasoning improvement
replacing TRM/QTRM latent recursion
fixing MSA/long-memory by itself
proving broad language ability without heldout generation gates
```

## Local Implementation Status

Implemented primitive:

```text
src/wgram_lm/tst.py
  next_token_bags
  superpose_embeddings
  multi_hot_cross_entropy

tests:
  tests/test_tst.py
```

This is wired into the native language bootstrap as an optional phase:

```text
scripts/354_train_qtrm_native_language_bootstrap.py
  --tst-phase-steps
  --tst-bag-size

scripts/335_train_qtrm_native_etd_probe.py
  NativeQTRMETDLM.forward_embeddings
```

First smoke:

```text
checkpoint:
  local_eval/qtrm_native_language_bootstrap_bilingual_bpe16k_d192_external4500_tst_b4_s3600_20260515/last.pt

bag_size:
  4

result:
  rejected

reason:
  bootstrap accepted, but bilingual/broad heldout gates rejected and depth4 loss
  was slightly worse than the non-TST external4500 baseline.
```

Follow-up sweeps:

```text
bag_size=2, TST=300, CE recovery=2400:
  checkpoint:
    local_eval/qtrm_native_language_bootstrap_bilingual_bpe16k_d192_external4500_tst_b2_s3600_20260515/last.pt

  result:
    rejected

  depth4_loss:
    1.6091

  heldout:
    bilingual rejected on semantic relevance
    broad unseen rejected on semantic relevance

bag_size=2, TST=150, CE recovery=2550:
  checkpoint:
    local_eval/qtrm_native_language_bootstrap_bilingual_bpe16k_d192_external4500_tst_b2_short_s3600_20260515/last.pt

  result:
    rejected before promotion

  depth4_loss:
    1.6811

  reason:
    worse than the non-TST baseline and worse than the longer b2 sweep.

current canonical baseline:
  local_eval/qtrm_native_language_bootstrap_bilingual_bpe16k_d192_external4500_s3600_20260515/last.pt

baseline depth4_loss:
  1.5977
```

## Required QTRM-Native Experiment

Next concrete experiment:

```text
qtrm_native_language_tst_phase_smoke

compare:
  baseline:
    normal byte-BPE 16k language bootstrap

  TST:
    phase 1 superposed bag training with bag_size in {2, 4}
    phase 2 recovery using normal next-token CE

accept if:
  bilingual core gate stays accepted
  broad unseen gate stays accepted
  direct English/Korean inference remains clean
  depth4/core-on still beats depth0/thinking-block-off
  equal or lower wall-clock/token budget reaches the same loss

reject if:
  TST improves loss but harms greedy answer quality
  TST creates Assistant/User marker leakage
  TST removes causal depth gain
```

Current local decision:

```text
Do not promote TST b4, b2, or b2-short. Keep TST as an implemented experimental
objective, but the current small QTRM-native language scaffold should use the
non-TST external4500 checkpoint as canonical.

Further TST work is justified only as a deliberate training-efficiency study
with a larger budget or a modified objective. It is not the next shortest path
to better English/Korean language quality.
```

## Boundary

TST is canonical only as an offline training objective. It must not add a
runtime donor, hidden retrieval path, sidecar solver, or visible chain-of-thought
target. The final model must still be:

```text
prompt tokens -> native embeddings/backbone -> mandatory recurrent core
-> native decoder/readout -> LM logits -> autoregressive text
```
