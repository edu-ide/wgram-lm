# Generation Verifier And Reranking Sources

Accessed: 2026-05-01.

This page tracks prior work used for QTRM generation-verifier and reranking
experiments. The current QTRM implementation is inspired by these mechanisms;
it is not an official implementation of any single paper.

## Degenerate Repetition

### Neural Text Generation with Unlikelihood Training

- URL: https://arxiv.org/abs/1908.04319
- Problem: maximum-likelihood training can assign too much probability to
  repeated or overly frequent tokens.
- Relevant mechanism: add token/sequence-level penalties for known bad
  repetitions.
- QTRM mapping: keep `loss_repeat_unlikelihood_weight` as a narrow guard, but
  do not rely on it as the whole solution. QTRM's current failure also needs an
  output-level verifier because repeated generations can be prompt/category
  dependent.

### A Contrastive Framework for Neural Text Generation

- URL: https://arxiv.org/abs/2202.06417
- Problem: representation-space degeneration can lead to unnatural repeated
  generations.
- Relevant mechanism: SimCTG contrastive training and contrastive search.
- QTRM mapping: useful as a future decoding/representation axis. It is a larger
  intervention than the current verifier split, because it affects generation
  scoring directly.

## Verifier-Guided Generation

### FUDGE: Controlled Text Generation With Future Discriminators

- URL: https://arxiv.org/abs/2104.05218
- Code: https://github.com/yangkevin2/naacl-2021-fudge-controlled-generation
- Problem: control generation with a discriminator that estimates whether a
  partial sequence will satisfy a future attribute.
- Relevant mechanism: discriminator-guided inference can steer generation
  without retraining the base LM.
- QTRM mapping: the generation verifier is a weaker first step. It scores full
  candidate completions now; a FUDGE-style future discriminator would later
  score partial prefixes during decoding.

## Candidate Reranking

### Lightweight Reranking for Language Model Generations

- URL: https://arxiv.org/abs/2307.06857
- Problem: selecting the best sample from multiple generations can improve
  generation quality.
- Relevant mechanism: rerank generated candidates with a lightweight scorer.
- QTRM mapping: before using a hard decoding gate, test the generation verifier
  as a candidate reranker on multiple QTRM completions.

### Reranking Laws for Language Generation

- URL: https://arxiv.org/abs/2409.07131
- Code: https://github.com/deep-spin/reranking-laws
- Problem: understand generator-reranker systems and how error changes with
  candidate count and reranker quality.
- Relevant mechanism: even imperfect rerankers can help when candidate diversity
  and reranker signal are sufficient.
- QTRM mapping: the next acceptance gate should report both verifier quality and
  candidate-rerank quality. A verifier with weak precision can still damage
  generation if used as a hard gate.

## Current Decision

Ranked architecture candidates:

| Rank | Candidate | Why | Risk |
| ---: | --- | --- | --- |
| 1 | Full-candidate verifier reranker | Smallest falsifiable step; matches current coda-text verifier head. | Requires candidate diversity and calibrated thresholds. |
| 2 | FUDGE-style prefix verifier | More direct generation control. | More invasive; partial-prefix labels are not built yet. |
| 3 | SimCTG/contrastive search path | Addresses representation degeneration. | Larger architecture/decoding change; may obscure whether QTRM verifier works. |

Current choice: keep the coda-text generation verifier as probe-only, add
train/calibration/holdout split tooling, and evaluate calibrated holdout before
any reranker or decoding gate.

## Answer-Channel / Thinking-Mode Control

### Qwen Quickstart And Transformers Docs

- URL: https://qwen.readthedocs.io/en/stable/getting_started/quickstart.html
- URL: https://qwen.readthedocs.io/en/stable/inference/transformers.html
- Problem: Qwen thinking-capable chat models may emit visible thinking content
  unless non-thinking mode is requested.
- Relevant mechanism: official examples use chat-template controls such as
  `enable_thinking=False`; Qwen also documents `/think` and `/no_think` soft
  switches.
- QTRM mapping: QTRM should not treat visible `<think>` leakage as only a
  reranking problem. Prompt formatting and decoding should enforce a visible
  answer channel before verifier reranking is trusted.

### Qwen3 Official Repo Quickstart

- URL:
  https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/quickstart.md
- Problem: some Qwen variants distinguish thinking and non-thinking behavior.
- Relevant mechanism: official docs describe hard non-thinking control through
  chat-template arguments and note non-thinking-only instruct variants.
- QTRM mapping: when using a Qwen donor, prefer an explicit non-thinking
  generation contract for user-facing answers. If chat-template control is not
  available in the raw base-donor path, suppress visible thinking markers as a
  runtime guard and train/evaluate format-aware targets.
