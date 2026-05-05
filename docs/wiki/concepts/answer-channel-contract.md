# Answer Channel Contract

Status: active runtime/eval decision, 2026-05-01.

## Root Architecture Claim

QTRM should not rely on a post-hoc verifier to clean up visible reasoning
leakage. The generator must have an explicit visible-answer channel contract at
prompt formatting and decoding time.

## Why This Became A Root-Structure Issue

The generation-verifier reranker failed on the first 8-prompt, 3-candidate
smoke. With format-aware labels, baseline first-candidate quality was `0.625`,
but verifier-reranked quality dropped to `0.250`; oracle quality was `0.750`.

That means the candidate pool contained good answers, but the current
post-hoc verifier did not reliably select them. The bigger failure was that the
generator was producing visible `<think>`-style text before the answer. A
reranker can hide this sometimes, but it does not fix the generator's channel
contract.

## Prior

Official Qwen documentation describes thinking and non-thinking modes. In Qwen
chat-template usage, `enable_thinking=False` disables thinking mode, and Qwen
also supports `/think` and `/no_think` style soft switches.

Local source page:
[Generation Verifier And Reranking](../sources/generation-verifier-reranking.md).

## Minimal Runtime Probe

Added eval/inference-time controls:

```text
scripts/92_eval_qtrm_logits.py --suppress-visible-reasoning-tokens
scripts/92_eval_qtrm_logits.py --no-repeat-ngram-size 2
scripts/92_eval_qtrm_logits.py --answer-contract direct
scripts/95_eval_memory_retrieval.py --suppress-visible-reasoning-tokens
scripts/95_eval_memory_retrieval.py --no-repeat-ngram-size 2
SUPPRESS_VISIBLE_REASONING=1 bash scripts/90_infer_with_donor.sh ...
NO_REPEAT_NGRAM_SIZE=2 bash scripts/90_infer_with_donor.sh ...
ANSWER_CONTRACT=direct bash scripts/90_infer_with_donor.sh ...
```

The accepted runtime guard is intentionally narrow:

- suppress tokenizer ids for `<think>` and `</think>`;
- optionally block repeated completion n-grams during decoding;
- do not ban normal step-by-step answers, because user prompts may explicitly
  ask for stepwise reasoning.

The `direct` prompt contract remains experimental. On the 20-prompt smoke it
reduced visible reasoning and repetition, but it caused new instruction drift in
several completions. Therefore it is not accepted as a default.

## Result

8 prompts, 3 sampled candidates, no suppression:

| Metric | Value |
| --- | ---: |
| visible reasoning rate | 0.458 |
| repeat failure rate | 0.167 |
| clean rate | 0.417 |
| group visible reasoning rate | 0.625 |
| group has clean candidate rate | 0.750 |

8 prompts, 1 sampled candidate, `<think>` suppression:

| Metric | Value |
| --- | ---: |
| visible reasoning rate | 0.000 |
| repeat failure rate | 0.000 |
| clean rate | 1.000 |

20 prompts, 1 sampled candidate, drift-aware format detector:

| Mode | Visible Reasoning | Repeat Failure | Answer Drift | Clean / Quality-Like |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.150 | 0.300 | 0.200 | 0.550 |
| suppress `<think>` + no-repeat-2 | 0.000 | 0.000 | 0.150 | 0.850 |
| direct prompt contract + suppress + no-repeat-2 | 0.000 | 0.000 | 0.300 | 0.700 |

Interpretation:

- answer-channel suppression directly fixes visible `<think>` leakage on this
  smoke;
- no-repeat-2 fixes n-gram repetition on this short smoke, but it is a decoding
  guard and must be stress-tested on longer answers;
- direct prompt suffixes can create instruction drift with the raw base donor;
- post-hoc verifier reranking is not accepted as an architecture improvement
  until it beats baseline candidate choice under format-aware labels.

MemoryOS follow-up:
the same suppression/no-repeat controls were added to
`scripts/95_eval_memory_retrieval.py`. On the logical-causal bottleneck quick
gate, guarded decoding still scored `0/4` and copied evidence/prompt text. That
means answer-channel guards are necessary but not sufficient for MemoryOS; the
model needs a trained short-answer extraction path.

## Architecture Direction

Ranked next steps:

1. **Answer-channel contract**:
   use official chat-template/non-thinking controls where available; in the
   current raw base-donor path, suppress visible reasoning markers during QTRM
   decoding.
2. **Prefix-time degeneration controller**:
   use no-repeat n-gram suppression as a runtime guard, then replace it with a
   learned prefix-time controller if larger probes justify it.
3. **Verifier reranker**:
   keep as diagnostic/rerank-only until it improves held-out candidate
   selection under format-aware labels.

Acceptance gate:

- visible reasoning rate stays near zero on held-out prompts;
- repeat failure does not regress;
- answer/instruction drift does not rise;
- output quality improves over raw donor/QTRM sampling;
- if verifier reranking is used, it must beat baseline candidate choice and
  approach oracle candidate quality.
