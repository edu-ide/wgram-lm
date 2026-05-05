# Evidence Span Boundary REVISE Truthcal 72

Date: 2026-05-02

## Decision

Accepted as a narrow answer-renderer improvement:

`ANSWER_DECISION + evidence_span_copy + evidence_span_boundary REVISE`

Rejected as an architecture path:

`reliability evidence-source governor`

## Why

After the in-model answer-decision head reached 62/72, the remaining misses were
not mostly hallucination. Several were span-rendering errors where the reader
selected the right evidence region but stopped inside an atomic identifier:

- `a Cho` instead of `Sena Cho`
- `Ember-Badge-11` instead of `Ember-Badge-110`
- `Frost-Badge-11` instead of `Frost-Badge-111`
- `Garnet-Badge-11` instead of `Garnet-Badge-112`
- `Iris-Badge-11` instead of `Iris-Badge-113`

The fix adds a deterministic `REVISE` branch after span copy. It only expands a
copied workspace span when tokenization appears to have cut an ASCII atomic
identifier at the left or right boundary. It does not search new evidence and it
does not change UNKNOWN blocking.

## Implementation

```text
scripts/95_eval_memory_retrieval.py
  --answer-revision evidence_span_boundary
  --answer-revision-max-left-tokens
  --answer-revision-max-right-tokens

tests/test_memory_eval_script.py
  span boundary right expansion
  span boundary left expansion
  whitespace crossing guard
```

## Accepted Result

Command:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src uv run python scripts/95_eval_memory_retrieval.py \
  --config configs/qwen35_2b_4090_answer_decision_head_truthcal_s200.yaml \
  --checkpoint runs/qwen35_2b_4090_answer_decision_head_truthcal_s200/last.pt \
  --cases data/eval/memory_reasoning_heldout_expanded_72.jsonl \
  --mode qtrm_residual_with_evidence \
  --mode qtrm_answer_decision_features_off_with_evidence \
  --mode qtrm_answer_decision_off_with_evidence \
  --jsonl-out docs/wiki/decisions/evidence-span-truthcal-72-boundary-revision-ablation-records.jsonl \
  --answer-channel evidence_span_copy --answer-revision evidence_span_boundary \
  --truth-gate --model-answer-decision \
  --evidence-mode all --evidence-injection workspace --retrieval-top-k 4 \
  --evidence-span-max-tokens 16 --evidence-span-no-answer-threshold 0.5 \
  --short-answer-governor --suppress-visible-reasoning-tokens
```

| Mode | Accuracy |
| --- | ---: |
| full in-model + boundary revise | 67 / 72 = 0.9306 |
| feature-off ablation | 55 / 72 = 0.7639 |
| decision-head-off ablation | 55 / 72 = 0.7639 |

Boundary revision changed 6 spans. Five became correct; one was later blocked
to UNKNOWN by the in-model answer-decision head.

## Rejected Source-Governor Result

An experimental `--evidence-source-governor reliability` was tested to prefer
signed/current/latest sources and prune anonymous/stale/decoy records.

```text
records: docs/wiki/decisions/evidence-span-truthcal-72-source-governor-records.jsonl
full mode: 48 / 72 = 0.6667
verdict: rejected
```

The failure is important: pruning evidence before the learned answer-decision
path shifted the model out of its calibrated evidence distribution and caused
many positive conflict cases to become `UNKNOWN`. Source selection should become
a learned selector/verifier head or a trained reranker, not a brittle pruning
heuristic.

## Remaining Failure Class

After accepted boundary revision, remaining misses are:

- source/authority selection errors in conflict cases;
- span no-answer false negatives on some positive authority cases;
- one multi-hop bridge/source selection miss.

Next step:
train or distill a source-selector/verifier objective that ranks evidence
records before span selection, while keeping the full evidence set available
unless the selector has calibrated confidence.
