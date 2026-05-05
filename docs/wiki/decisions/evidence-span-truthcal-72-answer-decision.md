# Evidence Span Truthcal 72 Answer Decision

Status: `accepted runtime integration`, 2026-05-02.

## Setup

This run wires the learned answer-decision head into the actual
`scripts/95_eval_memory_retrieval.py` runtime path. It is not only a replay of
recorded telemetry.

```text
config: configs/qwen35_2b_4090_evidence_span_reader_truthcal_s300.yaml
qtrm checkpoint: runs/qwen35_2b_4090_evidence_span_reader_truthcal_s300/last.pt
answer decision checkpoint: runs/qwen35_2b_4090_answer_decision_head_truthcal_train144_eval72/last.pt
cases: data/eval/memory_reasoning_heldout_expanded_72.jsonl
records: docs/wiki/decisions/evidence-span-truthcal-72-answer-decision-records.jsonl
answer_channel: evidence_span_copy
truth_gate: true
evidence_injection: workspace
retrieval_top_k: 4
threshold: 0.63
```

Command:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src uv run python scripts/95_eval_memory_retrieval.py \
  --config configs/qwen35_2b_4090_evidence_span_reader_truthcal_s300.yaml \
  --checkpoint runs/qwen35_2b_4090_evidence_span_reader_truthcal_s300/last.pt \
  --cases data/eval/memory_reasoning_heldout_expanded_72.jsonl \
  --mode qtrm_residual_with_evidence \
  --jsonl-out docs/wiki/decisions/evidence-span-truthcal-72-answer-decision-records.jsonl \
  --answer-channel evidence_span_copy --truth-gate \
  --answer-decision-checkpoint runs/qwen35_2b_4090_answer_decision_head_truthcal_train144_eval72/last.pt \
  --evidence-mode all --evidence-injection workspace --retrieval-top-k 4 \
  --evidence-span-max-tokens 16 --evidence-span-no-answer-threshold 0.5 \
  --short-answer-governor --suppress-visible-reasoning-tokens
```

## Result

| Metric | Value |
| --- | ---: |
| baseline span/truth answer channel | 49 / 72 = 0.6806 |
| runtime answer-decision path | 62 / 72 = 0.8611 |
| blocked candidates | 14 |
| expected-unknown false positives | 0 |
| human audit cases | 10 / 72 = 0.1389 |
| retrieved target rate | 72 / 72 = 1.0000 |
| all targets retrieved rate | 72 / 72 = 1.0000 |

By task family:

| Family | Accuracy |
| --- | ---: |
| abstention | 24 / 24 = 1.0000 |
| conflict | 20 / 24 = 0.8333 |
| multi_hop | 18 / 24 = 0.7500 |

## Interpretation

The previous static threshold gate failed because it over-abstained on
held-out cases. The learned decision head is stricter in the useful direction:
it blocks unsupported answer candidates that should become `UNKNOWN` without
blocking the positive held-out answers in this run.

This closes the first runtime wiring step for:

```text
PROPOSE_ANSWER -> VERIFY_ANSWER -> ANSWER_DECISION -> ANSWER | ABSTAIN
```

It still does not solve positive wrong answers. The remaining 10 misses are
mostly conflict or multi-hop answer-selection errors where the candidate should
be revised or searched again, not merely blocked to `UNKNOWN`.

## Next Step

Move from post-hoc runtime decision to an ablatable in-model QTRM
answer-decision head:

```text
span/truth telemetry + latent state
-> QTRM answer decision head
-> ANSWER | ABSTAIN | REVISE | SEARCH_MORE
```

Accept the next stage only if:

- it preserves or beats `62 / 72`;
- verifier-off and decision-head-off ablations drop;
- `REVISE` or `SEARCH_MORE` reduces the remaining positive wrong answers.
