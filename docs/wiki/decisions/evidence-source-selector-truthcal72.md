# Evidence Source Selector

Status: `accepted`

## Setup

- train cases: `data/filtered/memory_reasoning_synth_train_cases.jsonl`
- eval cases: `data/eval/memory_reasoning_heldout_expanded_72.jsonl`
- selected threshold: `0.74`

## Metrics

| Split | Case Success | F1 | Precision | Recall | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| eval | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |

## Boundary

This selector learns which retrieved source record contains the answer-bearing span. It should be applied as a span-logit mask while preserving the full workspace evidence context, not as pre-forward evidence pruning.

## Runtime Answer Result

The selector was then wired into `scripts/95_eval_memory_retrieval.py` as a
span-logit mask:

```text
full workspace evidence -> QTRM forward
learned source selector -> selected source text token mask
evidence span reader logits -> masked span argmax
boundary REVISE -> model answer decision
```

This is intentionally different from the rejected reliability source governor:
the full evidence context remains visible to the model, and only the final copied
answer span is constrained.

Command:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src uv run python scripts/95_eval_memory_retrieval.py \
  --config configs/qwen35_2b_4090_answer_decision_head_truthcal_s200.yaml \
  --checkpoint runs/qwen35_2b_4090_answer_decision_head_truthcal_s200/last.pt \
  --cases data/eval/memory_reasoning_heldout_expanded_72.jsonl \
  --mode qtrm_residual_with_evidence \
  --jsonl-out docs/wiki/decisions/evidence-source-selector-span-mask-truthcal72-thr065-records.jsonl \
  --answer-channel evidence_span_copy --answer-revision evidence_span_boundary \
  --evidence-source-selector-checkpoint runs/evidence_source_selector_truthcal_s500/selector.pt \
  --evidence-source-selector-mode span_mask \
  --truth-gate --model-answer-decision --model-answer-decision-threshold 0.65 \
  --evidence-mode all --evidence-injection workspace --retrieval-top-k 4 \
  --evidence-span-max-tokens 16 --evidence-span-no-answer-threshold 0.5 \
  --short-answer-governor --suppress-visible-reasoning-tokens
```

Result:

```text
records: docs/wiki/decisions/evidence-source-selector-span-mask-truthcal72-thr065-records.jsonl
answer accuracy: 71 / 72 = 0.9861
unknown negatives: 24 / 24
remaining miss: synthetic-authority-vault-0102, no_answer false negative
```

For comparison:

| Path | Accuracy |
| --- | ---: |
| in-model answer decision only | 62 / 72 = 0.8611 |
| boundary REVISE | 67 / 72 = 0.9306 |
| reliability source pruning | 48 / 72 = 0.6667 |
| learned source span-mask + boundary REVISE | 71 / 72 = 0.9861 |

## Remaining Failure

The only remaining miss has the correct selected source
`signed_garnet_vault.md`, but the span reader emits `no_answer` with
`no_answer_prob=0.9707`. Raising the no-answer threshold to `0.99` did not
change the result. The next fix is not source selection; it is recalibrating the
span reader's no-answer head under source-masked decoding.
