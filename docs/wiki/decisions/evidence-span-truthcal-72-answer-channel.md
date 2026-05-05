# Evidence Span Truthcal 72 Answer Channel

Status: useful answer-formation component with remaining false positives,
2026-05-01.

## Run

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src uv run python \
  scripts/95_eval_memory_retrieval.py \
  --config configs/qwen35_2b_4090_evidence_span_reader_truthcal_s300.yaml \
  --checkpoint runs/qwen35_2b_4090_evidence_span_reader_truthcal_s300/last.pt \
  --cases data/eval/memory_reasoning_heldout_expanded_72.jsonl \
  --mode qtrm_residual_with_evidence \
  --mode qtrm_evidence_span_reader_off_with_evidence \
  --mode donor_only_with_evidence \
  --jsonl-out docs/wiki/decisions/evidence-span-truthcal-72-answer-channel-records.jsonl \
  --answer-channel evidence_span_copy \
  --truth-gate \
  --evidence-mode all \
  --evidence-injection workspace \
  --retrieval-top-k 4 \
  --evidence-span-max-tokens 16 \
  --evidence-span-no-answer-threshold 0.5 \
  --short-answer-governor \
  --suppress-visible-reasoning-tokens
```

## Result

```text
qtrm_residual_with_evidence: 49 / 72 = 0.6806
qtrm_evidence_span_reader_off_with_evidence: 24 / 72 = 0.3333
donor_only_with_evidence: 24 / 72 = 0.3333
retrieved_target_rate: 1.0000
all_targets_retrieved_rate: 1.0000
```

## Interpretation

This is a real answer-formation signal. The span/truth answer channel more than
doubles donor-only accuracy on this held-out set, and disabling the span reader
collapses back to donor/UNKNOWN behavior.

It is not solved:

- QTRM still has false positives on redaction and missing-authority cases;
- the truth gate often allows spans with only moderate missing probability;
- multi-hop and conflict categories remain much weaker than abstention cases.

The next architecture should combine the stable transition-state action loop
with a verifier-controlled answer-decision loop, rather than assuming the
fixed `ANSWER` step is enough.
