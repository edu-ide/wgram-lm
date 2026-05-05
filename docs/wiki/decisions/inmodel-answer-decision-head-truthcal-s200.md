# In-Model Answer Decision Head Truthcal S200

Status: `accepted in-model causal gate`, 2026-05-02.

## Setup

This moves the answer-decision gate from a post-hoc sidecar into the QTRM
checkpoint as an ablatable model output:

```text
config: configs/qwen35_2b_4090_answer_decision_head_truthcal_s200.yaml
checkpoint: runs/qwen35_2b_4090_answer_decision_head_truthcal_s200/last.pt
train mix: data/filtered/memory_reasoning_answer_decision_truthcal_train144.jsonl
eval cases: data/eval/memory_reasoning_heldout_expanded_72.jsonl
records: docs/wiki/decisions/evidence-span-truthcal-72-inmodel-answer-decision-records.jsonl
```

Architecture correction:

- hidden-only in-model decision head failed: `49 / 72`, no useful blocks;
- linear telemetry projection was weak: `51 / 72`, only 2 useful blocks;
- feature MLP with raw telemetry and class-balanced bootstrap is accepted.

The accepted path uses answer-channel telemetry as the model-visible decision
signal. Full mode uses the feature MLP logit; `qtrm_answer_decision_features_off`
falls back to the hidden-only decision path, so the causal contribution is
measurable.

## Commands

```bash
PYTHONPATH=src uv run python scripts/164_bootstrap_answer_decision_feature_head.py \
  --config configs/qwen35_2b_4090_answer_decision_head_truthcal_s200.yaml \
  --init-checkpoint runs/qwen35_2b_4090_evidence_span_reader_truthcal_s300/last.pt \
  --train-jsonl data/filtered/memory_reasoning_answer_decision_truthcal_train144.jsonl \
  --out-checkpoint runs/qwen35_2b_4090_answer_decision_head_truthcal_s200/last.pt \
  --epochs 400 --lr 3.0e-3
```

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src uv run python scripts/95_eval_memory_retrieval.py \
  --config configs/qwen35_2b_4090_answer_decision_head_truthcal_s200.yaml \
  --checkpoint runs/qwen35_2b_4090_answer_decision_head_truthcal_s200/last.pt \
  --cases data/eval/memory_reasoning_heldout_expanded_72.jsonl \
  --mode qtrm_residual_with_evidence \
  --mode qtrm_answer_decision_features_off_with_evidence \
  --mode qtrm_answer_decision_off_with_evidence \
  --jsonl-out docs/wiki/decisions/evidence-span-truthcal-72-inmodel-answer-decision-records.jsonl \
  --answer-channel evidence_span_copy --truth-gate --model-answer-decision \
  --evidence-mode all --evidence-injection workspace --retrieval-top-k 4 \
  --evidence-span-max-tokens 16 --evidence-span-no-answer-threshold 0.5 \
  --short-answer-governor --suppress-visible-reasoning-tokens
```

## Result

| Mode | Accuracy |
| --- | ---: |
| full in-model answer decision | 62 / 72 = 0.8611 |
| feature-off ablation | 49 / 72 = 0.6806 |
| decision-head-off ablation | 49 / 72 = 0.6806 |

Additional gates:

| Metric | Value |
| --- | ---: |
| blocked candidates | 14 |
| block improved | 13 |
| block harmed | 0 |
| remaining expected-unknown false positives | 0 |
| train bootstrap accuracy | 0.9792 |
| retrieved target rate | 72 / 72 = 1.0000 |

Decision probability distribution in full mode:

```text
min / p25 / p50 / p75 / p90 / max
1.75e-13 / 1.37e-06 / 0.0078 / 0.1733 / 0.9774 / 1.0000
```

## Interpretation

Accepted, but with a precise limitation: the causal improvement comes from
answer-channel telemetry inside the QTRM checkpoint, not yet from latent hidden
state alone. This is still useful because it turns the prior sidecar gate into
an ablatable model path and proves that `ANSWER_DECISION` can be part of the
model's forward contract.

The failed hidden-only attempt is evidence against relying on a final pooled
state for truth/abstention. The model needs explicit verifier telemetry, raw
feature scale, and balanced positive UNKNOWN-block training.

## Next Step

The remaining failures are not UNKNOWN false positives. They are positive
wrong answers in conflict or multi-hop cases. The next architecture step is:

```text
PROPOSE_ANSWER
-> VERIFY_ANSWER
-> ANSWER_DECISION
-> ANSWER | ABSTAIN | REVISE | SEARCH_MORE
```

The next gate should add a learned `REVISE` or `SEARCH_MORE` branch and accept
only if it improves the remaining 10 held-out misses without reducing the
62/72 answer-decision score.
