# Truth-Gated Evidence Answer Channel

Status: implementation wiring, 2026-05-01.

## Claim Boundary

Looped reasoning models and TRM-style recursive cores provide repeated latent
state updates. They do not automatically judge truth. In QTRM, truth judgment
must be an explicit answer-path contract:

```text
hidden workspace evidence
-> prompt-conditioned evidence span reader
-> span confidence floor
-> support/refute/missing/causal truth gate
-> copied answer span or Answer: UNKNOWN
```

The recursive core can become useful for this only if ablations show that
turning it off changes truth-gated answers. Until then, the current accepted
causal path is workspace evidence plus the evidence-span reader, not the core.

## Runtime Gate

`scripts/95_eval_memory_retrieval.py` now exposes:

```text
--truth-gate
--truth-support-threshold
--truth-causal-threshold
--truth-refute-threshold
--truth-missing-threshold
```

When `--truth-gate` is enabled, `evidence_span_copy` copies the selected span
only if:

```text
sigmoid(evidence_support_logits) >= support_threshold
sigmoid(evidence_causal_gate_logits) >= causal_threshold
sigmoid(evidence_refute_logits) < refute_threshold
sigmoid(evidence_missing_logits) < missing_threshold
```

Otherwise it returns `Answer: UNKNOWN` with
`answer_channel_meta.status=truth_gate_blocked`.

## Why This Change Matters

The previous confidence floor answered "did the span reader pick a strong
span?" This new gate asks a different question: "do the logical evidence heads
allow this span to become the answer?"

That distinction matters because workspace evidence is context engineering, not
truth by itself. Truth-sensitive answering needs a reasoned gate over support,
contradiction, missing evidence, and relevance.

## Current Risk

The heads are wired into the answer channel, but their calibration is not yet
proven. The gate is disabled by default in the runner so the previous accepted
confidence-gate result remains reproducible.

To test it:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src \
  TRUTH_GATE=1 \
  OUT=runs/eval/reasoning_safe_span_copy_truth_gate_16.jsonl \
  AUDIT_OUT=runs/eval/reasoning_safe_span_copy_truth_gate_16_audit.jsonl \
  bash scripts/153_run_reasoning_safe_span_copy_gate.sh
```

Accept only if the truth-gated run preserves or improves held-out accuracy and
the metadata shows sensible block reasons for refuted or missing evidence.
