# Canonical Decision Token S120 Result

Status: rejected as canonical answer architecture, 2026-05-02.

## Failure Ledger

Failure:
decision-token SFT improved training/logprob causality but did not improve
greedy answer behavior.

Evidence:

```text
data: data/filtered/memory_reasoning_canonical_decision_tokens.jsonl
rows: 144
decision targets: ANSWER=96, ABSTAIN=48
checkpoint: runs/qwen35_2b_4090_canonical_decision_tokens_s120/last.pt
```

Training reached a positive canonical logprob margin:

```text
final canonical_causal_margin: 0.0725
```

Direct residual-logprob probe confirmed that the latent path changes answer
logprobs on the decision-token data:

```text
script: scripts/169_probe_canonical_causal_margin.py
data: data/filtered/memory_reasoning_canonical_decision_tokens.jsonl
records: 8
core_off margin:      0.0792 causal=true
workspace_off margin: 0.0933 causal=true
```

But the canonical greedy answer gate rejected the checkpoint:

```text
report: docs/wiki/decisions/canonical-decision-tokens-s120-qscale030-answer-gate-8.md
QTRM residual: 5 / 8
donor-only:    5 / 8
workspace_off: 5 / 8
core_off:      5 / 8
status: rejected
```

Known limitation class:
logprob-level latent causality without greedy answer-level causality.

Root architecture hypothesis:
the QTRM residual is too weak and too late in the fused donor decoding path.
It can slightly reshape label logprobs but cannot reliably change selected
tokens or improve answer decisions.

Could the big structure be wrong?:
yes. Training explicit decision text is not enough if the final user-facing
answer path still lets donor logits dominate.

Information path needed:

```text
canonical prompt tokens
-> latent workspace/core/verifier state
-> forced answer decision policy
-> final greedy answer tokens
```

Current information path:

```text
canonical prompt tokens
-> donor logits dominate
-> weak QTRM residual nudges logprobs
-> same greedy answer as donor/core-off/workspace-off
```

Local fix budget already spent:

- core-to-text bridge;
- answer bottleneck;
- safe residual gate;
- selective residual gate;
- canonical decision-token SFT.

Recommended candidate:
do not add another margin-only run. Move the answer decision into a causal
forward path that can force final answer behavior while preserving the donor as
default.

## SSOT/KISS/DRY Check

SSOT source:
one visible prompt stream. Retrieval/evidence is compiled into `prompt`; no
hidden `workspace_context` field is used in the decision-token rows.

Smallest path:
decision-token SFT plus existing QTRM residual/core. This was intentionally
small and was rejected.

Duplicated logic avoided:
no span-copy answer channel, no MemoryOS sidecar answer path, no post-hoc
decision script in the canonical claim.

Canonical gate:
`scripts/166_run_canonical_ssot_answer_gate.sh` with
`--evidence-injection ssot` and `--answer-channel greedy`.

## Evaluation Correction

The first gate run used `scripts/166_run_canonical_ssot_answer_gate.sh` default
`QTRM_LOGITS_SCALE=0.10`, while the config set `qtrm_logits_scale: 0.30`.
That was an evaluation mismatch.

The generic gate runner no longer hardcodes a QTRM scale override; it uses the
config unless `QTRM_LOGITS_SCALE` is explicitly set. The decision-token runner
sets the experiment-specific scale:

```text
scripts/166_run_canonical_ssot_answer_gate.sh: no hardcoded QTRM scale
scripts/171_run_canonical_decision_token_train.sh:
QTRM_LOGITS_SCALE=0.30
```

The corrected `0.30` gate still rejected the checkpoint.

## Next Architecture

Ranked candidates:

1. Causal answer-policy governor inside the model forward path.
   Train `ALLOW | ABSTAIN | REVISE | SEARCH_MORE` as model outputs, but make
   the final answer logits condition on this state, not only append text labels.

2. Forced workspace answer bottleneck v2.
   Reuse the narrow causal success of `answer_bottleneck`, but make it
   selective: donor remains default, and QTRM may override only when the
   verifier/decision state predicts a supported improvement.

3. Decision-token prompt mode as a diagnostic only.
   Let the model visibly emit `Verify`, `Decision`, and `Answer`, then score the
   `Answer:` line. This can test process learning, but it is not the canonical
   short-answer LLM path.

Reject if:

- donor-only matches full QTRM;
- core/workspace-off keeps the same hit count;
- only logprob probes pass while greedy completions stay unchanged.

## Hidden Answer-Decision Head Probe

Follow-up:
train only a hidden-state in-model answer-decision head while keeping the same
SSOT visible prompt path.

```text
config: configs/qwen35_2b_4090_canonical_ssot_hidden_answer_decision_s200.yaml
data: data/filtered/memory_reasoning_answer_decision_truthcal_train144.jsonl
init: runs/qwen35_2b_4090_canonical_decision_tokens_s120/last.pt
checkpoint: runs/qwen35_2b_4090_canonical_ssot_hidden_answer_decision_s200/last.pt
trainable params: 513
```

Result:

```text
report: docs/wiki/decisions/canonical-ssot-hidden-answer-decision-s200-gate-8.md
status: rejected
qtrm_residual_with_evidence: 5 / 8
donor_only_with_evidence:    5 / 8
workspace_off_with_evidence: 5 / 8
core_off_with_evidence:      5 / 8
```

The head did not block any bad answers:

```text
qtrm_residual_with_evidence block count: 0 / 8
block probability range: 0.00255..0.00280
```

Interpretation:
with the current frozen representation, answer validity is not linearly
available to a hidden-only decision head. The previously accepted decision
result depended on rich answer-channel telemetry/features. To make this
canonical, the model must either learn those verifier features internally or
make final logits pass through a stronger verifier-conditioned answer policy.
