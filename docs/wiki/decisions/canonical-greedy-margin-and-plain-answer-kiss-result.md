# Canonical Greedy-Margin And Plain-Answer KISS Result

Status: rejected, 2026-05-02.

This page records two canonical SSoT answer-path experiments after the
decision-token result. Both were designed to test whether the QTRM latent
workspace/core path can become causally necessary for ordinary greedy
autoregressive answers without using span-copy, hidden workspace evidence, or a
post-hoc answer channel.

## Experiment 1: Greedy-Token Margin On Decision Tokens

Artifacts:

```text
config: configs/qwen35_2b_4090_canonical_greedy_margin_s120.yaml
runner: scripts/172_run_canonical_greedy_margin_train.sh
data: data/filtered/memory_reasoning_canonical_decision_tokens.jsonl
init: runs/qwen35_2b_4090_canonical_decision_tokens_s120/last.pt
checkpoint: runs/qwen35_2b_4090_canonical_greedy_margin_s120/last.pt
report: docs/wiki/decisions/canonical-greedy-margin-s120-answer-gate-8.md
summary: docs/wiki/decisions/canonical-greedy-margin-s120-answer-gate-8-summary.json
```

Result:

```text
status: rejected
donor_only_with_evidence: 5/8
qtrm_residual_with_evidence: 5/8
qtrm_core_to_text_off_with_evidence: 6/8
causal modes: none
exact match: 0/8
human audit: 8/8
```

Failure:
the greedy-token margin can change local training argmax behavior, but it did
not make the latent workspace/core path necessary on the held-out canonical
answer gate. `core_to_text_off` improved the hit count, which is direct evidence
that the current core-to-text path is still harmful or leaky in this setup.

Additional issue:
the training rows used `Verify: ... Decision: ... Answer: ...`, while the
canonical answer gate expects a short greedy `Answer:` completion. That contract
mismatch is enough to create source/evidence continuation leakage even when the
retrieved evidence is correct.

## Experiment 2: Plain-Answer KISS Reset

Artifacts:

```text
data builder: scripts/173_build_canonical_plain_answer_data.py
data: data/filtered/memory_reasoning_canonical_plain_answer.jsonl
config: configs/qwen35_2b_4090_canonical_plain_answer_kiss_s120.yaml
runner: scripts/174_run_canonical_plain_answer_kiss_train.sh
init: runs/qwen35_2b_4090_donor_residual_s010_1000/last.pt
checkpoint: runs/qwen35_2b_4090_canonical_plain_answer_kiss_s120/last.pt
report: docs/wiki/decisions/canonical-plain-answer-kiss-s120-answer-gate-8.md
summary: docs/wiki/decisions/canonical-plain-answer-kiss-s120-answer-gate-8-summary.json
```

Result:

```text
status: rejected
donor_only_with_evidence: 5/8
qtrm_residual_with_evidence: 4/8
qtrm_workspace_off_with_evidence: 5/8
qtrm_core_off_with_evidence: 5/8
causal modes: none
```

Failure:
the plain-answer contract reduced the format mismatch, but the residual QTRM
path still degraded donor behavior. Workspace/core-off ablations matched or
improved the full model, so this cannot support the claim that QTRM is doing
causal latent reasoning on the final answer.

Observed generation failures:

- empty `Answer:` on some cases;
- `비공개입니다.` instead of canonical `UNKNOWN` on an expected-unknown case;
- donor-correct multi-hop answers harmed by the residual path.

## Failure Ledger

```text
Failure:
  Repeated local losses changed training metrics but did not create held-out
  answer-level latent causality.

Evidence:
  Greedy-margin QTRM tied donor at 5/8 and core_to_text_off improved to 6/8.
  Plain-answer KISS QTRM fell to 4/8 while workspace/core-off rose to 5/8.

Known limitation class:
  Donor-fused residual bypass. The donor/prompt path can answer without using
  the claimed latent workspace/core path, and the residual can still harm donor
  correct outputs.

Root architecture hypothesis:
  The final answer signal is not forced through a causally necessary latent
  evidence/workspace state.

Could the big structure be wrong?:
  Yes. The current residual-fusion path may be a useful adapter probe, but it is
  not yet a proof of an actual reasoning core.

Information path needed:
  canonical prompt tokens -> latent evidence/reasoning state -> answer renderer
  -> greedy autoregressive answer.

Current information path:
  canonical prompt tokens -> donor logits + optional QTRM residual logits
  -> greedy answer. The donor path can bypass the QTRM latent path.

Local fix budget already spent:
  CE, student LM, repeat guard, canonical causal margin, decision tokens,
  selective/safe residual gates, greedy-token margin, and plain-answer reset.

Likely component:
  answer renderer / residual fusion boundary.

Alternative explanations:
  data too small, short training, bad heldout split, answer-template mismatch.
  These do not explain why component-off ablations match or improve full QTRM.

Prior work to check:
  verifier-controlled generation, latent-token reasoning, recurrent transition
  state controllers, mixture-of-depth/halting, and evidence bottleneck models.

Architecture candidates:
  A. forced latent answer renderer;
  B. donor-preserving residual governor with verifier-conditioned write access;
  C. recurrent transition-state planner feeding a plain answer renderer.

Recommended candidate:
  Start with A+B: force only the answer-token residual path through a latent
  answer state, but let a learned governor choose zero-residual fallback when
  donor is already correct.

Smallest next experiment:
  Train an answer renderer on plain-answer rows where QTRM residual logits for
  answer positions are produced only from latent state. Evaluate donor, full,
  latent-off, governor-off, and renderer-off modes on the same 8-case gate.

Acceptance gate:
  full QTRM > donor-only;
  workspace/core/renderer-off reduce held-out accuracy or successful examples;
  no empty Answer:;
  no source/evidence continuation;
  expected-unknown cases answer UNKNOWN.

Kill criterion:
  If full QTRM does not beat donor or latent-off matches/improves full QTRM,
  reject the candidate and stop local loss tuning.
```

## Decision

Do not run another margin-only or threshold-only local tuning experiment on the
current donor-fused residual path.

The next accepted candidate must change the causal answer path, not only the
loss weight. The simplest acceptable direction is:

```text
canonical prompt tokens
-> donor hidden states
-> QTRM latent evidence/reasoning state
-> verifier/governor decides whether QTRM may write answer-token residuals
-> greedy autoregressive answer
```

The donor remains useful for language preservation, but the QTRM claim is only
valid if disabling the latent state, renderer, or governor measurably harms the
held-out answer behavior.
