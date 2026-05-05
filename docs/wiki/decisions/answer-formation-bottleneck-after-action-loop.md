# Answer Formation Bottleneck After Action Loop

Status: root-architecture bottleneck identified, 2026-05-01.

## Claim

The learned transition-state action loop is now trainable and causal, but it
does not by itself improve answer quality when it calls the same answer renderer
as the scripted baseline.

In short:

```text
stable retrieve -> verify -> answer control is necessary,
but not sufficient for better reasoning.
```

## Evidence

### Action-First Runtime Controller

Action-first strict runtime retraining fixed the controller instability:

```text
checkpoint: runs/qwen35_2b_4090_transition_state_controller_runtime_actionfirst_s200/last.pt
train sequences: 256
state_loss_weight: 0.2
strict_runtime_state_inputs: true
held-out action accuracy: 1.0000
held-out RETRIEVE_MEMORY: 72 / 72
held-out VERIFY_EVIDENCE: 72 / 72
held-out ANSWER: 72 / 72
zero_transition_state: 0.3333
transition_state_drop: 0.6667
state_prediction_binary_accuracy: 0.9913
gate: accepted
```

### Task-Level Answer Gate

When the stable controller is used for answer generation, the action loop no
longer produces accidental UNKNOWN answers. The learned loop matches the
scripted baselines instead of beating them:

```text
report: docs/wiki/decisions/learned-state-answer-loop-runtime-actionfirst-gate.md
learned_state_qtrm: 4 / 8 = 0.5000
scripted_qtrm_answer_channel: 4 / 8 = 0.5000
scripted_donor_answer_channel: 4 / 8 = 0.5000
state_off: 2 / 8 = 0.2500
action_success_rate: 1.0000
gate: rejected
failed: learned_state_does_not_beat_scripted_qtrm, learned_state_does_not_beat_scripted_donor
```

This falsifies the idea that action-policy learning alone is enough for answer
reward.

### Span/Truth Answer Channel

A separate evidence-span/truth-calibrated answer channel does improve answer
formation on the 72-case held-out set:

```text
config: configs/qwen35_2b_4090_evidence_span_reader_truthcal_s300.yaml
checkpoint: runs/qwen35_2b_4090_evidence_span_reader_truthcal_s300/last.pt
records: docs/wiki/decisions/evidence-span-truthcal-72-answer-channel-records.jsonl
answer_channel: evidence_span_copy
truth_gate: true
evidence_injection: workspace

qtrm_residual_with_evidence: 49 / 72 = 0.6806
qtrm_evidence_span_reader_off_with_evidence: 24 / 72 = 0.3333
donor_only_with_evidence: 24 / 72 = 0.3333
retrieved_target_rate: 1.0000
all_targets_retrieved_rate: 1.0000
```

This means the answer renderer path is causally useful: disabling the span
reader drops to the donor/UNKNOWN baseline.

## Root Architecture Lesson

The current action loop has only this successful policy:

```text
RETRIEVE_MEMORY -> VERIFY_EVIDENCE -> ANSWER
```

If `ANSWER` always delegates to the same renderer, the learned controller cannot
beat the scripted controller except through accidental failures. To make the
loop genuinely useful, verification must causally alter answer formation:

```text
candidate answer
-> support/refute/missing/authority/temporal verifier
-> choose ANSWER, ABSTAIN, REVISE, or SEARCH_MORE
-> renderer uses the verified evidence span or returns UNKNOWN
```

## Next Architecture Candidate

Add an answer-decision loop, not another action-only loop:

```text
RETRIEVE_MEMORY
-> VERIFY_EVIDENCE
-> PROPOSE_ANSWER
-> VERIFY_ANSWER
-> ANSWER | ABSTAIN | REVISE | SEARCH_MORE
```

Minimal prototype:

- keep the accepted action-first transition-state controller;
- attach the evidence-span/truth answer channel as the renderer;
- add a learned or calibrated answer-decision head that can block unsupported
  candidates before final `ANSWER`;
- evaluate against donor, scripted greedy, scripted span-copy, state-off, and
  verifier-off baselines.

Accept only if:

- learned answer-decision loop beats scripted span-copy on held-out answer
  accuracy or reduces false positives at matched recall;
- state-off and verifier-off ablations drop;
- retrieved-target rate remains high, so gains are not retrieval leakage.

Reject if:

- the learned loop merely matches scripted span-copy;
- gains come only from returning UNKNOWN too often;
- thresholds are tuned on the same 8-case smoke without holding out a larger
  validation set.

## Static Threshold Probe

I tested a threshold-only answer-decision gate over the existing truth-gate
probabilities:

```text
report: docs/wiki/decisions/answer-decision-gate-truthcal-72.md
records: docs/wiki/decisions/evidence-span-truthcal-72-answer-channel-records.jsonl
calibration baseline -> gated: 0.6111 -> 0.8056
heldout baseline -> gated: 0.7500 -> 0.4167
heldout false positives: 2 -> 0
heldout blocked positives: 16
gate: rejected
```

This rejects the easy threshold solution. The verifier must be learned or
trained with richer counterfactual negatives; otherwise it trades hallucination
reduction for excessive abstention.

## Learned Decision Probe

A post-hoc MLP answer-decision head over the same recorded telemetry passes on
the separate heldout set:

```text
report: docs/wiki/decisions/answer-decision-head-truthcal-train144-eval72.md
train records: docs/wiki/decisions/evidence-span-truthcal-train144-answer-channel-records.jsonl
eval records: docs/wiki/decisions/evidence-span-truthcal-72-answer-channel-records.jsonl
checkpoint: runs/qwen35_2b_4090_answer_decision_head_truthcal_train144_eval72/last.pt
baseline eval: 49 / 72 = 0.6806
learned-decision eval: 62 / 72 = 0.8611
false positives: 13 -> 0
block improved: 13
block harmed: 0
status: accepted
```

Decision:
the next architecture step is justified. Add an in-loop answer-decision stage:

```text
PROPOSE_ANSWER -> ANSWER_DECISION -> ANSWER | ABSTAIN | REVISE | SEARCH_MORE
```

The post-hoc MLP is not the final architecture. It is evidence that the signal
is learnable and should be moved into QTRM as an ablatable head.

## Runtime-Wired Decision Path

The learned decision head has now been wired into the actual memory retrieval
eval path:

```text
report: docs/wiki/decisions/evidence-span-truthcal-72-answer-decision.md
records: docs/wiki/decisions/evidence-span-truthcal-72-answer-decision-records.jsonl
baseline span/truth: 49 / 72 = 0.6806
runtime decision: 62 / 72 = 0.8611
blocked candidates: 14
expected-unknown false positives: 0
retrieved_target_rate: 1.0000
```

This upgrades the probe from offline replay to runtime behavior. The next
bottleneck is positive wrong answers: cases where the model should revise the
candidate span or search again instead of only deciding `ANSWER` versus
`UNKNOWN`.

## In-Model Decision Path

The answer-decision gate has now been moved into the QTRM checkpoint as an
ablatable model output:

```text
report: docs/wiki/decisions/inmodel-answer-decision-head-truthcal-s200.md
checkpoint: runs/qwen35_2b_4090_answer_decision_head_truthcal_s200/last.pt
records: docs/wiki/decisions/evidence-span-truthcal-72-inmodel-answer-decision-records.jsonl
full in-model: 62 / 72 = 0.8611
feature-off: 49 / 72 = 0.6806
decision-head-off: 49 / 72 = 0.6806
block improved: 13
block harmed: 0
```

Interpretation:
the answer decision is now causal inside the model checkpoint, but the signal
comes from raw answer-channel telemetry rather than hidden state alone. This
supports the next answer-loop stage, but it does not remove the need for
`REVISE` or `SEARCH_MORE` on positive wrong answers.

## Boundary REVISE Result

The first `REVISE` branch is accepted as a narrow answer-renderer fix:

```text
report: docs/wiki/decisions/evidence-span-boundary-revise-truthcal72.md
records: docs/wiki/decisions/evidence-span-truthcal-72-boundary-revision-ablation-records.jsonl
full in-model + boundary revise: 67 / 72 = 0.9306
feature-off: 55 / 72 = 0.7639
decision-head-off: 55 / 72 = 0.7639
```

Interpretation:
some post-decision positive misses were not reasoning failures. They were
atomic identifier boundary errors in the span-copy renderer. The accepted fix
expands only tokenization-cut ASCII identifiers inside the existing workspace
span.

A non-label reliability source governor was also tested and rejected:

```text
records: docs/wiki/decisions/evidence-span-truthcal-72-source-governor-records.jsonl
full mode: 48 / 72 = 0.6667
```

This rejects hard evidence pruning as the next architecture step. Source
selection needs a learned selector/verifier objective rather than brittle
pre-answer context filtering.

## Learned Source Span-Mask Result

The source-selection step is now accepted as a learned span-mask, not as
evidence pruning:

```text
report: docs/wiki/decisions/evidence-source-selector-truthcal72.md
selector checkpoint: runs/evidence_source_selector_truthcal_s500/selector.pt
records: docs/wiki/decisions/evidence-source-selector-span-mask-truthcal72-thr065-records.jsonl
answer accuracy: 71 / 72 = 0.9861
unknown negatives: 24 / 24
```

Interpretation:
the model should keep full retrieved evidence in the workspace, but the final
copy operation should be constrained by a learned answer-source selector. This
solves most positive wrong-source errors without causing the distribution break
seen in pre-forward pruning.

Remaining bottleneck:
one positive authority case still returns `UNKNOWN` because the span reader's
no-answer head fires despite the correct source being selected. The next
architecture step is no-answer calibration conditioned on source-masked span
selection.
