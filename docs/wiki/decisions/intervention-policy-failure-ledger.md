# Intervention Policy Failure Ledger

Status: active root-architecture blocker, 2026-05-02.

## Failure

The current donor-backed QTRM residual path can change answers, but it has not
learned when it is allowed to intervene.

## Evidence

Audit script:

```text
scripts/179_audit_intervention_policy.py
```

Diagnostic preference builder:

```text
scripts/180_build_intervention_preferences.py
```

Preference checkpoint:

```text
eval: runs/eval/canonical_answer_preference_s160_answer_gate_8.jsonl
audit: docs/wiki/decisions/canonical-answer-preference-s160-intervention-audit.json
donor_hit_qtrm_miss: 1 / 8
qtrm_hit_donor_miss: 1 / 8
core_off_beats_qtrm: 1 / 8
qtrm_beats_core_off: 0 / 8
qtrm_changed_donor_completion: 7 / 8
```

The critical case:

```text
id: synthetic-authority-vault-0100
gold: opal-river
donor-only: Answer: opal-river
core_off: Answer: opal-river
full QTRM: Answer: stone-arch.
```

Residual-scale sweep:

```text
eval: runs/eval/canonical_answer_preference_s160_qscale030_answer_gate_8.jsonl
audit: docs/wiki/decisions/canonical-answer-preference-s160-qscale030-intervention-audit.json
donor_hit_qtrm_miss: 0 / 8
qtrm_hit_donor_miss: 0 / 8
core_off_beats_qtrm: 0 / 8
qtrm_beats_core_off: 0 / 8
qtrm_changed_donor_completion: 4 / 8
```

Lowering QTRM scale avoids the `opal-river -> stone-arch` mistake, but it also
removes all measured QTRM advantage and all causal drops.

Diagnostic intervention rows built from the held-out eval:

```text
diagnostic only: docs/wiki/decisions/canonical-answer-preference-s160-intervention-preferences.diagnostic.jsonl
rows: 3
allow_qtrm: 1
preserve_donor: 1
suppress_core_override: 1
```

These rows prove the training signal shape, but they must not be used for a
final held-out claim. The same builder should be applied to train-split
on-policy eval records, then the resulting checkpoint must be tested on held-out
cases.

Train-split intervention rows:

```text
train eval: runs/eval/canonical_answer_preference_s160_train24_answer_gate.jsonl
audit: docs/wiki/decisions/canonical-answer-preference-s160-train24-intervention-audit.json
data: data/filtered/memory_reasoning_intervention_preferences_train24.jsonl
case_count: 24
donor_hit_qtrm_miss: 2 / 24
qtrm_hit_donor_miss: 2 / 24
core_off_beats_qtrm: 2 / 24
qtrm_beats_core_off: 2 / 24
rows: 6
preserve_donor: 2
allow_qtrm: 2
suppress_core_override: 2
```

On-policy intervention fine-tune:

```text
config: configs/qwen35_2b_4090_intervention_preference_train24_s080.yaml
script: scripts/181_run_intervention_preference_train.sh
checkpoint: runs/qwen35_2b_4090_intervention_preference_train24_s080/last.pt
report: docs/wiki/decisions/intervention-preference-train24-s080-answer-gate-8.md
status: rejected
causal_gate_status: rejected
full QTRM: 5 / 8
donor-only: 5 / 8
core_off: 6 / 8
answer_residual_governor_off: 6 / 8
intervention audit: docs/wiki/decisions/intervention-preference-train24-s080-intervention-audit.json
donor_hit_qtrm_miss: 0 / 8
qtrm_hit_donor_miss: 0 / 8
core_off_beats_qtrm: 1 / 8
```

Mixed finding:

```text
synthetic-authority-vault-0100 improved from stone-arch to opal-river.
```

But the checkpoint is still rejected because `core_off` and
`answer_residual_governor_off` beat full QTRM. This means the intervention
preference signal can repair specific overrides, but the current latent/core
answer path is still not a promoted architecture.

The `core_off` signal was then expressed as an explicit coreless lower-bound,
not as the final target:

```text
config: configs/qwen35_2b_4090_coreless_workspace_answer_candidate.yaml
script: scripts/182_run_coreless_workspace_answer_candidate_gate.sh
8-case report: docs/wiki/decisions/coreless-workspace-answer-candidate-gate-8.md
16-case report: docs/wiki/decisions/coreless-workspace-answer-candidate-gate-16.md
8-case status: accepted
16-case status: accepted
16-case QTRM coreless: 11 / 16
16-case donor-only: 10 / 16
16-case workspace_off: 10 / 16
```

This does not rescue the old recursive-core claim. It gives a teacher/baseline
for a safer mandatory core.

Mandatory identity-safe core candidate:

```text
config: configs/qwen35_2b_4090_mandatory_identity_core_candidate.yaml
script: scripts/183_run_mandatory_identity_core_candidate_gate.sh
report: docs/wiki/decisions/mandatory-identity-core-candidate-gate-8.md
full mandatory core: 6 / 8
donor-only: 5 / 8
core_off: 6 / 8
workspace_off: 5 / 8
```

The full path runs the core (`core_steps=2`) while preserving the lower-bound
behavior. This is a safety start, not a core-causality proof. The next step is
training the core output blend to open only when the latent loop improves the
answer over the lower-bound teacher.

Mandatory core causal training:

```text
config: configs/qwen35_2b_4090_mandatory_identity_core_causal_s080.yaml
script: scripts/184_run_mandatory_identity_core_causal_train.sh
report: docs/wiki/decisions/mandatory-identity-core-causal-s080-gate-8.md
status: rejected
full mandatory core: 5 / 8
donor-only: 5 / 8
core_off: 5 / 8
workspace_off: 5 / 8
core_off same completions: 8 / 8
```

This rejected the identity-blend-only fix. The core ran, but the answer path
still behaved as if the core were optional.

Strict mandatory answer bottleneck:

```text
config: configs/qwen35_2b_4090_mandatory_core_answer_bottleneck_causal_s120.yaml
script: scripts/185_run_mandatory_core_answer_bottleneck_causal_train.sh
report: docs/wiki/decisions/mandatory-core-answer-bottleneck-causal-s120-gate-8.md
status: rejected
full mandatory core: 5 / 8
donor-only: 5 / 8
core_off: 5 / 8
workspace_off: 5 / 8
core_off same completions: 1 / 8
```

This fixed the bypass: disabling the core no longer produced the same strings.
It was still rejected because the full model did not beat donor-only.

Clean intervention preference on the strict path:

```text
builder: scripts/186_build_clean_intervention_preferences.py
data: data/filtered/memory_reasoning_intervention_preferences_clean_train24.jsonl
config: configs/qwen35_2b_4090_mandatory_core_intervention_preference_s080.yaml
script: scripts/187_run_mandatory_core_intervention_preference_train.sh
report: docs/wiki/decisions/mandatory-core-intervention-preference-s080-gate-8.md
status: accepted
causal_gate_status: accepted
full mandatory core: 6 / 8
donor-only: 5 / 8
core_off: 5 / 8
workspace_off: 5 / 8
```

Current resolution:
the promoted small-gate candidate is now the strict mandatory-core answer path.
Coreless remains a lower-bound/teacher only. The remaining blocker is scale:
the 8-case accepted result must survive 16/32/72 held-out gates.

Scale-up result:

```text
16-case full mandatory core: 13 / 16
16-case donor/core_off/workspace_off: 10 / 16

32-case full mandatory core: 20 / 32
32-case donor/core_off/workspace_off: 17 / 32

72-case full mandatory core: 50 / 72
72-case donor/core_off/workspace_off: 39 / 72
72-case core/workspace hit drop: 11
```

Updated resolution:
the strict mandatory-core answer path now passes the full 72-case held-out root
gate. The intervention-policy blocker is no longer "core is optional" on this
candidate. The next blocker is narrower: improve abstention/UNKNOWN and
temporal-conflict calibration without losing the 72-case core/workspace causal
drop.

## Root Hypothesis

The missing component is not another global residual scale. It is a learned
intervention policy:

```text
When donor is likely correct: preserve donor.
When donor is likely wrong and QTRM has verified evidence: allow QTRM residual.
When neither side is verified: abstain/search/revise.
```

## Big-Structure Doubt Gate

One-sentence root claim:

```text
QTRM is useful only if its latent state can decide when to override donor.
```

Falsifier:

```text
donor-only or core_off matches/beats full QTRM on held-out gates.
```

Current result:

```text
falsified for the current residual path.
```

## Architecture Candidates

Candidate A: On-policy intervention training

```text
Use generated donor/full/core-off records to build rows where:
- donor hit and QTRM miss -> close QTRM intervention
- QTRM hit and donor miss -> open QTRM intervention
- core_off hit and full miss -> penalize latent-core override
```

Implementation scaffold:

```text
scripts/180_build_intervention_preferences.py
```

Risk: can overfit small eval records unless generated on a separate training
split. Held-out diagnostic rows are for analysis only.

Candidate B: Verifier-conditioned residual permit

```text
QTRM residual logits are multiplied by an in-model permit gate trained from
support/refute/missing/conflict labels and donor/QTRM disagreement telemetry.
```

Risk: needs real verifier labels, not only thresholded confidence.

Candidate C: Donor-default candidate renderer

```text
The model keeps donor as default and emits QTRM answer tokens only after a
learned permit token/state. The answer still goes through the ordinary greedy
language path.
```

Risk: more architectural churn, but it directly targets the failure.

## Decision

Do not promote the current preference/governor checkpoint. Do not spend the
next run on another global scale, margin-only, or threshold-only change. The
on-policy intervention fine-tune has now also been tested and rejected, so the
next run must change the causal answer path itself.

The next accepted experiment for the full recursive path must prove:

```text
full QTRM > donor-only
full QTRM >= core_off
at least one latent/permit/governor ablation drop
no donor-hit -> QTRM-miss override on the held-out gate
```

The current lower-bound candidate is narrower:

```text
coreless workspace-answer path > donor-only
workspace_off drops below coreless path
scale to 32/72 before promotion beyond candidate status
```

The actual QTRM target is stricter:

```text
mandatory identity-safe core > coreless teacher
core_off drops below mandatory core
workspace_off drops below mandatory core
```
