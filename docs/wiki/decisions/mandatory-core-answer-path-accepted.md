# Mandatory-Core Answer Path Accepted Gate

Status: accepted 72-case held-out root gate, 2026-05-02.

## Claim

QTRM's target architecture must require the latent recursive core on the answer
path. `coreless` remains a diagnostic lower-bound/teacher only.

## Accepted Candidate

```text
canonical prompt tokens
-> frozen Qwen donor hidden states/logits
-> QTRM workspace
-> mandatory recursive core
-> core-conditioned answer bottleneck
-> QTRM residual logits
-> donor-fused greedy autoregressive answer
```

Implementation:

```text
src/qtrm_mm/config.py::QTRMConfig.answer_bottleneck_requires_core
src/qtrm_mm/qtrm_model.py strict answer-bottleneck residual gate
configs/qwen35_2b_4090_mandatory_core_intervention_preference_s080.yaml
scripts/187_run_mandatory_core_intervention_preference_train.sh
```

Key rule:

```text
answer_bottleneck_requires_core: true
```

Under `core_off` or `workspace_off`, the answer-bottleneck residual is zeroed.
Those ablations therefore cannot keep using a coreless QTRM shortcut.

## Gate Result

```text
report: docs/wiki/decisions/mandatory-core-intervention-preference-s080-gate-8.md
summary: docs/wiki/decisions/mandatory-core-intervention-preference-s080-gate-8-summary.json
eval: runs/eval/mandatory_core_intervention_preference_s080_gate_8.jsonl
audit: runs/eval/mandatory_core_intervention_preference_s080_gate_8_audit.jsonl
checkpoint: runs/qwen35_2b_4090_mandatory_core_intervention_preference_s080/last.pt
```

Scores:

```text
full mandatory core: 6/8
donor-only: 5/8
core_off: 5/8
workspace_off: 5/8
status: accepted
causal_gate_status: accepted
```

The strict checks passed because full QTRM beat donor-only by one hit and both
critical ablations lost one hit.

Scale-up gates:

```text
16-case report: docs/wiki/decisions/mandatory-core-intervention-preference-s080-gate-16.md
16-case full mandatory core: 13/16
16-case donor-only: 10/16
16-case core_off: 10/16
16-case workspace_off: 10/16

32-case report: docs/wiki/decisions/mandatory-core-intervention-preference-s080-gate-32.md
32-case full mandatory core: 20/32
32-case donor-only: 17/32
32-case core_off: 17/32
32-case workspace_off: 17/32

72-case report: docs/wiki/decisions/mandatory-core-intervention-preference-s080-gate-72.md
72-case full mandatory core: 50/72
72-case donor-only: 39/72
72-case core_off: 39/72
72-case workspace_off: 39/72
72-case same completions core_off/workspace_off: 18/72
```

The 72-case gate is the current strongest evidence that the mandatory latent
core is now on the causal answer path. Disabling the core or workspace removes
the whole 11-hit advantage over donor-only.

## Why Previous Attempts Failed

```text
mandatory identity causal s080:
  rejected because core_off produced the same completions as full on 8/8.

mandatory core answer bottleneck causal s120:
  rejected because bypass was removed, but full QTRM still tied donor-only.

mandatory core intervention preference s080:
  accepted because cleaned preference rows preserved donor-correct behavior
  while keeping the core/workspace ablations worse than full.
```

## SSOT/KISS/YAGNI/DRY Check

```text
SSOT source: canonical prompt token stream.
Smallest path: donor + QTRM workspace + mandatory core + one answer residual.
Needed now because: coreless cannot be the QTRM target.
Duplicated logic removed: no hidden MemoryOS/evidence answer path is promoted.
Canonical gate: greedy autoregressive answer with donor/core/workspace ablations.
```

## Limit

This is not enough to claim ASI or a complete robust architecture. It is enough
to promote this checkpoint as the current canonical strict mandatory-core
candidate. The next promotion requires broader capability and failure-class
work:

```text
abstention: negative/missing cases still frequently answer with a span.
temporal conflict: Korean temporal tasks still confuse date/code/location.
format: some answers still leak prompt/source fragments.
language preservation: donor-fused path is still needed for fluent output.
```

Observed QTRM misses on the 72-case gate are concentrated in:

```text
negative_authority_missing_synth: 4
negative_missing_synth: 4
negative_temporal_location_ko_synth: 3
negative_missing_ko_synth: 3
temporal_location_ko_synth: 2
temporal_conflict_ko_synth: 2
negative_authority_location_ko_synth: 2
```
