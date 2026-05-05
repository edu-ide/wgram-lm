# ASI Sufficiency Gate 2026-05-02

Status: rejected for ASI claim; accepted only as an ASI-oriented research
scaffold.

## Direct Answer

The current QTRM architecture is not sufficient for a general ASI claim.

It is a useful direction because the model/runtime boundary is now cleaner:

```text
QTRM model:
canonical token stream -> donor -> QTRM workspace/core/residual -> answer

QTRM runtime:
optional MemoryOS/tools -> context compiler -> canonical token stream -> model
```

But this is still a donor-backed residual cognitive adapter plus optional
runtime memory. It is not yet a generally intelligent, self-improving,
autonomous, verified reasoning system.

## Why This Is Not Enough

| Missing proof | Why it matters | Current status |
| --- | --- | --- |
| Plain-prompt language competence | A general model must work without MemoryOS. | Stability scripts exist, but every new checkpoint must pass them. |
| Causal latent reasoning | A looped core must improve answers, not merely compute. | Some controller gates show drops, but answer-level reasoning causality is not broadly proven. |
| Generalization beyond synthetic MemoryOS tasks | ASI claims need broad task transfer. | Current strongest evidence is narrow evidence/answer gates. |
| Verified self-improvement | Self-generated traces can collapse or pollute memory. | Trace buffers/gates exist, but no accepted learning loop yet. |
| Robust truth handling | The model must compare sources, detect contradiction, and abstain/search. | Evidence gates and span-copy exist, but calibration remains narrow. |
| Long-horizon agency | ASI requires repeated observe-act-verify-learn loops. | Controller/action-loop scaffolds exist, but task-level reward is not accepted. |
| Donor independence or reliable donor preservation | Either replace donor safely or preserve it without damage. | Current canonical path is donor-backed, not donor-free. |

## Root Architecture Judgment

The architecture is no longer obviously wrong after the MemoryOS boundary fix.
The clean split is:

```text
model intelligence path: donor + QTRM latent/core/residual
runtime support path: MemoryOS/tools/context compiler/history
```

However, it is not enough to keep adding modules. The next progress must come
from falsifiable gates:

```text
detect -> route/search -> reason -> verify -> answer/act -> learn -> regress-test
```

## Ranked Next Architecture Candidates

### Candidate 1: Model-Only Competence Gate

Limitation solved:
QTRM must be useful without MemoryOS.

Architecture change:
No new model component. Enforce a gate: every checkpoint must pass donor-only
and QTRM-residual plain-prompt language stability.

Minimal prototype:

```text
bash scripts/152_run_residual_language_stability_sweep.sh
```

Accept if:

- no repetition collapse;
- no visible reasoning leakage;
- Korean and English prompts remain coherent;
- QTRM residual does not damage donor baseline.

Reject if:

- QTRM only works when MemoryOS evidence is present;
- residual scale produces repeated tokens or `UNKNOWN` on ordinary prompts.

### Candidate 2: Causal Latent Answer Gate

Limitation solved:
The looped latent core must matter for answers, not only for telemetry.

Architecture change:
Use SSOT context as canonical input, then require `core_off`, `workspace_off`,
or relevant bottleneck-off ablations to reduce held-out answer score.

Accept if:

```text
full QTRM > donor-only
full QTRM > core_off/workspace_off
drop is on held-out cases, not only held-in data
```

Reject if:

```text
donor-only or prompt-only matches full QTRM
```

### Candidate 3: Verified Self-Improvement Runtime

Limitation solved:
ASI-oriented systems need learning from mistakes without hallucination
pollution.

Architecture change:
Keep self-improvement outside the model update path until verified:

```text
candidate trace
-> verifier/executable/citation gate
-> quarantine or accepted training buffer
-> held-out regression
-> update accepted or rejected
```

Accept if:

- self-generated data improves held-out tasks;
- false memory writes are rejected;
- regression suite stays stable.

Reject if:

- self-generated corrections improve only in-sample;
- memory writes contain unsupported facts;
- later checkpoints regress plain-prompt language stability.

## Immediate Priority

The next work should not be "add another big component." It should be:

1. Run the model-only language stability gate.
2. Run SSOT answer evaluation against donor-only and ablations.
3. Only then add or train new architecture pieces.

## First Model-Only Sweep Result

Command:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm \
OUT_DIR=runs/language_stability/asi_sufficiency_20260502 \
bash scripts/152_run_residual_language_stability_sweep.sh
```

Checkpoint:

```text
runs/qwen35_2b_4090_donor_residual_s010_1000/last.pt
```

Artifacts:

```text
runs/language_stability/asi_sufficiency_20260502/summary.jsonl
runs/language_stability/asi_sufficiency_20260502/scale_0p0.jsonl
runs/language_stability/asi_sufficiency_20260502/scale_0p05.jsonl
runs/language_stability/asi_sufficiency_20260502/scale_0p10.jsonl
```

Metric result:

| Residual scale | Records | Clean rate | Repeat failure | Visible reasoning | Answer drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 4 | 1.000 | 0.000 | 0.000 | 0.000 |
| 0.05 | 4 | 1.000 | 0.000 | 0.000 | 0.000 |
| 0.10 | 4 | 1.000 | 0.000 | 0.000 | 0.000 |

Interpretation:
this is only a narrow language-stability pass. It shows that the residual path
does not reproduce the previous `Freeze Freeze` / repeated-token collapse on
four plain prompts at scales 0.00, 0.05, and 0.10. It does not prove reasoning
competence.

Observed limitation:
the arithmetic probe still produced an incorrect answer:

```text
Prompt: Solve step by step: if x + 3 = 7, what is x?
Output: To solve for x, subtract 3 from both sides: x = 7 - 4 = x + 1.
Expected: x = 4
```

Decision:
Candidate 1 is provisionally accepted only as a repetition/format sanity gate.
It is not accepted as a reasoning gate, and it does not upgrade the architecture
claim. The next required gate is answer-level correctness with donor-only,
full-QTRM, `core_off`, `workspace_off`, and bottleneck-off comparisons.

## Canonical Answer Gate

The next answer-level gate must use the SSoT autoregressive contract:

```bash
bash scripts/166_run_canonical_ssot_answer_gate.sh
```

This script calls `scripts/95_eval_memory_retrieval.py` with:

```text
--require-canonical-ssot
--evidence-injection ssot
--answer-channel greedy
```

The point is to test the real model answer path:

```text
compiled prompt tokens -> donor + QTRM -> autoregressive answer
```

not:

```text
hidden workspace evidence -> span-copy extractor -> short answer
```

Span-copy and hidden-evidence gates remain useful probes, but they cannot be
used as evidence that QTRM has become a general autoregressive reasoning model.

## Canonical Causal Training Step

The first corrective training step after the rejected canonical answer smoke is:

```bash
bash scripts/168_run_canonical_ssot_greedy_causal_train.sh
```

This trains:

```text
CE(chosen canonical token stream)
+ student LM on QTRM logits
+ repeat guard
+ canonical causal contrast:
   full path > core_off/workspace_off/evidence_bottleneck_off
```

The config is:

```text
configs/qwen35_2b_4090_canonical_ssot_greedy_causal_s050.yaml
```

Acceptance condition remains held-out evaluation, not training loss:

```text
full QTRM must beat donor-only
full QTRM must drop when core/workspace/bottleneck paths are disabled
greedy answer path only
```

### Core-To-Text Bridge Smoke

The first canonical causal run showed the loss was attached too weakly to the
latent answer path. The loss now reads `qtrm_residual_logits`, and the stronger
candidate config turns on a direct latent-to-text bridge:

```text
configs/qwen35_2b_4090_canonical_ssot_coretotext_causal_s050.yaml
```

Observed training signal:
`canonical_causal_margin` moved away from `0.0000` and reached about `0.047` on
one batch, so the latent bridge can affect residual logits.

Held-out 4-case gate:

```text
report: docs/wiki/decisions/canonical-ssot-coretotext-answer-gate-after-causal-train-4.md
status: rejected
full QTRM: 2/4
donor-only with evidence: 2/4
workspace/core/core_to_text off: 2/4
```

Interpretation:
this is progress at the wiring level, not acceptance. The bridge can perturb
text, but it has not yet improved held-out answer correctness.

### Smoke Result

Command:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm \
MAX_CASES=2 \
OUT=runs/eval/canonical_ssot_answer_gate_smoke2.jsonl \
AUDIT_OUT=runs/eval/canonical_ssot_answer_gate_smoke2_audit.jsonl \
ROOT_MD=docs/wiki/decisions/canonical-ssot-answer-gate-smoke2.md \
ROOT_JSON=docs/wiki/decisions/canonical-ssot-answer-gate-smoke2-summary.json \
bash scripts/166_run_canonical_ssot_answer_gate.sh
```

Result:

```text
status: rejected
qtrm_residual_with_evidence accuracy: 1/2 = 0.500
donor_only_with_evidence accuracy:    1/2 = 0.500
critical ablation drops:              0
critical same-completion rate:        1.000 for every critical mode
missing critical modes:               none
```

Failure ledger:

```text
Failure:
  canonical SSoT greedy answer path is not causally using QTRM workspace/core.

Evidence:
  workspace_off, core_off, workspace_memory_off, core_context_off,
  evidence_bottleneck_off, and span_reader_off all match full QTRM output.

Known limitation class:
  non-causal latent workspace/core on answer-level autoregressive generation.

Root architecture hypothesis:
  the donor-backed residual path can currently behave as a donor/template path
  while the latent core computes without changing the answer.

Information path needed:
  canonical prompt tokens -> workspace/core state update -> changed answer logits.

Current information path:
  canonical prompt tokens -> donor logits dominate -> QTRM ablations do not alter
  answer strings in this smoke.

Recommended candidate:
  train the canonical greedy answer path with forced answer-level causal pressure,
  not span-copy extraction.

Acceptance gate:
  full QTRM must beat donor-only and must drop under core/workspace/evidence
  bottleneck ablations on held-out SSoT prompts.
```

Until these pass, the honest label remains:

```text
QTRM = ASI-oriented donor-backed residual cognitive adapter
     + optional external MemoryOS/runtime scaffold
```

not:

```text
QTRM = general ASI
```

## 2026-05-02 Update: First Accepted Narrow SSoT Causal Answer Gate

The `core_to_text` bridge alone was insufficient: it made answer-token
log-probs move, but greedy outputs stayed identical under critical ablations.
That means the donor/prompt/coda path still bypassed the latent workspace/core
for top-1 generation.

The next candidate forced the QTRM residual answer logits through a latent
answer bottleneck while keeping the input contract SSoT:

```text
canonical prompt tokens
-> donor hidden states + QTRM prelude
-> latent workspace/core z_h
-> answer bottleneck cross-attends text queries to z_h
-> residual answer logits + donor logits
-> greedy answer
```

Artifacts:

```text
config: configs/qwen35_2b_4090_canonical_ssot_core_answer_bottleneck_s150.yaml
checkpoint: runs/qwen35_2b_4090_canonical_ssot_core_answer_bottleneck_s150/last.pt
margin probe: runs/eval/canonical_causal_margin_probe_core_answer_bottleneck_s150_4_summary.json
gate report: docs/wiki/decisions/canonical-ssot-core-answer-bottleneck-s150-answer-gate-4.md
```

Result on 4 held-out cases:

```text
status: accepted
qtrm_residual_with_evidence: 3/4
donor_only_with_evidence:    2/4
workspace_off_with_evidence: 2/4
workspace_off same outputs:  0/4
```

Residual logprob probe:

```text
core_off margin:      0.6939
workspace_off margin: 2.8459
```

Interpretation:
this is the first narrow answer-level proof that the canonical SSoT path can
make latent workspace/core state causally matter for greedy output. It is still
not ASI sufficiency. The next gate must scale beyond 4 cases, preserve
multi-hop accuracy, and make core-off reduce quality rather than only changing
strings.

## 2026-05-02 Update: 8-Case Scale-Up Rejected

The same `core_answer_bottleneck_s150` checkpoint failed the next 8-case
canonical SSoT greedy gate:

```text
report: docs/wiki/decisions/canonical-ssot-core-answer-bottleneck-s150-answer-gate-8.md
status: rejected
qtrm_residual_with_evidence: 5/8
donor_only_with_evidence:    6/8
workspace_off_with_evidence: 6/8
core_off_with_evidence:      6/8
causal modes:                none
```

The failure is informative. The latent answer bottleneck is strong enough to
change greedy output, but not yet reliable enough to improve over the donor:
it helps some abstention-style cases and harms some donor-correct multi-hop or
conflict cases.

A safe residual-gate eval preserved donor-like behavior but also removed the
causal improvement:

```text
config: configs/qwen35_2b_4090_canonical_ssot_core_answer_bottleneck_safe_gate_eval.yaml
report: docs/wiki/decisions/canonical-ssot-core-answer-bottleneck-safe-gate-eval-answer-gate-8.md
status: rejected
qtrm_residual_with_evidence: 5/8
donor_only_with_evidence:    5/8
causal modes:                none
```

Decision:

```text
Do not claim ASI or promote answer_bottleneck as the canonical default.
Use the 4-case acceptance only as a diagnostic proof that the path can be
causal, not as proof that it is generally useful.
```

Next architecture requirement:

```text
selective residual control
    donor remains the default fluent answer policy
    QTRM intervenes only when its verifier/reasoning signal is calibrated
    held-out QTRM > donor-only
    intervention-off ablation drops accuracy
```

## 2026-05-02 Update: Selective-Gate Fine-Tune Rejected

Tried a conservative selective residual-gate fine-tune from the rejected strong
answer-bottleneck checkpoint:

```text
config: configs/qwen35_2b_4090_canonical_ssot_core_answer_bottleneck_selective_gate_s150.yaml
checkpoint: runs/qwen35_2b_4090_canonical_ssot_core_answer_bottleneck_selective_gate_s150/last.pt
report: docs/wiki/decisions/canonical-ssot-core-answer-bottleneck-selective-gate-s150-answer-gate-8.md
```

Result:

```text
status: rejected
qtrm_residual_with_evidence: 5/8
donor_only_with_evidence:    5/8
core_off_with_evidence:      6/8
causal modes:                none
```

Interpretation:
simple residual gating plus local canonical-margin training does not solve the
tradeoff. In this run, disabling the core improved the 8-case score, which is a
clear rejection of the current latent-core answer path.

Next required design change:

```text
Do not keep tuning this answer bottleneck checkpoint.
Move to explicit verifier/decision targets:
PROPOSE -> VERIFY -> DECIDE(allow/abstain/revise/search-more) -> ANSWER.
```
