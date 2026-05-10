# KISS YAGNI DRY SSOT Contract

Status: canonical engineering contract, 2026-05-02.

This page defines how QTRM architecture work should stay simple enough to
debug.

## One-Sentence Rule

```text
Only the simplest single-stream path is allowed to make a model claim.
```

This contract is now paired with the
[Universal LLM Causal Path Contract](universal-llm-causal-path-contract.md):
QTRM may add latent reasoning, memory, typed state, and metacognition, but the
canonical claim must still run from prompt tokens to model logits to
autoregressive text.

The canonical answer path is:

```text
compiled prompt tokens
-> frozen Qwen donor + QTRM latent workspace/core/residual
-> greedy autoregressive answer
```

The code-level source of truth for this contract is:

```text
src/qtrm_mm/eval/ssot_contract.py
```

Canonical evals must use:

```text
--require-canonical-ssot
--evidence-injection ssot
--answer-channel greedy
```

## KISS

Keep the main path small:

```text
Context Compiler -> one token stream -> donor + QTRM -> greedy answer
```

Do not draw or describe MemoryOS, rerankers, source selectors, span-copy heads,
or hidden workspace evidence as internal model blocks.

## YAGNI

A component is not promoted to the canonical architecture until it passes a
held-out gate with component-off ablations.

Probe-only components include:

- `--evidence-injection workspace`
- `--evidence-injection dual`
- `--answer-channel evidence_span_copy`
- source selector masks
- post-hoc answer-decision checkpoints
- deterministic span-boundary revision

These can be useful diagnostics, but they are not proof that QTRM is a general
autoregressive reasoning model.

## DRY

Do not duplicate the canonical answer-path rule in every script. The constants
live in `qtrm_mm.eval.ssot_contract`:

```text
CANONICAL_EVIDENCE_INJECTION = "ssot"
CANONICAL_ANSWER_CHANNEL = "greedy"
```

Scripts should import the validator instead of rewriting the checks.

## SSOT

There is one semantic model input:

```text
canonical token stream
```

MemoryOS/retrieval/rerank may run before the model, but their output must be
compiled into that stream. Source metadata may annotate stream tokens for
evaluation, but it must not become a second hidden reality for model claims.

Typed registers, operation selectors, verifier heads, and memory readers follow
the same rule. They may be internal train/eval bottlenecks, but they must be
derived from the canonical token stream or QTRM latent state and must feed the
causal answer path. They must not become external calculators or hidden answer
channels.

## Promotion Gate

A new component can move from probe-only to canonical only if:

```text
full QTRM > donor-only
full QTRM > core_off/workspace_off/evidence_bottleneck_off
held-out result, not only training or smoke data
greedy autoregressive answer, not span-copy extraction
no sidecar/rule solver computes the final answer outside the LLM path
```

If those checks fail, the component stays diagnostic.

The root gate now separates two statuses:

```text
causal_gate_status = accepted only means some critical ablation worsened output
status             = strict promotion status when donor/ablation checks are on
```

This prevents a weak architecture from being promoted just because one local
component is useful.

## Canonical Causal Training

The canonical path should not merely be evaluated after training. It now has a
direct training pressure:

```text
full canonical SSOT greedy path
> core_off / workspace_off / evidence_bottleneck_off ablations
```

Implementation:

```text
src/qtrm_mm/losses.py::canonical_causal_ablation_loss
configs/qwen35_2b_4090_canonical_ssot_greedy_causal_s050.yaml
scripts/168_run_canonical_ssot_greedy_causal_train.sh
```

The loss is computed on `qtrm_residual_logits`, not donor-fused logits. Donor
logits are useful for language preservation, but they can hide whether the
QTRM path itself changes the answer. The ablation log-probs are stop-gradient
targets. This avoids the bad shortcut where shared weights learn to make
ablated paths worse instead of making the full path better.

This is still not an ASI proof. It is the minimum causal-pressure gate needed
before claiming that the recursive/workspace path participates in the answer.

## Current Causal Bridge Finding

The first canonical causal config without `core_to_text` produced
`canonical_causal_margin=0.0000`: full, `core_off`, and `workspace_off` had the
same residual answer log-probs.

The corrected candidate enables:

```text
core_context_enabled: true
core_to_text_enabled: true
coda_attn_every: 1
```

Config:

```text
configs/qwen35_2b_4090_canonical_ssot_coretotext_causal_s050.yaml
```

This makes workspace/core ablations change some completions, but the 4-case
held-out greedy gate is still rejected because answer accuracy does not drop
when the components are disabled.

## Current Tested Candidate

The first accepted narrow canonical candidate is not the plain `core_to_text`
bridge. That bridge made residual log-probs causal, but donor/prompt/coda paths
still produced the same greedy strings under ablation.

The accepted candidate is:

```text
canonical prompt tokens
-> donor hidden states + QTRM prelude
-> latent workspace/core z_h
-> answer bottleneck cross-attends text queries to z_h
-> QTRM residual answer logits + bounded donor logits
-> greedy answer
```

Config:

```text
configs/qwen35_2b_4090_canonical_ssot_core_answer_bottleneck_s150.yaml
```

What changed:

- `answer_bottleneck_enabled: true`
- `answer_bottleneck_requires_workspace_memory: false`
- evidence is still SSoT prompt text, not hidden workspace evidence
- `donor_logits_scale` is annealed from `1.0` to `0.8`
- `qtrm_residual_logits` are produced by the latent answer bottleneck, not the
  ordinary text coda path

4-case gate:

```text
report: docs/wiki/decisions/canonical-ssot-core-answer-bottleneck-s150-answer-gate-4.md
status: accepted
full QTRM: 3/4
donor-only with evidence: 2/4
workspace_off: 2/4
workspace_off same-completion rate: 0/4
```

Margin probe:

```text
core_off residual logprob margin:      0.6939
workspace_off residual logprob margin: 2.8459
```

Remaining limitation:
`core_off` changes all completions but does not yet reduce the 4-case hit
count. `workspace_off` is the first answer-level causal drop. The candidate is
only the current best narrow architecture candidate, not an ASI proof.

## Scale-Up Correction

The 4-case result did not survive the next 8-case gate:

```text
report: docs/wiki/decisions/canonical-ssot-core-answer-bottleneck-s150-answer-gate-8.md
status: rejected
full QTRM: 5/8
donor-only with evidence: 6/8
workspace_off: 6/8
core_off: 6/8
```

Failure class:

- strong answer bottleneck can fix UNKNOWN/abstention cases, but it also
  overwrites donor-correct multi-hop or conflict answers;
- weaker residual/safe-gate variants preserve donor behavior, but then QTRM
  stops providing a causal answer improvement;
- `workspace_memory_off`, `core_context_off`, `core_to_text_off`,
  `evidence_bottleneck_off`, and `span_reader_off` still match full output in
  the scaled gate.

Decision:

```text
Do not promote answer_bottleneck as the canonical default yet.
Keep it as a diagnostic checkpoint.
The next accepted architecture must beat donor-only while dropping under a
real workspace/core ablation on more than a smoke split.
```

The safe-gate eval config documents the other side of the tradeoff:

```text
configs/qwen35_2b_4090_canonical_ssot_core_answer_bottleneck_safe_gate_eval.yaml
report: docs/wiki/decisions/canonical-ssot-core-answer-bottleneck-safe-gate-eval-answer-gate-8.md
status: rejected
full QTRM: 5/8
donor-only with evidence: 5/8
causal drops: none
```

This confirms the current design tension:

```text
strong residual = can change answers, but may harm donor-correct outputs
weak residual   = preserves donor, but does not prove reasoning improvement
```

A learned selective-gate fine-tune was also rejected:

```text
config: configs/qwen35_2b_4090_canonical_ssot_core_answer_bottleneck_selective_gate_s150.yaml
checkpoint: runs/qwen35_2b_4090_canonical_ssot_core_answer_bottleneck_selective_gate_s150/last.pt
report: docs/wiki/decisions/canonical-ssot-core-answer-bottleneck-selective-gate-s150-answer-gate-8.md
status: rejected
full QTRM: 5/8
donor-only with evidence: 5/8
core_off: 6/8
causal modes: none
```

This is stronger evidence that local CE/canonical-margin tuning is not enough.
The next architecture should not try another threshold-only or margin-only
variant. It needs explicit verifier/decision supervision:

```text
PROPOSE answer
-> VERIFY support/refute/missing/conflict
-> DECIDE allow | abstain | revise | search-more
-> emit through the ordinary greedy language path only after the decision
```

## Latest Local-Tuning Rejections

Two more local fixes were tested and rejected:

```text
config: configs/qwen35_2b_4090_canonical_greedy_margin_s120.yaml
report: docs/wiki/decisions/canonical-greedy-margin-s120-answer-gate-8.md
status: rejected
full QTRM: 5/8
donor-only with evidence: 5/8
core_to_text_off: 6/8
causal modes: none
```

```text
config: configs/qwen35_2b_4090_canonical_plain_answer_kiss_s120.yaml
report: docs/wiki/decisions/canonical-plain-answer-kiss-s120-answer-gate-8.md
status: rejected
full QTRM: 4/8
donor-only with evidence: 5/8
workspace_off: 5/8
core_off: 5/8
causal modes: none
```

Decision:

```text
Stop margin-only and threshold-only tuning on the current donor-fused residual
path. The next candidate must change the causal answer path itself.
```

The new root hypothesis is:

```text
canonical prompt tokens
-> donor hidden states
-> QTRM latent evidence/reasoning state
-> verifier/governor-controlled answer-token renderer
-> greedy autoregressive answer
```

The donor can preserve language, but the QTRM claim requires an ablation drop
when the latent state, renderer, or governor is disabled.

## Canonical Decision-Token Rejection

The first SSOT decision-token implementation kept the evidence and decision
targets in one visible prompt stream:

```text
builder: scripts/170_build_canonical_decision_token_data.py
data: data/filtered/memory_reasoning_canonical_decision_tokens.jsonl
config: configs/qwen35_2b_4090_canonical_decision_tokens_s120.yaml
checkpoint: runs/qwen35_2b_4090_canonical_decision_tokens_s120/last.pt
```

It passed a residual logprob causality probe on the decision-token data:

```text
core_off margin:      0.0792
workspace_off margin: 0.0933
```

But it failed the canonical greedy answer gate even after correcting the eval
scale to `QTRM_LOGITS_SCALE=0.30`:

```text
report: docs/wiki/decisions/canonical-decision-tokens-s120-qscale030-answer-gate-8.md
status: rejected
qtrm_residual_with_evidence: 5 / 8
donor_only_with_evidence:    5 / 8
workspace_off_with_evidence: 5 / 8
core_off_with_evidence:      5 / 8
```

Decision:
decision-token SFT is useful supervision, but not a sufficient architecture by
itself. The next candidate must make the decision state causally control final
answer logits instead of only nudging label logprobs under a donor-dominated
decode.

## Answer Residual Governor Result

The next causal-answer-path change added a governor over the answer-bottleneck
residual logits:

```text
donor answer logits
+ answer_residual_governor * QTRM answer-bottleneck residual logits
-> greedy autoregressive answer
```

Training target:

```text
open governor when donor top-token != gold next-token
close governor when donor top-token == gold next-token
```

Config:

```text
configs/qwen35_2b_4090_canonical_answer_governor_s120.yaml
```

Strict 8-case gate:

```text
report: docs/wiki/decisions/canonical-answer-governor-s120-answer-gate-8.md
status: rejected
causal_gate_status: accepted
full QTRM: 5/8
donor-only with evidence: 5/8
answer_residual_governor_off: 4/8
core_off: 6/8
```

Decision:

```text
Promote answer_residual_governor only as a useful causal/protective diagnostic
component. Do not promote the full architecture, because full QTRM does not
beat donor-only and core_off beats full QTRM.
```

Next architecture requirement:

```text
The latent core must earn its place:
full QTRM > donor-only
full QTRM >= every critical component-off mode
at least one real workspace/core/governor ablation drop
```

## Donor Preservation And Preference Follow-Up

Two follow-up fixes were tested:

```text
config: configs/qwen35_2b_4090_canonical_answer_governor_preserve_s120.yaml
report: docs/wiki/decisions/canonical-answer-governor-preserve-s120-answer-gate-8.md
status: rejected
full QTRM: 5/8
donor-only with evidence: 5/8
core_off: 5/8
answer_residual_governor_off: 4/8
```

```text
config: configs/qwen35_2b_4090_canonical_answer_preference_s160.yaml
report: docs/wiki/decisions/canonical-answer-preference-s160-answer-gate-8.md
status: rejected
causal_gate_status: accepted
full QTRM: 5/8
donor-only with evidence: 5/8
core_off: 6/8
answer_residual_governor_off: 4/8
```

The preserve run made the core-off result less damaging, but it still did not
beat donor-only. The preference run kept a causal governor signal, but made the
core-off ablation better than the full path again.

The repeated held-out failure is:

```text
case: synthetic-authority-vault-0100
gold: opal-river
donor-only: opal-river
core_off: opal-river
full QTRM: stone-arch
```

Decision:

```text
Do not keep stacking local losses on this residual path.
Treat the latent core as unsafe to override donor-correct answers until a
redesigned core/gate passes donor-only and core-off strict comparisons.
```

A residual-scale sweep on the same preference checkpoint was also rejected:

```text
override: QTRM_LOGITS_SCALE=0.30
report: docs/wiki/decisions/canonical-answer-preference-s160-qscale030-answer-gate-8.md
status: rejected
causal_gate_status: rejected
full QTRM: 5/8
donor-only with evidence: 5/8
causal modes: none
```

This is an important negative result. Lowering QTRM scale fixed the
`opal-river` authority-conflict override, but it did not create a promoted
architecture. It erased the causal drop and damaged abstention/UNKNOWN
behavior. The problem is therefore not a single residual-scale setting. The
missing piece is an intervention policy: the latent path needs evidence that
an override is warranted before it can move the greedy answer.

## On-Policy Intervention Preference Rejection

The train-split intervention preference experiment tested the most direct local
repair for unsafe overrides:

```text
train eval: runs/eval/canonical_answer_preference_s160_train24_answer_gate.jsonl
data: data/filtered/memory_reasoning_intervention_preferences_train24.jsonl
rows: 6
config: configs/qwen35_2b_4090_intervention_preference_train24_s080.yaml
script: scripts/181_run_intervention_preference_train.sh
```

Held-out result:

```text
report: docs/wiki/decisions/intervention-preference-train24-s080-answer-gate-8.md
active-path report: docs/wiki/decisions/intervention-preference-train24-s080-active-path-gate-8.md
status: rejected
causal_gate_status: rejected
full QTRM: 5/8
donor-only: 5/8
core_off: 6/8
answer_residual_governor_off: 6/8
```

The old authority-conflict override improved:

```text
synthetic-authority-vault-0100: stone-arch -> opal-river
```

But this did not produce a promoted architecture. Several critical paths still
match full output exactly, and disabling the core/governor improves hit count.

SSOT/KISS/YAGNI/DRY implication:

```text
SSOT source: still the canonical token stream.
Smallest path: donor + one QTRM residual answer path.
Needed now because: intervention preference fixed a symptom, not causality.
Avoided duplicate logic: no MemoryOS/hidden evidence promotion.
Canonical gate: rejected, so the path stays diagnostic.
```

Next root change:

```text
Do not add another local loss. Either remove the non-causal sidecars from the
canonical claim, or force a single ablatable latent decision/answer renderer
that must carry the answer improvement under the strict root gate.
```

Gate tooling:

```text
scripts/148_build_root_architecture_gate.py --critical-mode ...
```

This keeps two views separate:

- broad root gate: includes probe-only paths to expose non-causal sidecars;
- active-path gate: includes only currently active SSOT answer components.

Both views reject this checkpoint.

## Mandatory Identity-Safe Core Candidate

The latent loop is mandatory for the QTRM target. The corrected design does not
make the core optional; it makes the core safe at initialization:

```text
workspace
-> recursive core loop always runs
-> core output blend starts near identity(workspace)
-> answer bottleneck / residual governor
-> greedy answer
```

Implementation:

```text
src/qtrm_mm/config.py::QTRMConfig.core_output_blend_enabled
src/qtrm_mm/qtrm_model.py::core_output_blend_gate
configs/qwen35_2b_4090_mandatory_identity_core_candidate.yaml
scripts/183_run_mandatory_identity_core_candidate_gate.sh
```

8-case result:

```text
report: docs/wiki/decisions/mandatory-identity-core-candidate-gate-8.md
full mandatory core: 6/8
donor-only: 5/8
core_off: 6/8
workspace_off: 5/8
```

Interpretation:

```text
core is mandatory and runs: yes
core is already causally improving answers: no
workspace-answer path is still causal: yes
```

This is the right starting point for latent-loop training. The next accepted
core experiment must satisfy:

```text
mandatory core > coreless teacher
mandatory core > donor-only
core_off < mandatory core
workspace_off < mandatory core
donor-correct override = 0
```

## Coreless Workspace-Answer Lower Bound

The failed full path produced a useful pruning signal: disabling the recursive
core was better than keeping the old unsafe core. This is not the final target;
it is a lower-bound/teacher path for the mandatory core.

```text
model.core_enabled: false
```

Implementation:

```text
src/qtrm_mm/config.py::QTRMConfig.core_enabled
src/qtrm_mm/qtrm_model.py forward effective disable_core
configs/qwen35_2b_4090_coreless_workspace_answer_candidate.yaml
scripts/182_run_coreless_workspace_answer_candidate_gate.sh
```

Result:

```text
8-case report: docs/wiki/decisions/coreless-workspace-answer-candidate-gate-8.md
8-case status: accepted
8-case QTRM coreless: 6/8
8-case donor-only: 5/8
8-case workspace_off: 5/8

16-case report: docs/wiki/decisions/coreless-workspace-answer-candidate-gate-16.md
16-case status: accepted
16-case QTRM coreless: 11/16
16-case donor-only: 10/16
16-case workspace_off: 10/16
```

SSOT/KISS/YAGNI/DRY implication:

```text
SSOT source: canonical prompt token stream.
Smallest diagnostic path: donor + workspace answer bottleneck + residual governor.
Needed now because: old recursive core is harmful on the strict gate.
Duplicated logic removed: no hidden runtime path.
Canonical target: mandatory identity-safe core, not coreless.
```

Limit:
the 16-case workspace-off same-completion rate is `15/16`; the accepted causal
drop comes from one case. This is enough to justify the candidate, not enough
to claim a robust architecture. Next gate should be 32/72 held-out with an
abstention-focused failure ledger.

## Accepted Mandatory-Core Answer Path

The latest correction removes the coreless shortcut from the answer residual.
The target architecture now requires a running core for the QTRM residual path:

```text
canonical prompt tokens
-> donor hidden/logits
-> QTRM workspace
-> mandatory recursive core
-> core-conditioned answer bottleneck
-> QTRM residual logits
-> donor-fused greedy answer
```

Implementation:

```text
config flag: QTRMConfig.answer_bottleneck_requires_core
config: configs/qwen35_2b_4090_mandatory_core_intervention_preference_s080.yaml
script: scripts/187_run_mandatory_core_intervention_preference_train.sh
report: docs/wiki/decisions/mandatory-core-intervention-preference-s080-gate-8.md
```

When `answer_bottleneck_requires_core: true`, `core_off` and `workspace_off`
zero the QTRM answer-bottleneck residual. This prevents the old failure mode
where `core_off` still used the same workspace answer path and matched the full
model.

Accepted held-out results:

```text
8-case:  full mandatory core 6/8,  donor/core_off/workspace_off 5/8
16-case: full mandatory core 13/16, donor/core_off/workspace_off 10/16
32-case: full mandatory core 20/32, donor/core_off/workspace_off 17/32
72-case: full mandatory core 50/72, donor/core_off/workspace_off 39/72
```

SSOT/KISS/YAGNI/DRY implication:

```text
SSOT source: canonical prompt token stream.
Smallest promoted path: donor + workspace + mandatory core + one residual.
Needed now because: latent loop is the QTRM claim and cannot be optional.
Duplicated logic removed: no MemoryOS hidden answer path, no span-copy channel.
Canonical gate: greedy answer must beat donor and drop under core/workspace off.
```

The 72-case scale-up makes this the current canonical strict mandatory-core
candidate. It does not prove ASI or broad reasoning. It proves a narrower but
important claim: this version's answer advantage is lost when the recursive
core or workspace is disabled.

Remaining blocker:

```text
negative/missing/UNKNOWN calibration is weak.
temporal Korean conflict handling is weak.
some generations still leak prompt/source fragments.
```
