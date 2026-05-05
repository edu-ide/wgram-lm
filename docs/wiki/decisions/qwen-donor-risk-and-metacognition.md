# Qwen Donor Risk And Metacognition Boundary

Status: canonical boundary decision, 2026-05-03.

## Question

Is using a Qwen donor itself a problem?

## Decision

No, not inherently.

Using a frozen Qwen donor is acceptable when it is treated as:

```text
tokenizer and embedding contract
frozen hidden-state provider
base language-policy baseline
bounded residual-fusion scaffold
fluency preservation reference
```

It becomes a problem when it is treated as evidence that QTRM itself has gained
raw intelligence.

The canonical QTRM claim is not:

```text
Qwen can answer, therefore QTRM can reason.
```

The canonical claim must be:

```text
QTRM's trainable recursive core, memory path, or metacognitive state causes a
held-out gain over donor-only and loses that gain under the matching ablation.
```

## Why Donor Use Is Still Rational

QTRM is not yet a stable donor-free language model. Current donor-free or
low-donor paths have repeatedly shown collapse or no held-out depth scaling in
raw recursive gates. Therefore, the donor is useful as a conservative scaffold:

- it preserves tokenizer and language-manifold compatibility;
- it gives a strong donor-only baseline that prevents fake progress;
- it lets QTRM test whether a small trainable cognitive path can improve a
  strong base policy without destroying fluency;
- it supports residual-adapter and donor-annealing experiments before full
  student replacement.

This is closer to side-tuning/residual-adapter style research than to LoRA or
full fine-tuning, because QTRM does not edit Qwen donor weights.

## When The Donor Becomes A Crutch

Treat donor use as a root-architecture problem if any of these happen:

```text
donor_only equals or beats full QTRM on the target gate
core_off or memory_off equals or beats full QTRM
fused logits look good while QTRM-only logits repeat or collapse
calibration improves only because the donor dominates the fused distribution
QTRM residual only changes answer style or format
QTRM can answer only when hidden evidence/retrieval is injected outside the
canonical token stream
teacher/donor outputs are used as gold labels without verifier or gold process
```

In those cases, the correct action is not to add more side heads. The correct
action is to either reduce the claim to "donor-backed residual adapter" or
redesign the causal path.

## Metacognition Boundary

Overconfidence is now a raw-intelligence failure. A model that cannot know when
it does not know is not robustly reasoning.

However, Qwen donor confidence and QTRM metacognition must be separated.

Required modes for metacognitive evaluation:

```text
donor_only
qtrm_fused
qtrm_only_or_low_donor
qtrm_core_off_fused
qtrm_core_off_qtrm_only
qtrm_memory_off, if trainable memory is enabled
```

Promotion requires:

```text
QTRM improves accuracy or selective accuracy over donor_only
QTRM improves ECE/Brier/unknown handling over donor_only
the improvement drops under core_off or memory_off when that component is
claimed as causal
the result survives held-out answerable, unknown, contradiction, OOD, and
adversarial-evidence cases
```

Threshold-only abstention, external rerankers, prompt rules, or verifier
sidecars are useful probes, but they are not canonical metacognition unless the
model's internal trainable state becomes causally responsible for better
answer/search/abstain decisions.

## Random-Noise Warm-Up Mapping

The random-noise calibration prior suggests this QTRM-safe adaptation:

```text
freeze Qwen donor
feed random token IDs or controlled random hidden states through the normal
QTRM path
train only QTRM trainable modules toward random labels or chance-level
confidence targets
then resume real raw-reasoning training
measure QTRM-only and fused calibration before/after
```

Important restriction:

```text
Do not update Qwen donor weights during this warm-up.
Do not call the donor's improved confidence a QTRM metacognitive gain.
Do not promote unless QTRM trainable-path ablations prove causality.
```

This mirrors the 2026 random-noise warm-up fine-tuning pattern where a
pretrained backbone can remain frozen while the newly added classifier is
calibrated.

## Distillation Safety Boundary

Same-family Qwen teacher/donor/student setups require extra caution. The
Subliminal Learning note in `docs/wiki/sources/donor-annealing-distillation.md`
warns that teacher-generated data can transfer hidden traits even when surface
content is filtered.

Canonical path:

```text
teacher/donor proposes candidates, critiques, hard negatives, or top-k logits
verifier / unit test / symbolic solver / retrieval evidence / human gold label
selects the accepted target
QTRM trains on verified labels or explicitly scoped preferences
```

Non-canonical path:

```text
Qwen teacher answer or CoT
-> content filter only
-> direct QTRM SFT/KL as gold
```

## Current Canonical Label

Until donor-free gates pass, call the system:

```text
Qwen-backed residual cognitive adapter with recursive-core and metacognitive
calibration probes
```

Do not call it:

```text
standalone Qwen replacement
donor-free ASI core
fully self-sufficient language model
proven metacognitive reasoner
```

## Next Experiments

1. Add a donor-boundary eval table to every raw-intelligence report:

```text
donor_only
qtrm_fused
qtrm_low_donor
qtrm_only
core_off_fused
core_off_qtrm_only
```

2. Add a random-noise warm-up option that updates only QTRM trainable modules.

Status: implemented as a training option in
`scripts/196_train_pure_recursive_depth_supervised.py` and
`scripts/197_run_pure_recursive_depth_supervised_train.sh`. The first smoke
checkpoint is
`runs/qwen35_2b_4090_noise_warmup_smoke_s001/last.pt`. This is plumbing only,
not an accepted calibration claim.

3. Add a metacognitive calibration gate:

```text
answerable accuracy
unknown accuracy
selective accuracy by confidence
ECE
Brier
OOD confidence
core_off/memory_off calibration drop
```

Status: first choice-score gate implemented in
`scripts/202_build_metacognitive_calibration_gate.py`. The 1-case smoke report
is `docs/wiki/decisions/noise-warmup-metacognitive-calibration-smoke-s001.md`.
It is accepted only as evaluator plumbing. The larger 40-case held-out report
is `docs/wiki/decisions/noise-warmup-metacognitive-calibration-heldout40-s001.md`
and rejects the current two-step warm-up because global ECE worsens even though
Brier slightly improves. A high-entropy variant,
`docs/wiki/decisions/noise-uniform-warmup-metacognitive-calibration-heldout40-s001.md`,
improves global ECE/Brier but is also rejected because the stricter gate shows
critical core-on/QTRM-only modes worsen. A direct known/unknown/OOD
forced-choice probe,
`docs/wiki/decisions/metacog-forced-choice-s080-calibration-heldout40.md`,
also rejects: it lowers overconfidence but drops QTRM-only/core-on answer
accuracy. Promotion now requires policy-preserving calibration, such as
teacher-depth KL from the no-warmup QTRM core path plus targeted calibration
loss. The first teacher-depth KL run,
`docs/wiki/decisions/metacog-teacher-kl-s080-v2-calibration-heldout40.md`,
preserves QTRM-only/core-on accuracy but still rejects because global and
low-donor fused calibration worsen. Unknown-only selective follow-ups, including
the conservative S040 run, preserve QTRM-only accuracy and can improve QTRM-only
ECE/Brier, but still reject because low-donor fused calibration worsens. The
same matched gate must pass before donor annealing or teacher distillation
resumes, or the claim must be explicitly narrowed to QTRM-only metacognition
with fused-generation calibration treated as a separate gate.

4. Continue donor annealing only after QTRM-only generation stops collapsing on
basic language-stability checks.

## Related Pages

- [Random Noise Calibration References](../sources/random-noise-calibration.md)
- [Donor Annealing And Distillation](../sources/donor-annealing-distillation.md)
- [QTRM Terminology](../concepts/qtrm-terminology.md)
- [Raw Intelligence Gates](raw-intelligence-gates.md)
- [Qwen Donor Integration](../components/qwen-donor-integration.md)
