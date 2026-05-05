# Donor Annealing Roadmap

This concept defines the long-term path from a donor-backed QTRM sidecar toward
a more independent QTRM student.

## Current Position

Current QTRM is not a standalone replacement for Qwen. It is a donor-backed
cognitive and memory adapter:

```text
Qwen donor hidden states -> QTRM workspace/core/coda -> QTRM residual logits
Qwen donor logits        -> base language policy
fused logits             -> donor logits + QTRM residual logits
```

This is the safest stage because QTRM can learn evidence selection, MemoryOS
behavior, latent recurrence, and halt control while Qwen preserves language
fluency.

More precise name:

```text
donor-backed residual adapter
```

or, when emphasizing the workspace/core path:

```text
donor-backed residual cognitive adapter
```

It is not a LoRA adapter because it does not update Qwen's internal weights.
The role is still adapter-like: Qwen supplies the base tokenizer, hidden states,
and language-policy logits; QTRM supplies bounded residual corrections on top.

## Validation Reset

The `donor_logits_scale=0.0` probes were useful stress tests, but they skipped
ahead from adapter validation to standalone-student validation. These are
different claims:

| Claim | What it means | Current status |
| --- | --- | --- |
| Donor-only baseline works | Qwen can answer/generate coherently without QTRM. | established as baseline |
| Residual adapter is safe | QTRM residual does not damage Qwen fluency. | mostly established with clamp/gate |
| Residual adapter is useful | QTRM beats donor-only on evidence-sensitive MemoryOS tasks. | primary next gate |
| Low-donor adapter is robust | QTRM remains stable when donor logits are reduced but nonzero. | partial, donor `0.25` remains fluent |
| Donor-free student works | QTRM owns the language policy at `donor_logits_scale=0`. | not established; currently collapses |

Near-term claims must stay at the residual-adapter level until the usefulness
gate is passed. A donor-free collapse is not by itself proof that the residual
adapter idea is wrong. It is proof that QTRM has not yet become a standalone
language-policy student.

Correct validation order:

1. Donor-only Qwen baseline.
2. Residual safety: no degradation in language fluency.
3. Residual usefulness: better than donor-only on MemoryOS/evidence tasks.
4. Component causality: workspace/core ablations explain the gain.
5. Low-donor robustness.
6. Donor-free student behavior after OPD/GKD/DistiLLM-style training.

## Target Direction

The long-term target is not to delete the donor immediately. The target is a
curriculum:

```text
donor-backed adapter
-> donor-logit annealed adapter
-> donor-hidden distillation
-> task-local QTRM standalone behavior
-> optional broader QTRM student pretraining
```

The first realistic success condition is narrower than "replace Qwen":

QTRM should answer selected MemoryOS/reasoning tasks with `donor_logits_scale=0`
while still using the same tokenizer and learned QTRM parameters. Only after
that passes should we reduce donor hidden-state dependence.

## Stages

| Stage | Donor hidden states | Donor logits | QTRM role | Gate |
| --- | --- | --- | --- | --- |
| 0. Donor baseline | used by donor only | 1.0 | none | donor-only eval |
| 1. Residual sidecar | used | 1.0 | evidence/memory residual | QTRM > donor on held-out MemoryOS |
| 2. Logit annealing | used | 1.0 -> 0.0 | student learns final distribution | no collapse at low donor scale |
| 3. Hidden distillation | reduced | 0.0 or low | QTRM mimics donor features | hidden reconstruction and task gates |
| 4. Task-local standalone | optional/none | 0.0 | QTRM handles bounded tasks | MemoryOS/reasoning pass without donor logits |
| 5. Broader student | none | none | independent small model | broad LM + retrieval + reasoning eval |

## Implemented Hook

`TrainConfig` now has:

```yaml
train:
  donor_logits_scale_start: 1.0
  donor_logits_scale_end: 0.0
  loss_student_lm_weight: 1.0
  loss_donor_kl_weight: 0.05
  donor_kl_beta: 1.0
  donor_kl_temperature: 1.0
```

The training loop linearly updates `model.cfg.donor_logits_scale` during the
run. If these fields are omitted, behavior is unchanged and the static
`model.donor_logits_scale` is used. The optional donor KL loss distills donor
logits into the QTRM student policy while the donor scale is reduced. The
student LM loss adds next-token CE on QTRM-only logits before donor-logit
fusion.

Probe config:

- `configs/qwen35_2b_4090_donor_anneal_probe.yaml`
- `configs/qwen35_2b_4090_student_lm_pretrain_probe.yaml`

Code paths:

- `src/qtrm_mm/training/train.py`: donor-logit schedule and train-loop wiring.
- `src/qtrm_mm/qtrm_model.py`: returns `qtrm_logits` before donor fusion.
- `src/qtrm_mm/losses.py`: student-only LM loss and generalized donor-logit
  distillation loss.

## Probe Result: Fused-Loss Failure Mode

The first 200-step linear anneal probe (`1.0 -> 0.0`) was informative but
failed:

| Checkpoint | Eval donor scale | Observed behavior |
| --- | --- | --- |
| fused-loss-only probe | `1.0` / `0.5` | coherent Korean, carried by donor logits |
| fused-loss-only probe | `0.25` | list/number pattern collapse |
| fused-loss-only probe | `0.0` | punctuation repetition collapse |

Root cause: the ordinary LM loss was computed on fused logits. At high donor
scale, Qwen donor logits could satisfy CE while QTRM-only logits remained poor.
When donor logits were reduced, the untrained student head was exposed.

After adding `qtrm_logits` and `loss_student_lm_weight`, the 200-step probe
started with `lm=2.22` but `student_lm=12.46`, confirming the diagnosis. By
step 200 it only reached `student_lm=11.42`; donor `0.0` still collapsed into
repetition, while donor `0.5` stayed fluent because the donor still carried the
base language policy.

The next fixed-donor student pretrain probe kept donor logits at `1.0`, used
`loss_student_lm_weight=1.0`, and trained 500 steps:

| Step | student_lm | Interpretation |
| --- | --- | --- |
| 0 | 12.44 | QTRM-only logits are mostly untrained |
| 100 | 11.21 | learning signal is working |
| 200 | 9.89 | fixed-donor pretrain beats full anneal at same step count |
| 300 | 8.58 | QTRM head is improving but not standalone |
| 400 | 7.85 | still above fluent standalone range |
| 450 | 8.00 | noisy but still near 8 |

Generation after this 500-step checkpoint:

| donor_logits_scale | qtrm_logits_scale | Behavior |
| --- | --- | --- |
| 0.0 | 0.5 | `and` / comma repetition collapse |
| 0.25 | 0.5 | `1.1.1...` repetition collapse |
| 0.5 | 0.5 | `1. 1. 1...` repetition collapse |
| 1.0 | 0.0 | fluent donor baseline |
| 1.0 | 0.1 | fluent; residual does not visibly damage donor |
| 0.5 | 0.1 | fluent |
| 0.25 | 0.1 | fluent |
| 0.5 | 0.5 + clamp/gate | fluent |
| 0.25 | 0.5 + clamp/gate | fluent |

Bounded residual 500-step probe:

| Setting | Observation |
| --- | --- |
| initial student_lm | `12.42` |
| final student_lm | `10.98` |
| learned unnormalized gate | collapsed to about `3.3e-6` |
| clamp-only, donor `0.25`, QTRM `0.5` | Korean smoke prompt repeated `1. **` |
| normalized gate + `0.05` floor, donor `0.25`, QTRM `0.5` | fluent smoke generations, gate about `0.061`, no argmax shift, repeated 2/3-gram rate `0.0` |
| normalized gate + `0.05` floor, donor `0.0`, QTRM `0.5` | still collapses to `,, and` repetition |

Decision: do not run full detach yet. Run longer student-only LM pretraining
with donor logits fixed as a safety rail, keep QTRM residual amplitude bounded,
then anneal donor only after student LM loss and donor-scale sweep gates
improve.

2K bounded student-LM continuation:

| Setting | Observation |
| --- | --- |
| config | `configs/qwen35_2b_4090_bounded_residual_studentlm_2k.yaml` |
| init | normalized-gate bounded 500-step checkpoint |
| key change | `qtrm_logits_scale: 1.0` with donor logits fixed at `1.0` |
| early loss signal | `student_lm` dropped from about `11.05` to the `5.7-6.2` range, much faster than the `qtrm_logits_scale: 0.1` probes |
| donor `1.0`, `0.5`, `0.25` | fluent greedy generations; gate about `0.063`; no donor argmax shift |
| donor `0.0` | still repeats `world of the world` / chapter-pattern text |

Interpretation: raising QTRM logit scale made the student-only LM loss learn
faster, but the bounded residual still behaves as a non-invasive sidecar when
donor logits are present. This is useful for safety, but it does not yet train
QTRM to own the language policy. The next detach attempt should move closer to
GKD/MiniLLM-style training: generate low-donor or QTRM-only continuations, let
the donor/teacher score or correct those student trajectories, and train on
that on-policy distribution instead of relying only on teacher-forced text.

## Breakthrough Search Update

The latest web search did not point to "more steps on the same objective" as
the main fix. It pointed to a distribution-mismatch fix:

```text
teacher-forced text training
  !=
donor-free QTRM autoregressive generation
```

The current `donor_logits_scale=0.0` collapse is exactly the kind of exposure
bias that GKD/OPD papers target. QTRM sees clean next-token contexts during
training, but at inference it must recover from its own bad tokens. Once it
enters a phrase such as `world of the world`, teacher-forced CE has not trained
it to escape.

The highest-priority paper-backed fix is therefore:

```text
collect QTRM failed rollouts
-> score/correct them with Qwen donor
-> train on student-visited states
-> explicitly push down repeated/collapsed student text
```

Candidate methods:

| Method | Core idea | QTRM use |
| --- | --- | --- |
| GKD / OPD | Train on student self-generated sequences with teacher feedback. | Use donor feedback on low-donor and donor-free QTRM rollouts. |
| 2026 OPD recipe | OPD succeeds when teacher/student patterns are compatible and teacher feedback adds new capability; use off-policy cold start and teacher-aligned prompt selection when OPD fails. | Start from bounded donor-backed checkpoint, not from fully collapsed donor-free rollouts. |
| DistiLLM | Skew-KL plus adaptive off-policy use of student outputs. | Cheaper route than full on-policy sampling every step. |
| DistiLLM-2 | Increase teacher-response likelihood and decrease bad student-response likelihood. | Treat `world of the world` rollouts as negative examples. |
| Residual KD | Distill compressed teacher representations and learn residual differences. | Strong fit for QTRM's donor-hidden residual architecture. |
| Concrete Score Distillation | Match relative logit score geometry instead of only softmax probabilities. | Possible replacement for plain donor KL if KL is too weak. |
| Unlikelihood / SimCTG | Penalize repeated tokens or anisotropic degenerate generation. | Add direct anti-repetition loss, but keep it secondary to OPD. |
| Capacity-gap / Minitron / Sheared LLaMA | Reduce teacher-student gap by using intermediate/compressed language backbones. | Fallback if QTRM's randomly initialized language policy remains too weak. |

Immediate implementation direction:

1. Add a rollout-generation script that samples QTRM with
   `donor_logits_scale=0.25` and `0.0`.
2. Store prompt, generated tokens, repetition metrics, donor top-k logits, and
   teacher/reference continuation.
3. Add a training mode that mixes clean text with low-donor student rollouts.
4. Add losses for teacher top-k skew/reverse KL, repeated n-gram unlikelihood,
   and DistiLLM-2-style contrastive negative response training.
5. Re-run the donor-scale sweep and require `donor=0.0` repeated 2/3-gram rates
   to drop before claiming donor-detach progress.

## Subliminal-Learning Safety Update

Subliminal Learning (`https://arxiv.org/abs/2507.14805`) adds a stronger risk
boundary around the OPD/GKD plan. Teacher-generated data is not automatically
safe just because visible content looks unrelated to alignment traits or passes
filters. The paper reports hidden trait transfer through number sequences, code,
and reasoning traces, especially when teacher and student share the same base
initialization or model family.

For QTRM this matters because the likely teacher/donor/student stack is
Qwen-family:

```text
Qwen3.6-27B teacher
Qwen3.5-2B donor hidden/logits
QTRM student head/core attached to Qwen tokenizer and donor states
```

Therefore the donor-annealing plan must split teacher use into two classes.

Canonical:

```text
teacher proposes candidates, critiques, or hard negatives
external verifier/gold process chooses the label
QTRM trains on the verified label or checked preference
```

Non-canonical until proven safe:

```text
teacher answer direct SFT
single-teacher CoT trace SFT
unbounded teacher-logit KL as the target distribution
large same-family synthetic datasets filtered only by surface content
```

This does not mean online distillation is abandoned. It means online
distillation must become verifier-gated:

```text
QTRM/student rollout
-> teacher suggests correction or critique
-> rule solver / unit test / symbolic checker / evidence verifier / human gold
   accepts or rejects it
-> train only from accepted labels and explicit rejected alternatives
```

Near-term consequence: before DGX Qwen3.6 direct online distillation, use
verified public datasets such as GSM8K, MATH-500, NuminaMath verifiable,
ProofWriter, CLUTRR, bAbI, MBPP, MBPP+, and HumanEval. Qwen3.6 can still be
used for candidate generation and hard-case mining, but not as the only source
of truth.

Bounded residual support:

```yaml
model:
  qtrm_residual_clamp: 1.0
  qtrm_residual_gate_enabled: true
  qtrm_residual_gate_init_bias: -2.0
  qtrm_residual_gate_normalize: true
  qtrm_residual_gate_min: 0.05
```

The gate input must be normalized before the gate linear layer. Without this,
the gate can saturate even when the learned bias remains close to its safe
initial value. The small minimum floor prevents the donor-preservation objective
from solving the task by closing the QTRM residual completely.

Reference map:

- [Donor Annealing And Distillation References](../sources/donor-annealing-distillation.md)

## Paper-Backed Interpretation

The current QTRM hook is a conservative first stage, not full MiniLLM/GKD yet.

| Prior work | What it contributes | Current QTRM status |
| --- | --- | --- |
| Annealing-KD | gradual teacher soft-target curriculum | implemented as donor-logit scale schedule |
| Pro-KD | staged teacher/checkpoint curriculum | not implemented yet |
| MiniLLM | reverse-KL/on-policy LLM distillation | reverse-KL direction available through `donor_kl_beta=1.0`; on-policy not yet |
| GKD | student self-generated sequences plus teacher feedback | not implemented yet |
| ULD / Multi-Level OT | cross-tokenizer logit distillation | future only if donor/student tokenizers differ |

## Required Evals

Do not claim donor replacement from a training loss alone. Required gates:

- donor scale sweep: `1.0`, `0.5`, `0.25`, `0.0`;
- held-out MemoryOS accuracy does not collapse as donor scale decreases;
- repetition and entropy remain in a healthy range;
- donor KL stays bounded without suppressing answer-level improvements;
- `qtrm_residual_with_evidence` still beats `qtrm_core_off_with_evidence`;
- Korean conflict and multi-hop cases improve, not only English/easy cases.

## Risks

- If QTRM only learns to copy donor logits, annealing to zero will collapse.
- If hidden-state dependence remains high, QTRM is still not standalone even
  when donor logits are zero.
- If the dataset is narrow, QTRM may become a task-specific extractor rather
  than a general cognitive core.

## Current Decision

Use Qwen as teacher and safety rail. Reduce donor logits first. Do not reduce
donor hidden-state dependence until MemoryOS held-out gates pass at low donor
logit scale. The next non-static step should follow GKD/MiniLLM more closely:
sample QTRM continuations at low donor scale, score them with the donor/teacher,
and train on those self-generated sequences.
