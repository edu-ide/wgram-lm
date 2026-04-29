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
  loss_donor_kl_weight: 0.05
  donor_kl_beta: 1.0
  donor_kl_temperature: 1.0
```

The training loop linearly updates `model.cfg.donor_logits_scale` during the
run. If these fields are omitted, behavior is unchanged and the static
`model.donor_logits_scale` is used. The optional donor KL loss distills donor
logits into the fused/student policy while the donor scale is reduced.

Probe config:

- `configs/qwen35_2b_4090_donor_anneal_probe.yaml`

Code paths:

- `src/qtrm_mm/training/train.py`: donor-logit schedule and train-loop wiring.
- `src/qtrm_mm/losses.py`: generalized donor-logit distillation loss.

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
