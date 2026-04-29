# Donor Annealing And Distillation References

Purpose: ground the QTRM donor-annealing path in paper-backed distillation
methods and local implementation references.

## Local References

Downloaded PDFs:

| Area | Local PDF | Source |
| --- | --- | --- |
| Foundational KD | `references/papers/donor_annealing/1503.02531.pdf` | `https://arxiv.org/abs/1503.02531` |
| Sequence-level KD | `references/papers/donor_annealing/1606.07947.pdf` | `https://arxiv.org/abs/1606.07947` |
| Annealing-KD | `references/papers/donor_annealing/2104.07163.pdf` | `https://arxiv.org/abs/2104.07163` |
| Pro-KD | `references/papers/donor_annealing/2110.08532.pdf` | `https://arxiv.org/abs/2110.08532` |
| Distilling step-by-step | `references/papers/donor_annealing/2305.02301.pdf` | `https://arxiv.org/abs/2305.02301` |
| MiniLLM | `references/papers/donor_annealing/2306.08543.pdf` | `https://arxiv.org/abs/2306.08543` |
| GKD / on-policy distillation | `references/papers/donor_annealing/2306.13649.pdf` | `https://arxiv.org/abs/2306.13649` |
| Universal Logit Distillation | `references/papers/donor_annealing/2402.12030.pdf` | `https://arxiv.org/abs/2402.12030` |
| Multi-Level OT | `references/papers/donor_annealing/2412.14528.pdf` | `https://arxiv.org/abs/2412.14528` |

Implementation snapshots:

| Area | Local path | Upstream | Commit |
| --- | --- | --- | --- |
| TRL GKD/MiniLLM/Distillation | `references/official/trl` | `https://github.com/huggingface/trl` | `574ebe0503f5` |
| MiniLLM original repo | `references/official/lmops/minillm` | `https://github.com/microsoft/LMOps` | `e3ef717c7f76` |
| Distilling step-by-step | `references/official/distilling-step-by-step` | `https://github.com/google-research/distilling-step-by-step` | `ef944263c9dd` |
| EasyDistill | `references/official/easydistill` | `https://github.com/modelscope/easydistill` | `acbe49b5d9b1` |
| MiniPLM | `references/official/miniplm` | `https://github.com/thu-coai/MiniPLM` | `ac2302f9a551` |
| Multi-Level OT | `references/official/multi-level-ot` | `https://github.com/2018cx/Multi-Level-OT` | `712cfb49aeae` |
| Teacher Assistant KD | `references/official/teacher-assistant-kd` | `https://github.com/imirzadeh/Teacher-Assistant-Knowledge-Distillation` | `0f8168f4e722` |

No official implementation was found for Annealing-KD or Pro-KD. The paper
logic is still useful for schedule design, but implementation should be checked
against TRL/MiniLLM/EasyDistill-style code.

## Prior Work Map

| Reference | Core idea | QTRM relevance |
| --- | --- | --- |
| Hinton et al., 2015 | Train a compact model on teacher soft targets, not only hard labels. | Justifies donor-logit supervision beyond next-token CE. |
| Kim and Rush, 2016 | Distill sequence-level behavior, not only single-token distributions. | Future QTRM should distill generated answers/traces, not only next-token logits. |
| Annealing-KD, 2021 | Feed teacher soft targets through a gradual temperature curriculum. | Supports gradual donor/teacher signal schedules instead of abrupt donor removal. |
| Pro-KD, 2022 | Follow teacher checkpoints/footprints to reduce capacity gap and checkpoint-search risk. | Suggests donor annealing should use staged checkpoints/evals, not only one final teacher. |
| Distilling step-by-step, 2023 | Use teacher rationales as additional supervision for smaller students. | Maps to CoT-to-latent trace supervision before full latent-only behavior. |
| MiniLLM, 2024 | Reverse-KL/on-policy KD for generative LMs, reducing exposure bias and over-wide low-probability imitation. | Preferred reference for low-donor-scale QTRM training when precision and calibration matter. |
| GKD, 2024 | Train on student self-generated sequences with teacher feedback to fix train-inference mismatch. | Direct next step after static donor annealing: QTRM rollouts should be corrected by donor/teacher feedback. |
| ULD / Multi-Level OT | Distill logits across models with different tokenizers using optimal transport. | Needed only if the donor changes to a model family with incompatible tokenizer/vocab. |
| EasyDistill / DistilQwen | Practical black-box, white-box, ranking, RL, and CoT distillation pipelines. | Useful implementation reference for Qwen-family distillation and stored top-k teacher logits. |
| MiniPLM | Pretraining-stage KD with Qwen teacher and reference-data filtering. | Relevant if QTRM moves from task-local replacement to broader student pretraining. |

## Implementation Lessons

Current QTRM implements only the first static step:

```text
fused_logits = donor_logits_scale * donor_logits
             + qtrm_logits_scale  * qtrm_logits

donor_logits_scale: 1.0 -> 0.0
optional donor KL loss: KL(student/fused || donor) or KL(donor || student)
```

The new config hooks are:

```yaml
train:
  donor_logits_scale_start: 1.0
  donor_logits_scale_end: 0.0
  loss_donor_kl_weight: 0.05
  donor_kl_beta: 1.0
  donor_kl_temperature: 1.0
```

`donor_kl_beta` follows the TRL GKD convention:

| beta | Meaning | When to prefer |
| --- | --- | --- |
| `0.0` | forward KL, teacher-to-student mass covering | preserving broad donor distribution |
| `1.0` | reverse KL, student-to-teacher mode seeking | precise generative distillation / MiniLLM-style behavior |
| `0.0 < beta < 1.0` | generalized JSD interpolation | controlled middle ground |

## Next QTRM Steps

1. Run a static donor anneal probe with `loss_donor_kl_weight > 0`.
2. Sweep `donor_logits_scale` at eval time: `1.0`, `0.5`, `0.25`, `0.0`.
3. Add an on-policy MemoryOS distillation mode:

```text
sample QTRM at low donor scale
-> score generated sequence with donor/teacher logits
-> train QTRM on the self-generated sequence with donor feedback
```

4. Add sequence-level trace distillation for answer format, evidence selection,
   and CoT-to-latent behavior.
5. Only after same-tokenizer Qwen donor replacement works should cross-tokenizer
   ULD/Multi-Level-OT be considered.

## Risk Boundary

Donor annealing does not by itself prove that QTRM became a standalone LLM. It
only tests whether QTRM can learn the final token policy for bounded tasks while
donor logits are reduced. Full donor replacement also requires reducing hidden
state dependence and passing donor-free language, MemoryOS, reasoning, and
fluency gates.
