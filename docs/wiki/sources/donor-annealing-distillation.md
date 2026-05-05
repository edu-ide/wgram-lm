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
| Subliminal Learning | `references/papers/donor_annealing/2507.14805.pdf` | `https://arxiv.org/abs/2507.14805` |

New web-searched references to fetch next:

| Area | Source | Why it matters now |
| --- | --- | --- |
| OPD failure/success recipe | `https://arxiv.org/abs/2604.13016` | Most directly addresses when on-policy distillation fails and how to recover it. |
| OPD survey | `https://arxiv.org/abs/2604.00626` | Organizes OPD by feedback signal, teacher access, and token/sequence granularity. |
| DistiLLM | `https://arxiv.org/abs/2402.03898` | Skew-KL and adaptive off-policy distillation reduce the cost of student-generated-output training. |
| DistiLLM-2 | `https://arxiv.org/abs/2503.07067` | Contrastive distillation increases teacher response likelihood while decreasing bad student responses. |
| Online KD | `https://arxiv.org/abs/2409.12512` | Adds small adaptive teacher-side modules to reduce immutable-teacher mismatch. |
| Residual KD | `https://openreview.net/forum?id=Dh6KxUxG20` | Two-stage projector and residual-learning distillation maps well to QTRM's donor-hidden residual design. |
| Concrete Score Distillation | `https://openreview.net/forum?id=bZBJFrxH1H` | Logit-level score matching may preserve richer teacher information than softmax KL. |
| Distillation scaling laws | `https://arxiv.org/abs/2502.08606` | Helps decide whether compute should go to more QTRM training, a better teacher, or a larger student. |
| Capacity gap law | `https://arxiv.org/abs/2311.07052` | Warns that the largest teacher is not always optimal for a small student. |
| Capacity-gap mitigation | `https://arxiv.org/abs/2305.12129` | MiniMoE-style extra capacity is one way to reduce teacher-student gap without large inference cost. |
| Minitron | `https://github.com/NVlabs/Minitron` | Practical pruning + KD recipe for deriving smaller language models from stronger bases. |
| Sheared LLaMA | `https://arxiv.org/abs/2310.06694` | Structured pruning + continued pretraining can beat training small models from scratch. |
| Unlikelihood training | `https://arxiv.org/abs/1908.04319` | Direct objective for reducing repetitive degenerate generations. |
| SimCTG | `https://arxiv.org/abs/2202.06417` | Contrastive training/decoding for neural text degeneration. |

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

New implementation references to clone next:

| Area | Upstream | Notes |
| --- | --- | --- |
| OPD 2026 recipe | `https://github.com/thunlp/OPD` | Uses student rollouts, teacher token-level rewards, top-k reward sets, and GRPO option. |
| DistiLLM | `https://github.com/jongwooko/distillm` | Official PyTorch implementation; DistiLLM-2 code is referenced from this project. |
| CSD | `https://github.com/aailab-kaist/CSD` | Referenced by the OpenReview page for Concrete Score Distillation. |
| Minitron | `https://github.com/NVlabs/Minitron` | Useful for a pruning/KD alternative if QTRM-only language policy remains too weak. |
| Sheared LLaMA | `https://github.com/princeton-nlp/LLM-Shearing` | Useful for structured pruning and continued-pretraining recipes. |
| Unlikelihood training | `https://github.com/facebookresearch/unlikelihood_training` | Direct anti-repetition objective implementation. |

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
| OPD recipe, 2026 | Successful OPD needs compatible student/teacher thinking patterns and teacher feedback that adds genuinely new capability; failing OPD can be recovered with off-policy cold start and teacher-aligned prompt selection. | Explains why QTRM should not jump straight from teacher-forced clean text to donor-free generation. |
| DistiLLM / DistiLLM-2 | Skew-KL, adaptive off-policy use of student generations, and contrastive use of teacher-positive vs student-negative responses. | Best fit for current `world of the world` collapse: failed QTRM rollouts become negative training examples instead of discarded samples. |
| Online KD | Make teacher feedback adapt to the student with small online modules instead of treating teacher distribution as immutable. | Future option if static Qwen donor feedback over- or under-guides the much smaller QTRM student. |
| Residual KD | Pretrain projectors to compress teacher representations, then train residual/differential prediction with supervised data. | Matches QTRM's donor-hidden adapter shape better than pure logit imitation. |
| Concrete Score Distillation | Match relative logit differences with score matching instead of only softmax probabilities. | Useful if donor KL keeps losing important low-rank/logit geometry. |
| Distillation scaling laws / capacity gap | Distillation benefit depends on student size, teacher quality, data, and compute; the best teacher is not always the largest one. | Warns against assuming Qwen teacher strength alone solves QTRM-only collapse. |
| Minitron / Sheared LLaMA | Prune a strong base model, then recover with continued training and distillation. | Alternative path if a randomly initialized small QTRM language head cannot become a stable standalone LM quickly. |
| Unlikelihood / SimCTG | Directly attack repetitive text degeneration through negative likelihood or contrastive representation/decoding objectives. | Local patch for repeated n-gram collapse, but not a complete replacement for OPD/GKD. |
| ULD / Multi-Level OT | Distill logits across models with different tokenizers using optimal transport. | Needed only if the donor changes to a model family with incompatible tokenizer/vocab. |
| EasyDistill / DistilQwen | Practical black-box, white-box, ranking, RL, and CoT distillation pipelines. | Useful implementation reference for Qwen-family distillation and stored top-k teacher logits. |
| MiniPLM | Pretraining-stage KD with Qwen teacher and reference-data filtering. | Relevant if QTRM moves from task-local replacement to broader student pretraining. |
| Subliminal Learning, 2025 | Teacher-generated data can transfer behavioral traits through hidden signals, even when the surface data is unrelated to the trait and filtered; the effect was observed for number sequences, code, and reasoning traces, and is strongest when teacher and student share or closely match the base model. | Direct warning for Qwen3.6-to-Qwen3.5/QTRM distillation: do not treat teacher outputs, CoT traces, or logits as safe gold labels merely because they pass content filters. |

## Implementation Lessons

### Subliminal-Learning Safety Boundary

Subliminal Learning changes the default QTRM distillation policy.

Risky path:

```text
teacher output / CoT / logits
-> content filter
-> direct SFT or KL target
-> QTRM student
```

This is no longer canonical for QTRM, especially for same-family Qwen teacher
and donor/student setups. The paper reports trait transfer through data that is
semantically unrelated to the trait, including number sequences, code, and
reasoning traces. Filtering is not enough as the sole safety mechanism.

Accepted path:

```text
teacher proposes candidates / critiques / hard cases
-> rule solver, unit test, symbolic verifier, retrieval evidence checker, or
   human-approved gold label decides the target
-> QTRM trains only on verified labels or explicitly scoped preferences
```

Implications:

- Qwen3.6 online direct-answer distillation is downgraded to a probe.
- CoT trace distillation from a single teacher is quarantined until a verifier
  or gold process converts it into checked supervision.
- Soft-logit KL from Qwen3.6 to QTRM is allowed only as a bounded auxiliary on
  verified states, not as the semantic source of truth.
- Same-family Qwen-to-Qwen/QTRM transfer needs extra trait, style, refusal,
  repetition, and alignment regression gates.
- Public datasets with gold answers or executable tests now rank above
  teacher-generated synthetic answers for raw recursive-reasoning training.

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

The current blocker is not simply insufficient step count. The 2K bounded
student-LM continuation improved teacher-forced loss, but donor-free generation
still collapsed. The next step should train on the student's own failure
distribution.

Priority recipe:

1. Build a rollout collector for low-donor and donor-free QTRM generations.
2. Mark degenerate rollouts using repeated n-gram rate, token-run metrics,
   entropy collapse, and known string loops such as `world of the world`.
3. Query Qwen donor/teacher logits on the same student-visited states.
4. Train with a hybrid objective:

```text
positive loss:
  teacher/reference continuation CE
  + teacher top-k skew-KL or reverse-KL on student-visited states

negative loss:
  unlikelihood for repeated n-grams
  + DistiLLM-2-style contrastive penalty on failed student continuations

stability loss:
  residual gate/clamp regularization
  + optional donor KL while donor scale is still nonzero
```

5. Mix data conservatively at first:

```text
60-70% clean/reference text
20-30% low-donor QTRM rollouts
5-10% donor-free failed QTRM rollouts
```

6. Sweep `donor_logits_scale` at eval time: `1.0`, `0.5`, `0.25`, `0.0`.
7. Only after donor-free repetition drops should we run a full donor anneal.

On-policy MemoryOS distillation mode:

```text
sample QTRM at low donor scale
-> score generated sequence with donor/teacher logits
-> train QTRM on the self-generated sequence with donor feedback
```

8. Add sequence-level trace distillation for answer format, evidence selection,
   and CoT-to-latent behavior.
9. Only after same-tokenizer Qwen donor replacement works should cross-tokenizer
   ULD/Multi-Level-OT be considered.

Fallback architecture path:

If QTRM-only language policy remains unstable after OPD/DistiLLM-style training,
do not keep increasing random small-student training blindly. Switch to a
compressed-language-backbone path inspired by Minitron/Sheared LLaMA:

```text
Qwen-compatible compressed LM backbone
-> QTRM latent workspace/core adapter
-> donor residual/teacher feedback
-> gradual donor-logit detach
```

This would trade architectural purity for a much stronger language-policy
starting point.

## Risk Boundary

Donor annealing does not by itself prove that QTRM became a standalone LLM. It
only tests whether QTRM can learn the final token policy for bounded tasks while
donor logits are reduced. Full donor replacement also requires reducing hidden
state dependence and passing donor-free language, MemoryOS, reasoning, and
fluency gates.
