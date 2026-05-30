# Donor-Preserving LoopLM Free-Generation Repair

Date: 2026-05-30

Purpose: turn the S040 rank-8 LoRA / depth-sweep clarification into a concrete
S041 repair path for autoregressive free generation.

## Local Failure To Explain

The S040 rank-8 Mythos LoRA checkpoint shows a causal forced-choice signal but
does not yet generate the answer freely:

```text
causal forced-choice depth sweep: nonzero, best depth around 2-8 depending on smoke
free generation depth sweep: 0/48, with answer loops, bang loops, and numeric attractors
```

Therefore the next question is not "can the recursive core ever find an answer
basin?"  It is:

```text
Can QTRM steer the fluent donor mouth without replacing it with a private,
collapsed renderer?
```

## Paper Findings

| Prior | Link | S041 implication |
| --- | --- | --- |
| Relaxed Recursive Transformers | <https://arxiv.org/abs/2410.20672> | Depth-wise LoRA is a valid way to relax a tied/looped transformer. This supports loop-indexed rank-8 adapters, but also warns that recurrence must remain on the LM path. |
| LoopUS | <https://arxiv.org/abs/2605.11011> | A pretrained LLM can be recast into encoder, looped reasoning block, and decoder with selective gating, random deep supervision, and confidence exit. This is the closest high-level recipe for converting donor-backed QTRM into a looped latent refinement system. |
| Ouro / LoopLM | <https://arxiv.org/abs/2510.25741> | Latent iterative computation must be trained as part of the causal LM path and paired with learned depth allocation. Side renderers are not enough. |
| LoopFormer | <https://arxiv.org/abs/2602.11451> | Variable-depth trajectories need shortcut consistency; depth sweeps need not be monotonic without trajectory conditioning. |
| Parcae | <https://arxiv.org/abs/2604.12946> | Looped residual streams can destabilize through injection spectral norms. S041 should clamp/bound deltas and track collapse/entropy. |
| ReFT / LoReFT | <https://arxiv.org/abs/2404.03592> | Frozen base models can be steered by small representation interventions. This supports donor-preserving hidden/logit interventions instead of a QTRM-only mouth. |
| Proxy-Tuning | <https://arxiv.org/abs/2401.08565> | A tuned-minus-untuned proxy delta can guide a stronger model. S041 maps this to `donor_logits + alpha * qtrm_delta_logits`. |
| ThinkLogit | <https://arxiv.org/abs/2510.09354> | Decoding-time logit arithmetic with a smaller reasoning guider is directly analogous to QTRM-as-guider over Qwen donor logits. |
| Unlikelihood Training | <https://arxiv.org/abs/1908.04319> | The observed repeated-token failures should become explicit negative tokens/spans during training. |
| Dynamic Scheduled Sampling / GKD | <https://arxiv.org/abs/2301.13753>, <https://huggingface.co/papers/2306.13649> | Teacher-forced success is insufficient; S041 must train or at least evaluate on self-generated prefixes. |

## S041 Design

Start with a bounded donor-logit guider:

```text
donor forward -> donor_logits
QTRM recurrent core + LoRA -> qtrm_delta_logits
conflict/uncertainty gate -> gate in [0, 1]
final_logits = donor_logits + alpha * gate * clamp(qtrm_delta_logits)
```

Training objective after the inference smoke:

```text
response-only CE on answer tokens
first-answer-token CE/margin on donor-wrong rows
donor-correct KL or margin preservation
unlikelihood on Answer:Answer / bang loops / numeric attractors
self-rollout repair on generated failure prefixes
depth-randomized supervision and shortcut consistency
```

Required evaluation:

```text
donor_only
core_off
delta_off / gate_off
depth sweep: 1, 2, 4, 8
alpha sweep: qtrm residual scale under donor scale 1.0
free generation exact match
causal forced-choice
repetition/collapse flags
donor-correct preservation
```

## Immediate Smoke

The first local smoke is intentionally inference-only:

```text
script: scripts/262_run_s041_donor_preserving_freegen_sweep.sh
checkpoint: S040 rank-8 Mythos LoRA checkpoint
scoring: generation
conflict gate: on
donor scale: 1.0 for guided modes
qtrm alpha sweep: 0.25, 0.5, 1.0
depth sweep: 2, 4, 8
anti-repeat decode guard: no_repeat_ngram_size=2
```

Promotion rule:

```text
Do not promote S041 from forced-choice alone.
Free generation must improve over donor-only while delta/core ablations prove causality.
```

## 2026-05-30 Smoke Result

Artifacts:

```text
reports/s041_donor_preserving_freegen/s041_conflict_gated_free_generation_smoke8.jsonl
reports/s041_donor_preserving_freegen/s041_conflict_gated_free_generation_smoke8.summary.md
```

Result:

| Family | Best exact | Read |
| --- | ---: | --- |
| donor-only | 2/8 | fluent path is alive but weak on reasoning |
| QTRM-only depth 2/4/8 | 0/8 exact | private renderer still collapses |
| donor + conflict-gated QTRM alpha sweep | 2/8 exact | preserves donor fluency better, but does not beat donor |

Interpretation:

```text
The inference-only donor-preserving gate reduces the worst QTRM-only collapse
patterns, but it does not create new free-generation reasoning wins.  This is a
negative promotion result and a positive design signal: S041 needs training on
the donor-preserving path, not just a decoding-time scale sweep.
```

UltraData implication:

```text
Training on all UltraData is likely useful for coverage and instruction format,
but not sufficient by itself.  The full-data run must include free-running
renderer objectives: first-answer-token margin, donor-correct preservation,
self-rollout repair, and unlikelihood against the observed collapse strings.
```
