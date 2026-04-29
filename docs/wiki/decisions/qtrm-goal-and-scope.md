# QTRM Goal And Scope

Decision:

QTRM's near-term goal is to build a small, testable cognitive/memory adapter
around Qwen3.5, not to replace Qwen3.5 as a standalone generator.

## Goal

Build a Qwen-donor-backed system that can:

- preserve donor tokenizer and generation behavior;
- add a trainable latent reasoning/memory path;
- use Qwen donor logits as the base language policy and train QTRM as a
  residual adapter;
- support multimodal/world-model objectives through reference-backed modules;
- connect to MemoryOS-style external memory;
- learn from clean, curated reasoning and retrieval traces;
- avoid repetition collapse under free generation.

## Non-Goals

- Do not claim frontier-model parity from a 1B parameter target.
- Do not treat rumored GPT-4 parameter counts as design requirements.
- Do not train the donor weights before the adapter path is verified.
- Do not add more architecture components to explain away a failed training run.
- Do not promote any component without ablation evidence.

## Current Architecture Position

| Axis | Current stance |
| --- | --- |
| Generator | Qwen/HF donor is the baseline and quality reference |
| Cognitive core | QTRM modules are the experimental reasoning/memory residual adapter |
| External memory | Prefer retrieval/MemoryOS for facts and episodes |
| Data | Clean trace shape matters more than just more steps |
| Mixer | Official Gated DeltaNet/FLA path preferred |
| World model | LeWM-style next-embedding prediction plus SIGReg preferred |
| Recursion | TRM/Parcae-style recurrence requires depth and stability telemetry |
| Fact verification | MemoryOS verifier handles evidence, source, conflict, and time labels |
| Critical synthesis | Value/religion questions require critique, preservation, risk checks, reframing, and positive conclusions |

## Naming

The current model should be described as a **donor-backed residual adapter**,
or more specifically a **donor-backed residual cognitive adapter**.

It should not be described as a standalone Qwen replacement yet. QTRM receives
Qwen hidden states, keeps Qwen donor logits as the base language policy, and
adds bounded residual logits. This is adapter-like in function, but it is not
LoRA: LoRA changes donor weights inside the donor model, while QTRM is an
external residual path around the frozen donor.

```text
Qwen hidden states -> QTRM workspace/core/coda -> residual logits
Qwen donor logits  -> base language policy
fused logits       -> donor logits + bounded QTRM residual
```

The immediate research question is therefore not "can QTRM replace Qwen?" but:

```text
Can QTRM, as a residual adapter, improve donor-only answers on tasks where
external memory/evidence matters, without damaging donor fluency?
```

The standalone-student question comes later, after OPD/GKD/DistiLLM-style
training has exposed QTRM to its own generation distribution.

## Immediate Priority Order

1. Establish donor-only Qwen generation and tokenizer tests.
2. Use donor logits as the base generation distribution.
3. Add tiny-overfit, target-token-rank, entropy, and repetition diagnostics.
4. Build a small clean trace dataset before another long run.
5. Run residual-adapter ablations.
6. Add memory/retrieval traces.
7. Add fact-verification traces with support/refute/conflict labels.
8. Add critical-synthesis traces for religion/value questions.
9. Scale steps only after diagnostics show real learning.

Do not treat `donor_logits_scale=0` as the next default gate. It is a later
standalone-student gate. The next default gate is residual-adapter usefulness:
`qtrm_residual_with_evidence` must beat `donor_only_with_evidence` on hard and
held-out MemoryOS probes.

## Success Criteria

A run is promising when:

- donor-only baseline is coherent;
- QTRM tiny-overfit converges quickly;
- validation loss and target-token rank improve together;
- logit entropy does not collapse;
- free generation avoids repeated-token attractors;
- at least one QTRM component improves a target metric without regressing donor
  baseline behavior.

A run is not promising when:

- loss decreases slowly but free generation repeats one token;
- validation probes are missing;
- the only evidence is a single sample;
- the tokenizer, labels, or hidden-size mapping are unverified.

## Practical Consequence

The next engineering work should focus on data and diagnostics before model
scale:

- make donor baseline scripts first-class;
- keep donor-logit passthrough as a required baseline for every generation run;
- create a curated micro-dataset for language, reasoning, retrieval, memory, and
  Korean prompts;
- log target-token rank and entropy every eval interval;
- compare against donor-only and component-disabled variants;
- only then increase steps, sequence length, or trainable parameter count.
