# Cognitive Core And Data Quality

The cognitive-core hypothesis says that a useful reasoning system should not
spend most of its capacity memorizing a noisy web snapshot. A compact reasoning
core should learn reusable thought, verification, planning, tool-use, and memory
access patterns, while factual and episodic knowledge can live in donor weights,
retrieval, or MemoryOS.

For QTRM this is a design constraint, not a promise of frontier capability.

## QTRM Interpretation

QTRM should be treated as:

- a small trainable reasoning/memory/world-model layer;
- attached to a frozen or mostly frozen Qwen donor;
- trained on curated traces and alignment targets;
- evaluated against donor-only generation before any full-model claims.

QTRM should not be treated as:

- a standalone LLM replacement;
- proof that 1B parameters are enough for frontier intelligence;
- a reason to skip baseline generation tests;
- a reason to keep training after collapse signals appear.

## Data Priority

| Data type | Role | Gate |
| --- | --- | --- |
| Donor continuation samples | Verify tokenizer, prompt format, and decoding | Qwen donor output must be coherent |
| Clean next-token text/code | Keep language modeling grounded | Validation loss and target-token rank improve |
| Reasoning traces | Teach multi-step state transitions | Teacher-forced next-token rank improves on reasoning prompts |
| Retrieval/tool traces | Teach "look up instead of memorize" behavior | Model selects retrieval/tool-use markers when needed |
| Memory read/write traces | Connect QTRM to MemoryOS behavior | Read/write decisions are deterministic on synthetic fixtures |
| Korean prompts | Prevent English-only overfitting | Korean eval prompts do not collapse to repeated tokens |
| Negative/corrupted traces | Detect shortcut learning | Model rejects or avoids known-bad transitions |

## Training Rule

More steps are justified only after these probes pass:

1. donor-only Qwen baseline generation;
2. tiny overfit on 32-128 fixed examples;
3. train/validation split;
4. target-token rank under teacher forcing;
5. logit entropy, repeated n-gram, and special-token rate;
6. component ablation against donor-only, no-JEPA, no-recursion, and no-mixer
   variants.

If free generation repeats one token such as `Freeze`, treat it as collapse or
misalignment until logs prove otherwise.

## Distillation Rule

Distillation is allowed when it has explicit labels:

- teacher model id;
- prompt format;
- target field;
- whether targets are logits, tokens, hidden states, or preference labels;
- filtering criteria;
- entropy/diversity checks;
- trainable parameter set.

Synthetic traces must be diversified. Single-teacher, single-template data can
look clean while silently collapsing to a narrow distribution.

Additional Subliminal Learning rule:

- teacher-generated answers, code, CoT traces, or logits are not gold labels by
  default;
- content filtering alone is not enough to make synthetic distillation data
  safe;
- same-family teacher/student setups need stronger suspicion because hidden
  teacher traits may transfer through non-semantic patterns;
- final labels should come from rule solvers, executable tests, symbolic
  verifiers, evidence checkers, or human-approved gold data;
- teacher models may propose candidates, critiques, hard negatives, and
  curricula, but the verifier/gold process owns the target.

QTRM policy:

```text
verified public datasets first
teacher as proposer/critic second
direct teacher imitation only as a quarantined probe
```

## Architecture Consequence

The cognitive-core direction favors:

- frozen donor first, trainable QTRM second;
- external memory before parametric memorization;
- small controlled datasets before large noisy dumps;
- recurrent/depth modules only after the LM path is known-good;
- JEPA/world-model losses only after target construction is validated.

It does not replace the official architecture axes:

- Qwen/HF remains the generator baseline.
- Gated DeltaNet remains the mixer reference.
- LeWorldModel remains the JEPA/world-model reference.
- TRM/Parcae remain recursive-depth references.
- Training diagnostics remain the go/no-go gate.

## Failure Modes

- **Param-count arithmetic**: treating `1.8T / 1B` as a measured efficiency law.
- **Memory starvation**: removing too much knowledge and forcing retrieval for
  basic reasoning context.
- **Synthetic collapse**: training on low-diversity generated traces until the
  student repeats high-probability phrases.
- **Adapter isolation failure**: the QTRM head learns a narrow token attractor
  while the donor hidden states are coherent.
- **Evaluation leakage**: judging success from one prompt instead of fixed eval
  sets and donor baselines.
