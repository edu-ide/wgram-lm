# Limit-Aware Model Objective

Status: architecture objective, 2026-05-01.

## Position

The phrase "a model with no limitations" must not be treated as a literal
claim. No fixed model can be limit-free across all tasks, data distributions,
time, modalities, tools, values, and adversarial inputs.

For QTRM, the practical target is a **limit-aware model**:

```text
When a limitation appears, the system should detect it, route around it,
verify the route, record the failure, and train or redesign the smallest
causal component needed so the same limitation becomes less likely next time.
```

The goal is not "never fail." The goal is:

```text
no known failure class remains unmeasured, unhandled, undocumented, or without
a falsifiable improvement plan.
```

## Why This Matters

Current QTRM failures show why this framing is necessary:

- free generation can collapse into repetition;
- retrieved evidence can be present but unused;
- counterfactual workspace evidence can trigger decoy copying;
- span-reader training can look healthy while a dataset conversion bug removes
  positive span supervision;
- component ablations can show that the claimed latent path is not causal.

A model that simply hides these failures with thresholds is not becoming
limit-free. It is becoming harder to debug.

## Operational Definition

A QTRM architecture improvement counts as progress toward the limit-aware goal
only if it adds at least one of the following:

| Capability | Meaning | Required proof |
| --- | --- | --- |
| Detect | The model or harness notices the failure class. | metric, head, audit reason, or eval split |
| Abstain | The model refuses unsupported answers. | UNKNOWN/no-answer accuracy under distractors |
| Search | The system requests evidence or more computation when needed. | search-needed labels or agentic retrieval trace |
| Verify | The answer is checked against evidence, source, time, logic, or value constraints. | verifier/evidence gate improves held-out behavior |
| Route | The forward path sends the needed signal through a causal module. | ablation drop or completion identity change |
| Learn | Training data/objective changes reduce recurrence of the failure. | held-out gate, not only train loss |
| Redesign | If local fixes fail, a replacement architecture is proposed. | root-architecture doubt ledger |

## Failure Surface Map

The model should keep an explicit map of known limits:

| Limit class | Current QTRM response |
| --- | --- |
| Hallucination / unsupported answer | evidence bottleneck, span-copy abstention, fact-verification traces |
| Retrieved-but-wrong answer | workspace swap gates, hard-negative span-reader training |
| Repetition collapse | repeat unlikelihood, generation verifier, entropy/repetition telemetry |
| Donor degradation | bounded residual, donor KL, donor-only baseline |
| Latent path not causal | workspace/core/evidence ablations and root gate |
| Unknown or missing evidence | no-answer head, UNKNOWN answer channel |
| Contradictory evidence | logical support/refute/missing heads and source authority labels |
| Long context overload | MemoryOS retrieval, reranking, sparse memory plan |
| Multimodal unverified path | visual ablation and multimodal held-out gates still required |
| Value/religion synthesis risk | critique, preservation, risk check, positive conclusion traces |

## Architecture Rule

Do not add components just to make the diagram more impressive. Each component
must close a named limitation and expose a falsifiable gate.

Bad:

```text
Add more memory, more heads, more losses, and more scripts because the model
still fails.
```

Good:

```text
Failure: workspace swap copies decoy evidence.
Hypothesis: no-answer head is not trained on prompt/evidence mismatch.
Change: train-only counterfactual hard negatives plus span-reader dataset fix.
Gate: residual mode outputs UNKNOWN on held-out swap while preserving normal
span-copy accuracy.
```

## Success Criteria

QTRM moves closer to the limit-aware objective when:

- every serious failure has a ledger entry;
- every claimed reasoning/memory path has an ablation gate;
- evidence use is causal, not just retrieved;
- unsupported answers become UNKNOWN rather than fluent guesses;
- prior research is checked before inventing a new mechanism;
- data conversion and label paths are tested before long training;
- wiki entries preserve commands, checkpoints, metrics, and conclusions.

QTRM moves away from the objective when:

- the same failure is attacked with another scalar loss after two failed local
  fixes;
- the model becomes more complex without a new measurable causal path;
- training loss improves but held-out behavior or ablation proof does not;
- a threshold hides the failure without teaching the model or changing the
  information path;
- claims outrun donor-only, retrieval-only, or ablated baselines.

## Research-Driven Loop

The research-driven architecture loop is now part of the model objective:

```text
observe limitation
-> write failure ledger
-> question the root architecture
-> search prior work
-> propose 2-3 concrete architecture candidates
-> implement the smallest falsification experiment
-> run held-out and ablation gates
-> update wiki and training plan
```

If this loop is skipped, the project is not building a limit-aware model; it is
only accumulating patches.

## Current Next Gate

For the current MemoryOS/span-reader line, the next proof remains:

```text
trainhardnegx2 span-reader checkpoint
-> normal held-out gate must remain accepted
-> workspace-swap gate must stop residual decoy copying
-> span-reader-off / workspace-memory-off ablations must show causal behavior
```

Only after that should the project expand to no-RAG latent reasoning,
multimodal gates, or larger donor/distillation runs.
