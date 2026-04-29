# Compositional Architecture

QTRM combines multiple research axes: Qwen3.5 donor generation, GatedDeltaNet
mixing, LeWM-style world modeling, and recursive reasoning. This is a valid
research direction, but it is not automatically a best architecture.

Working rule:

- Treat every imported paper idea as a hypothesis until it passes a local
  ablation.
- Keep axes separable so each component can be switched off, replaced, or
  measured independently.
- Prefer official implementations for each axis before inventing a local
  approximation.
- Do not add a new component to fix a failure until the current failure has a
  diagnostic signal.

Minimum ablation grid:

| Variant | Purpose |
| --- | --- |
| Donor-only Qwen | Verify tokenizer, prompt format, and baseline generation |
| QTRM without JEPA | Isolate language-model adapter behavior |
| QTRM without recursive core | Test whether recurrence destabilizes logits |
| QTRM without delta mixer | Test whether mixer state is causing collapse |
| QTRM with official GatedDeltaNet | Compare local fallback vs reference mixer |
| QTRM with LeWM head only | Verify future-embedding objective independently |
| Full QTRM | Measure interaction after components pass individually |

Required measurements:

- training loss and validation loss
- tiny-overfit loss
- target-token rank under teacher forcing
- logit entropy and repeated n-gram rate
- free-running generation samples
- wall-clock and memory per step

Decision criterion:

Only promote a component from experimental to default when it improves at least
one target metric without causing collapse or regression in the baseline probes.
