# Actual Reasoning Architecture Roadmap

Status: architecture decision, 2026-04-30.

## Decision

The current QTRM direction should be treated as an attempt to build a small
cognitive model around a donor LLM, not merely as a model that imitates
reasoning-looking text.

The operational target is:

```text
prompt/evidence -> latent workspace -> recursive state update -> halt/search/
verify/logit-change decision -> answer
```

This means QTRM should learn a process over latent states. But the architecture
is not allowed to claim "actual reasoning" until ablations prove that the
latent process is both necessary and useful.

## Why This Is Not Just CoT Pattern Learning

Visible CoT training teaches a model to produce reasoning-shaped text. That can
be useful as teacher supervision, but it does not prove that hidden computation
is doing the work.

QTRM separates the roles:

| Role | QTRM use |
| --- | --- |
| Explicit CoT / verifier trace | Teacher signal, audit signal, and dataset structure. |
| Latent workspace/core loop | Runtime computation path. |
| Halt/search/verify labels | Controls for when the latent state is sufficient. |
| Residual logits | Measurable effect on the donor policy. |

The desired behavior is that the model can answer with a short output while
the hidden workspace/core state carries the decision work. That is closer to a
compact cognitive process than to a long visible reasoning transcript.

## Current Architecture Is Sufficient For The Next Proof

The current architecture already has the minimum hooks needed for the next
scientific gate:

- donor-only baseline;
- bounded donor-logit residual;
- LatentWorkspace;
- workspace-only evidence path;
- gated workspace memory update;
- recursive `z_l`/`z_h` core;
- gated core context injection;
- coda path from latent prefix to text logits;
- component ablation modes;
- halt and depth telemetry hooks.

Therefore the next move is not to add another major architecture family. The
next move is to prove or reject the current causal path.

## Improvements Still Needed

The architecture still has important missing pieces. They should be added only
after the current evidence-path gate is measured.

| Improvement | Why it matters | Gate before promotion |
| --- | --- | --- |
| True TRM-style persistent carry | Makes recurrent state survive across steps and reset per halted sequence. | Per-sequence halt works without answer regression. |
| Per-sequence ACT instead of batch-level halt | Avoids forcing easy and hard samples to share the same loop count. | Same accuracy with fewer average core steps. |
| Latent distillation from CoT/verifier traces | Transfers explicit reasoning supervision into hidden state dynamics. | Latent-distilled model beats trace-SFT-only at equal token budget. |
| State-trajectory consistency loss | Reduces inference cliffs by making shallow/deep latent states converge. | Teacher-depth target predicts safe early exit. |
| Residual usefulness gate | Prevents QTRM from changing donor logits when no evidence-sensitive correction is needed. | Low donor argmax shift on ordinary text, high useful shift on memory tasks. |
| Retrieval/rerank feedback | Turns MemoryOS from external lookup into a trainable context-use loop. | Retrieval found-but-wrong cases improve. |
| Donor annealing | Eventually tests whether QTRM can survive with less donor policy support. | QTRM-only/student LM no longer collapses. |

## Inference Cliff Control

QTRM should assume that inference cliffs will appear. The control strategy is
not to declare them impossible, but to expose them early:

```text
tiny overfit -> donor-only smoke -> residual smoke -> held-out MemoryOS
-> workspace/core/context ablations -> depth/halt sweep -> donor-scale sweep
```

Promotion requires all of these to remain stable:

- answer accuracy;
- repetition metrics;
- entropy and KL-to-donor bounds;
- residual argmax-shift usefulness;
- workspace/core/context ablation gap;
- early-exit accuracy.

## Next Executable Gate

The next proof should answer one question:

```text
Does full QTRM with workspace-only evidence and gated core context outperform
the same checkpoint when core context or workspace memory is disabled?
```

Minimum comparison:

```text
qtrm_residual_with_evidence
qtrm_workspace_memory_off_with_evidence
qtrm_core_context_off_with_evidence
donor_only_with_evidence
donor_only_no_evidence
```

Minimum telemetry:

```text
latent_gates.workspace_update_gate_mean
latent_gates.core_context_gate_mean
core_halt.core_steps
first_step_logit_shift
workspace_memory_token_count
```

If full QTRM does not beat the ablations, the current architecture is not yet
using the new cognitive path. In that case, improve data/targets/telemetry
before adding another module.

## Claim Boundary

Allowed claim after the next gate passes:

```text
QTRM shows evidence of a causally useful latent workspace/core path for
evidence-sensitive residual correction.
```

Not allowed yet:

```text
QTRM is a standalone reasoning model.
QTRM has solved inference cliffs.
QTRM replaces the donor.
QTRM proves human-like cognition.
```

## Related Pages

- [QTRM Terminology](../concepts/qtrm-terminology.md)
- [QTRM Goal And Scope](qtrm-goal-and-scope.md)
- [QTRM Limitations And Mitigation Roadmap](limitations-mitigation-roadmap.md)
- [CoT To Latent Transfer](../concepts/cot-to-latent-transfer.md)
- [Gated Core Context Injection](../concepts/gated-core-context-injection.md)
- [Workspace Evidence Path](../concepts/workspace-evidence-path.md)
