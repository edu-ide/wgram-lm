# Gated Core Context Injection

Status: implemented as an ablatable model path on 2026-04-30.

## Purpose

The workspace evidence path proved that retrieved evidence can be hidden from
the donor-visible prompt and injected into QTRM's workspace context. The next
problem was narrower: the recursive cognitive core did not directly read the
prompt/evidence context. It only received the latent workspace state.

Gated core context injection adds a conservative cross-attention path from the
recursive `z_l`/`z_h` core back to the prelude context sequence.

## Architecture

```text
MemoryOS evidence -> donor evidence encoder -> workspace context
visible prompt -> donor/text context
workspace context + prompt context -> prelude context
prelude context -> LatentWorkspace -> workspace
prelude context -> gated core cross-attention
workspace + gated context -> z_l/z_h recursive core
z_h + visible text context -> coda -> residual logits
```

The important boundary is that the coda/text path still does not receive hidden
workspace evidence as direct tokens. The core can read that context only through
an ablatable gated path.

## Implementation

- `QTRMConfig.core_context_enabled`
  - Enables the core cross-context path.
- `QTRMConfig.core_context_gate_init_bias`
  - Initializes the per-slot context gates conservatively.
- `QTRMRecursiveCore`
  - Adds `context_cross_l` and `context_cross_h`.
  - Adds per-token sigmoid gates for low/high state context deltas.
  - Returns `context_gate_mean` telemetry.
- `QTRMMultimodalModel.forward`
  - Passes the prelude context to the core.
  - Adds `disable_core_context`.
  - Returns `core_context_gate_mean`.
- `scripts/95_eval_memory_retrieval.py`
  - Adds `qtrm_core_context_off_with_evidence`.
  - Writes `latent_gates.core_context_gate_*` and
    `latent_gates.workspace_update_gate_*` telemetry into eval JSONL records.
- `scripts/114_run_expanded_strict_causality_ablation.sh`
  - Includes `qtrm_core_context_off_with_evidence` in the strict causality
    ablation set.

## Reference Mapping

- LM2: preserve the main Transformer path while adding a complementary memory
  lane with cross-attention and gates.
- G-MemLLM: use a frozen backbone plus trainable gated latent memory.
- RETRO/Atlas: retrieved evidence should enter architectural attention paths,
  not only prompt strings.

This is not a faithful implementation of any one paper. It is a small QTRM
adaptation of the shared design principle:

```text
keep the base route intact
add a gated context/memory route
measure it with an off-switch
```

## Required Proof

The claim "the cognitive core uses context directly" is valid only if:

```text
qtrm_residual_with_evidence > qtrm_core_context_off_with_evidence
```

on held-out MemoryOS evidence tasks after training with
`core_context_enabled: true`.

If `core_context_off` does not reduce score, then the new path is present but
not behaviorally important yet.

The eval record should also show non-empty `latent_gates.core_context_gate_*`
telemetry for full QTRM runs when `core_context_enabled: true`. A score drop
without gate telemetry is not interpretable.
