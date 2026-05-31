# Gated Latent Memory

> **Context (2026-06)**: This mechanism is now interpreted as one component of the MSA + 5.56 Adaptive Rehearsal latent memory policy inside the OneBodyParallelHybridBlock. See canonical [Memory Architecture (MSA First-Class)](memory-architecture-msa.md) for the integrated view and current experimental status.

QTRM now has a first local implementation of gated latent memory inside
`LatentWorkspace`.

## Why Add This

Perceiver/OpenFlamingo/Q-Former learned slots are useful as a compact connector
and in-context workspace, but they are no longer the main modern-memory claim.
By themselves they mostly answer "what should the latent slots attend to?" They
do not explicitly answer "what should be preserved, overwritten, or reset?"

Newer memory references push in that direction:

| Reference | QTRM reading |
| --- | --- |
| LM2 | Preserve the base Transformer path and add a complementary gated memory lane. |
| G-MemLLM | Frozen backbone plus trainable latent memory bank with GRU-style updates. |
| Titans / ATLAS / MIRAS | Test-time memory should manage surprise, retention, and forgetting. |
| MSA | Very large memory should be routed sparsely instead of placed entirely in the prompt. |
| LightMem / MemCoT | External MemoryOS should filter, consolidate, and iteratively search memory. |

Current policy:

```text
Keep learned slots as the connector baseline.
Do not add more Perceiver depth as the next research step.
Move the claim-bearing path to gated memory causality and sparse external memory.
```

## Current Implementation

Code:

- `src/wgram_lm/workspace.py`
- `src/wgram_lm/wgram_model.py`
- `configs/qwen35_2b_4090_current_arch_pretrain_probe.yaml`

Config:

```yaml
model:
  workspace_memory_gate_enabled: true
  workspace_memory_gate_init_bias: -2.0
```

Layer update:

```text
prev_latent
cross_updated = prev_latent + cross_attention(prev_latent, context)
update_gate   = sigmoid(Wu([prev_latent, cross_updated]))
reset_gate    = sigmoid(Wr([prev_latent, cross_updated]))
candidate     = tanh(Wc([reset_gate * prev_latent, cross_updated]))
next_latent   = (1 - update_gate) * prev_latent + update_gate * candidate
```

The negative update-gate bias makes the initial model conservative: it tends to
preserve latent state until training learns useful overwrites.

## What This Does Not Prove

This is not full LM2, G-MemLLM, Titans, or MSA.

- It is not persistent long-term memory.
- It does not update weights at test time.
- It does not replace MemoryOS retrieval.
- It does not prove latent reasoning until gate-off ablations show a drop.

## Required Gate

Any gated-workspace checkpoint must be evaluated with:

```text
qtrm_residual_with_evidence
qtrm_workspace_gate_off_with_evidence
qtrm_workspace_off_with_evidence
qtrm_core_off_with_evidence
qtrm_residual_head_off_with_evidence
```

For the stricter evidence-only workspace path, run:

```text
--evidence-injection workspace
qtrm_workspace_memory_off_with_evidence
```

Interpretation:

- If `workspace_gate_off` matches full residual, the gate is not yet causally
  important.
- If `workspace_gate_off` drops but `workspace_off` also drops, the broader
  workspace path matters.
- If `workspace_memory_off` drops in workspace-injection mode, retrieved
  evidence is actually flowing through workspace-side memory.
- If only residual-head-off drops, the model is still mainly a residual adapter
  without proven gated workspace reasoning.
