# TRM Canonical Core / Mythos Demotion

Status: canonical architecture decision, 2026-05-07.

## Decision

QTRM will use the TRM-inspired recursive core as the primary reasoning core.
There must not be a second answer-side reasoning core that independently
decides the answer after the TRM core has already run.

Canonical reasoning path:

```text
canonical prompt token stream
-> donor hidden states / token-aligned context
-> latent workspace
-> mandatory TRM/QTRM recursive core
-> answer-state readout loop
-> LM logits
-> autoregressive text
```

The answer-state loop is a renderer/readout/control layer. It may expose halt
or depth gating, but it is not allowed to become a separate Mythos/OpenMythos
reasoning path unless ablations show that it improves raw reasoning and
generation through the same LM-logit path.

## Why

The current strongest raw-recursive evidence comes from the Ouro answer-halt
TRM-style path: depth and halt-gate ablations can collapse performance while
the full path solves the forced-choice smoke gate. That is the first defensible
foundation for QTRM raw reasoning.

The later Mythos-style answer-loop joint decoder experiment did not promote:

```text
generation smoke8: 0/8 in every mode
causal forced-choice smoke4:
  donor_only: 0/4
  core_off: 0/4
  core8 full: 2/4
  decoder_off: 4/4
  halt_gate_off: 2/4
```

That result says the extra answer-side loop is not the reliable reasoning
source. It can even obscure the accepted halt/core signal.

## Rule

OpenMythos/Mythos and Parcae-style recurrence may be used only as stability
ideas inside or around the TRM core:

- stable input injection;
- loop-index conditioning;
- contraction telemetry;
- no-grad recurrence followed by short backprop-through-depth;
- ACT/PonderNet-style halting.

They are not canonical as a separate answer loop until these gates pass:

```text
full > donor_only
full > core_off
full > answer_loop_off / mythos_off
deeper or halted TRM recursion improves held-out reasoning
greedy/autoregressive generation improves, not only forced-choice scoring
```

## Next Work

1. Keep `core_world_model_enabled=false` and MemoryOS off for raw recursive
   gates.
2. Make the TRM carry/ACT gap explicit: carry object, reset-on-halt,
   no-grad inner cycles, detached carry continuation, per-sequence halt.
3. Use Mythos/Parcae only to improve the stability of the TRM state update,
   not to create a second answer-deciding module.
4. Treat renderer work as a separate claim from raw reasoning: a checkpoint can
   be good at forced-choice reasoning and still fail generation.
