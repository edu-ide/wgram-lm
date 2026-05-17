# DGX QTRM-Native Fastlane

Status: proposed execution path, 2026-05-17.

## Goal

Use DGX for the fastest credible route from the current QTRM-native scaffold to
a stronger reasoning model. The target is not another architecture-shopping
round. The target is:

```text
QTRM-native causal path
-> measurable recurrent-depth gain
-> scale from small slices to 256/512+ cases
-> preserve ordinary language generation
```

## Current Bottleneck

M6 is accepted only as a scoped synthetic raw-reasoning win. The DGX Qwen3.6
MTP GGUF proxy is weak on that custom suite, but this is not public benchmark
parity.

M7B is the current public-style bottleneck:

```text
64-case M7B:
  depth0: 0.09375
  depth4: 0.140625
  accepted small-slice core-depth signal

256-case M7B:
  depth0: 0.15234375
  depth4: 0.09375
  rejected scale-out
```

Therefore the fastest high-value work is core-depth scale-out repair.

## Fastest Credible Method

Use a three-layer fastlane:

```text
1. Slow weights:
   Train the QTRM-native recurrent core and minimal LM path with low drift.

2. Fast data/context:
   Use DGX Qwen3.6 only to generate or filter hard examples, not as a runtime
   donor. Final inference remains native token -> QTRM core -> LM logits.

3. Depth curriculum:
   Train and evaluate across depth0/1/2/4/8, with acceptance requiring
   depth4 or depth8 to beat depth0 and shallow depths on held-out cases.
```

This matches the useful part of recent prior work:

- Recurrent-depth latent reasoning: iterate a recurrent block to spend more
  compute in latent space before emitting tokens.
- Looped transformers: reasoning tasks often need effective depth more than
  raw parameter count.
- Attractor-style loop solving: stabilize recurrent models by encouraging
  convergence rather than treating each depth as an unrelated rollout.
- Fast-slow training: keep slow weights stable while fast task-specific data or
  context accelerates adaptation.
- Dynamic early exit: do not assume more loops are always better; learn or
  measure when to stop.

References:

```text
Scaling up Test-Time Compute with Latent Reasoning
https://arxiv.org/abs/2502.05171

Reasoning with Latent Thoughts
https://arxiv.org/abs/2502.17416

Solve the Loop: Attractor Models for Language and Reasoning
https://arxiv.org/abs/2605.12466

Learning, Fast and Slow
https://arxiv.org/abs/2605.12484

Dynamic Early Exit in Reasoning Models
https://arxiv.org/abs/2504.15895
```

## What Not To Do

Do not spend the next DGX cycle on:

```text
- more Qwen/Ouro transition-prior shopping without a 256-case core-depth gate;
- MemoryOS/RAG as a benchmark shortcut;
- answer-only CE that improves formatting but erases core-depth gain;
- teacher CoT imitation, because QTRM reasoning is supposed to remain latent;
- runtime donor claims, because the canonical goal is QTRM-native.
```

## DGX Execution Order

Use:

```bash
bash scripts/408_dgx_qtrm_native_fastlane.sh status
bash scripts/408_dgx_qtrm_native_fastlane.sh sync
bash scripts/408_dgx_qtrm_native_fastlane.sh m7-final-token-repair
bash scripts/408_dgx_qtrm_native_fastlane.sh m7-depth-256
bash scripts/410_run_trm_breakthrough_gate.sh
```

The first serious pass should use:

```text
M7 final-token repair:
  steps: 1200
  batch: 32
  train cases: available validation/auxiliary only
  eval: held-out balanced 256
  think_steps: 8

  objective:
    primary option-letter CE at depth8
    multi-depth CE at depth4 and depth8
    depth-gain margin: depth8 gold score > depth0/1/2/4
    trajectory KL: depth6 should be close to depth8

M7 depth gate:
  depths: 0, 1, 2, 4, 8
  pass condition:
    full depth gain >= 0.03 over depth0
    full depth gain >= 0.03 over best shallow
    invalid/prompt echo/pred collapse below limits
```

If this fails, do not widen the model first. Inspect by-family and by-depth
failure reports, then add only one of these:

```text
1. trajectory consistency loss:
   depth2/depth4/depth8 should converge toward the same answer distribution
   when the answer is stable.

2. attractor residual loss:
   penalize destructive latent movement after the answer distribution stabilizes.

3. family-balanced hard-negative replay:
   oversample only the families where depth4 loses to depth0.
```

## Promotion Rules

The fastlane is accepted only when:

```text
1. M7B 256 passes as a standalone rerun.
2. 512-case rerun keeps a positive core-depth gain.
3. language gate still passes.
4. core_off/depth0/state ablations remove the gain.
5. the checkpoint is reproducible across at least two seeds or one seed plus
   checkpoint-soup/non-regression audit.
```

Anything weaker is diagnostic, not a model-performance claim.

The hard final promotion gate is documented separately:

```text
docs/wiki/decisions/trm-like-breakthrough-bar.md
```
