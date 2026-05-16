# Diffusion, Token Superposition, And Fast-Slow LLMs 2026

## Sources

```text
Mercury-2 product note
  url: https://www.inceptionlabs.ai/blog/introducing-mercury-2
  source type: company blog / product announcement

Mercury: Ultra-Fast Language Models Based on Diffusion
  arxiv: https://arxiv.org/abs/2506.17298
  source type: arXiv paper

Token Superposition Training
  arxiv: https://arxiv.org/abs/2605.06546
  source type: arXiv paper
  detailed note: sources/token-superposition-training.md

Fast-Slow Training of Language Models
  arxiv: https://arxiv.org/abs/2605.12484
  source type: arXiv paper

TiDAR: Think in Diffusion, Talk in Autoregression
  arxiv: https://arxiv.org/abs/2511.08923
  source type: arXiv / NVIDIA tech report
```

## Mercury / Diffusion LLM Direction

Claim summary:

```text
Mercury-style diffusion language models replace strictly one-token-at-a-time
decoding with parallel token refinement. The primary advertised benefit is
low-latency generation and high token throughput, not a direct proof of
stronger recursive reasoning.
```

QTRM interpretation:

```text
Useful for:
  decoder/readout bottleneck
  multi-token answer refinement
  reducing exposure-bias-like failures in answer generation
  comparing AR greedy answer decoding against parallel refinement decoding

Not sufficient for:
  proving that the recursive QTRM core became more intelligent
  replacing the mandatory native token->core->logits path
  bypassing core ablations with a separate answer sampler
```

Canonical boundary:

```text
Allowed candidate:
  prompt tokens
  -> native encoder
  -> mandatory QTRM/TRM recursive core
  -> core-conditioned diffusion/parallel refinement readout
  -> shared LM logits or shared token projection
  -> final text

Rejected shortcut:
  prompt
  -> standalone diffusion decoder
  -> answer

Reason:
  The shortcut would make the diffusion decoder the real model and would no
  longer prove QTRM-native recursive-core intelligence.
```

Future gate:

```text
qtrm_native_parallel_refinement_decoder_gate

Compare:
  AR greedy decoder
  AR beam decoder
  core-conditioned diffusion/parallel refinement decoder

Accept only if:
  generation_exact improves;
  language non-regression holds;
  core_off/state_reset/z_L_zero/z_H_zero/refinement_off ablations reduce the
  same final generation metric;
  no separate solver, renderer, or hidden answer channel computes the answer.
```

## Token Superposition Training

Claim summary:

```text
Token Superposition Training uses two phases. First, it averages contiguous
input token embeddings into bags and trains one prediction against the next
non-overlapping bag of tokens with multi-hot cross entropy. Second, it switches
back to ordinary next-token pretraining for recovery. It is mainly a
throughput/pretraining-efficiency method.
```

QTRM interpretation:

```text
Useful for:
  later QTRM-native language pretraining
  faster token consumption when training the native backbone from scratch
  multi-token auxiliary targets after the architecture is stable

Not the immediate fix for:
  the current L4 recursive-core causality gate
  z_L/z_H ablation weakness
  renderer collapse from a harmful core state
```

Candidate use:

```text
After bilingual language scaffold promotion:
  add TST-style phase-1 superposed bag pretraining
  then recover with normal next-token CE
  use normal next-token CE as the reference objective
  require language non-regression and generation_exact non-regression

Do not:
  claim raw-intelligence improvement from TST unless depth/core ablations show
  the same recursive gain.
```

Local status:

```text
pdf:
  references/papers/2605.06546-efficient-pre-training-with-token-superposition.pdf

primitive:
  src/qtrm_mm/tst.py

tests:
  tests/test_tst.py

next:
  wire a qtrm_native_language_tst_phase_smoke into the native language bootstrap
  after the larger English/Korean external corpus gate is recorded.
```

## Fast-Slow Training

Claim summary:

```text
Fast-Slow Training treats long-context adaptation as a two-level process:
fast context-level adaptation and slow parameter-level learning. The useful
idea for QTRM is not a sidecar prompt trick; it is the separation between fast
task/state updates and slower durable model updates.
```

QTRM interpretation:

```text
Directly relevant:
  z_L can be interpreted as fast working state
  z_H can be interpreted as slower abstract state
  nested learned update MLPs are a local fast/slow optimizer hypothesis

Potential repair for current bottleneck:
  z_H can solve too much without z_L in some accepted nested-MHA probes.
  Fast-Slow-style pressure can force z_L to carry fast per-instance state while
  z_H carries slower abstract control.
```

Candidate experiment:

```text
qtrm_native_fast_slow_latent_update_l4_repair

Architecture:
  prompt tokens
  -> native embeddings
  -> native encoder
  -> z_L fast state updated every L cycle
  -> z_H slow state updated every H step
  -> learned fast-slow consistency/anti-collapse losses
  -> native decoder/readout
  -> LM logits

Loss ideas:
  fast-state counterfactual:
    corrupt/reset z_L and require answer degradation

  slow-state retention:
    keep z_H stable across small prompt perturbations that preserve task rule

  fast/slow disentanglement:
    penalize z_H-only solving when z_L is zeroed;
    penalize z_L-only solving when high-level control is needed

  replay/KL:
    prevent slow backbone forgetting when fast-state update is strengthened
```

Acceptance rule:

```text
Promote only if:
  full_generation_exact passes the L4 threshold;
  full_minus_worst_ablation passes;
  z_L_zero and z_H_zero both reduce the same LM-logit generation metric;
  language non-regression holds;
  the final answer remains ordinary autoregressive text or a separately gated
  core-conditioned refinement decoder.
```

Implemented first repair slice:

```text
script:
  scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py

new loss:
  fast_slow_latent_counterfactual_loss

new CLI:
  --fast-slow-latent-loss-weight
  --fast-slow-latent-every
  --fast-slow-z-l-margin
  --fast-slow-z-h-margin
  --fast-slow-z-l-weight
  --fast-slow-z-h-weight

mechanism:
  full answer logprob must beat z_L-zero and z_H-zero ablation logprobs by
  configurable margins. This is the first minimal QTRM-native adaptation of
  Fast-Slow Training: it does not optimize external prompts, but pressures
  fast z_L and slow z_H to both matter in the normal LM-logit path.

runner gate:
  qtrm_native_fast_slow_latent_update_l4_repair

smoke report:
  local_eval/research_gate_runner/qtrm_native_fast_slow_latent_update_l4_repair_smoke/report.json

smoke decision:
  smoke_passed_fast_slow_latent_update_l4_repair
```

Boundary:

```text
This is not a full reproduction of the FST paper, because the paper's fast
weights are textual prompt/context populations optimized with GEPA while QTRM's
first repair slice uses latent z_L/z_H ablation pressure.

Claim level:
  faithful QTRM adaptation hypothesis, not official reproduction.
```

## Priority For QTRM

```text
Immediate:
  keep the current H=3/L=6 official-schedule split-mixer standard gate running.

If split-mixer passes:
  run seed stability and language non-regression before adding diffusion/TST.

If split-mixer fails:
  do not resume architecture shopping.
  try Fast-Slow-style latent update pressure first because it directly targets
  the z_L/z_H causal bottleneck.

Later:
  test Mercury-style parallel refinement only as a decoder/readout alternative.
  test TST only as a native pretraining throughput method.
```

## TiDAR: Think In Diffusion, Talk In Autoregression

Claim summary:

```text
TiDAR is a sequence-level hybrid LLM architecture. It drafts candidate tokens
in a diffusion-style parallel section, then samples/verifies final output
autoregressively. The point is not replacing AR language modeling; it is using
"free token slots" in one forward pass to improve decoding throughput while
keeping AR quality.

Reported scale:
  1.5B and 8B models

Reported claim:
  closes the quality gap with AR models while giving about 4.71x to 5.91x more
  tokens per second.
```

Core mechanism:

```text
prefix tokens:
  causal / AR attention, KV cache preserved

draft tokens from previous step:
  AR rejection sampling / verification

pre-draft tokens for next step:
  diffusion-style masked block, bidirectional inside the draft block

training:
  structured hybrid attention mask
  next-token prediction loss on causal prefix
  diffusion loss on fully masked draft section
```

QTRM interpretation:

```text
Useful for:
  later QTRM-native decoding throughput
  multi-token pre-draft / draft verification
  serving-friendly answer generation after language ability exists
  avoiding pure diffusion as the final answer path

Not the immediate fix for:
  QTRM-native language acquisition
  raw recursive reasoning
  z_L/z_H causal usefulness
  semantic generalization
```

Possible QTRM-native adaptation:

```text
prompt tokens
-> native embeddings
-> mandatory QTRM/TRM recurrent core
-> AR answer token logits
-> parallel draft slots conditioned on the same core state
-> AR verification / acceptance
-> final autoregressive text
```

Promotion rule:

```text
Treat TiDAR as a decoder/serving candidate only after:
  1. QTRM-native language bootstrap scales beyond tiny controlled answers;
  2. raw reasoning gates still pass with the same model;
  3. AR greedy baseline is stable;
  4. TiDAR-style draft/verify improves tokens/sec or tokens/forward without
     reducing answer quality;
  5. core_off/state_reset/refinement_off ablations reduce the same final
     answer metric.
```

Rejected shortcut:

```text
Do not replace QTRM-native with a standalone TiDAR/diffusion LM and call that
QTRM progress. TiDAR is useful only if the mandatory QTRM recurrent core remains
inside the causal answer path.
```
