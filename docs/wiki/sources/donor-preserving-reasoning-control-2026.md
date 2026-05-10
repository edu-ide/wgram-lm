# Donor-Preserving Reasoning Control 2026

Date: 2026-05-06

Purpose: identify current paper-backed methods for improving reasoning while
preserving a frozen donor language model's fluent autoregressive path.

## Why This Source Page Exists

The Ouro renderer probes rejected the private-QTRM-renderer direction:

```text
QTRM private hidden/head -> autoregressive text
```

The next direction should preserve the donor decoder/language path and inject
QTRM only as a bounded controller:

```text
donor hidden/logits remain the language path
QTRM recursive core supplies reasoning/control signal
small bounded residual intervention changes donor behavior
```

## Relevant 2025-2026 Prior

| Prior | Link | Useful idea for QTRM |
| --- | --- | --- |
| ThinkLogit / logit arithmetic | <https://arxiv.org/abs/2510.09354> | A smaller guider can improve a larger frozen target at decoding time by logit arithmetic. This is the closest method to QTRM-as-reasoning-guider over donor logits. |
| Proxy-Tuning | <https://arxiv.org/abs/2401.08565> | Apply a tuned-minus-untuned proxy logit delta to a stronger base model. QTRM can be tested as a learned delta over Qwen donor logits. |
| ReFT / LoReFT | <https://arxiv.org/abs/2404.03592>, code: <https://github.com/stanfordnlp/pyreft> | Freeze the base model and learn interventions on residual representations. QTRM should inject into donor-compatible residual states, not only into a private head. |
| BREP-ReFT | <https://arxiv.org/abs/2511.10707>, code: <https://github.com/LiangThree/BREP> | For math/reasoning, constrain representation interventions and focus early-prefix behavior to avoid disturbing numeric encodings. This warns against large hidden bridges. |
| Reasoning Representation Engineering | <https://arxiv.org/abs/2504.19483> | Reasoning performance can be modulated through residual-stream control vectors. QTRM should first act like a learned, example-conditioned control vector. |
| GLoRE | <https://arxiv.org/abs/2503.11314> | Long-CoT reasoning appears as steerable representation patterns plus domain-specific representation. QTRM can learn a prompt-conditioned reasoning-mode vector. |
| Reasoning finetuning repurposes latent representations | <https://arxiv.org/abs/2507.12638> | Reasoning finetuning reuses existing base-model directions. Prefer finding and modulating donor directions over relearning a separate renderer. |
| Small Vectors, Big Effects | <https://arxiv.org/abs/2509.06608> | Lightweight residual-stream vectors trained with RL can match larger behavioral shifts; layer choice matters. QTRM intervention should be small and layer-targeted. |
| Bottom-up Policy Optimization | <https://arxiv.org/abs/2512.19673>, code: <https://github.com/Trae1ounG/BuPO> | Treat hidden layers as internal policies; optimize intermediate policy distributions, not only final logits. Useful for Qwen because the paper reports Qwen-series progressive policy structure. |
| Dead Weights, Live Signals | <https://arxiv.org/abs/2604.08335> | Frozen LMs can communicate through learned continuous projections and residual-stream hooks. Supports QTRM as a frozen-donor graph node/controller instead of standalone decoder. |
| Speculative Thinking | <https://arxiv.org/abs/2504.12329> | Stronger model guidance can be applied only at reflective/reasoning points. QTRM can learn when to intervene rather than always overriding donor. |
| Thinking Intervention | <https://arxiv.org/abs/2503.24370> | Reasoning can be controlled by targeted interventions at thinking points. For QTRM this maps to a learned gate over donor intervention sites/tokens. |

## Method Ranking For QTRM

### 1. Donor-Logit Guider Delta

Formula:

```text
final_logits = donor_logits + alpha * gate * qtrm_delta_logits
```

Training target:

```text
qtrm_delta should increase correct answer/reasoning tokens
qtrm_delta should be near zero on donor-correct or uncertain rows
```

Why first:

```text
It preserves donor fluency and exactly matches ThinkLogit/Proxy-Tuning style
guidance. It also avoids the failed private QTRM renderer.
```

Required ablations:

```text
donor_only
delta_off
gate_off
core_off
alpha sweep
entropy/KL/repetition guard
```

### 2. Donor Residual-Stream Intervention

Formula:

```text
donor_hidden[layer, token] += alpha * gate * P(qtrm_core_state)
```

Why second:

```text
ReFT, steering vectors, GLoRE, BuPO, and Dead Weights all point to residual
stream intervention as the better place to preserve language while steering
reasoning.
```

Risk:

```text
Requires hooking the Qwen donor forward path and choosing stable layers/tokens.
Must start as eval/probe only, with a small alpha and donor-KL guard.
```

### 3. Internal-Policy / Layer-Policy Distillation

Formula:

```text
intermediate donor layer logits should move toward correct policy before final
layer convergence
```

Why later:

```text
BuPO is promising but heavier. It needs access to donor intermediate states and
layerwise logit-lens scoring, so it is a second-stage training objective after
safe intervention hooks exist.
```

## Rejection Rule

Reject any candidate that:

```text
improves forced-choice but keeps greedy generation at 0
damages donor fluency/repetition
requires QTRM private head to produce final text
works only when an external rule solver or sidecar computes the answer
```

## Next Implementation Candidate

Implement a bounded donor-logit guider first:

```text
Qwen donor forward -> donor_logits
QTRM core -> qtrm_delta_logits
QTRM gate -> scalar/token gate
final_logits = donor_logits + alpha * gate * clamp(qtrm_delta_logits)
```

Acceptance smoke:

```text
generation > donor_only on held-out reasoning rows
delta_off returns donor_only behavior
core_off loses the improvement
donor-correct rows do not regress
repeat/entropy/KL guard does not worsen
```

Only after that passes should we add residual-stream intervention hooks.

## 2026-05-08 Update: Final-Hidden ReFT-Lite Rejected

Implemented and tested a cheap ReFT-lite bridge:

```text
QTRM core_loop_readout_hidden
-> low-rank projection
-> add to donor final hidden state
-> frozen donor lm_head
```

Artifacts:

```text
scripts/303_train_donor_hidden_reft_lite.py
docs/wiki/decisions/donor-hidden-reft-lite-renderer-reject.md
```

Decision:

```text
rejected
```

Reason:

```text
Multi-token training slightly improved teacher-forced top1 but kept greedy
generation at 0/8. First-token-only training reached teacher-forced top1 1.0,
but core_off also reached 1.0, so the apparent gain was a non-causal adapter
bias rather than recursive-core reasoning.
```

Architecture implication:

```text
Final hidden-state addition is too late and too easy to collapse into a format
bias. The next donor-preserving experiment must intervene inside the donor
transformer path, either through an internal residual-stream hook or a
core-conditioned soft-prefix processed by frozen donor layers.
```

## 2026-05-08 Update: Core Soft-Prefix Also Rejected

Implemented a donor-internal soft-prefix falsifier:

```text
QTRM core_loop_readout_hidden
-> virtual donor embedding tokens
-> frozen Qwen transformer via inputs_embeds
-> donor lm_head
```

Artifacts:

```text
scripts/304_train_core_soft_prefix_donor.py
docs/wiki/decisions/core-soft-prefix-donor-renderer-reject.md
```

Decision:

```text
rejected
```

Reason:

```text
The path is live and trains, but held-out greedy generation remains 0/4. The
model improves teacher-forced token accuracy yet still emits nearby
intermediate numeric values under autoregressive rollout.
```

Architecture implication:

```text
The renderer bottleneck is now more specifically exposure-bias / rollout
stability, not merely "wrong injection location." The next candidate needs
self-rollout or scheduled-sampling pressure through the donor path, or a reset
to a simpler greedy-generation renderer gate before mixed list arithmetic.
```
