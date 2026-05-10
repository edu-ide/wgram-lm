# Core Soft-Prefix Donor Renderer Reject

Date: 2026-05-08

Status: rejected L0/L1 donor-internal renderer falsifier.

## Gate

```text
Target level:
  L0/L1 donor-internal renderer falsifier

Major bottleneck:
  latent-core-to-autoregressive-text

Baseline to beat:
  donor_only_no_evidence
  soft_core_off_no_evidence

Required score:
  soft_full greedy generation must beat donor and core_off

Required ablation drop:
  soft_full > soft_core_off
```

## Implementation

Added:

```text
scripts/304_train_core_soft_prefix_donor.py
tests/test_core_soft_prefix_donor.py
```

Path:

```text
prompt tokens
-> frozen Qwen hidden states for QTRM
-> QTRM core_loop_readout_hidden
-> core-conditioned virtual donor embedding tokens
-> frozen Qwen transformer via inputs_embeds
-> donor lm_head
-> greedy generation
```

This is stronger than final-hidden ReFT-lite because the QTRM-conditioned
signal passes through donor transformer layers, not only the final lm_head.

## Smoke

Artifact:

```text
local_eval/research_gate_runner/core_soft_prefix_smoke/report.json
```

Result:

```text
inputs_embeds forward/backward path works
adapter-only gradient path works
no OOM on 1-step smoke
```

## Small Gate

Artifact:

```text
local_eval/research_gate_runner/core_soft_prefix_s080/report.json
```

Result:

```text
decision: rejected
train token_acc:              1.0 by step 20
held-out soft_full_token_acc: 0.6667
held-out core_off_token_acc:  0.4583
generation donor:            0/4
generation soft_core_off:     0/4
generation soft_full:         0/4
```

Example completions:

```text
gold 300015 -> soft_full 100009 / 50007
gold 400037 -> soft_full 100013 / 50009
```

## Interpretation

Soft-prefix can move donor token probabilities through the internal transformer
path, but this does not yet survive greedy autoregression.

The current failure is not simply "QTRM signal is outside donor." Even when the
signal is inside donor layers via virtual embeddings, single-pass teacher-forced
training still suffers from exposure-bias and numeric continuation instability.

## Decision

Reject soft-prefix as-is. Do not promote it as canonical.

## Next Candidate

The next experiment must change the training/eval target, not just the
injection site:

```text
1. self-rollout / scheduled-sampling training through the donor path, or
2. true internal donor residual-layer hook with layer/token sweep, or
3. reset to a simpler renderer task where held-out greedy generation can be
   solved before returning to mixed list arithmetic.
```

Promotion still requires:

```text
greedy generation > donor
greedy generation > core_off
the improvement disappears under core_off
no hidden answer channel
```

## First-Token Focus Gate

Artifact:

```text
local_eval/research_gate_runner/core_soft_prefix_firsttok_s120/report.json
```

Result:

```text
decision: rejected
train token_acc:             unstable, 1.0 at step 30 then 0.0 later
held-out soft_full_token_acc: 0.0000
generation soft_full:         0/4
```

Interpretation:

```text
Increasing scale and focusing only on the first answer token does not create a
stable causal bridge. It makes the adapter brittle and still fails greedy
generation.
```

Updated decision:

```text
Stop local soft-prefix variants. The next step is either a true donor residual
layer hook with a layer sweep, or a smaller greedy-generation renderer reset
task before returning to mixed list arithmetic.
```

## Reset Exception: Arithmetic Short-Surface Gate

The mixed-list soft-prefix path remains rejected, but a smaller arithmetic-only
renderer reset produced the first causal greedy-generation gain:

```text
docs/wiki/decisions/arithmetic-renderer-reset-soft-prefix.md
```

This does not promote the mixed-list renderer. It narrows the next work to a
short-surface reset gate where generation can be improved quickly and measured
with donor/core_off ablations.
