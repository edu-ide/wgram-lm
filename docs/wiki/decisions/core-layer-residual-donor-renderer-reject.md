# Core Layer Residual Donor Renderer Reject

Date: 2026-05-08

Status: rejected L0/L1 donor-internal renderer candidate.

## Gate

```text
Target level:
  L0/L1 donor-internal renderer falsifier

Major bottleneck:
  latent-core-to-autoregressive-text

Baseline to beat:
  donor_only_no_evidence
  layer_core_off_no_evidence

Required score:
  layer_full greedy generation must beat donor and core_off

Required ablation drop:
  layer_full > layer_core_off
```

## Implementation

Added:

```text
scripts/306_train_core_layer_residual_donor.py
tests/test_core_layer_residual_donor.py
```

Path:

```text
prompt tokens
-> frozen Qwen hidden states for QTRM
-> QTRM core_loop_readout_hidden
-> low-rank gated residual delta
-> frozen Qwen language decoder layer hidden state at answer-prediction positions
-> remaining frozen Qwen decoder path
-> frozen donor lm_head
-> greedy generation
```

The Qwen3.5 donor language layers are under:

```text
model.language_model.layers
```

## Smoke

Artifact:

```text
local_eval/research_gate_runner/core_layer_residual_arith_smoke/report.json
```

Result:

```text
layer: 20 / 24
steps: 5
teacher-forced full:     0.7059
teacher-forced core_off: 0.4118
generation donor:        0/4
generation core_off:     0/4
generation full:         0/4
```

Interpretation:

```text
The hook is wired and causally changes token probabilities, but the short smoke
does not prove generation improvement.
```

## Heldout4 Layer 20

Artifact:

```text
local_eval/research_gate_runner/core_layer_residual_arith_s240_eval4/report.json
```

Result:

```text
layer: 20 / 24
steps: 240
teacher-forced full:     0.6471
teacher-forced core_off: 0.4118
generation donor:        0/4
generation core_off:     0/4
generation full:         0/4
```

Example misses:

```text
417  -> 420
632  -> 624
851  -> 840
1074 -> 1094
```

## Heldout4 Layer 23

Artifact:

```text
local_eval/research_gate_runner/core_layer_residual_arith_s240_eval4_lastlayer/report.json
```

Result:

```text
layer: 23 / 24
steps: 240
teacher-forced full:     0.7059
teacher-forced core_off: 0.4118
generation donor:        0/4
generation core_off:     0/4
generation full:         0/4
```

## Decision

Reject layer-residual hook as the next shortest renderer path.

Reason:

```text
soft-prefix arithmetic heldout4 already reached 3/4 greedy generation, while
the layer-residual hook reaches 0/4 under two nearby language-layer positions.
```

Do not continue broad layer sweeps until a stronger reason appears. The current
shortest path remains the arithmetic soft-prefix reset with exposure-bias
control:

```text
best current heldout18:
  soft-prefix scheduled_sampling=0.3
  generation full:     4/18
  generation core_off: 1/18
  generation donor:    0/18

rejected:
  scheduled_sampling=0.6 -> full 0/18
  scheduled_sampling=0.3, 960 steps -> full 1/18
  layer residual hook -> full 0/4
```

Next shortest candidates:

```text
1. Keep soft-prefix scheduled_sampling near 0.3.
2. Add a warm-up schedule instead of applying rollout pressure from the start.
3. Add strict exact/first-number telemetry so loose contains does not hide
   renderer errors.
4. Only after heldout18 improves, return to heldout128 and mixed-list gates.
```
