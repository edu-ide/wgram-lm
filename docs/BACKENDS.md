# Backend Strategy

## Prototype mode

```yaml
attention_backend: sdpa
delta_backend: torch_gated_delta
strict_backends: false
```

This mode is trainable and useful for smoke tests, but it does not prove production throughput.

## Production candidate mode

```yaml
attention_backend: flash_attn
delta_backend: fla_gated_delta
strict_backends: true
```

If FlashAttention or FLA cannot be imported, model construction fails.

For Gated DeltaNet, the adapter follows the official FLA path:

```python
from fla.layers import GatedDeltaNet
```

The local `torch_gated_delta` backend remains smoke/debug only. It is not a
faithful Gated DeltaNet implementation.

KDA remains available as `delta_backend: fla_kda` if the installed FLA version
provides `KimiDeltaAttention`.

## Backend policy

```text
- Official kernels are used as backends, not as complete model replacements.
- Qwen/TRM/Parcae/Coconut full models are not blindly imported as architecture.
- Adapters expose stable interfaces so backends are replaceable.
- `strict_backends: true` must be used before any production or long training
  run that claims official backend coverage.
```
