# 2026-05-25 Official GDN2 Runtime Contract

Status: active launch policy.

## Plain-Language Decision

An official GDN2 experiment must use the official GDN2 engine.

If the official module, CUDA kernel path, ptxas, or checkpoint is not clean, the
run must stop. It must not silently continue with a Torch fallback or an old
fallback checkpoint while keeping an official experiment name.

In ordinary language:

```text
Do not let the student take the exam with a different engine hidden under the
desk, then file the score under the official-engine experiment.
```

## Runtime Rule

For DGX GB10 official GDN2 runs:

```bash
REQUIRED_TRITON_PTXAS_PATH=/usr/local/cuda-13.2/bin/ptxas
TRITON_PTXAS_PATH=/usr/local/cuda-13.2/bin/ptxas
```

Both variables must be explicit. The code must not auto-discover or default to a
different ptxas path.

For local runs, use the same explicit pair with the local installed ptxas path.

Current local RTX 4090 note:

```bash
REQUIRED_TRITON_PTXAS_PATH=/usr/local/cuda-12.8/bin/ptxas
TRITON_PTXAS_PATH=/usr/local/cuda-12.8/bin/ptxas
```

Do not use `/usr/local/cuda/bin/ptxas` on the current local machine for Triton
3.3.1 official-GDN2 runs. It points at CUDA 13.0 and produced:

```text
RuntimeError: Triton only support CUDA 10.0 or higher, but got CUDA version: 13.0
```

Plain-language read:

```text
This is not an architecture failure and not a reason to fall back to Torch.
It is a toolchain pin. Use the compatible local ptxas and keep the official
engine honest.
```

## Checkpoint Rule

Fallback checkpoints are legacy evidence. They are not clean resume bases for
official GDN2 one-body training.

Reject a resume checkpoint if it contains any of these keys:

```text
*.mixer.runtime_fallback.*
*.mixer.impl.in_proj.*
*.mixer.impl.gate_proj.*
*.mixer.impl.out_proj.*
```

These are Torch fallback mixer traces, not official GDN2 traces.

## Executable Guard

Use the plain-language preflight before continuing a Stage95-style run:

```bash
REQUIRED_TRITON_PTXAS_PATH=/usr/local/cuda-13.2/bin/ptxas \
TRITON_PTXAS_PATH=/usr/local/cuda-13.2/bin/ptxas \
bash scripts/559_run_stage95_blt_partial_then_full_dgx.sh preflight
```

The underlying checker is:

```text
scripts/613_preflight_official_gdn2_contract.py
```

It checks:

- ptxas contract is explicit and executable;
- resume checkpoint has no legacy fallback keys;
- checkpoint/report runtime says `actual_delta_runtime=official_runtime`;
- fallback counters are zero;
- one-body runs are not accidentally resumed from `decoder_latent_mode=add`.

## Legacy Evidence

On DGX, the previous Stage95B resume path is blocked by this guard:

```text
checkpoint:
  local_eval/20260524_STAGE95B_DGX_1B_BLT2_PARTIAL_MODELONLY/last_model.pt

blockers:
  legacy fallback delta-mixer keys are present
  checkpoint decoder_latent_mode=add, expected=one_body

warnings:
  checkpoint delta runtime summary missing
  report delta runtime summary missing
```

Decision:

```text
Do not resume new official-GDN2 one-body work from that checkpoint.
Start a clean official-GDN2 run or resume only from a checkpoint that passes
the preflight.
```

Current launcher policy:

```text
scripts/559_run_stage95_blt_partial_then_full_dgx.sh now defaults to clean
20260525 Stage95G/H output directories, not the legacy Stage95B/C paths.
The run path calls preflight automatically before partial training and again
before full continuation from the selected resume checkpoint.
```

## Why This Matters

The project is trying to understand whether the architecture works. If the
runtime silently swaps the engine, then a good or bad loss curve no longer
explains the architecture. That is not robustness. It is evidence pollution.
