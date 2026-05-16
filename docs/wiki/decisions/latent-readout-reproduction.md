# Latent Readout Reproduction

Date: 2026-05-11

Status: accepted L1 minimal reproduction, not QTRM L4 promotion.

## Purpose

The current QTRM blocker is not retrieval, MemoryOS, MSA, donor size, or typed
field supervision. It is:

```text
recurrent latent state -> stable autoregressive next-token synthesis
```

This reproduction isolates that bottleneck outside QTRM. It asks whether a
small recurrent decoder can read token-aligned latent states and emit the
correct digit tokens plus EOS under greedy rollout.

## Prior Mapping

Prior family:

```text
Scheduled Sampling / exposure-bias control
Looped Language Models / recurrent latent computation
latent-thought-to-LM-logits training
```

Local source notes:

```text
docs/wiki/sources/output-embedding-renderer.md
docs/wiki/sources/ouro-looplm.md
```

## Script

```text
scripts/331_train_latent_readout_reproduction.py
```

The script builds synthetic numeric answer cases, constructs fixed
token-aligned latent states, trains a small GRU readout, and reports both:

```text
teacher_forced_token_acc
teacher_forced_exact
greedy_token_acc
greedy_exact
```

Acceptance requires greedy exact generation, not teacher-forced accuracy alone.

## Results

Teacher-forcing-only profile:

```text
out:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  latent_readout_repro_teacher_forcing_s200/report.json

scheduled_sampling_prob: 0.0
train_steps: 200
teacher_forced_token_acc: 1.0
teacher_forced_exact: 1.0
greedy_token_acc: 1.0
greedy_exact: 1.0
decision: accepted
```

Scheduled-sampling profile:

```text
out:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  latent_readout_repro_scheduled_s200/report.json

scheduled_sampling_prob: 0.5
train_steps: 200
teacher_forced_token_acc: 1.0
teacher_forced_exact: 1.0
greedy_token_acc: 1.0
greedy_exact: 1.0
decision: accepted
```

Example greedy output:

```text
target:     600000
prediction: 600000
tokens:     6 0 0 0 0 0 <eos>
```

## Interpretation

This is an L1 reproduction only:

```text
It proves that a small recurrent latent readout can learn greedy exact
digit/EOS generation when the latent state is token-aligned and sufficient.
```

It does not prove:

```text
QTRM core states are token-aligned;
QTRM core states contain the correct scalar answer;
QTRM answer_state_loop has the right readout objective;
QTRM can solve mixed non-copy reasoning;
L4/general-LM promotion.
```

## QTRM Porting Criterion

The next QTRM candidate should not add more typed fields, renderers, or side
solvers. It should port the reproduction's minimal contract:

```text
core trajectory state
-> explicit per-output-step latent readout state
-> recurrent next-token decoder/readout
-> LM logits
-> greedy autoregressive answer
```

Promotion requires the same strict gate:

```text
full > donor-only
full > core_off
full > core_state_zero
full > answer_recurrent_off
greedy exact improves on held-out mixed non-copy cases
```

If QTRM still fails after this port while the standalone reproduction passes,
the missing piece is upstream: QTRM core states are not yet token-aligned or do
not contain the final scalar answer.
