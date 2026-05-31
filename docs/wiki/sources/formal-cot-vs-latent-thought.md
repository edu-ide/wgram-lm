# Formal CoT Vs Latent Thought

Status: ingested, 2026-05-03.

## Source

| Item | Local artifact | Upstream |
| --- | --- | --- |
| Paper | `references/papers/recurrent_depth/formal_comparison_cot_latent_thought_2509.25239.pdf` | <https://arxiv.org/abs/2509.25239> |
| Official code | `references/official/cot-vs-loop` at `783fa90` | <https://github.com/kevin671/cot-vs-loop> |

Paper:

```text
A Formal Comparison Between Chain of Thought and Latent Thought
Kevin Xu, Issei Sato
arXiv:2509.25239v2, 2026-01-29
```

## Core Claim

This paper is important because it does not say "latent thought dominates
CoT". It gives a formal separation:

- Latent thought / looped transformers are better suited to parallelizable
  computation. With enough loop iterations, the model can update many hidden
  positions together.
- CoT is better suited to inherently sequential stochastic procedures,
  especially approximate counting and sampling, because sampled tokens can act
  as explicit random choices.
- Therefore CoT and latent thought are not interchangeable. A robust reasoning
  system should know which paradigm a task needs.

## Local Code Reading

The official repo contains three mechanisms relevant to QTRM:

| Mechanism | File | QTRM implication |
| --- | --- | --- |
| Hidden-state feedback loop | `references/official/cot-vs-loop/models/looplm.py` | The previous hidden states become the next loop's input embeddings. This is closer to a real loop LM than a passive latent prefix. |
| Loop-index modulation | `references/official/cot-vs-loop/models/tmlt.py` | Each recurrence step receives a timestep embedding, matching QTRM's need for depth-conditioned state transitions. |
| CoT internalization curriculum | `references/official/cot-vs-loop/experiments/train_distill.py` | It shortens explicit CoT while increasing latent steps. This supports staged QTRM training instead of jumping directly to all-latent reasoning. |

## QTRM Design Consequences

1. Keep the mandatory recursive core. The paper strengthens the case that
   recurrence is useful for parallelizable latent computation.
2. Do not claim that pure latent recursion replaces all CoT. That would
   contradict the paper's separation result.
3. Split raw-intelligence gates by problem family:

```text
latent-favorable:
  graph reachability/connectivity
  circuit/DAG propagation
  dynamic-programming-like table updates
  local constraint propagation

CoT-favorable or stochastic-favorable:
  approximate counting
  sampling
  serial arithmetic traces
  tasks requiring explicit random choices
```

4. Treat CoT as a teacher and audit trace, not as something to delete
   unconditionally. The correct target is staged internalization:

```text
visible trace -> shortened trace -> latent steps -> short answer
```

5. Add task metadata to future raw gates:

```text
reasoning_family
expected_paradigm
requires_stochasticity
parallel_depth_estimate
serial_trace_length_estimate
```

6. The current QTRM staged loop-readout result should be described as a weak
   positive signal, not as broad raw intelligence. It passed a tiny 8-case
   gate but failed to scale to 16 cases. This paper says the next evaluation
   must ask whether the cases are latent-favorable or CoT-favorable before
   concluding that the loop architecture itself is dead.

Implemented 2026-05-03:

```text
scripts/190_build_pure_recursive_reasoning_cases.py
scripts/192_eval_raw_intelligence.py
scripts/194_build_pure_recursive_reasoning_preferences.py
src/wgram_lm/eval/raw_intelligence_gate.py
```

The raw-intelligence case builder now emits:

```text
reasoning_family
expected_paradigm
requires_stochasticity
parallel_depth_estimate
serial_trace_length_estimate
```

The eval script preserves these fields in JSONL records, and the raw gate
summary now includes `by_expected_paradigm` so a future report can distinguish
latent-favorable failures from CoT-favorable failures.

## Architecture Implication

The next serious QTRM architecture should be hybrid but still SSOT:

```text
one canonical prompt token stream
-> donor/prelude encoder
-> mandatory recurrent state-machine core
-> loop-state readout
-> answer/action logits
```

The hybrid part is not a second evidence path. It is the training/evaluation
policy:

- latent loop for parallelizable internal state updates;
- short explicit state/action tokens when the task is inherently sequential;
- optional stochastic decode only when the task formally needs sampling.

## Promotion Rule

QTRM should not be promoted as a latent-reasoning architecture unless it passes
both families:

```text
parallelizable gate:
  core8 > core1/core_off/donor
  depth changes answers or calibrated confidence

sequential/stochastic gate:
  hybrid trace/latent mode >= CoT baseline at equal or lower token budget
  pure latent mode is not required to win if the task is CoT-favorable
```

This is a cleaner claim than "all latent thought is better": QTRM should learn
when recurrence helps, and avoid pretending that one internal format solves
every reasoning class.
