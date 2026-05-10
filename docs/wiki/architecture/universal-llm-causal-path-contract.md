# Universal LLM Causal Path Contract

Status: canonical architecture principle, 2026-05-05.

## One-Sentence Rule

```text
QTRM may add latent reasoning, memory, typed state, and metacognition, but the
canonical model claim must still be a general LLM path from prompt tokens to LM
logits to autoregressive text.
```

## Canonical Path

All semantic input enters the model through one prompt/token route:

```text
chat template / user prompt / compiled context
-> tokenizer
-> token embeddings or frozen donor hidden states
-> QTRM latent workspace / recursive core / trainable memory
-> LM logits
-> autoregressive text generation
```

This means retrieval, MemoryOS, tools, validators, and dataset solvers may
prepare context or labels, but they must not become a second hidden answer
path.

## Single Reasoning Core Rule

The canonical QTRM reasoning claim has one learned recursive reasoning core:
the TRM/QTRM latent core. Additional loops may exist only as readout,
stability, or halting mechanisms unless ablations prove that they add raw
reasoning through the same LM-logit path.

Reject a design if it creates two competing answer-deciding routes:

```text
TRM core decides one state
and
answer-loop/Mythos/runtime sidecar decides another answer
```

The accepted route is:

```text
TRM/QTRM recursive core -> answer-state readout -> LM logits
```

The answer-state loop must depend on the core. It must not be promoted as a
parallel reasoning core unless `answer_loop_off`, `core_off`, depth, and
generation gates all prove its causal value.

## What Structured Modules Are Allowed To Do

Typed registers, operation selectors, algorithmic heads, verifier heads,
planner states, and memory readers are allowed when they satisfy all of these:

```text
1. Their inputs come from the canonical token stream, donor states, or QTRM
   latent states.
2. Their outputs feed the model's causal answer path or a measured internal
   state path.
3. They are ablatable at inference.
4. Held-out performance drops when they are disabled or corrupted.
5. They do not compute the final answer as an external rule solver.
```

For the next typed-register executor candidate, this contract means:

```text
prompt tokens
-> role binder
-> mandatory recurrent core
-> learned operation selector
-> typed registers
-> verifier-checked register update during training/eval
-> role-value readout
-> LM answer path
```

The verifier may judge whether a register update is correct. It must not
replace the learned update at inference.

## What Is Rejected

Reject a candidate as non-general LLM architecture if:

```text
external code computes the answer and the model only formats it;
retrieval/span-copy bypasses autoregressive reasoning for the model claim;
typed operations are hardcoded into the runtime instead of learned or selected
from model state;
the component cannot be ablated without changing the task harness;
donor-only or a simpler scripted baseline matches the claimed system;
the result only works because labels contain operation names unavailable at
real inference.
```

Synthetic tasks are still useful, but only as falsifiable gates. They test
whether the learned latent path can acquire primitive state-transition ability
before scaling to broader language tasks.

## Typed CE Scope

Typed CE, register fields, action labels, and algorithmic slots are not
universal LLM objectives by themselves. They are internal process probes or
auxiliary losses.

They can be promoted only when:

```text
the same prompt-token path feeds them;
their outputs affect the LM hidden/logit path;
module-off/core-off ablations hurt held-out LM answers;
the final answer is still produced by LM logits or autoregressive text.
```

If typed fields improve while the LM answer path does not improve, the result
is diagnostic, not canonical architecture progress.

## Promotion Gate

A structured internal module can become canonical only if:

```text
full QTRM > donor-only/simple baseline
full QTRM > module_off/core_off/memory_off
held-out performance improves, not only train loss
the final answer is produced through LM logits/autoregressive text
depth, memory, or metacognitive ablations prove causal contribution
```

For raw recursive reasoning, the strongest signal remains:

```text
core_steps=8 > core_steps=1
and
core_off < full
```

## Design Implication

The target is not a fixed calculator attached to a language model. The target
is a general LLM whose latent core can learn state updates, memory use,
uncertainty, and verification while keeping one causal token-to-logit answer
path.
