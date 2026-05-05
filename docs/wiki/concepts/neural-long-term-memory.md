# Neural Long-Term Memory

Neural long-term memory is the memory axis where the model stores information
outside the immediate prompt attention window, often through a learned memory
module or fast weights.

## QTRM Memory Layers

QTRM should keep these memory concepts separate:

| Layer | Current status | Purpose |
| --- | --- | --- |
| Donor KV/context | inherited from Qwen donor | base language modeling over the current prefix |
| LatentWorkspace | implemented | in-context working memory over projected donor/text states |
| Gated latent update | implemented, optional | LM2/G-MemLLM-inspired update/preserve/overwrite control inside workspace layers |
| Recursive core | implemented | looped latent computation over workspace slots |
| Retrieval/MemoryOS | future | explicit external evidence and episodic records |
| Titans-style NeuralMemory | future ablation | learned long-term/test-time memory state |
| In-Place TTT | future ablation | donor-side adaptive fast-weight updates |

## What Counts As Latent-Space Inference

QTRM can be described as performing latent-space computation because:

- Qwen hidden states are projected into QTRM-width context tokens.
- Learned workspace slots cross-attend to that context.
- Optional gated workspace layers decide how much latent state to preserve or
  overwrite after cross-attention.
- The recursive core updates `z_l` and `z_h` latent states for multiple cycles.
- The resulting `z_h` states are prepended before coda decoding and influence
  QTRM residual logits.

This should not yet be overstated as proven latent reasoning:

- The donor is still the base generation policy.
- The current residual ablation mostly verifies stability, not reasoning gains.
- We have not yet shown a task where the latent workspace improves over donor
  alone under controlled metrics.
- Hidden latent recurrence is not equivalent to visible chain-of-thought.

Precise wording:

```text
QTRM performs looped latent-workspace computation over donor representations.
It is being trained as a residual reasoning/memory adapter over Qwen donor
logits. It is not yet a standalone latent-reasoning LM.
```

## Current Gated Memory Fit

The implemented gated workspace is the conservative first step:

```text
previous latent slot
+ cross-attended context update
+ update/reset/candidate gates
-> next latent slot
```

It is local to the current forward pass. It does not yet persist across user
turns, documents, or inference calls. Its contribution must be measured with
`qtrm_workspace_gate_off_with_evidence`, not assumed from architecture alone.

## Titans-Style Memory Fit

Titans-style memory should be evaluated as a plug-in after residual stability is
established. The cleanest integration boundary is:

```text
Qwen hidden states
  -> QTRM projection
  -> LatentWorkspace
  -> optional NeuralMemory read/write
  -> Recursive core / coda
  -> residual logits
  -> donor-logit fusion
```

Initial ablations should be small:

- no neural memory;
- read-only neural memory state;
- train-time write only;
- inference-time write with detached memory state;
- retrieval/MemoryOS baseline.

## Evaluation Gates

Do not add Titans-style memory to the main path unless it improves a memory
metric without damaging donor stability:

- exact-match or answer-F1 on a local-context memory task;
- needle retrieval over synthetic long contexts;
- Korean fact recall from provided evidence;
- repetition rate and entropy remain near donor baseline;
- no regression on residual scale stability tests.
