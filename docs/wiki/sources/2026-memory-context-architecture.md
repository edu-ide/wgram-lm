# 2026 Memory And Context Architecture Update

Date: 2026-05-01

Status: current research-backed direction for QTRM MemoryOS architecture.

## Primary Sources Checked

| Source | Date | Relevant mechanism | QTRM implication |
| --- | --- | --- | --- |
| [MSA: Memory Sparse Attention for Efficient End-to-End Memory Model Scaling to 100M Tokens](https://arxiv.org/abs/2603.23516) | 2026-03 | end-to-end trainable sparse memory, document-wise RoPE, KV compression, Memory Interleaving | memory must be a trainable routing path, not only prompt stuffing |
| [G-MemLLM](https://arxiv.org/abs/2602.00015) | 2026-01 | frozen LLM backbone plus trainable gated latent memory bank | donor-backed QTRM can stay modular, but memory update/read gates must be causal |
| [MemCoT](https://arxiv.org/abs/2604.08216) | 2026-04 | task-conditioned short-term memories and evidence localization/expansion | memory reads should be conditioned by the task/question, not workspace-only |
| [LM2: Large Memory Models](https://arxiv.org/abs/2502.06049) | 2025-02 | decoder Transformer with auxiliary memory module, cross-attention, gated updates | auxiliary memory is compatible with the main LM path when it preserves original flow |
| [OpenAI File Search](https://developers.openai.com/api/docs/guides/tools-file-search) | accessed 2026-05-01 | model-triggered retrieval from vector stores before generation | production systems still keep retrieval outside the base weights, then feed relevant evidence |
| [Claude Context Windows](https://platform.claude.com/docs/en/build-with-claude/context-windows) | accessed 2026-05-01 | context window as working memory; more context can degrade recall without curation | larger context is not enough; context selection remains part of the architecture |
| [Gemini Long Context](https://ai.google.dev/gemini-api/docs/long-context) | accessed 2026-05-01 | 1M+ token input windows for long context use cases | commercial long context exists, but it is still a visible-context interface, not lifetime memory |
| [Qwen3.5-2B-Base model card](https://huggingface.co/Qwen/Qwen3.5-2B-Base) | accessed 2026-05-01 | Qwen3.5 uses efficient hybrid architecture with Gated Delta Networks and sparse MoE | Qwen donor is not itself a MemoryOS latent workspace; QTRM adds a separate memory/reasoning layer |

## Corrected 2026 Claim

Latest LLM systems do not usually expose a literal `prompt` versus
`workspace` split as a standard user-facing model architecture. The common
pattern is:

```text
visible request/context
+ optional tool or retrieval results
+ model internal hidden states
-> generation
```

Research memory architectures, however, increasingly separate memory capacity
from reasoning compute. The important difference is that the memory lane is
still read through a prompt-conditioned query/router. A workspace-only sidecar
that is not causally conditioned by the question is weak.

Therefore QTRM should not be described as "latest LLMs also split prompt and
workspace this exact way." The stronger and more accurate claim is:

```text
QTRM is evolving from an external MemoryOS/RAG sidecar into a
prompt-conditioned trainable memory reader, aligned with 2025-2026 memory-LM
directions such as LM2, G-MemLLM, MemCoT, and MSA.
```

## Design Rule For QTRM

The visible prompt and hidden workspace must not be independent paths. The
visible prompt/question must condition:

- which evidence tokens are selected;
- whether the evidence supports, refutes, or lacks the answer;
- whether to abstain with `UNKNOWN`;
- which memory span is copied or decoded into the answer channel;
- whether the QTRM residual is allowed to influence donor logits.

The updated canonical path is:

```text
visible prompt/question states
-> query projection

hidden workspace evidence tokens / compressed memory blocks
-> prompt-conditioned selector/span reader
-> support/refute/missing and UNKNOWN gate
-> selected answer span or no-answer state
-> answer-only copy/decoder channel
-> donor logits for surface fluency only
```

This is different from the rejected sidecar path:

```text
hidden workspace evidence
-> latent workspace
-> free-form residual logits
```

The rejected path can learn answer style or generic priors without proving that
the workspace evidence is causally used.

## Architecture Ranking After 2026 Update

1. Prompt-conditioned evidence span reader

   Best next small experiment. Many MemoryOS facts have literal answer spans.
   Span labels provide a sharper causal signal than free-form CE.

2. Gated latent memory bank

   Next after span-reader proof. This is closer to G-MemLLM/LM2: persistent or
   episode-level latent memory with gated write/read, tested by memory-off and
   swapped-memory ablations.

3. MSA-style sparse memory fork

   Best for 100M-token scale, but it is a root architecture fork. It likely
   requires healing/continual pretraining because replacing donor attention
   breaks direct weight compatibility.

4. External verifier/RAG sidecar

   Useful for user-visible correctness, but it proves a RAG pipeline rather
   than a learned internal QTRM memory reasoner.

## Acceptance Gates

For the span-reader direction to count as architecture progress:

- normal held-out MemoryOS score must improve over the previous `1/4` causal
  answer-bottleneck result;
- workspace-swap cases must return `UNKNOWN` when the swapped hidden evidence
  lacks the requested answer;
- `workspace_memory_off`, `evidence_span_reader_off`, or equivalent ablations
  must reduce performance;
- `evidence_bottleneck_off` must not be the mode that improves correctness;
- repeated-token collapse must remain absent.

If those gates fail twice, stop tuning the current answer bottleneck and move
to either a G-MemLLM/LM2-style trainable memory lane or an external verifier
sidecar depending on whether the goal is research proof or product accuracy.
