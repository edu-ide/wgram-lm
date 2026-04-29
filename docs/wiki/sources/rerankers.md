# Reranker References

Source links:

- Qwen3-Reranker-0.6B:
  <https://huggingface.co/Qwen/Qwen3-Reranker-0.6B>
- Qwen3 Embedding/Reranker paper:
  <https://arxiv.org/abs/2506.05176>
- NVIDIA Llama Nemotron Rerank 1B v2:
  <https://huggingface.co/nvidia/llama-nemotron-rerank-1b-v2>
- Contextual AI Reranker v2:
  <https://huggingface.co/ContextualAI/ctxl-rerank-v2-instruct-multilingual-1b>
- BGE reranker v2 m3:
  <https://huggingface.co/BAAI/bge-reranker-v2-m3>
- Jina reranker m0:
  <https://huggingface.co/jinaai/jina-reranker-m0>

## Decision

Use `Qwen/Qwen3-Reranker-0.6B` as the first real reranker for QTRM MemoryOS.

Reasons:

- It is Apache 2.0 and open-weight.
- It supports 100+ languages and 32k context.
- It is in the Qwen family, matching the current Qwen donor lineage.
- The official model card supports `sentence_transformers.CrossEncoder`.
- The 0.6B size is small enough for local experiments while remaining strong.

## Comparison Candidates

- `BAAI/bge-reranker-v2-m3`: practical lightweight multilingual baseline.
- `Qwen/Qwen3-Reranker-4B`: higher-quality Qwen comparison after 0.6B.
- `nvidia/llama-nemotron-rerank-1b-v2`: NVIDIA production-oriented candidate.
- `ContextualAI/ctxl-rerank-v2-instruct-multilingual-1b`: interesting
  instruction-following/conflict-aware reranker, but non-commercial license.
- `jinaai/jina-reranker-m0`: multimodal/document-image reranking candidate for
  future multimodal MemoryOS.

## QTRM Integration Pattern

The intended MemoryOS path is:

1. Dense retrieval with Harrier 270M.
2. Candidate pool, e.g. top 20.
3. Cross-encoder reranking with Qwen3-Reranker.
4. Evidence prompt injection with the top 3 reranked records.
5. Donor-only versus QTRM-residual generation comparison.

This separates candidate recall from evidence precision. It also makes it
possible to test whether QTRM benefits from better evidence ordering or merely
copies whatever the donor sees.
