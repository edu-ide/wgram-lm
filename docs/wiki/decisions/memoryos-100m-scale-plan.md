# MemoryOS 100M Scale Plan

Decision:

MemoryOS should scale beyond 100M tokens as an external memory pool. QTRM should
not attempt to read 100M tokens as a direct prompt on a 4090-class local setup.

## Scale Estimate

Default planning command:

```bash
PYTHONPATH=src .venv/bin/python scripts/97_plan_memoryos_scale.py --total-tokens 100000000
```

Staged planning command before a 100M build:

```bash
PYTHONPATH=src .venv/bin/python scripts/98_benchmark_memoryos_scale.py --token-targets 1M,10M
```

Default staged output:

- `runs/eval/memoryos_scale_plan_1m_10m.jsonl`

Current estimate with 512-token chunks and 64-token overlap:

- Estimated chunks: 223,215
- Harrier 270M embedding dimension: 640
- Float32 embedding storage: about 0.532 GiB
- Raw text at 4 bytes/token: about 0.373 GiB
- Build backend recommendation: `faiss_hnsw`
- Serving pattern: `retrieve-rerank-compress`

This means 100M tokens is not blocked by embedding storage. The real bottlenecks
are build time, index sharding, retrieval quality, reranking cost, evidence
compression, abstention, and multi-hop memory use.

## Architecture Direction

Stage 1: scalable RAG-like MemoryOS.

- Ingest text into chunks with stable source metadata.
- Build Harrier embeddings using sharded jobs for large corpora.
- Use FAISS/HNSW for local approximate search.
- Retrieve top-n, rerank with Qwen3-Reranker, and pass only selected evidence to
  the generator/QTRM residual path.

Stage 2: QTRM latent memory.

- Convert retrieved evidence into compact latent workspace inputs.
- Train traces where QTRM must abstain, resolve conflicts, and chain evidence.
- Measure donor-only versus QTRM residual on the same retrieved evidence.

Stage 3: MSA-style sparse latent memory.

- Add document ids and document-wise position treatment.
- Store compressed latent/KV summaries for memory blocks.
- Route top-k memory blocks into the upper QTRM layers or latent workspace.
- Add Memory Interleave-style repeated retrieval/context expansion for multi-hop
  questions.

Stage 4: long-horizon agent harness.

- Add an explicit inference mode router:
  `non_rlm_direct`, `memory_rag`, `rlm_no_subcalls`, and `rlm_recursive`.
- Keep huge context outside the prompt as indexed evidence/context variables.
- Allow RLM-style recursive subcalls only after non-recursive MemoryOS evidence
  use is stable.
- Require sandboxed execution, max step budgets, max subcall budgets, timeouts,
  trace logging, and verification gates.
- Store reflections and reusable skills separately from factual evidence.

## Go/No-Go Gates

- Retrieval recall: target evidence appears in top-n before rerank.
- Rerank recall: target evidence remains in top-k after rerank.
- Answer accuracy: generated answer hits aliases without prompt leakage.
- Evidence insufficiency: closed-evidence evals return `UNKNOWN`; open-world
  agent mode routes to `NEEDS_SEARCH` and expands retrieval/search before any
  final user response.
- Latency: top-n retrieval plus rerank must stay practical before scaling corpus
  size further.

Do not claim 100M-token reasoning until the hard memory reasoning probe passes
retrieval, rerank, answer, and abstention gates.

Immediate order:

1. Keep the small hard MemoryOS reasoning probe as the primary correctness
   gate.
2. Use `scripts/98_benchmark_memoryos_scale.py` only as a planning artifact for
   1M and 10M ingestion.
3. Move to 100M only after the model stops failing cases where the answer is
   absent, contradicted, or superseded by newer/signed evidence.
4. Treat RLM-style execution as an inference harness feature, not as a reason to
   skip retrieval, reranking, abstention, and held-out evaluation gates.
