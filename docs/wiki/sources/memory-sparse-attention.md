# Memory Sparse Attention

Source links:

- Paper: <https://arxiv.org/abs/2603.23516>
- GitHub: <https://github.com/EverMind-AI/MSA>
- Local reference clone: `references/official/msa`
- Local clone commit checked during this pass: `30405b2`

## Source Facts

MSA proposes Memory Sparse Attention as an end-to-end trainable latent-memory
framework for memory contexts up to 100M tokens.

Important source claims:

- Full-attention LLMs are still effectively limited around 128K to 1M tokens for
  practical long-memory workloads.
- MSA separates memory capacity from reasoning by routing over document latent
  states rather than placing all tokens directly in the active context.
- The implementation uses document-wise RoPE, sparse top-k memory selection,
  compressed K/V states, and a Memory Parallel inference path.
- The paper reports less than 9% degradation when scaling from 16K to 100M
  tokens and 100M-token inference on 2xA800 GPUs.
- Memory Interleave alternates retrieval/context expansion/generation for
  scattered multi-hop evidence.

## QTRM Decision

MSA should be treated as the closest current reference for the next MemoryOS
architecture step. The project now also keeps an aggressive Qwen3.5-2B full-MSA
fork path for turning the current donor itself into an MSA-style donor.

Immediate QTRM mapping:

- Keep MemoryOS as an external memory pool that can exceed 100M tokens.
- Use Harrier embeddings + FAISS/HNSW + Qwen3 reranking as the current auditable
  retrieval layer.
- Add scale planning before large builds so chunk count, embedding storage, and
  backend choice are explicit.
- Train/evaluate QTRM latent-memory behavior only after retrieval/rerank recall
  and abstention probes are stable.

Future QTRM mapping:

- Add document ids and document-wise position handling to memory traces.
- Store compressed latent/KV summaries per memory block, not only raw text.
- Route top-k memory blocks into the QTRM latent workspace.
- Add a Memory Interleave eval mode for multi-hop retrieval over scattered
  evidence.

## Qwen3.5-2B Full-MSA Fork Track

Decision page:
[Qwen3.5 Full-MSA Fork](../decisions/qwen35-full-msa-fork.md).

Implemented scaffold:

- `src/wgram_lm/msa_qwen35.py`
- `scripts/129_prepare_qwen35_full_msa_fork.py`
- `tests/test_qwen35_full_msa_fork.py`

The scaffold rewrites the Qwen3.5-2B text config so all 24 layers target
Hugging Face's allowed `sparse` layer type with `qtrm_full_msa_fork=true`.
It records that the 6 original full-attention layers can seed a Qwen3.5-native
MSA implementation, while the 18 original linear-attention layers require
reinitialization/healing.

This is not yet a trained full-MSA donor. It is the necessary conversion
boundary before implementing the custom Qwen3.5-native MSA layer and donor
healing run.
