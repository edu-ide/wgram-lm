# Workspace Memory Architecture

Current interpretation:

QTRM has two different memory concepts that should not be conflated.

1. In-context working memory.
   This is the latent workspace and recursive core inside the forward pass. It
   compresses the current donor/text context into trainable latent slots and
   lets the recursive core operate on those slots.

2. External or persistent memory.
   This is MemoryOS/retrieval-style evidence, or future RMT/ARMT-style memory
   tokens carried across segments. It can also include Titans-style neural
   memory or donor-side In-Place TTT. It is not yet trained as a QTRM read/write
   policy.

Design correction:

- Qwen donor remains the base generator/base policy.
- QTRM should not replace Qwen's language head from scratch.
- QTRM should add a small residual over donor logits and route workspace/memory
  information into that residual.
- The workspace should follow Perceiver/OpenFlamingo/Q-Former patterns: learned
  queries, repeated cross-attention, latent processing, and feed-forward blocks.

Implementation status:

- `src/qtrm_mm/workspace.py` now supports Perceiver/OpenFlamingo-style repeated
  latent layers through `workspace_layers`, `workspace_ff_mult`, and
  `workspace_include_latents_in_kv`.
- `src/qtrm_mm/qtrm_model.py` supports donor-logit residual generation through
  `donor_logits_scale` and `qtrm_logits_scale`.
- `configs/qwen35_2b_4090_donor_passthrough.yaml` verifies that direct donor
  logits preserve Qwen generation and remove the `world of the world` collapse.
- `configs/qwen35_2b_4090_donor_residual_workspace_pilot.yaml` verifies that a
  small QTRM residual can be attached without immediately destroying generation.
- `scripts/95_eval_memory_retrieval.py` runs a fixed-evidence MemoryOS-style
  probe and scores only generated completion text to avoid counting answers
  leaked in the prompt. The first probe shows Qwen donor logits already solve
  the evidence-copy task, while QTRM residual `0.10` preserves that behavior.
- `scripts/96_build_memory_retrieval_probe_index.py` builds a real MemoryOS
  vector index from probe cases using Harrier 270M embeddings. The paired eval
  mode `--evidence-mode memoryos` retrieves from that FAISS index and can
  case-filter synthetic probe records to avoid cross-case leakage.
- `src/qtrm_mm/memoryos/rerank.py` adds a reranking stage. The current real
  reranker candidate is `Qwen/Qwen3-Reranker-0.6B` through
  SentenceTransformers `CrossEncoder`; lightweight `none` and `lexical` modes
  exist for tests and fast probes.
- `data/eval/memory_reasoning_probe.jsonl` tests whether the MemoryOS path can
  support temporal conflict, authority conflict, multi-hop evidence chaining,
  and missing-answer abstention. The first run shows retrieval succeeds but
  abstention fails in both donor-only and QTRM residual modes.
- `src/qtrm_mm/memoryos/scale_plan.py` estimates large MemoryOS builds before
  ingestion. With 100M tokens, 512-token chunks, 64-token overlap, and Harrier
  270M's 640-dimensional embeddings, the current estimate is 223,215 chunks and
  about 0.532 GiB of float32 embedding storage.

100M-token interpretation:

- 100M+ tokens belongs in external MemoryOS, not in the direct model prompt.
- The first scalable serving pattern is retrieval -> rerank -> compress ->
  QTRM/donor generation.
- MSA is the current closest reference for the later end-to-end latent-memory
  step: document-wise position handling, sparse top-k memory routing, compressed
  K/V or latent memory blocks, and interleaved multi-hop retrieval.
- This means MemoryOS scale and model context length are separate axes. A
  100M-token memory pool can still feed a 4K/8K/16K working context.

Next ablations:

- `donor_logits_scale=1.0`, `qtrm_logits_scale=0.0`: donor passthrough baseline.
- `donor_logits_scale=1.0`, `qtrm_logits_scale=0.05/0.1`: residual adapter.
- `workspace_layers=1` versus `3` versus `6`.
- `workspace_include_latents_in_kv=false` versus `true`.
- MemoryOS prompt retrieval versus no retrieval on fixed tasks.
- Distractor retrieval and trained memory traces that can distinguish donor
  evidence copying from a QTRM-specific memory/residual policy.
- Cache Harrier retrieval inside long eval runs; the current correctness path
  now caches the embedder and cross-encoder process-locally, but long-running
  evals should still batch retrieval and reranking across cases.
- Train/evaluate explicit abstention and contradiction traces before claiming
  MemoryOS reasoning. Evidence recall alone is not enough.
- Later: RMT/ARMT-style segment memory tokens, Titans-style neural memory, or
  In-Place TTT donor-side fast weights, only after residual generation is
  stable.
- Later: MSA-style document-wise sparse latent memory after retrieval/rerank and
  abstention gates pass.
