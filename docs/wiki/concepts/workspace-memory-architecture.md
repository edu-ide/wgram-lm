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
   memory or donor-side In-Place TTT. MemoryOS is a runtime/system layer, not a
   QTRM model block, and it is not yet trained as a QTRM read/write policy.

Design correction:

- Qwen donor remains the base generator/base policy.
- QTRM should not replace Qwen's language head from scratch.
- QTRM should add a small residual over donor logits and route workspace/memory
  information into that residual.
- The workspace can keep Perceiver/OpenFlamingo/Q-Former patterns as a compact
  connector baseline: learned queries, repeated cross-attention, latent
  processing, and feed-forward blocks. These patterns are not enough to claim
  modern long-term memory or latent reasoning.
- The next workspace step is LM2/G-MemLLM-style gated latent memory: preserve the
  donor/base path, add a complementary memory lane, and make update/preserve/
  overwrite behavior measurable with ablations.
- The long-memory scale step is MSA-style sparse memory routing, not a deeper
  Perceiver stack.

Implementation status:

- `src/qtrm_mm/workspace.py` now supports Perceiver/OpenFlamingo-style repeated
  latent layers through `workspace_layers`, `workspace_ff_mult`, and
  `workspace_include_latents_in_kv`.
- `src/qtrm_mm/workspace.py` now also supports an optional gated latent memory
  update through `workspace_memory_gate_enabled` and
  `workspace_memory_gate_init_bias`. The gate is GRU-like: each latent slot can
  preserve the previous state, reset part of it, or overwrite with a candidate
  state computed from the cross-attended context.
- `src/qtrm_mm/qtrm_model.py` exposes `workspace_update_gate_mean` telemetry and
  supports `disable_workspace_memory_gate=True` for causal ablation.
- `src/qtrm_mm/qtrm_model.py` also accepts `workspace_text_states` and
  `workspace_attention_mask` for workspace/dual ablation probes. This input is
  not the canonical model architecture and should not be drawn as MemoryOS
  inside the model.
- `src/qtrm_mm/data/jsonl_dataset.py` and `src/qtrm_mm/training/train.py`
  support `train.workspace_evidence_injection=true` for supervised rows built
  from `MemoryOS evidence ... User prompt:` prompts.
- `configs/qwen35_2b_4090_current_arch_pretrain_probe.yaml` enables the gated
  workspace path for the next current-architecture training probe.
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

100M-token runtime interpretation:

- 100M+ tokens belongs in external MemoryOS, not in the direct model prompt.
- The first scalable serving pattern is retrieval -> rerank -> compress ->
  QTRM/donor generation.
- The QTRM model still receives one compiled working context; MemoryOS scale is
  a runtime capacity axis, not a model parameter axis.
- MSA is the current closest reference for the later end-to-end large-memory
  step: document-wise position handling, sparse top-k memory routing, compressed
  K/V or latent memory blocks, and interleaved multi-hop retrieval.
- LightMem and MemCoT are external MemoryOS references, not replacements for the
  trainable QTRM core. They inform retrieval filtering, consolidation, and
  iterative memory search.
- This means MemoryOS scale and model context length are separate axes. A
  100M-token memory pool can still feed a 4K/8K/16K working context.

Next ablations:

- `donor_logits_scale=1.0`, `qtrm_logits_scale=0.0`: donor passthrough baseline.
- `donor_logits_scale=1.0`, `qtrm_logits_scale=0.05/0.1`: residual adapter.
- `workspace_layers=1` versus `3` versus `6`.
- `workspace_include_latents_in_kv=false` versus `true`.
- `workspace_memory_gate_enabled=false` versus `true`.
- `qtrm_workspace_gate_off_with_evidence`: runtime gate ablation for trained
  gated-workspace checkpoints.
- `--evidence-injection workspace` plus
  `qtrm_workspace_memory_off_with_evidence`: strict gate for whether evidence
  reaches the answer through workspace-side memory rather than the prompt.
- MemoryOS prompt retrieval versus no retrieval on fixed tasks.
- Distractor retrieval and trained memory traces that can distinguish donor
  evidence copying from a QTRM-specific memory/residual policy.
- Cache Harrier retrieval inside long eval runs; the current correctness path
  now caches the embedder and cross-encoder process-locally, but long-running
  evals should still batch retrieval and reranking across cases.
- Train/evaluate explicit abstention and contradiction traces before claiming
  MemoryOS reasoning. Evidence recall alone is not enough.
- Later: RMT/ARMT-style segment memory tokens, Titans/ATLAS-style neural memory,
  or In-Place TTT donor-side fast weights, only after residual generation is
  stable.
- Later: MSA-style document-wise sparse latent memory after retrieval/rerank and
  abstention gates pass.
- Later: full LM2-style explicit memory module as a bounded adapter lane. The
  current gated workspace is only the first local step, not a faithful LM2 port.

Prior decision:

- Perceiver/Q-Former/OpenFlamingo-style learned slots should not be deleted, but
  they are now classified as a connector/bottleneck baseline. The modern memory
  claim must come from LM2/G-MemLLM-style gated causality and MSA/MemoryOS-style
  sparse external memory routing. See
  `docs/wiki/decisions/latent-workspace-prior-decision.md`.
