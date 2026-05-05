# Workspace And Memory References

Purpose: ground QTRM's latent workspace and memory path in known architectures
instead of treating the current learned-query cross-attention block as a proven
memory module.

Downloaded references:

- `references/official/deepmind-research-perceiver`: DeepMind Perceiver /
  Perceiver IO implementation.
- `references/official/lavis`: Salesforce LAVIS implementation of BLIP-2 and
  Q-Former.
- `references/official/open_flamingo`: OpenFlamingo implementation with
  `PerceiverResampler`.
- `references/official/associative-recurrent-memory-transformer`: ARMT
  implementation for recurrent memory tokens and associative updates.
- `references/official/lm2`: Large Memory Models implementation, local commit
  `5f56b197b735`.
- `references/official/lightmem`: LightMem implementation for external agent
  memory, local commit `b11eccd23c7c`.
- `references/papers/memory_workspace/perceiver_2103.03206.pdf`
- `references/papers/memory_workspace/perceiver_io_2107.14795.pdf`
- `references/papers/memory_workspace/perceiver_ar_2202.07765.pdf`
- `references/papers/memory_workspace/blip2_qformer_2301.12597.pdf`
- `references/papers/memory_workspace/flamingo_2204.14198.pdf`
- `references/papers/memory_workspace/open_flamingo_2308.01390.pdf`
- `references/papers/memory_workspace/recurrent_memory_transformer_2207.06881.pdf`
- `references/papers/memory_workspace/associative_recurrent_memory_transformer_2407.04841.pdf`
- `references/papers/memory_workspace/lm2_2502.06049.pdf`
- `references/papers/memory_workspace/g_memllm_2602.00015.pdf`
- `references/papers/long_term_memory/lightmem_2510.18866.pdf`
- `references/papers/long_term_memory/memcot_2604.08216.pdf`
- `references/papers/long_term_memory/atlas_2505.23735.pdf`
- `references/papers/long_term_memory/miras_2504.13173.pdf`

Key implementation notes:

- Perceiver maps large inputs into a compact latent bottleneck and then performs
  deep processing in latent space.
- OpenFlamingo's `PerceiverResampler` repeats latent cross-attention and feed
  forward blocks over learned latents.
- BLIP-2 Q-Former uses learned query tokens and cross-attention to bridge frozen
  encoders and frozen language models.
- These learned-query adapters are now treated as connector baselines, not as
  sufficient evidence of modern long-term memory. Recent VLMs also often use
  simpler dynamic-resolution ViT plus MLP merger paths, as in Qwen2.5-VL.
- RMT/ARMT append memory tokens to base model inputs and carry or update memory
  across segments. This is closer to persistent/long-context memory than QTRM's
  current in-context workspace.
- LM2 adds an auxiliary memory module to a decoder-only Transformer. The memory
  acts as a contextual representation repository, interacts with input tokens
  through cross-attention, and updates through gating. It deliberately preserves
  the original Transformer information flow while adding a complementary memory
  pathway.
- G-MemLLM is the closest architectural shape to the next QTRM workspace step:
  frozen LLM backbone plus trainable latent memory bank with GRU-style gated
  update, preserve, and overwrite behavior.
- MSA belongs to the large external-memory axis: sparse top-k memory routing,
  document-wise position handling, compressed K/V, and interleaved multi-hop
  memory access.
- LightMem and MemCoT belong mostly to the agentic MemoryOS layer: memory
  filtering, consolidation, iterative search, and trajectory state outside the
  trainable QTRM forward pass.

QTRM implication:

The old `LatentWorkspace` was a single learned-query cross-attention block. That
was only a sketch of a workspace. The current direction is:

```text
Perceiver/Q-Former learned query slots
+ LM2/G-MemLLM-style gated latent update
+ separate MemoryOS/MSA/LightMem external memory layer
```

The decision is to keep the learned slots but demote their role:

```text
learned slots = connector/bottleneck
gated memory = internal memory causality target
MSA/MemoryOS = large external memory routing target
```

LM2 is relevant to the explicit-memory axis because it validates a design
principle close to the current QTRM adapter stance:

```text
preserve the base Transformer path
add a separate memory pathway
fuse memory information through cross-attention/gates
verify that general ability is not degraded
```

This should be treated as memory-architecture evidence, not as an immediate
donor-free language-policy fix. LM2 helps answer "how should QTRM memory be
attached?" more than "how does QTRM become a standalone LM?"
