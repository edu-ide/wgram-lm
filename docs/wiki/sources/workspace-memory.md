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
- `references/papers/memory_workspace/perceiver_2103.03206.pdf`
- `references/papers/memory_workspace/perceiver_io_2107.14795.pdf`
- `references/papers/memory_workspace/perceiver_ar_2202.07765.pdf`
- `references/papers/memory_workspace/blip2_qformer_2301.12597.pdf`
- `references/papers/memory_workspace/flamingo_2204.14198.pdf`
- `references/papers/memory_workspace/open_flamingo_2308.01390.pdf`
- `references/papers/memory_workspace/recurrent_memory_transformer_2207.06881.pdf`
- `references/papers/memory_workspace/associative_recurrent_memory_transformer_2407.04841.pdf`

Key implementation notes:

- Perceiver maps large inputs into a compact latent bottleneck and then performs
  deep processing in latent space.
- OpenFlamingo's `PerceiverResampler` repeats latent cross-attention and feed
  forward blocks over learned latents.
- BLIP-2 Q-Former uses learned query tokens and cross-attention to bridge frozen
  encoders and frozen language models.
- RMT/ARMT append memory tokens to base model inputs and carry or update memory
  across segments. This is closer to persistent/long-context memory than QTRM's
  current in-context workspace.

QTRM implication:

The old `LatentWorkspace` was a single learned-query cross-attention block. That
is only a sketch of a workspace. The production direction should use a
Perceiver/Q-Former-style repeated query adapter for in-context working memory,
and separately evaluate RMT/ARMT-style memory tokens for persistent or
segment-level memory.
