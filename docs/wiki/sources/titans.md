# Titans / titans-pytorch

Source status:

- Paper: `references/papers/long_term_memory/titans_2501.00663.pdf`
- Paper URL: https://arxiv.org/abs/2501.00663
- Repo: `references/official/titans-pytorch`
- Repo URL: https://github.com/lucidrains/titans-pytorch
- Local commit: `714a14c`
- Implementation status: unofficial PyTorch implementation.

## What The Source Claims

Titans frames attention as accurate short-term memory over the current context
and neural memory as a longer-term memory that can memorize historical context.
The paper introduces a family of architectures that combine attention with a
learned neural memory module and reports long-context gains, including
needle-in-haystack-style evaluations.

## Repo Surface

The repo exposes:

- `NeuralMemory`
- `NeuralMemState`
- `mem_state_detach`
- `MemoryAsContextTransformer`
- memory model variants such as `MemoryMLP`, `MemoryAttention`,
  `MemorySwiGluMLP`, and `GatedResidualMemoryMLP`

Important files:

- `titans_pytorch/neural_memory.py`: differentiable neural memory module.
- `titans_pytorch/mac_transformer.py`: Memory-As-Context transformer.
- `titans_pytorch/memory_models.py`: candidate memory networks.

## Relevant Implementation Ideas

`NeuralMemory` stores information by updating the weights of a small memory
model from key/value-derived surprise gradients. Key implementation features:

- chunked memory reads/writes through `chunk_size`;
- adaptive step size from the outer network;
- optional momentum and learned momentum combination;
- optional per-parameter learning-rate modulation;
- optional surprise gradient normalization;
- explicit `NeuralMemState` for inference-time memory state.

`MemoryAsContextTransformer` combines:

- local/segmented attention;
- persistent memory tokens;
- long-term memory tokens;
- optional neural memory layers;
- optional sliding-window attention.

## QTRM Relevance

Titans is relevant to QTRM as a long-term/test-time memory axis, not as an
immediate replacement for the current donor-logit residual path.

Current QTRM memory stack:

1. Qwen donor hidden states and logits: stable base language policy.
2. QTRM `LatentWorkspace`: in-context working memory over the current prompt.
3. QTRM recursive core: looped latent computation over workspace slots.
4. Future memory axis: retrieval, MemoryOS, RMT/ARMT, Titans-style neural
   memory, or In-Place TTT donor adaptation.

Recommended order:

1. Finish donor-logit residual scale ablation (`0.05` versus `0.10`).
2. Build a small memory/retrieval trace eval.
3. Add Titans as a separate memory ablation, not fused into the baseline.

## Risk

The implementation is not official. Treat it as a readable reference and
prototype source, not as ground truth. The QTRM integration should keep a local
interface boundary so a Titans-style memory can be swapped out without changing
donor integration, latent workspace, or logit fusion.
