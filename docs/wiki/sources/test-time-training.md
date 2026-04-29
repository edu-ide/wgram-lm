# Test-Time Training References

Purpose: track inference-time adaptation methods that may complement QTRM's
Qwen-donor residual adapter and MemoryOS path.

Downloaded references:

- `references/official/in-place-ttt`: ByteDance Seed In-Place TTT implementation.
- `references/papers/test_time_training/in_place_ttt_2604.06169.pdf`

Primary source:

- In-Place Test-Time Training, arXiv:2604.06169, submitted 2026-04-07,
  accepted as ICLR 2026 Oral according to the repository/paper metadata.

Key idea:

In-Place TTT updates fast weights at inference time without adding external
memory modules. In the released Qwen3 implementation, the fast-weight path lives
inside selected MLP blocks by adapting the MLP down-projection computation over
chunks. The objective is aligned with autoregressive next-token prediction
rather than a generic reconstruction loss.

Implementation notes from the official repo:

- Qwen3 support is implemented under `hf_models/hf_qwen3`.
- Config fields include `ttt_layers`, `ttt_mode`, `ttt_proj`, `ttt_lr`,
  `ttt_chunk`, and `ttt_target`.
- The README recommends `ttt_target: input_embed` for from-scratch pretraining
  and `ttt_target: hidden_states` for continual training.
- Their long-context stack targets Qwen3-8B and LLaMA-3.1-8B with RULER
  evaluation up to 256K context.

QTRM relevance:

- This is not a replacement for QTRM's latent workspace or MemoryOS retrieval.
- It is a strong candidate for **donor-side adaptive memory**: Qwen remains the
  base language policy, and selected internal MLP projections adapt during
  inference.
- The most conservative QTRM path is to keep current donor-logit residual
  generation stable first, then evaluate In-Place TTT as an optional donor
  adapter for long-context/read-new-information tasks.
