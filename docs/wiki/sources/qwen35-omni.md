# Qwen3.5 Omni Technical Report

Source:

- Paper: `references/papers/qwen35_omni_technical_report_2604.15804.pdf`
- URL: `https://arxiv.org/abs/2604.15804`
- Version checked: v2, revised 2026-04-21

QTRM relevance:

- This is not the exact `Qwen/Qwen3.5-2B-Base` donor config.
- It is important for the future multimodal/omni MemoryOS path.
- It covers Qwen3.5-Omni scaling, long context, audio/video understanding,
  Hybrid Attention MoE, Thinker/Talker split, ARIA speech alignment, and
  temporal audio-visual grounding.

Architecture signal:

- Qwen3.5-Omni uses a Hybrid Attention Mixture-of-Experts framework for both
  Thinker and Talker.
- It supports long audio/video contexts and structured temporal grounding.

QTRM implication:

- For text/image/video only, use the Qwen3.5 2B HF configs as the direct donor
  reference.
- For future audio/video MemoryOS and omni-agent work, use this report as the
  high-level architecture reference.
