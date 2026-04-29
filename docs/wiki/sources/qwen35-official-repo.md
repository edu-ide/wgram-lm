# Qwen3.5 Official Repo

Source:

- Repo: `references/official/qwen35`
- Upstream: `https://github.com/QwenLM/Qwen3.5`
- Commit: `f1443092c299`

QTRM relevance:

- Official release notes for Qwen3.5 and Qwen3.6.
- Confirms Qwen3.5 positioning: native multimodal agents, unified
  vision-language foundation, efficient hybrid architecture.
- Confirms release timeline:
  - 2026-02-16: initial Qwen3.5 release
  - 2026-02-24: 122B/35B/27B release
  - 2026-03-02: 9B/4B/2B/0.8B release
- Points to official Hugging Face and ModelScope weights.
- Recommends Transformers, SGLang, vLLM, and common fine-tuning frameworks.

QTRM implication:

- Use Qwen3.5 repo/model cards for donor behavior, not generic Qwen3-only
  assumptions.
- Keep HF/Transformers generation and tokenizer behavior as the baseline.
