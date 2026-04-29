# LLM Wiki

LLM Wiki means the architecture knowledge is accumulated as markdown pages
rather than rediscovered from raw sources every time.

For QTRM:

- `references` is source truth.
- `docs/wiki` is accumulated synthesis.
- `docs/REFERENCE_BASELINE.md` is the high-level source index and review rules.
- `docs/wiki/log.md` records what changed.

This matters because QTRM spans several independent research axes. Without a
wiki, it is easy to mix up which source supports which component.

Current correction rule:

- If a claim enters the project from a podcast, social post, or summary, record
  the corrected version in `sources/*` and the QTRM consequence in `concepts/*`.
- Do not let catchy parameter-count claims become architecture assumptions
  unless there is a paper, official code, or local ablation behind them.

## Current Long-Horizon Agent Bundle

The current long-horizon/agentic reference bundle is filed in:

- [Long-Horizon Agent References](../sources/long-horizon-agent-references.md)
- [Long-Horizon Agent Architecture](long-horizon-agent-architecture.md)

Stable synthesis:

- The system should not rely on one giant prompt or a model "thinking for hours"
  inside a single context.
- Long-running capability should be externalized into memory, skills, protocols,
  evidence indices, trace logs, and verification gates.
- QTRM should stay a donor-backed residual latent-workspace adapter while
  MemoryOS and an agent harness manage long-lived state.
- RLM-style recursion is a mode in the inference harness, not the default model
  architecture.
