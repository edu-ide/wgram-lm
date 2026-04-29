# Test-Time Adaptation

QTRM should separate three adaptation mechanisms:

1. Donor base policy.
   Qwen donor logits provide the stable language distribution.

2. QTRM residual adapter.
   QTRM adds a small trainable residual over donor logits using workspace,
   recursion, world-model, and later MemoryOS evidence.

3. Donor-side test-time adaptation.
   In-Place TTT updates selected internal fast weights during inference. This is
   closer to adaptive long-context memory than ordinary prompt retrieval, but it
   changes the donor computation path and therefore must be gated carefully.

Why In-Place TTT matters:

- It is designed for standard Transformer LLMs, including a released Qwen3 path.
- It targets MLP down-projection fast weights instead of adding a side memory
  module.
- Its update objective is aligned with next-token prediction, which is a better
  fit for autoregressive donors than reconstruction-only TTT objectives.

QTRM decision:

Do not merge In-Place TTT directly into the current residual pilot. First keep
the donor-logit passthrough and QTRM residual baselines stable. Then add a
separate ablation axis:

- donor logits only;
- donor logits + QTRM residual;
- donor logits + In-Place TTT donor;
- donor logits + In-Place TTT donor + QTRM residual.

Minimum safety checks:

- compare against donor-only generation on the same prompts;
- run RULER-style long-context tasks or fixed synthetic retrieval tasks;
- log whether test-time updates improve target-token rank without increasing
  repetition;
- reset fast weights between unrelated sessions unless the experiment explicitly
  studies continual memory.
