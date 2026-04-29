# CoT To Latent Reasoning Sources

Status: reference synthesis, 2026-04-29.

Purpose: track papers and code for transferring explicit chain-of-thought
supervision into latent-space reasoning, plus halting references for adaptive
latent loops.

## Reference Map

| Area | Source | Local artifact | QTRM relevance |
| --- | --- | --- | --- |
| Continuous latent reasoning | Coconut, "Training Large Language Models to Reason in a Continuous Latent Space" | `references/papers/latent_reasoning/coconut_2412.06769.pdf` | Uses hidden states as continuous thoughts instead of decoding every reasoning step. Supports the idea that QTRM should not emit long CoT at inference. |
| CoT compression | CODI, "Compressing Chain-of-Thought into Continuous Space via Self-Distillation" | `references/papers/latent_reasoning/codi_2025.emnlp-main.36.pdf`, `references/official/codi` at `2c23146` | Directly matches the QTRM training target: explicit CoT as teacher, latent state as student. |
| Hybrid text/latent reasoning | HybridCoT | `references/papers/latent_reasoning/hybridcot_4mfGbMzTwu.pdf` | Warns against over-compressing all reasoning; critical symbols or operations may remain textual while semantic reasoning is latent. |
| Looped latent thoughts | Reasoning with Latent Thoughts / Looped Transformers | `references/papers/recurrent_depth/reasoning_with_latent_thoughts_looped_transformers_2502.17416.pdf` | Theoretical and empirical support for looped models simulating CoT-like steps through repeated latent computation. |
| Reliability warning | Do Latent Tokens Think? | `references/papers/latent_reasoning/do_latent_tokens_think_2512.21711.pdf` | Cautions that latent tokens may become shortcut placeholders. QTRM must prove latent contribution with ablations and adversarial/distractor evals. |
| Halting reference | TinyRecursiveModels | `references/official/tiny-recursive-models` at `c011037` | Provides the closest official-style reference for z_H/z_L carry, `q_halt/q_continue`, `halt_max_steps`, exploration, and halt loss. |
| ACT reference implementation | Associative Recurrent Memory Transformer ACT utilities | `references/official/associative-recurrent-memory-transformer/modeling_amt/act_utils.py` | Useful for classic ACT probability/remainder mechanics, but QTRM should start from TRM's simpler Q-head pattern. |

## Paper Links

- Coconut: <https://arxiv.org/abs/2412.06769>
- CODI: <https://aclanthology.org/2025.emnlp-main.36/>
- CODI code: <https://github.com/zhenyi4/codi>
- HybridCoT: <https://openreview.net/forum?id=4mfGbMzTwu>
- Reasoning with Latent Thoughts: <https://arxiv.org/abs/2502.17416>
- Do Latent Tokens Think?: <https://arxiv.org/abs/2512.21711>
- TinyRecursiveModels: <https://github.com/SamsungSAILMontreal/TinyRecursiveModels>

## QTRM Takeaways

1. CoT should be a teacher signal, not necessarily the final inference format.
2. Latent reasoning needs a transfer objective: hidden-state alignment,
   designated latent tokens/workspace states, answer correctness, and residual
   telemetry are all stronger than simply hiding CoT text.
3. Hybrid text/latent reasoning is safer than all-latent reasoning at the start.
   Keep short structured traces for verification and critical operations while
   moving repeated semantic reasoning into the workspace.
4. Latent reasoning claims require adversarial checks. Shortcut behavior is a
   known failure mode.
5. Adaptive computation should follow TRM-style halt supervision before any
   purely hand-written norm threshold is treated as the main halting mechanism.

## TRM Halting Details To Preserve

From the local TinyRecursiveModels reference:

- `halt_max_steps` caps recurrence.
- `halt_exploration_prob` prevents the halt head from seeing only one depth.
- carry stores `z_H`, `z_L`, `steps`, `halted`, and `current_data`.
- halted sequences reset their carry before receiving a new problem.
- the inner model emits `q_halt_logits` and `q_continue_logits`.
- the default TRM config sets `no_ACT_continue: True`, using the halt logit
  itself rather than a halt-vs-continue comparison.
- the loss includes task loss plus a halt-head loss against sequence
  correctness.

QTRM has the first implementation steps: a core halt head, telemetry, and a
generic halt loss that can train supplied `core_halt_targets`. It does not yet
implement persistent carry, per-sequence reset, exploration, or automatic
verifier-derived halt targets.
