# OpenMythos And Recurrent-Depth Sources

Code references:

| Area | Local path | Upstream | Commit | Status |
| --- | --- | --- | --- | --- |
| OpenMythos | `references/official/openmythos` | <https://github.com/kyegomez/OpenMythos> | `8c68c1f` | Community/speculative implementation, not official Anthropic evidence |
| Parcae | `references/official/parcae` | <https://github.com/sandyresearch/parcae> | `dee8363` | Paper-backed stable looped language model reference |

Paper PDFs:

| Area | Local PDF | Upstream |
| --- | --- | --- |
| Parcae stable looped LMs | `references/papers/recurrent_depth/parcae_stable_looped_lm_2604.12946.pdf` | <https://arxiv.org/abs/2604.12946> |
| Looped transformers for learning algorithms | `references/papers/recurrent_depth/looped_transformers_learning_algorithms_2311.12424.pdf` | <https://arxiv.org/abs/2311.12424> |
| Reasoning with latent thoughts | `references/papers/recurrent_depth/reasoning_with_latent_thoughts_looped_transformers_2502.17416.pdf` | <https://arxiv.org/abs/2502.17416> |
| Negative/limited latent-CoT probing | `references/papers/recurrent_depth/latent_cot_depth_recurrent_transformer_2507.02199.pdf` | <https://arxiv.org/abs/2507.02199> |

QTRM relevance:

- OpenMythos is useful as an integration sketch: Prelude -> recurrent block ->
  Coda, input re-injection, loop-index signal, ACT-like halting, GQA/MLA
  attention options, and recurrent-block MoE.
- OpenMythos must not be treated as evidence for Claude internals or as an
  official architecture. Its README labels the project as theoretical and
  speculative.
- Parcae is the stronger reference for stable looped language models. It uses a
  diagonal injection of the form `x_{t+1} = exp(-dt * A) * x_t + dt * B @ e`,
  exposes spectral/contraction telemetry, and evaluates recurrence depth.
- The negative latent-CoT probing paper is important: more recurrence does not
  automatically mean interpretable or strong latent chain-of-thought.
