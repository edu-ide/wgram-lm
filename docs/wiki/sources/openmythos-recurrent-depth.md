# OpenMythos And Recurrent-Depth Sources

Code references:

| Area | Local path | Upstream | Commit | Status |
| --- | --- | --- | --- | --- |
| OpenMythos | `references/official/openmythos` | <https://github.com/kyegomez/OpenMythos> | `8c68c1f` | Community/speculative implementation, not official Anthropic evidence |
| Parcae | `references/official/parcae` | <https://github.com/sandyresearch/parcae> | `dee8363` | Paper-backed stable looped language model reference |
| CoT vs looped latent thought | `references/official/cot-vs-loop` | <https://github.com/kevin671/cot-vs-loop> | `783fa90` | Official implementation for formal CoT/latent separation experiments |

Paper PDFs:

| Area | Local PDF | Upstream |
| --- | --- | --- |
| Parcae stable looped LMs | `references/papers/recurrent_depth/parcae_stable_looped_lm_2604.12946.pdf` | <https://arxiv.org/abs/2604.12946> |
| Looped transformers for learning algorithms | `references/papers/recurrent_depth/looped_transformers_learning_algorithms_2311.12424.pdf` | <https://arxiv.org/abs/2311.12424> |
| Reasoning with latent thoughts | `references/papers/recurrent_depth/reasoning_with_latent_thoughts_looped_transformers_2502.17416.pdf` | <https://arxiv.org/abs/2502.17416> |
| Negative/limited latent-CoT probing | `references/papers/recurrent_depth/latent_cot_depth_recurrent_transformer_2507.02199.pdf` | <https://arxiv.org/abs/2507.02199> |
| Formal CoT/latent separation | `references/papers/recurrent_depth/formal_comparison_cot_latent_thought_2509.25239.pdf` | <https://arxiv.org/abs/2509.25239> |

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
- The formal CoT/latent separation paper is now a guardrail: latent loops are
  not a blanket replacement for visible/token-level CoT. QTRM must evaluate
  parallelizable and stochastic/sequential task families separately.

## 2026-05-01 QTRM Design Mapping

Observed QTRM failure:

- The general residual path was alive after the evidence-gate fix, but the text
  logits could still mostly follow the donor/text bypass.
- In the first HF warmup config, the coda had two layers and inherited
  `attn_every: 4`, so no explicit coda attention layer was guaranteed.
- This made the recursive `z_H` workspace prefix too easy to ignore.

OpenMythos mechanism adapted:

- OpenMythos captures the prelude output as `e`, loops the recurrent state, then
  sends recurrent output to coda. There is no separate unchanged text hidden
  bypass around the recurrent block.
- QTRM cannot simply replace the token stream with `z_H`, because it is a
  donor-backed residual adapter with a separate latent workspace. The safe
  adaptation is a gated `z_H -> text_context` cross-attention path before coda.

Implemented QTRM mechanism:

- `QTRMConfig.core_to_text_enabled`
- `QTRMConfig.core_to_text_gate_init_bias`
- `QTRMConfig.core_to_text_gate_min`
- `QTRMMultimodalModel.forward(..., disable_core_to_text=True)` ablation
- output telemetry: `core_to_text_gate_mean`
- v2 warmup config:
  `configs/qwen35_2b_4090_hf_first_wave_warmup_v2_s400.yaml`

The v2 config also sets `coda_attn_every: 1`, making the coda explicitly able
to attend from text tokens to the recurrent workspace prefix. This is still not
a standalone OpenMythos clone; it is a conservative donor-backed residual
adaptation of the OpenMythos recurrent-output-to-coda principle.
