# Training Diagnostics Sources

Paper PDFs:

| Area | Local PDF | Upstream |
| --- | --- | --- |
| LLM scaling laws | `references/papers/training_diagnostics/scaling_laws_neural_language_models_2001.08361.pdf` | <https://arxiv.org/abs/2001.08361> |
| Compute-optimal LLM training | `references/papers/training_diagnostics/chinchilla_compute_optimal_training_2203.15556.pdf` | <https://arxiv.org/abs/2203.15556> |
| Gradient noise scale | `references/papers/training_diagnostics/empirical_model_large_batch_training_1812.06162.pdf` | <https://arxiv.org/abs/1812.06162> |
| Learning curve extrapolation | `references/papers/training_diagnostics/lc_pfn_learning_curve_extrapolation_2310.20447.pdf` | <https://arxiv.org/abs/2310.20447> |
| LLM training dynamics | `references/papers/training_diagnostics/training_dynamics_llm_scaling_laws_acl2025.pdf` | <https://aclanthology.org/2025.acl-long.1366/> |
| Early stopping criteria | `references/papers/training_diagnostics/early_stopping_but_when_prechelt_1997.pdf` | <https://page.mi.fu-berlin.de/prechelt/Biblio/stop_tricks1997.pdf> |

QTRM relevance:

- Scaling-law papers give a sanity check for whether loss movement is plausible
  for the amount of data, compute, and parameter count used.
- Chinchilla-style compute/data framing warns against interpreting more steps as
  progress if the token budget or data quality is the limiting factor.
- Gradient noise scale is the reference axis for deciding whether larger batches
  are useful or just hiding optimization noise.
- Learning-curve extrapolation supports short pilot runs before long adapter
  training.
- Early-stopping work grounds validation-loss patience instead of picking a
  stopping rule ad hoc.
- LLM training dynamics papers motivate logging loss acceleration/deceleration,
  not just raw loss snapshots.
