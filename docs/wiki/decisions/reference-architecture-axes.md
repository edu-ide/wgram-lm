# Reference Architecture Axes

QTRM should not be validated against one paper. It combines separate axes:

| Axis | Primary reference | QTRM component |
| --- | --- | --- |
| Generator baseline | Qwen + HF Transformers | donor generation / tokenizer |
| Delta mixer | Gated DeltaNet + FLA | `mixers.py`, `blocks.py` |
| Recursive core | Tiny Recursive Models / HRM | `core.py`, controller heads |
| World model | LeWorldModel | `world_model.py`, JEPA loss |
| Memory/wiki workflow | Karpathy LLM Wiki | `docs/wiki`, MemoryOS docs |
| Cognitive core/data quality | Karpathy/Dwarkesh cognitive core + Phi/TinyStories/Chinchilla | data mix, distillation traces, retrieval/memory gates |
| Architecture composition | NAS / design spaces / transformer modification transfer | ablation matrix, config variants |

Decision:

- Do not call a component official unless it maps to its axis reference.
- If a component is useful but not reference-faithful, label it experimental.
- Long training should wait until the target axes have tests and docs.
- Full QTRM should be treated as a composition experiment until individual axes
  pass local ablations and the combined model beats the simpler baselines.
- The 1B cognitive-core idea is a hypothesis about memory/reasoning separation,
  not a license to skip donor baselines, clean data construction, or collapse
  diagnostics.
