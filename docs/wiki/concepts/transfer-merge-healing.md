# Transfer, Merge, Healing

This axis covers how QTRM should recover and transfer capability after attaching
new modules to a Qwen donor.

Terminology:

- **Transfer learning**: start from a pretrained donor and train a smaller set of
  new parameters or adapters.
- **Checkpoint merging**: combine weights or deltas from compatible checkpoints.
- **Frankenmerge / passthrough**: assemble layer slices from multiple models,
  usually with MergeKit.
- **Adapter merging**: merge LoRA/PEFT deltas before or after training.
- **Healing tune**: project-local term for low-risk post-change recovery
  training. It should map to a concrete objective: continued pretraining,
  supervised fine-tuning, merge-friendly fine-tuning, or adapter-only recovery.

Qwopus note, 2026-05-15:

- `Jackrong/Qwopus3.5-27B-v3` is publicly described as Qwen3.5-27B fine-tuned
  with Unsloth SFT + LoRA, not as a layer-stack frankenmerge.
- The explicit frankenmerge/healing example found in the Qwopus family is
  `Qwopus-GLM-18B-Merged`: two Qwen3.5-9B finetunes stacked into a 64-layer
  ~18B model, then repaired with a 1000-step QLoRA healing tune.
- For QTRM, this distinction matters: base-model post-training is useful as a
  native-language bootstrap precedent; frankenmerge healing is useful only as a
  model-surgery recovery precedent and must not be counted as evidence for
  QTRM-native reasoning.

QTRM rules:

- Do not frankenmerge models with incompatible tokenizer/config/hidden layouts.
- For Qwen3.5 donor work, keep tokenizer and generation behavior fixed unless a
  reference explicitly requires otherwise.
- Prefer adapter-only or QTRM-module-only healing before touching donor weights.
- If merging full checkpoints, require same base family, same vocab, same hidden
  shapes, and explicit eval gates.
- Healing data must include general text/code/reasoning plus the target QTRM
  traces, otherwise the model can overfit narrow behavior or forget instruction
  following.

Recommended healing mix for QTRM experiments:

- 60-70% general high-quality text/code
- 15-25% reasoning/math
- 10-20% QTRM traces, retrieval, verification, multimodal tasks
- small learning rate, short schedule, frequent eval

Open issue:

- "Healing tune" is not a single official method. Every run must specify the
  exact objective, trainable parameters, data mix, and eval gates.
