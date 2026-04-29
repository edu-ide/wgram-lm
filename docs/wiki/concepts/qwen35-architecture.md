# Qwen3.5 Architecture

Qwen3.5 is now a primary architecture reference for QTRM because the donor model
is `Qwen/Qwen3.5-2B-Base`.

Direct donor facts from the HF config/model card:

- architecture class: `Qwen3_5ForConditionalGeneration`
- native multimodal model with a vision encoder
- text hidden size: 2048
- text layers: 24
- vocabulary: 248320 padded tokens
- FFN intermediate size: 6144
- context length: 262144
- hidden layout: `6 x [3 x GatedDeltaNet + 1 x GatedAttention]`
- attention: 8 Q heads, 2 KV heads, head dimension 256
- vision encoder hidden size: 1024, output hidden size 2048

QTRM design constraints:

- Do not flatten Qwen3.5 config as if it were a plain text-only CausalLM config;
  read nested `text_config` and `vision_config`.
- QTRM `d_model` should match the donor hidden size when consuming donor hidden
  states directly.
- GatedDeltaNet placement should follow the 3:1 GDN/full-attention pattern if we
  are claiming Qwen3.5 alignment.
- Tokenizer and chat template must come from Qwen3.5, not a generic Qwen3/Qwen2
  tokenizer assumption.
- The QTRM generator baseline remains HF/Qwen generation, not the custom QTRM
  `lm_head`.
