# Qwen3.5 2B Configs

Sources:

- Base model card/config: `references/model_configs/qwen35_2b_base`
- Chat model card/config: `references/model_configs/qwen35_2b_chat`
- Base upstream: `https://huggingface.co/Qwen/Qwen3.5-2B-Base`
- Chat upstream: `https://huggingface.co/Qwen/Qwen3.5-2B`

Downloaded files:

- `README.md`
- `config.json`
- `tokenizer_config.json`

Notes:

- `generation_config.json` was not present in either HF repo at download time.
- The architecture fields live under nested `text_config` and `vision_config`,
  not at the top level.

Key 2B Base fields:

- `architectures`: `Qwen3_5ForConditionalGeneration`
- `model_type`: `qwen3_5`
- `text_config.model_type`: `qwen3_5_text`
- `text_config.hidden_size`: 2048
- `text_config.num_hidden_layers`: 24
- `text_config.vocab_size`: 248320
- `text_config.intermediate_size`: 6144
- `text_config.num_attention_heads`: 8
- `text_config.num_key_value_heads`: 2
- `text_config.head_dim`: 256
- `text_config.max_position_embeddings`: 262144
- `vision_config.hidden_size`: 1024
- `vision_config.out_hidden_size`: 2048
- `vision_config.depth`: 24
- `image_token_id`: 248056
- `video_token_id`: 248057
- `vision_start_token_id`: 248053
- `vision_end_token_id`: 248054

Model-card architecture summary:

- Type: causal language model with vision encoder.
- Hidden layout: six repeats of `3 x (Gated DeltaNet -> FFN)` followed by
  `1 x (Gated Attention -> FFN)`.
- Native context length: 262,144 tokens.
