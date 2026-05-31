# Qwen Donor Integration

Current code:

- `src/wgram_lm/qwen_donor.py`
- `scripts/08_train_donor_adapter.sh`
- `scripts/90_infer_with_donor.sh`

Reference sources:

- `docs/wiki/sources/qwen35-2b-configs.md`
- `docs/wiki/sources/qwen35-omni.md`

Status:

- Partially aligned. Needs stronger config/tokenizer/generation tests.

Boundary:

- Qwen donor use is valid as a frozen tokenizer/hidden-state/base-logit
  scaffold.
- Qwen donor use is not evidence that QTRM has raw recursive reasoning,
  trainable memory intelligence, or metacognitive calibration.
- Any QTRM capability claim must compare donor-only, fused QTRM, low-donor or
  QTRM-only, core-off, and memory-off modes when relevant.
- See `docs/wiki/decisions/qwen-donor-risk-and-metacognition.md`.

Known constraints:

- Qwen3.5 config uses nested `text_config` and `vision_config`.
- Donor hidden size for 2B is 2048.
- Vocab size is 248320.
- Native hidden layout is 3 GatedDeltaNet blocks per 1 GatedAttention block.
- Direct `Qwen3.5-2B-Base` generation is not instruction-tuned behavior; use the
  chat model or official chat template when evaluating conversational quality.

Gates before long training:

- Add tests that compare loaded tokenizer vocab/special tokens against the
  stored HF tokenizer config.
- Add tests that assert model `d_model` equals donor hidden size when using
  donor hidden states.
- Add an HF `generate()` baseline script for Qwen3.5 without QTRM.
- Do not use custom QTRM `lm_head` output as the quality baseline.
- Do not promote calibration gains unless QTRM-only/low-donor and core-off
  ablations show the trainable QTRM path caused the change.
