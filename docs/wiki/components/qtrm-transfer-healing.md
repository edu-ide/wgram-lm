# QTRM Transfer And Healing

Current status:

- No formal transfer/merge/healing pipeline exists yet.
- Current QTRM training is adapter/scaffold training, not a validated healing
  tune.

Reference source:

- `docs/wiki/sources/transfer-merge-healing.md`

Near-term plan:

1. Establish Qwen3.5 HF `generate()` baseline.
2. Train QTRM modules with donor frozen.
3. Run short healing tune on QTRM modules only.
4. Only then consider LoRA/PEFT on donor layers.
5. Avoid full checkpoint/frankenmerge until config/tokenizer/layout gates pass.

Gates:

- tokenizer identity check
- vocab/special-token check
- hidden-size compatibility
- no causal leakage
- Qwen baseline preservation
- before/after eval on general, reasoning, Korean, code, retrieval, and
  repetition-collapse prompts

Allowed merge experiments:

- same Qwen3.5 base and same tokenizer
- LoRA delta merge
- model soup over nearby checkpoints from the same training run
- MergeKit passthrough only as an explicitly experimental branch

Disallowed by default:

- merging Qwen3.5 with unrelated families
- changing tokenizer without retokenization/eval plan
- frankenmerge into production without a healing/eval stage
- judging success by one prompt sample
