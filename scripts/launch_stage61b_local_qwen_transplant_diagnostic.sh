#!/usr/bin/env bash
set -euo pipefail

# Stage61B: local Qwen transplant nerve diagnostic.
#
# Plain-language contract:
#   Qwen is the fluent reader/speaker. The added working-memory organ must pass
#   through Qwen's LM mouth, while a diagnostic fixed renderer reads the same
#   numeric/list ledger directly. If direct ledger succeeds but Qwen-mouth
#   fails, the transplant nerve is the problem. If both fail, the ledger/executor
#   is the problem.

OUT_DIR="${OUT_DIR:-/tmp/stage61b_local_qwen_transplant_directledger_$(date +%Y%m%d_%H%M%S)}"
EPOCHS="${EPOCHS:-2}"

PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python scripts/530_train_final_typed_register_answerer.py \
  --out-dir "${OUT_DIR}" \
  --checkpoint /mnt/sdc1/tripleyoung/qtrm_eval/20260522_092733_LOCAL_STAGE54B_oracle_guard_mixedall_seed42/best_stochastic_oracle.pt \
  --qwen-model-id Qwen/Qwen3.5-0.8B-Base \
  --train-jsonl scratch/stage59/shuffled_choices_train.jsonl \
  --eval-jsonl scratch/stage59/shuffled_choices_eval.jsonl \
  --train-limit "${TRAIN_LIMIT:-64}" \
  --eval-limit "${EVAL_LIMIT:-32}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE:-8}" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-8}" \
  --lr "${LR:-3e-4}" \
  --max-length 160 \
  --n-steps 14 \
  --answer-path lm_head \
  --workspace-pooling sequence \
  --recurrent-readout-pooling sharp_attention \
  --working-register-enabled \
  --working-register-slots 4 \
  --use-source-number-slots \
  --source-number-slots 8 \
  --typed-digit-registers \
  --typed-digit-register-digits 6 \
  --typed-digit-register-trace-weight 0.05 \
  --digit-transition-executor \
  --digit-transition-executor-mode dual_axis_lbecp \
  --digit-transition-executor-trace-weight 1.0 \
  --digit-transition-pretrain-epochs 2 \
  --residual-thought-graft \
  --residual-thought-graft-base register_mean \
  --qwen-lm-mouth-answerer \
  --qwen-lm-mouth-max-answer-tokens 16 \
  --qwen-lm-mouth-context-mode register_prefix \
  --qwen-lm-mouth-logit-mode qwen_plus_low_rank \
  --qwen-lm-mouth-ledger-token-reader \
  --qwen-lm-mouth-ledger-token-reader-scale 1.0 \
  --eval-qwen-lm-mouth-direct-ledger-renderer

echo "Stage61B local diagnostic wrote: ${OUT_DIR}"
