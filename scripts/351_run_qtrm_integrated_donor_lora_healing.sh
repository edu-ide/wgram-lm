#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=${1:-configs/qwen35_2b_4090_qtrm_integrated_donor_lora_healing_s080.yaml}

echo "=== QTRM-integrated donor LoRA healing tune ==="
echo "Config: $CONFIG"
echo "Mode: Qwen donor LoRA + trainable QTRM, donor logits annealed toward QTRM-only"

exec bash scripts/08_train_donor_adapter.sh "$CONFIG" "${@:2}"
