#!/usr/bin/env bash
set -euo pipefail

cd ~/qtrm-workspace/wgram-lm
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi
export PYTHONPATH=${PYTHONPATH:-$PWD/src}
export HF_HOME=${HF_HOME:-~/.cache/huggingface}

CONFIG=${CONFIG:-configs/qwen35_2b_4090.yaml}
MODEL_ID=${MODEL_ID:-}
PROMPT=${1:-"Explain quantum entanglement in simple terms."}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
LOAD_IN_4BIT=${LOAD_IN_4BIT:-}
DO_SAMPLE=${DO_SAMPLE:-0}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.9}
REPETITION_PENALTY=${REPETITION_PENALTY:-1.05}

echo "============================================================"
echo "Donor-only generation baseline"
echo "============================================================"
echo "Config: $CONFIG"
echo "Model override: ${MODEL_ID:-<from config>}"
echo "Prompt: $PROMPT"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "============================================================"

python - "$CONFIG" "$MODEL_ID" "$PROMPT" "$MAX_NEW_TOKENS" "$LOAD_IN_4BIT" "$DO_SAMPLE" "$TEMPERATURE" "$TOP_P" "$REPETITION_PENALTY" <<'PYEOF'
from __future__ import annotations

import sys
import torch

from wgram_lm.config import load_config

config_path, model_override, prompt = sys.argv[1], sys.argv[2], sys.argv[3]
max_new_tokens = int(sys.argv[4])
load_in_4bit_arg = sys.argv[5]
do_sample = sys.argv[6] == "1"
temperature = float(sys.argv[7])
top_p = float(sys.argv[8])
repetition_penalty = float(sys.argv[9])

cfg = load_config(config_path)
model_id = model_override or cfg.donor.model_id
if not model_id:
    raise SystemExit("No donor model id. Set MODEL_ID or donor.model_id in config.")

load_in_4bit = cfg.donor.load_in_4bit if load_in_4bit_arg == "" else load_in_4bit_arg == "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=cfg.donor.trust_remote_code)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

quantization_config = None
if load_in_4bit and torch.cuda.is_available():
    try:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    except Exception as exc:
        print(f"[warn] bitsandbytes unavailable, loading full precision: {exc}")

kwargs = {
    "trust_remote_code": cfg.donor.trust_remote_code,
    "dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    "device_map": "auto" if torch.cuda.is_available() else None,
}
if quantization_config is not None:
    kwargs["quantization_config"] = quantization_config

try:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
except Exception as causal_exc:
    print(f"[warn] AutoModelForCausalLM failed, trying AutoModelForImageTextToText: {causal_exc}")
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)

model.eval()
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(model.device if hasattr(model, "device") else device) for k, v in inputs.items()}

generate_kwargs = {
    "max_new_tokens": max_new_tokens,
    "do_sample": do_sample,
    "repetition_penalty": repetition_penalty,
    "pad_token_id": tokenizer.pad_token_id,
}
if tokenizer.eos_token_id is not None:
    generate_kwargs["eos_token_id"] = tokenizer.eos_token_id
if do_sample:
    generate_kwargs.update({"temperature": temperature, "top_p": top_p})

print(f"[donor] model={model_id}, device={device}, 4bit={bool(quantization_config)}")
with torch.no_grad():
    output_ids = model.generate(**inputs, **generate_kwargs)

text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("-" * 60)
print(text)
print("-" * 60)
print(f"Generated {output_ids.shape[1] - inputs['input_ids'].shape[1]} new tokens")
PYEOF
