#!/usr/bin/env bash
set -euo pipefail

cd ~/qtrm-workspace/qtrm_multimodal_memoryos
source .venv/bin/activate
export PYTHONPATH=$PWD/src

# Default values
CHECKPOINT=${CHECKPOINT:-runs/qwen35_2b_4090/last.pt}
CONFIG=${CONFIG:-configs/qwen35_2b_4090.yaml}
PROMPT=${1:-"Explain quantum entanglement in simple terms."}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
DONOR_LOGITS_SCALE=${DONOR_LOGITS_SCALE:-}
QTRM_LOGITS_SCALE=${QTRM_LOGITS_SCALE:-}
QTRM_RESIDUAL_CLAMP=${QTRM_RESIDUAL_CLAMP:-}
QTRM_RESIDUAL_GATE=${QTRM_RESIDUAL_GATE:-}
QTRM_RESIDUAL_GATE_BIAS=${QTRM_RESIDUAL_GATE_BIAS:-}

echo "============================================================"
echo "Inference: QTRM + Qwen3.5-2B Donor"
echo "============================================================"
echo "Prompt: $PROMPT"
echo "Checkpoint: $CHECKPOINT"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Donor logits scale: ${DONOR_LOGITS_SCALE:-config}"
echo "QTRM logits scale: ${QTRM_LOGITS_SCALE:-config}"
echo "QTRM residual clamp: ${QTRM_RESIDUAL_CLAMP:-config}"
echo "QTRM residual gate: ${QTRM_RESIDUAL_GATE:-config}"
echo "============================================================"

python - "$CHECKPOINT" "$CONFIG" "$PROMPT" "$MAX_NEW_TOKENS" "$DONOR_LOGITS_SCALE" "$QTRM_LOGITS_SCALE" "$QTRM_RESIDUAL_CLAMP" "$QTRM_RESIDUAL_GATE" "$QTRM_RESIDUAL_GATE_BIAS" <<'PYEOF'
import sys, torch
from qtrm_mm.config import load_config
from qtrm_mm.qtrm_model import QTRMMultimodalModel
from qtrm_mm.qwen_donor import QwenDonorAdapter
from transformers import AutoTokenizer

checkpoint = sys.argv[1]
config_path = sys.argv[2]
prompt = sys.argv[3]
max_new = int(sys.argv[4])
donor_scale_override = sys.argv[5]
qtrm_scale_override = sys.argv[6]
residual_clamp_override = sys.argv[7]
residual_gate_override = sys.argv[8]
residual_gate_bias_override = sys.argv[9]

def parse_bool(value):
    return value.strip().lower() in {"1", "true", "yes", "on"}

cfg = load_config(config_path)
if donor_scale_override:
    cfg.model.donor_logits_scale = float(donor_scale_override)
if qtrm_scale_override:
    cfg.model.qtrm_logits_scale = float(qtrm_scale_override)
if residual_clamp_override:
    cfg.model.qtrm_residual_clamp = float(residual_clamp_override)
if residual_gate_override:
    cfg.model.qtrm_residual_gate_enabled = parse_bool(residual_gate_override)
if residual_gate_bias_override:
    cfg.model.qtrm_residual_gate_init_bias = float(residual_gate_bias_override)
device = "cuda"

# Load donor
print("\n[1] Loading donor (4bit frozen)...")
donor = QwenDonorAdapter(cfg.donor)

# Load tokenizer
print("[2] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(cfg.donor.model_id, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

# Get initial token IDs from prompt
encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
input_ids = encoded["input_ids"].to(device)
attn_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(device)

use_donor_logits = cfg.model.donor_logits_scale != 0.0

def donor_kwargs(ids, mask):
    result = donor.encode_inputs(
        input_ids=ids,
        attention_mask=mask,
        return_logits=use_donor_logits,
    )
    out = {"text_states": result["text_states"].to(device)}
    if result.get("visual_features") is not None:
        out["visual_features"] = result["visual_features"].to(device)
    if use_donor_logits and result.get("logits") is not None:
        out["donor_logits"] = result["logits"].to(device)
    return out

# Encode prompt via donor
print("[3] Encoding prompt via donor...")
prompt_extra = donor_kwargs(input_ids, attn_mask)
print(f"    donor output: text_states={prompt_extra['text_states'].shape}")

# Load QTRM model
print("[4] Loading QTRM model...")
model = QTRMMultimodalModel(cfg.model)
if donor_scale_override:
    model.cfg.donor_logits_scale = float(donor_scale_override)
if qtrm_scale_override:
    model.cfg.qtrm_logits_scale = float(qtrm_scale_override)
if residual_clamp_override:
    model.cfg.qtrm_residual_clamp = float(residual_clamp_override)
if residual_gate_override:
    model.cfg.qtrm_residual_gate_enabled = parse_bool(residual_gate_override)
state = torch.load(checkpoint, map_location=device, weights_only=False)
model.load_state_dict(state.get("model", state), strict=False)
if residual_gate_bias_override:
    model.residual_gate.bias.data.fill_(float(residual_gate_bias_override))
model = model.to(device)
model.eval()
print(f"    loaded from {checkpoint}")
print(f"    qtrm_logits_scale={model.cfg.qtrm_logits_scale}")
print(f"    donor_logits_scale={model.cfg.donor_logits_scale}")
print(f"    qtrm_residual_clamp={model.cfg.qtrm_residual_clamp}")
print(f"    qtrm_residual_gate_enabled={model.cfg.qtrm_residual_gate_enabled}")

print("\n[5] Generating...")
print("-" * 50)

with torch.no_grad():
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        generated_ids = input_ids.tolist()[0]

        for step in range(max_new):
            cur_input_ids = torch.tensor([generated_ids], device=device)
            cur_mask = torch.ones_like(cur_input_ids)
            cur_extra = donor_kwargs(cur_input_ids, cur_mask)
            outputs = model(cur_input_ids, attention_mask=cur_mask, **cur_extra)
            last_logit = outputs["logits"][:, -1, :]
            next_id = last_logit.argmax(dim=-1).item()

            if next_id == tokenizer.eos_token_id:
                break

            generated_ids.append(next_id)

        # Decode
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

print(f"\n{generated_text}")
print("-" * 50)
print(f"\nGenerated {len(generated_ids) - len(input_ids[0])} new tokens")
print("============================================================")
PYEOF
