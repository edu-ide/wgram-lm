#!/usr/bin/env bash
set -euo pipefail

cd ~/qtrm-workspace/wgram-lm
source .venv/bin/activate
export PYTHONPATH=$PWD/src

# Default values
CHECKPOINT=${CHECKPOINT:-runs/qwen35_2b_4090/last.pt}
CONFIG=${CONFIG:-configs/qwen35_2b_4090.yaml}
PROMPT=${1:-"Explain quantum entanglement in simple terms."}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
LANGUAGE_SAFE=${LANGUAGE_SAFE:-1}
if [[ "$LANGUAGE_SAFE" == "1" ]]; then
  DONOR_LOGITS_SCALE=${DONOR_LOGITS_SCALE:-1.0}
  QTRM_LOGITS_SCALE=${QTRM_LOGITS_SCALE:-0.0}
  QTRM_RESIDUAL_CLAMP=${QTRM_RESIDUAL_CLAMP:-0.0}
  SUPPRESS_VISIBLE_REASONING=${SUPPRESS_VISIBLE_REASONING:-1}
  NO_REPEAT_NGRAM_SIZE=${NO_REPEAT_NGRAM_SIZE:-2}
  STOP_AFTER_SENTENCE=${STOP_AFTER_SENTENCE:-1}
  MIN_NEW_TOKENS_BEFORE_STOP=${MIN_NEW_TOKENS_BEFORE_STOP:-16}
else
  DONOR_LOGITS_SCALE=${DONOR_LOGITS_SCALE:-}
  QTRM_LOGITS_SCALE=${QTRM_LOGITS_SCALE:-}
  QTRM_RESIDUAL_CLAMP=${QTRM_RESIDUAL_CLAMP:-}
  SUPPRESS_VISIBLE_REASONING=${SUPPRESS_VISIBLE_REASONING:-0}
  NO_REPEAT_NGRAM_SIZE=${NO_REPEAT_NGRAM_SIZE:-0}
  STOP_AFTER_SENTENCE=${STOP_AFTER_SENTENCE:-0}
  MIN_NEW_TOKENS_BEFORE_STOP=${MIN_NEW_TOKENS_BEFORE_STOP:-16}
fi
QTRM_RESIDUAL_GATE=${QTRM_RESIDUAL_GATE:-}
QTRM_RESIDUAL_GATE_BIAS=${QTRM_RESIDUAL_GATE_BIAS:-}
ANSWER_CONTRACT=${ANSWER_CONTRACT:-none}
HISTORY_JSONL=${HISTORY_JSONL:-auto}

echo "============================================================"
echo "Inference: QTRM + Qwen3.5-2B Donor"
echo "============================================================"
echo "Prompt: $PROMPT"
echo "Checkpoint: $CHECKPOINT"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Language safe mode: $LANGUAGE_SAFE"
echo "Donor logits scale: ${DONOR_LOGITS_SCALE:-config}"
echo "QTRM logits scale: ${QTRM_LOGITS_SCALE:-config}"
echo "QTRM residual clamp: ${QTRM_RESIDUAL_CLAMP:-config}"
echo "QTRM residual gate: ${QTRM_RESIDUAL_GATE:-config}"
echo "Suppress visible reasoning: $SUPPRESS_VISIBLE_REASONING"
echo "No repeat ngram size: $NO_REPEAT_NGRAM_SIZE"
echo "Stop after sentence: $STOP_AFTER_SENTENCE"
echo "Answer contract: $ANSWER_CONTRACT"
echo "History JSONL: $HISTORY_JSONL"
echo "============================================================"

python - "$CHECKPOINT" "$CONFIG" "$PROMPT" "$MAX_NEW_TOKENS" "$DONOR_LOGITS_SCALE" "$QTRM_LOGITS_SCALE" "$QTRM_RESIDUAL_CLAMP" "$QTRM_RESIDUAL_GATE" "$QTRM_RESIDUAL_GATE_BIAS" "$SUPPRESS_VISIBLE_REASONING" "$NO_REPEAT_NGRAM_SIZE" "$ANSWER_CONTRACT" "$HISTORY_JSONL" "$LANGUAGE_SAFE" "$STOP_AFTER_SENTENCE" "$MIN_NEW_TOKENS_BEFORE_STOP" <<'PYEOF'
import re, sys, torch
from wgram_lm.config import load_config
from wgram_lm.history import append_generation_history
from wgram_lm.wgram_model import QTRMMultimodalModel
from wgram_lm.qwen_donor import QwenDonorAdapter
from transformers import AutoTokenizer

checkpoint = sys.argv[1]
config_path = sys.argv[2]
original_prompt = sys.argv[3]
prompt = original_prompt
max_new = int(sys.argv[4])
donor_scale_override = sys.argv[5]
qtrm_scale_override = sys.argv[6]
residual_clamp_override = sys.argv[7]
residual_gate_override = sys.argv[8]
residual_gate_bias_override = sys.argv[9]

def parse_bool(value):
    return value.strip().lower() in {"1", "true", "yes", "on"}

suppress_visible_reasoning = parse_bool(sys.argv[10])
no_repeat_ngram_size = int(sys.argv[11])
answer_contract = sys.argv[12]
history_jsonl = sys.argv[13]
language_safe = parse_bool(sys.argv[14])
stop_after_sentence = parse_bool(sys.argv[15])
min_new_tokens_before_stop = int(sys.argv[16])

def apply_answer_contract(text, mode):
    if mode == "none":
        return str(text)
    if mode == "direct":
        return str(text).rstrip() + (
            "\n\n/no_think\n"
            "Answer directly. Do not reveal hidden reasoning. "
            "Do not create multiple-choice options or a new question."
        )
    raise ValueError(f"unknown ANSWER_CONTRACT: {mode}")

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

def visible_reasoning_token_ids(tokenizer):
    ids = []
    for marker in ("<think>", "</think>"):
        try:
            ids.extend(tokenizer.encode(marker, add_special_tokens=False))
        except Exception:
            pass
    return sorted(set(int(token_id) for token_id in ids))

suppressed_token_ids = visible_reasoning_token_ids(tokenizer) if suppress_visible_reasoning else []
if suppressed_token_ids:
    print(f"    suppressing visible reasoning token ids: {suppressed_token_ids}")

def no_repeat_ngram_banned_tokens(generated, ngram_size):
    n = int(ngram_size)
    if n <= 0 or len(generated) < n - 1:
        return []
    if n == 1:
        return sorted(set(int(token_id) for token_id in generated))
    prefix = tuple(int(token_id) for token_id in generated[-(n - 1):])
    banned = set()
    for idx in range(0, len(generated) - n + 1):
        ngram = tuple(int(token_id) for token_id in generated[idx:idx + n])
        if ngram[:-1] == prefix:
            banned.add(ngram[-1])
    return sorted(banned)

def generated_completion_text(generated_ids):
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    if text.startswith(prompt):
        return text[len(prompt):].strip()
    return text.strip()

def should_stop_after_sentence(generated_ids, new_token_count):
    if not stop_after_sentence or new_token_count < min_new_tokens_before_stop:
        return False
    completion = generated_completion_text(generated_ids)
    if not completion:
        return False
    return re.search(r"[.!?。！？]\s*$", completion) is not None

# Get initial token IDs from prompt
prompt = apply_answer_contract(prompt, answer_contract)
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
            if suppressed_token_ids:
                last_logit[:, suppressed_token_ids] = -torch.inf
            banned_repeat_ids = no_repeat_ngram_banned_tokens(generated_ids[len(input_ids[0]):], no_repeat_ngram_size)
            if banned_repeat_ids:
                last_logit[:, banned_repeat_ids] = -torch.inf
            next_id = last_logit.argmax(dim=-1).item()

            if next_id == tokenizer.eos_token_id:
                break

            generated_ids.append(next_id)
            if should_stop_after_sentence(
                generated_ids,
                len(generated_ids) - len(input_ids[0]),
            ):
                break

        # Decode
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

completion_text = generated_text[len(prompt):].strip() if generated_text.startswith(prompt) else generated_text
history_mode = (
    "language_safe_donor"
    if language_safe and model.cfg.donor_logits_scale == 1.0 and model.cfg.qtrm_logits_scale == 0.0
    else "qtrm_residual"
)
append_generation_history(
    history_jsonl,
    source="infer_with_donor",
    checkpoint=checkpoint,
    config=config_path,
    prompt=original_prompt,
    output=generated_text,
    mode=history_mode,
    completion=completion_text,
    metadata={
        "max_new_tokens": max_new,
        "generated_new_tokens": len(generated_ids) - len(input_ids[0]),
        "donor_logits_scale": model.cfg.donor_logits_scale,
        "qtrm_logits_scale": model.cfg.qtrm_logits_scale,
        "qtrm_residual_clamp": model.cfg.qtrm_residual_clamp,
        "qtrm_residual_gate_enabled": model.cfg.qtrm_residual_gate_enabled,
        "language_safe": language_safe,
        "suppress_visible_reasoning": suppress_visible_reasoning,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "stop_after_sentence": stop_after_sentence,
        "min_new_tokens_before_stop": min_new_tokens_before_stop,
        "answer_contract": answer_contract,
    },
)

print(f"\n{generated_text}")
print("-" * 50)
print(f"\nGenerated {len(generated_ids) - len(input_ids[0])} new tokens")
print("============================================================")
PYEOF
