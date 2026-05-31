#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from wgram_lm.qwen_scope import (
    decode_token_texts,
    load_qwen_scope_sae_from_hub,
    qwen_scope_feature_records,
)


DEFAULT_PROMPTS = [
    "Explain quantum entanglement in simple terms.",
    "양자 컴퓨팅이란 무엇인가요?",
    "Determine whether the claim is supported by the evidence. Claim: The Eiffel Tower is in Berlin.",
]


class LayerAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        current = getattr(namespace, self.dest, None)
        if current is None or current == self.default:
            current = []
        current.append(int(values))
        setattr(namespace, self.dest, current)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Log Qwen-Scope SAE feature activations for Qwen donor residual streams."
    )
    ap.add_argument("--model-id", default="Qwen/Qwen3.5-2B-Base")
    ap.add_argument(
        "--sae-repo",
        default="Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_100",
    )
    ap.add_argument("--layer", action=LayerAction, type=int, default=[23])
    ap.add_argument("--prompt", action="append", default=None)
    ap.add_argument("--prompt-file", default=None, help="Optional UTF-8 text file, one prompt per line.")
    ap.add_argument("--out", default="runs/qwen_scope/qwen35_2b_base_sae_probe.jsonl")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="Use float32 for exact SAE probing unless memory requires lower precision.",
    )
    return ap


def collect_prompts(args: argparse.Namespace) -> list[str]:
    prompts = []
    if args.prompt:
        prompts.extend(args.prompt)
    if args.prompt_file:
        path = Path(args.prompt_file)
        prompts.extend(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return prompts or list(DEFAULT_PROMPTS)


def select_device(requested: str) -> str:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return "cuda"
    if requested == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def select_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


@torch.no_grad()
def run_probe(args: argparse.Namespace) -> list[dict]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = select_device(args.device)
    dtype = select_dtype(args.dtype)
    prompts = collect_prompts(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_kwargs["device_map"] = "auto" if device == "cuda" else None
        model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs).to(device)
    model.eval()

    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    outputs = model(**encoded, output_hidden_states=True, use_cache=False)
    hidden_states = outputs.hidden_states
    token_texts = decode_token_texts(tokenizer, encoded["input_ids"])

    records = []
    for layer in args.layer:
        hidden_index = int(layer) + 1
        if hidden_index >= len(hidden_states):
            raise ValueError(
                f"layer {layer} is outside hidden_states range 0..{len(hidden_states) - 2}"
            )
        sae = load_qwen_scope_sae_from_hub(
            args.sae_repo,
            layer=layer,
            device=device,
            dtype=dtype,
        )
        layer_records = qwen_scope_feature_records(
            sae,
            hidden_states[hidden_index],
            prompts=prompts,
            token_ids=encoded["input_ids"],
            token_texts=token_texts,
            attention_mask=encoded.get("attention_mask"),
            top_k=args.top_k,
            token_position="last_nonpad",
        )
        for row in layer_records:
            row["model_id"] = args.model_id
            row["sae_repo"] = args.sae_repo
        records.extend(layer_records)
    return records


def main() -> None:
    args = build_arg_parser().parse_args()
    records = run_probe(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} records to {out}")


if __name__ == "__main__":
    main()
