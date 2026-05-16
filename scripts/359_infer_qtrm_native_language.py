#!/usr/bin/env python3
"""Run donorless QTRM-native language inference from a saved checkpoint."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

import torch


def load_eval_module():
    path = Path(__file__).with_name("356_eval_qtrm_native_language_generalization.py")
    spec = importlib.util.spec_from_file_location("qtrm_native_language_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_eval_args(eval_module, checkpoint_args: dict[str, object], args: argparse.Namespace):
    overrides = Namespace(
        device=str(args.device),
        out_dir="local_eval/qtrm_native_language_infer_tmp",
        eval_think_steps=int(args.think_steps),
        max_new_chars=int(args.max_new_chars),
        repair_prompt_count=1,
        eval_seed_texts="",
        eval_seed_expectations="{}",
        eval_jsonl="",
        min_on_policy_continuation_chars=0,
        min_on_policy_keyword_hits=0,
        min_on_policy_loop_check_lines=4,
        min_on_policy_unique_line_fraction=0.55,
        max_on_policy_repeated_block_fraction=0.24,
        max_on_policy_repeated_line_fraction=0.30,
    )
    return eval_module.merged_checkpoint_args(checkpoint_args, overrides)


def build_seed_text(prompt: str, *, raw: bool) -> str:
    text = str(prompt)
    if raw or "Assistant:" in text:
        return text
    return f"User: {text}\nAssistant:"


@torch.no_grad()
def run_inference(args: argparse.Namespace) -> str:
    eval_module = load_eval_module()
    checkpoint = torch.load(str(args.checkpoint), map_location="cpu")
    checkpoint_args = checkpoint.get("args", {})
    if not isinstance(checkpoint_args, dict):
        checkpoint_args = {}
    eval_args = build_eval_args(eval_module, checkpoint_args, args)
    tokenizer = eval_module.tokenizer_from_checkpoint(
        checkpoint.get("tokenizer", {}),
        eval_args,
    )
    device = torch.device(str(args.device))
    model = eval_module._text_probe.build_model(
        eval_args,
        vocab_size=tokenizer.vocab_size,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    seed_text = build_seed_text(str(args.prompt), raw=bool(args.raw_prompt))
    sample = eval_module._text_probe.generate_text(
        model,
        tokenizer,
        seed_text=seed_text,
        seq_len=int(eval_args.seq_len),
        think_steps=int(args.think_steps),
        max_new_chars=int(args.max_new_chars),
        device=device,
    )
    if bool(args.answer_only) and sample.startswith(seed_text):
        return sample[len(seed_text) :].strip()
    return sample


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run donorless QTRM-native language inference."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Why should evidence be checked before trusting a claim?",
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "local_eval/qtrm_native_language_bootstrap_bilingual_bpe16k_d192_"
            "external4500_s3600_20260515/last.pt"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--think-steps", type=int, default=4)
    parser.add_argument("--max-new-chars", type=int, default=220)
    parser.add_argument("--raw-prompt", action="store_true")
    parser.add_argument("--show-prompt", dest="answer_only", action="store_false")
    parser.set_defaults(answer_only=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    print(run_inference(args))


if __name__ == "__main__":
    main()
