#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch

from qtrm_mm.config import load_config
from qtrm_mm.diagnostics import next_token_diagnostics, repetition_stats, topk_token_report
from qtrm_mm.qtrm_model import QTRMMultimodalModel
from qtrm_mm.qwen_donor import QwenDonorAdapter


DEFAULT_PROMPTS = [
    "Explain quantum entanglement in simple terms.",
    "양자 컴퓨팅이란 무엇인가요?",
    "Solve step by step: if x + 3 = 7, what is x?",
]


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Evaluate QTRM logits, target-token rank, entropy, and repetition.")
    ap.add_argument("--config", default="configs/qwen35_2b_4090.yaml")
    ap.add_argument("--checkpoint", default="runs/qwen35_2b_4090/last.pt")
    ap.add_argument("--prompt", action="append", default=None, help="Prompt/text to evaluate. Can be repeated.")
    ap.add_argument("--data-jsonl", nargs="*", default=None, help="Optional JSONL files with text or prompt/answer fields.")
    ap.add_argument("--max-samples", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--no-donor", action="store_true", help="Do not pass donor hidden states into QTRM.")
    ap.add_argument(
        "--refresh-donor-each-step",
        action="store_true",
        help="Deprecated compatibility flag; donor states are refreshed by default.",
    )
    ap.add_argument(
        "--fixed-donor-during-generation",
        action="store_true",
        help="Keep initial prompt donor states fixed during greedy generation.",
    )
    ap.add_argument("--json", action="store_true", help="Emit JSON records instead of pretty text.")
    return ap


def iter_jsonl_texts(paths: Iterable[str]) -> Iterable[str]:
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = row.get("text")
                if not text:
                    prompt = row.get("prompt") or ""
                    answer = row.get("answer") or ""
                    text = f"{prompt}\n\n{answer}".strip()
                if text:
                    yield text


def collect_texts(args: argparse.Namespace) -> list[str]:
    if args.prompt:
        return args.prompt[: args.max_samples]
    if args.data_jsonl:
        texts = []
        for text in iter_jsonl_texts(args.data_jsonl):
            texts.append(text)
            if len(texts) >= args.max_samples:
                break
        if texts:
            return texts
    return DEFAULT_PROMPTS[: args.max_samples]


def select_device(cfg_device: str, requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return "cuda"
    return "cuda" if torch.cuda.is_available() and cfg_device in {"auto", "cuda"} else "cpu"


def load_qtrm(config_path: str, checkpoint_path: str, device: str) -> QTRMMultimodalModel:
    cfg = load_config(config_path)
    model = QTRMMultimodalModel(cfg.model)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state.get("model", state), strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)}")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)}")
    return model.to(device).eval()


def prepare_inputs(tokenizer, text: str, max_length: int, device: str) -> dict[str, torch.Tensor]:
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=True,
    )
    return {k: v.to(device) for k, v in enc.items()}


@torch.no_grad()
def donor_kwargs(
    donor: QwenDonorAdapter | None,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: str,
    *,
    return_logits: bool = False,
):
    if donor is None:
        return {}
    encoded = donor.encode_inputs(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_logits=return_logits,
    )
    out = {"text_states": encoded["text_states"].to(device)}
    if encoded.get("visual_features") is not None:
        out["visual_features"] = encoded["visual_features"].to(device)
    if return_logits and encoded.get("logits") is not None:
        out["donor_logits"] = encoded["logits"].to(device)
    return out


@torch.no_grad()
def greedy_generate(
    model: QTRMMultimodalModel,
    donor: QwenDonorAdapter | None,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    device: str,
    max_new_tokens: int,
    refresh_donor_each_step: bool,
    return_donor_logits: bool,
) -> tuple[list[int], list[dict]]:
    generated = input_ids[0].detach().cpu().tolist()
    prompt_len = len(generated)
    fixed_donor = donor_kwargs(
        donor,
        input_ids,
        attention_mask,
        device,
        return_logits=return_donor_logits,
    )
    steps = []
    for step in range(max_new_tokens):
        cur_ids = torch.tensor([generated], dtype=torch.long, device=device)
        cur_mask = torch.ones_like(cur_ids)
        extra = (
            donor_kwargs(
                donor,
                cur_ids,
                cur_mask,
                device,
                return_logits=return_donor_logits,
            )
            if refresh_donor_each_step
            else fixed_donor
        )
        with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.bfloat16):
            outputs = model(cur_ids, attention_mask=cur_mask, **extra)
        last_logits = outputs["logits"][0, -1].float()
        next_id = int(last_logits.argmax(dim=-1).detach().cpu().item())
        steps.append(
            {
                "step": step,
                "next_id": next_id,
                "next_token": tokenizer.decode([next_id], skip_special_tokens=False),
                "top": topk_token_report(last_logits, tokenizer=tokenizer, k=5),
            }
        )
        if tokenizer.eos_token_id is not None and next_id == tokenizer.eos_token_id:
            break
        generated.append(next_id)
    return generated, steps


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = load_config(args.config)
    device = select_device(cfg.train.device, args.device)
    if not cfg.donor.model_id:
        raise SystemExit("donor.model_id is required for tokenizer loading")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.donor.model_id, trust_remote_code=cfg.donor.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_length = args.max_length or cfg.train.seq_len
    model = load_qtrm(args.config, args.checkpoint, device)
    donor = None if args.no_donor else QwenDonorAdapter(cfg.donor)
    texts = collect_texts(args)
    refresh_donor_each_step = args.refresh_donor_each_step or not args.fixed_donor_during_generation
    use_donor_logits = bool(cfg.model.donor_logits_scale != 0.0)

    if not args.json:
        print("=" * 72)
        print("QTRM logit diagnostics")
        print(f"checkpoint={args.checkpoint}")
        print(f"config={args.config}")
        print(
            f"device={device}, donor={'off' if donor is None else 'on'}, "
            f"max_length={max_length}, refresh_donor_each_step={refresh_donor_each_step}, "
            f"donor_logits_scale={cfg.model.donor_logits_scale}"
        )
        print("=" * 72)

    for idx, text in enumerate(texts):
        inputs = prepare_inputs(tokenizer, text, max_length, device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
        extra = donor_kwargs(
            donor,
            input_ids,
            attention_mask,
            device,
            return_logits=use_donor_logits,
        )

        with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.bfloat16):
            outputs = model(input_ids, attention_mask=attention_mask, **extra)
        offset = outputs["logits"].shape[1] - input_ids.shape[1]
        metrics = next_token_diagnostics(
            outputs["logits"],
            input_ids,
            offset=offset,
            attention_mask=attention_mask,
        )
        last_logits = outputs["logits"][0, -1].float()
        top = topk_token_report(last_logits, tokenizer=tokenizer, k=args.top_k)
        generated, steps = greedy_generate(
            model,
            donor,
            tokenizer,
            input_ids,
            attention_mask,
            device=device,
            max_new_tokens=args.max_new_tokens,
            refresh_donor_each_step=refresh_donor_each_step,
            return_donor_logits=use_donor_logits,
        )
        rep = repetition_stats(generated, prompt_len=input_ids.shape[1])
        record = {
            "sample": idx,
            "text": text,
            "input_tokens": int(input_ids.shape[1]),
            "offset": int(offset),
            "teacher_forced": metrics,
            "next_token_topk": top,
            "greedy_text": tokenizer.decode(generated, skip_special_tokens=True),
            "greedy_repetition": rep,
            "greedy_steps": steps[: min(5, len(steps))],
        }

        if args.json:
            print(json.dumps(record, ensure_ascii=False))
            continue

        print(f"\n[{idx}] {text[:180]}")
        print(
            "teacher_forced: "
            f"loss={metrics['loss']:.4f} ppl={metrics['ppl']:.2f} "
            f"rank_mean={metrics['target_rank_mean']:.2f} top1={metrics['target_top1_acc']:.3f} "
            f"top5={metrics['target_top5_acc']:.3f} entropy={metrics['entropy_mean']:.3f} "
            f"max_prob={metrics['max_prob_mean']:.3f}"
        )
        print("next top-k:")
        for item in top:
            print(f"  id={item['token_id']:>6} prob={item['prob']:.4f} token={item['token']!r}")
        print(
            "greedy repetition: "
            f"completion_tokens={rep['completion_tokens']} max_run={rep['max_token_run']} "
            f"common={rep['most_common_token_id']}x{rep['most_common_token_count']} "
            f"rep2={rep['repeated_2gram_rate']:.3f} rep3={rep['repeated_3gram_rate']:.3f}"
        )
        print("greedy text:")
        print(record["greedy_text"])


if __name__ == "__main__":
    main()
