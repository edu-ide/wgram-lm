#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
from typing import Iterable

import torch

from qtrm_mm.config import load_config
from qtrm_mm.diagnostics import (
    next_token_diagnostics,
    repetition_stats,
    residual_logit_telemetry,
    topk_token_report,
)
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
        "--ablation-mode",
        default="residual",
        choices=["residual", "donor_only", "workspace_off", "core_off"],
        help=(
            "Evaluation mode: residual uses configured scales; donor_only forces "
            "QTRM logits off and donor logits on; workspace_off removes latent "
            "workspace prefix; core_off bypasses recursive core updates."
        ),
    )
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
    ap.add_argument(
        "--enable-core-halt",
        action="store_true",
        help="Enable the learned recursive-core halt decision during forward/eval generation.",
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


def apply_ablation_mode(model: QTRMMultimodalModel, mode: str) -> None:
    if mode == "donor_only":
        model.cfg.qtrm_logits_scale = 0.0
        model.cfg.donor_logits_scale = 1.0


def forward_ablation_kwargs(mode: str) -> dict[str, bool]:
    return {
        "disable_workspace": mode == "workspace_off",
        "disable_core": mode == "core_off",
    }


def select_device(cfg_device: str, requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return "cuda"
    return "cuda" if torch.cuda.is_available() and cfg_device in {"auto", "cuda"} else "cpu"


def core_halt_telemetry(outputs: dict[str, torch.Tensor], *, enabled: bool) -> dict:
    q_halt = outputs.get("core_q_halt_logits")
    q_continue = outputs.get("core_q_continue_logits")
    core_steps = outputs.get("core_steps")
    core_halted = outputs.get("core_halted")

    record = {
        "enabled": bool(enabled),
        "core_steps": None,
        "core_halted": None,
        "q_halt_steps": 0,
        "q_halt_last_mean": None,
        "q_continue_steps": 0,
        "q_continue_last_mean": None,
    }
    if core_steps is not None:
        record["core_steps"] = core_steps.detach().cpu().tolist()
    if core_halted is not None:
        record["core_halted"] = core_halted.detach().cpu().tolist()
    if q_halt is not None and q_halt.numel() > 0:
        record["q_halt_steps"] = int(q_halt.shape[1]) if q_halt.ndim >= 2 else int(q_halt.numel())
        q_halt_last = q_halt[:, -1] if q_halt.ndim >= 2 else q_halt[-1:]
        record["q_halt_last_mean"] = float(q_halt_last.float().mean().detach().cpu().item())
    if q_continue is not None and q_continue.numel() > 0:
        record["q_continue_steps"] = int(q_continue.shape[1]) if q_continue.ndim >= 2 else int(q_continue.numel())
        q_continue_last = q_continue[:, -1] if q_continue.ndim >= 2 else q_continue[-1:]
        record["q_continue_last_mean"] = float(q_continue_last.float().mean().detach().cpu().item())
    return record


def load_qtrm(config_path: str, checkpoint_path: str, device: str) -> QTRMMultimodalModel:
    cfg = load_config(config_path)
    model = QTRMMultimodalModel(cfg.model)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state.get("model", state), strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)}", file=sys.stderr)
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)}", file=sys.stderr)
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


def build_donor(cfg, *, no_donor: bool, json_mode: bool) -> QwenDonorAdapter | None:
    if no_donor:
        return None
    if json_mode:
        with redirect_stdout(sys.stderr):
            return QwenDonorAdapter(cfg.donor)
    return QwenDonorAdapter(cfg.donor)


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
    ablation_mode: str,
    enable_core_halt: bool,
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
            outputs = model(
                cur_ids,
                attention_mask=cur_mask,
                **extra,
                **forward_ablation_kwargs(ablation_mode),
                enable_core_halt=enable_core_halt,
            )
        last_logits = outputs["logits"][0, -1].float()
        next_id = int(last_logits.argmax(dim=-1).detach().cpu().item())
        steps.append(
            {
                "step": step,
                "next_id": next_id,
                "next_token": tokenizer.decode([next_id], skip_special_tokens=False),
                "top": topk_token_report(last_logits, tokenizer=tokenizer, k=5),
                "core_halt": core_halt_telemetry(outputs, enabled=enable_core_halt),
            }
        )
        if tokenizer.eos_token_id is not None and next_id == tokenizer.eos_token_id:
            break
        generated.append(next_id)
    return generated, steps


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.no_donor and args.ablation_mode == "donor_only":
        raise SystemExit("--ablation-mode donor_only requires donor logits; remove --no-donor.")

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
    apply_ablation_mode(model, args.ablation_mode)
    donor = build_donor(cfg, no_donor=args.no_donor, json_mode=args.json)
    texts = collect_texts(args)
    refresh_donor_each_step = args.refresh_donor_each_step or not args.fixed_donor_during_generation
    use_donor_logits = bool(model.cfg.donor_logits_scale != 0.0)
    fwd_ablation = forward_ablation_kwargs(args.ablation_mode)

    if not args.json:
        print("=" * 72)
        print("QTRM logit diagnostics")
        print(f"checkpoint={args.checkpoint}")
        print(f"config={args.config}")
        print(
            f"device={device}, donor={'off' if donor is None else 'on'}, "
            f"max_length={max_length}, refresh_donor_each_step={refresh_donor_each_step}, "
            f"ablation_mode={args.ablation_mode}, "
            f"enable_core_halt={args.enable_core_halt}, "
            f"donor_logits_scale={model.cfg.donor_logits_scale}, "
            f"qtrm_logits_scale={model.cfg.qtrm_logits_scale}"
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
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                **extra,
                **fwd_ablation,
                enable_core_halt=args.enable_core_halt,
            )
        offset = outputs["logits"].shape[1] - input_ids.shape[1]
        metrics = next_token_diagnostics(
            outputs["logits"],
            input_ids,
            offset=offset,
            attention_mask=attention_mask,
        )
        last_logits = outputs["logits"][0, -1].float()
        top = topk_token_report(last_logits, tokenizer=tokenizer, k=args.top_k)
        residual_telemetry = None
        if use_donor_logits and extra.get("donor_logits") is not None:
            residual_telemetry = residual_logit_telemetry(
                extra["donor_logits"][0, -1],
                last_logits,
                tokenizer=tokenizer,
                donor_logits_scale=model.cfg.donor_logits_scale,
            )
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
            ablation_mode=args.ablation_mode,
            enable_core_halt=args.enable_core_halt,
        )
        rep = repetition_stats(generated, prompt_len=input_ids.shape[1])
        core_halt = core_halt_telemetry(outputs, enabled=args.enable_core_halt)
        record = {
            "sample": idx,
            "ablation_mode": args.ablation_mode,
            "core_halt": core_halt,
            "text": text,
            "input_tokens": int(input_ids.shape[1]),
            "offset": int(offset),
            "teacher_forced": metrics,
            "next_token_topk": top,
            "residual_telemetry": residual_telemetry,
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
        if residual_telemetry is not None:
            print(
                "residual telemetry: "
                f"argmax_changed={residual_telemetry['argmax_changed']} "
                f"donor_top={residual_telemetry['donor_top_token']!r}/"
                f"{residual_telemetry['donor_top_prob']:.3f} "
                f"fused_top={residual_telemetry['fused_top_token']!r}/"
                f"{residual_telemetry['fused_top_prob']:.3f} "
                f"kl_fd={residual_telemetry['kl_fused_to_donor']:.4f} "
                f"res_l2={residual_telemetry['residual_l2_norm']:.3f} "
                f"res_linf={residual_telemetry['residual_linf_norm']:.3f}"
            )
        print(
            "core halt: "
            f"enabled={core_halt['enabled']} "
            f"steps={core_halt['core_steps']} "
            f"halted={core_halt['core_halted']} "
            f"q_halt_steps={core_halt['q_halt_steps']} "
            f"q_halt_last_mean={core_halt['q_halt_last_mean']}"
        )
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
