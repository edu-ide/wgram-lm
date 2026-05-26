#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import re
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
from qtrm_mm.eval.general_answer_interface import extract_answer_candidate_text
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
    ap.add_argument("--num-candidates", type=int, default=1)
    ap.add_argument("--do-sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--no-repeat-ngram-size", type=int, default=0)
    ap.add_argument("--donor-logits-scale", type=float, default=None)
    ap.add_argument("--qtrm-logits-scale", type=float, default=None)
    ap.add_argument("--qtrm-residual-clamp", type=float, default=None)
    ap.add_argument("--stop-after-sentence", action="store_true")
    ap.add_argument("--min-new-tokens-before-stop", type=int, default=16)
    ap.add_argument("--answer-contract", choices=["none", "direct"], default="none")
    ap.add_argument(
        "--suppress-visible-reasoning-tokens",
        action="store_true",
        help="Suppress common visible reasoning tokens such as <think> during generation.",
    )
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
    ap.add_argument(
        "--stage59-candidates-jsonl",
        default="",
        help=(
            "Optional output JSONL in the Stage59 general-answer candidate contract: "
            "{id, candidates, raw_completions}. Requires --data-jsonl rows with id/case_id."
        ),
    )
    ap.add_argument("--json", action="store_true", help="Emit JSON records instead of pretty text.")
    return ap


DIRECT_ANSWER_CONTRACT = (
    "\n\n/no_think\n"
    "Answer directly. Do not reveal hidden reasoning. "
    "Do not create multiple-choice options or a new question."
)


def apply_answer_contract(text: str, mode: str) -> str:
    if mode == "none":
        return str(text)
    if mode == "direct":
        return str(text).rstrip() + DIRECT_ANSWER_CONTRACT
    raise ValueError(f"unknown answer contract: {mode}")


def _top_p_filtered_logits(logits: torch.Tensor, *, top_p: float) -> torch.Tensor:
    top_p = float(top_p)
    if top_p >= 1.0:
        return logits
    if top_p <= 0.0:
        raise ValueError("top_p must be > 0")
    sorted_logits, sorted_indices = torch.sort(logits.float(), descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = sorted_probs.cumsum(dim=-1)
    remove = cumulative > top_p
    remove[1:] = remove[:-1].clone()
    remove[0] = False
    filtered = logits.float().clone()
    filtered[sorted_indices[remove]] = -torch.inf
    return filtered


def apply_token_suppression(
    logits: torch.Tensor,
    suppressed_token_ids: Iterable[int] | None,
) -> torch.Tensor:
    ids = [int(token_id) for token_id in (suppressed_token_ids or []) if 0 <= int(token_id) < logits.shape[-1]]
    if not ids:
        return logits
    filtered = logits.float().clone()
    filtered[torch.tensor(ids, device=filtered.device, dtype=torch.long)] = -torch.inf
    return filtered


def no_repeat_ngram_banned_tokens(generated: Sequence[int], ngram_size: int) -> list[int]:
    n = int(ngram_size)
    if n <= 0 or len(generated) < n - 1:
        return []
    if n == 1:
        return sorted(set(int(token_id) for token_id in generated))
    prefix = tuple(int(token_id) for token_id in generated[-(n - 1) :])
    banned: set[int] = set()
    for idx in range(0, len(generated) - n + 1):
        ngram = tuple(int(token_id) for token_id in generated[idx : idx + n])
        if ngram[:-1] == prefix:
            banned.add(ngram[-1])
    return sorted(banned)


def generated_completion_text(tokenizer, generated: Sequence[int], *, prompt_len: int) -> str:
    full_text = tokenizer.decode(generated, skip_special_tokens=True)
    prompt_text = tokenizer.decode(generated[:prompt_len], skip_special_tokens=True)
    if prompt_text and full_text.startswith(prompt_text):
        return full_text[len(prompt_text) :].strip()
    return tokenizer.decode(generated[prompt_len:], skip_special_tokens=True).strip()


def should_stop_after_sentence(
    generated: Sequence[int],
    *,
    prompt_len: int,
    tokenizer,
    enabled: bool,
    min_new_tokens_before_stop: int,
) -> bool:
    if not enabled or len(generated) - int(prompt_len) < int(min_new_tokens_before_stop):
        return False
    completion = generated_completion_text(tokenizer, generated, prompt_len=prompt_len)
    if not completion:
        return False
    return re.search(r"[.!?。！？]\s*$", completion) is not None


def select_next_token(
    logits: torch.Tensor,
    *,
    do_sample: bool,
    temperature: float,
    top_p: float,
    generator: torch.Generator | None,
    suppressed_token_ids: Iterable[int] | None = None,
) -> int:
    logits = apply_token_suppression(logits.float(), suppressed_token_ids)
    if not do_sample:
        return int(logits.argmax(dim=-1).detach().cpu().item())
    temperature = max(float(temperature), 1e-6)
    filtered = _top_p_filtered_logits(logits / temperature, top_p=top_p)
    probs = torch.softmax(filtered, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1, generator=generator)
    return int(next_id.detach().cpu().item())


def visible_reasoning_token_ids(tokenizer, *, enabled: bool) -> list[int]:
    if not enabled:
        return []
    markers = ("<think>", "</think>")
    ids: list[int] = []
    for marker in markers:
        try:
            encoded = tokenizer.encode(marker, add_special_tokens=False)
        except Exception:
            encoded = []
        ids.extend(int(token_id) for token_id in encoded)
    return sorted(set(ids))


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


def _row_id(row: dict, fallback: str) -> str:
    for key in ("id", "case_id", "example_id", "uid"):
        if row.get(key) is not None:
            return str(row[key])
    return fallback


def _row_prompt(row: dict) -> str:
    for key in ("prompt", "qwen_prompt", "question", "text"):
        if row.get(key):
            return str(row[key])
    prompt = row.get("prompt") or ""
    answer = row.get("answer") or ""
    return f"{prompt}\n\n{answer}".strip()


def iter_stage59_rows(paths: Iterable[str]) -> Iterable[dict]:
    ordinal = 0
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
                if not isinstance(row, dict):
                    continue
                text = _row_prompt(row)
                if not text:
                    continue
                yield {
                    "id": _row_id(row, f"row-{ordinal}"),
                    "text": text,
                    "task_family": row.get("task_family") or row.get("family") or row.get("category") or "unknown",
                }
                ordinal += 1


def collect_texts(args: argparse.Namespace) -> list[str]:
    if args.prompt:
        return [apply_answer_contract(text, args.answer_contract) for text in args.prompt[: args.max_samples]]
    if args.data_jsonl:
        texts = []
        for text in iter_jsonl_texts(args.data_jsonl):
            texts.append(apply_answer_contract(text, args.answer_contract))
            if len(texts) >= args.max_samples:
                break
        if texts:
            return texts
    return [apply_answer_contract(text, args.answer_contract) for text in DEFAULT_PROMPTS[: args.max_samples]]


def collect_stage59_items(args: argparse.Namespace) -> list[dict]:
    if not args.data_jsonl:
        raise SystemExit("--stage59-candidates-jsonl requires --data-jsonl")
    items = []
    for row in iter_stage59_rows(args.data_jsonl):
        items.append({**row, "text": apply_answer_contract(str(row["text"]), args.answer_contract)})
        if len(items) >= args.max_samples:
            break
    if not items:
        raise SystemExit("no Stage59 rows loaded from --data-jsonl")
    return items


def apply_ablation_mode(model: QTRMMultimodalModel, mode: str) -> None:
    if mode == "donor_only":
        model.cfg.qtrm_logits_scale = 0.0
        model.cfg.donor_logits_scale = 1.0


def apply_logit_scale_overrides(model: QTRMMultimodalModel, args: argparse.Namespace) -> None:
    if args.donor_logits_scale is not None:
        model.cfg.donor_logits_scale = float(args.donor_logits_scale)
    if args.qtrm_logits_scale is not None:
        model.cfg.qtrm_logits_scale = float(args.qtrm_logits_scale)
    if args.qtrm_residual_clamp is not None:
        model.cfg.qtrm_residual_clamp = float(args.qtrm_residual_clamp)


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


def residual_gate_telemetry(outputs: dict[str, torch.Tensor]) -> dict:
    gate = outputs.get("qtrm_residual_gate")
    if gate is None:
        return {"enabled": False, "values": None, "mean": None}
    gate = gate.detach().float().cpu()
    return {
        "enabled": True,
        "values": gate.tolist(),
        "mean": float(gate.mean().item()),
        "min": float(gate.min().item()),
        "max": float(gate.max().item()),
    }


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
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    seed: int | None = None,
    suppressed_token_ids: Iterable[int] | None = None,
    no_repeat_ngram_size: int = 0,
    stop_after_sentence: bool = False,
    min_new_tokens_before_stop: int = 16,
) -> tuple[list[int], list[dict]]:
    generated = input_ids[0].detach().cpu().tolist()
    prompt_len = len(generated)
    generator = None
    if do_sample:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed if seed is not None else 0))
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
        step_suppressed_ids = list(suppressed_token_ids or [])
        step_suppressed_ids.extend(
            no_repeat_ngram_banned_tokens(generated[prompt_len:], no_repeat_ngram_size)
        )
        next_id = select_next_token(
            last_logits,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            generator=generator,
            suppressed_token_ids=step_suppressed_ids,
        )
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
        if should_stop_after_sentence(
            generated,
            prompt_len=prompt_len,
            tokenizer=tokenizer,
            enabled=stop_after_sentence,
            min_new_tokens_before_stop=min_new_tokens_before_stop,
        ):
            break
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
    apply_logit_scale_overrides(model, args)
    apply_ablation_mode(model, args.ablation_mode)
    donor = build_donor(cfg, no_donor=args.no_donor, json_mode=args.json)
    stage59_mode = bool(str(args.stage59_candidates_jsonl).strip())
    items = (
        collect_stage59_items(args)
        if stage59_mode
        else [
            {"id": f"sample-{index}", "text": text, "task_family": "manual"}
            for index, text in enumerate(collect_texts(args))
        ]
    )
    if args.num_candidates < 1:
        raise SystemExit("--num-candidates must be >= 1")
    refresh_donor_each_step = args.refresh_donor_each_step or not args.fixed_donor_during_generation
    use_donor_logits = bool(model.cfg.donor_logits_scale != 0.0)
    fwd_ablation = forward_ablation_kwargs(args.ablation_mode)
    suppressed_token_ids = visible_reasoning_token_ids(
        tokenizer,
        enabled=args.suppress_visible_reasoning_tokens,
    )

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
            f"num_candidates={args.num_candidates}, do_sample={args.do_sample}, "
            f"temperature={args.temperature}, top_p={args.top_p}, seed={args.seed}, "
            f"no_repeat_ngram_size={args.no_repeat_ngram_size}, "
            f"stop_after_sentence={args.stop_after_sentence}, "
            f"answer_contract={args.answer_contract}, "
            f"suppressed_token_ids={suppressed_token_ids}, "
            f"donor_logits_scale={model.cfg.donor_logits_scale}, "
            f"qtrm_logits_scale={model.cfg.qtrm_logits_scale}"
        )
        print("=" * 72)

    stage59_records = []
    for idx, item in enumerate(items):
        text = str(item["text"])
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
        core_halt = core_halt_telemetry(outputs, enabled=args.enable_core_halt)
        residual_gate = residual_gate_telemetry(outputs)
        records = []
        for candidate_id in range(args.num_candidates):
            candidate_seed = int(args.seed) + idx * 1000 + candidate_id
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
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=candidate_seed,
                suppressed_token_ids=suppressed_token_ids,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                stop_after_sentence=args.stop_after_sentence,
                min_new_tokens_before_stop=args.min_new_tokens_before_stop,
            )
            rep = repetition_stats(generated, prompt_len=input_ids.shape[1])
            completion = generated_completion_text(tokenizer, generated, prompt_len=input_ids.shape[1])
            answer_candidate = extract_answer_candidate_text(completion)
            record = {
                "sample": idx,
                "candidate_id": candidate_id,
                "id": item["id"],
                "ablation_mode": args.ablation_mode,
                "core_halt": core_halt,
                "residual_gate": residual_gate,
                "text": text,
                "completion": completion,
                "answer_candidate": answer_candidate,
                "input_tokens": int(input_ids.shape[1]),
                "offset": int(offset),
                "teacher_forced": metrics,
                "next_token_topk": top,
                "residual_telemetry": residual_telemetry,
                "decoding": {
                    "do_sample": bool(args.do_sample),
                    "temperature": float(args.temperature),
                    "top_p": float(args.top_p),
                    "seed": candidate_seed,
                    "suppressed_token_ids": suppressed_token_ids,
                    "no_repeat_ngram_size": int(args.no_repeat_ngram_size),
                    "stop_after_sentence": bool(args.stop_after_sentence),
                    "min_new_tokens_before_stop": int(args.min_new_tokens_before_stop),
                    "answer_contract": args.answer_contract,
                },
                "greedy_text": tokenizer.decode(generated, skip_special_tokens=True),
                "greedy_completion": completion,
                "greedy_repetition": rep,
                "greedy_steps": steps[: min(5, len(steps))],
            }
            records.append(record)
            if args.json:
                print(json.dumps(record, ensure_ascii=False))

        if stage59_mode:
            stage59_records.append(
                {
                    "id": item["id"],
                    "task_family": item.get("task_family", "unknown"),
                    "ablation_mode": args.ablation_mode,
                    "candidates": [str(record.get("answer_candidate", "")) for record in records],
                    "raw_completions": [str(record.get("greedy_completion", "")) for record in records],
                    "full_generations": [str(record.get("greedy_text", "")) for record in records],
                    "num_candidates": len(records),
                    "decoding": records[0]["decoding"] if records else {},
                }
            )

        if args.json:
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
            "residual gate: "
            f"enabled={residual_gate['enabled']} "
            f"mean={residual_gate['mean']} "
            f"min={residual_gate.get('min')} "
            f"max={residual_gate.get('max')}"
        )
        print(
            "core halt: "
            f"enabled={core_halt['enabled']} "
            f"steps={core_halt['core_steps']} "
            f"halted={core_halt['core_halted']} "
            f"q_halt_steps={core_halt['q_halt_steps']} "
            f"q_halt_last_mean={core_halt['q_halt_last_mean']}"
        )
        for record in records:
            rep = record["greedy_repetition"]
            print(
                f"candidate {record['candidate_id']} repetition: "
                f"completion_tokens={rep['completion_tokens']} max_run={rep['max_token_run']} "
                f"common={rep['most_common_token_id']}x{rep['most_common_token_count']} "
                f"rep2={rep['repeated_2gram_rate']:.3f} rep3={rep['repeated_3gram_rate']:.3f}"
            )
            print(f"candidate {record['candidate_id']} text:")
            print(record["greedy_text"])

    if stage59_mode:
        out_path = Path(args.stage59_candidates_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in stage59_records),
            encoding="utf-8",
        )
        if not args.json:
            print(f"\nWrote Stage59 candidate JSONL: {out_path}")


if __name__ == "__main__":
    main()
