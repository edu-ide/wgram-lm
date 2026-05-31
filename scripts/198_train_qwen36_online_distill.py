#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import torch

from wgram_lm.distill.online_qwen36 import (
    build_online_teacher_prompt,
    clean_teacher_answer,
    teacher_answer_record,
)


def _load_depth_train_module():
    path = Path(__file__).resolve().parent / "196_train_pure_recursive_depth_supervised.py"
    spec = importlib.util.spec_from_file_location("depth_supervised_train", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    spec.loader.exec_module(module)
    return module


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Online distill QTRM from a local Qwen3.6 teacher. The teacher is "
            "called inside the training loop to produce hard answer targets."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--cases-jsonl", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--qwen36-model-path", required=True)
    parser.add_argument("--tokenizer-model-id", default="Qwen/Qwen3.5-2B-Base")
    parser.add_argument("--teacher-tokenizer-path", default="")
    parser.add_argument("--teacher-device-map", default="auto")
    parser.add_argument("--teacher-dtype", choices=["auto", "bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--teacher-max-new-tokens", type=int, default=24)
    parser.add_argument("--teacher-temperature", type=float, default=0.0)
    parser.add_argument("--teacher-top-p", type=float, default=1.0)
    parser.add_argument("--teacher-use-chat-template", action="store_true")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--depth-steps", default="1,2,4,8")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--save-teacher-jsonl", default="")
    parser.add_argument("--causal-prefix-max-target-tokens", type=int, default=1)
    parser.add_argument("--final-logit-ce-weight", type=float, default=1.0)
    parser.add_argument("--all-depth-ce-weight", type=float, default=0.0)
    parser.add_argument("--progress-margin-weight", type=float, default=0.25)
    parser.add_argument("--progress-margin", type=float, default=0.10)
    parser.add_argument("--log-every", type=int, default=1)
    return parser


def load_case_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt = str(row.get("prompt", "")).strip()
            if not prompt:
                raise ValueError(f"{path}:{line_no}: missing prompt")
            rows.append(row)
    if not rows:
        raise ValueError(f"no rows found in {path}")
    return rows


def _teacher_dtype(value: str):
    if value == "auto":
        return "auto"
    if value == "bf16":
        return torch.bfloat16
    if value == "fp16":
        return torch.float16
    return torch.float32


def _teacher_device(model) -> torch.device:
    return next(model.parameters()).device


def generate_teacher_answer(
    *,
    teacher_model,
    teacher_tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    use_chat_template: bool,
) -> str:
    teacher_prompt = build_online_teacher_prompt(prompt)
    if use_chat_template and getattr(teacher_tokenizer, "chat_template", None):
        text = teacher_tokenizer.apply_chat_template(
            [{"role": "user", "content": teacher_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = teacher_prompt
    enc = teacher_tokenizer(text, return_tensors="pt")
    device = _teacher_device(teacher_model)
    enc = {key: value.to(device) for key, value in enc.items()}
    do_sample = float(temperature) > 0.0
    with torch.no_grad():
        out = teacher_model.generate(
            **enc,
            max_new_tokens=int(max_new_tokens),
            do_sample=do_sample,
            temperature=float(temperature) if do_sample else None,
            top_p=float(top_p) if do_sample else None,
            pad_token_id=teacher_tokenizer.eos_token_id,
        )
    new_tokens = out[0, enc["input_ids"].shape[1] :]
    return clean_teacher_answer(teacher_tokenizer.decode(new_tokens, skip_special_tokens=True))


def load_teacher_model(
    model_path: str,
    *,
    teacher_dtype: str,
    device_map: str,
    auto_model_for_causal_lm=None,
    auto_model_for_image_text_to_text=None,
    auto_model=None,
):
    if auto_model_for_causal_lm is None:
        from transformers import AutoModelForCausalLM

        auto_model_for_causal_lm = AutoModelForCausalLM
    if auto_model_for_image_text_to_text is None:
        try:
            from transformers import AutoModelForImageTextToText
        except Exception:
            AutoModelForImageTextToText = None
        auto_model_for_image_text_to_text = AutoModelForImageTextToText
    if auto_model is None:
        try:
            from transformers import AutoModel
        except Exception:
            AutoModel = None
        auto_model = AutoModel

    kwargs = {
        "trust_remote_code": True,
        "torch_dtype": _teacher_dtype(teacher_dtype),
        "device_map": device_map,
    }
    candidates = [
        ("AutoModelForCausalLM", auto_model_for_causal_lm),
        ("AutoModelForImageTextToText", auto_model_for_image_text_to_text),
        ("AutoModel", auto_model),
    ]
    errors: list[str] = []
    for label, loader in candidates:
        if loader is None:
            continue
        try:
            model = loader.from_pretrained(model_path, **kwargs)
            print(f"[teacher] loader={label}")
            return model
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
    joined = "\n".join(errors)
    raise RuntimeError(f"failed to load teacher model from {model_path}\n{joined}")


def main() -> None:
    args = build_arg_parser().parse_args()
    depth_train = _load_depth_train_module()

    from transformers import AutoTokenizer

    from wgram_lm.config import load_config
    from wgram_lm.wgram_model import QTRMMultimodalModel
    from wgram_lm.qwen_donor import QwenDonorAdapter
    from wgram_lm.training.train import configure_trainable_parameters, load_initial_checkpoint

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() and cfg.train.device in {"auto", "cuda"} else "cpu"
    rows = load_case_rows(args.cases_jsonl)
    student_tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_model_id,
        trust_remote_code=cfg.donor.trust_remote_code,
    )
    if student_tokenizer.pad_token_id is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token

    model = QTRMMultimodalModel(cfg.model).to(device)
    missing, unexpected = load_initial_checkpoint(model, args.init_checkpoint, map_location=device)
    if missing:
        print(f"[init] missing keys: {len(missing)}")
    if unexpected:
        print(f"[init] unexpected keys: {len(unexpected)}")
    donor = QwenDonorAdapter(cfg.donor)
    trainable_names = configure_trainable_parameters(model, cfg.train.trainable_param_policy)
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise ValueError("no trainable parameters selected")
    print(
        f"[trainable] policy={cfg.train.trainable_param_policy} "
        f"params={sum(p.numel() for p in params):,} tensors={len(trainable_names)}"
    )

    teacher_tokenizer_path = args.teacher_tokenizer_path or args.qwen36_model_path
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        teacher_tokenizer_path,
        trust_remote_code=True,
    )
    teacher_model = load_teacher_model(
        args.qwen36_model_path,
        teacher_dtype=args.teacher_dtype,
        device_map=args.teacher_device_map,
    )
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad_(False)
    print(f"[teacher] loaded {args.qwen36_model_path}")

    opt = torch.optim.AdamW(
        params,
        lr=float(args.lr if args.lr is not None else cfg.train.lr),
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.train.use_amp and device == "cuda"))
    steps = int(args.steps if args.steps is not None else cfg.train.steps)
    depth_steps = depth_train.parse_depth_steps(args.depth_steps)
    max_length = int(args.max_length or cfg.train.seq_len)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    teacher_jsonl = Path(args.save_teacher_jsonl) if args.save_teacher_jsonl else None
    if teacher_jsonl is not None:
        teacher_jsonl.parent.mkdir(parents=True, exist_ok=True)
    model.train()

    for step in range(steps):
        row_index, core_steps = depth_train.scheduled_row_and_core_steps(
            step,
            row_count=len(rows),
            depth_steps=depth_steps,
        )
        prompt = str(rows[row_index]["prompt"])
        answer = generate_teacher_answer(
            teacher_model=teacher_model,
            teacher_tokenizer=teacher_tokenizer,
            prompt=prompt,
            max_new_tokens=args.teacher_max_new_tokens,
            temperature=args.teacher_temperature,
            top_p=args.teacher_top_p,
            use_chat_template=args.teacher_use_chat_template,
        )
        if teacher_jsonl is not None:
            record = teacher_answer_record(
                prompt=prompt,
                answer=answer,
                teacher_model=str(args.qwen36_model_path),
            )
            with teacher_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

        train_examples = depth_train._prepare_causal_prefix_answer_examples(
            student_tokenizer,
            prompt,
            answer,
            max_length=max_length,
            device=device,
            max_target_tokens=args.causal_prefix_max_target_tokens,
        )
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(cfg.train.use_amp and device == "cuda"), dtype=torch.bfloat16):
            losses = []
            metric_sums = {}
            metric_counts = {}
            for input_ids, attention_mask, target_ids, target_start, target_end in train_examples:
                donor_out = donor.encode_inputs(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_logits=False,
                )
                old_outer_steps = int(model.cfg.outer_steps)
                model.cfg.outer_steps = int(core_steps)
                try:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        text_states=donor_out["text_states"].detach().to(device),
                        return_core_depth_logits=True,
                        return_core_depth_text_logits=True,
                    )
                finally:
                    model.cfg.outer_steps = old_outer_steps
                offset = outputs["logits"].shape[1] - input_ids.shape[1]
                final_text_logits = outputs["logits"][
                    :,
                    offset + target_start - 1 : offset + target_end - 1,
                    :,
                ]
                depth_text_logits = outputs["core_depth_text_logits"][
                    :,
                    :,
                    target_start - 1 : target_end - 1,
                    :,
                ]
                loss, metrics = depth_train.depth_sequence_supervision_loss(
                    depth_text_logits,
                    final_text_logits,
                    target_ids,
                    final_logit_ce_weight=args.final_logit_ce_weight,
                    all_depth_ce_weight=args.all_depth_ce_weight,
                    progress_margin_weight=args.progress_margin_weight,
                    progress_margin=args.progress_margin,
                )
                losses.append(loss)
                for key, value in metrics.items():
                    detached = value.detach()
                    if key not in metric_sums:
                        metric_sums[key] = detached.new_zeros(())
                        metric_counts[key] = 0
                    metric_sums[key] = metric_sums[key] + detached
                    metric_counts[key] += 1
            total_loss = losses[0]
            for extra_loss in losses[1:]:
                total_loss = total_loss + extra_loss
            total_loss = total_loss / float(len(losses))
            metrics = {
                key: metric_sums[key] / float(metric_counts[key])
                for key in metric_sums
            }
        scaler.scale(total_loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        if step % max(1, int(args.log_every)) == 0:
            desc = " ".join(
                f"{key}={float(value):.4f}"
                for key, value in {
                    "loss": total_loss.detach(),
                    "core_steps": total_loss.detach().new_tensor(float(core_steps)),
                    "teacher_answer_tokens": total_loss.detach().new_tensor(float(len(train_examples))),
                    **metrics,
                }.items()
            )
            print(f"step={step} answer={answer!r} {desc}")

    torch.save({"model": model.state_dict()}, out_dir / "last.pt")
    print(f"saved {out_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
