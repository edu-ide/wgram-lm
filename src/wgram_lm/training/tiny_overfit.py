from __future__ import annotations

import argparse
import json
from itertools import cycle
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from wgram_lm.config import load_config
from wgram_lm.diagnostics import next_token_diagnostics
from wgram_lm.losses import qtrm_smoke_loss
from wgram_lm.wgram_model import QTRMMultimodalModel
from wgram_lm.qwen_donor import QwenDonorAdapter
from wgram_lm.training.train import prepare_donor_batch


DEFAULT_TEXTS = [
    "Quantum entanglement means two particles can share one linked state.",
    "양자 컴퓨팅은 큐비트를 사용해 특정 계산을 병렬적으로 탐색하는 방식입니다.",
    "If x + 3 = 7, then x = 4.",
    "A model should use retrieval when the answer depends on external facts.",
    "Memory write: store the user's project goal. Memory read: recall it later.",
    "Reasoning trace: state the known facts, derive the next step, verify the result.",
    "Clean data should avoid repeated boilerplate and broken prompt boundaries.",
    "The donor model remains the generation baseline while QTRM learns adapters.",
]


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run a fixed-shard QTRM tiny-overfit diagnostic.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-jsonl", nargs="*", default=None)
    ap.add_argument("--text", action="append", default=None)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--no-donor", action="store_true")
    ap.add_argument("--save", default=None, help="Optional checkpoint path for the overfit run.")
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
    texts = list(args.text or [])
    if args.data_jsonl:
        for text in iter_jsonl_texts(args.data_jsonl):
            texts.append(text)
            if len(texts) >= args.samples:
                break
    if not texts:
        texts = DEFAULT_TEXTS.copy()
    while len(texts) < args.samples:
        texts.extend(texts)
    return texts[: args.samples]


def select_device(cfg_device: str, requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return "cuda"
    return "cuda" if torch.cuda.is_available() and cfg_device in {"auto", "cuda"} else "cpu"


def build_fixed_dataset(tokenizer, texts: list[str], max_length: int) -> TensorDataset:
    rows = []
    masks = []
    for text in texts:
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding="max_length",
            add_special_tokens=True,
        )
        rows.append(enc["input_ids"][0].to(dtype=torch.long))
        masks.append(enc["attention_mask"][0].to(dtype=torch.long))
    return TensorDataset(torch.stack(rows), torch.stack(masks))


def eval_fixed_batch(model, batch, donor, device: str):
    input_ids, attention_mask = (x.to(device) for x in batch)
    model_batch = {"input_ids": input_ids, "attention_mask": attention_mask}
    if donor is not None:
        model_batch.update(prepare_donor_batch(donor, model_batch))
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.bfloat16):
            loss, metrics, outputs = qtrm_smoke_loss(model, **model_batch)
        offset = outputs["logits"].shape[1] - input_ids.shape[1]
        diag = next_token_diagnostics(
            outputs["logits"],
            input_ids,
            offset=offset,
            attention_mask=attention_mask,
        )
    return metrics, diag


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = load_config(args.config)
    if not cfg.donor.model_id:
        raise SystemExit("donor.model_id is required for tokenizer loading")

    device = select_device(cfg.train.device, args.device)
    batch_size = args.batch_size or cfg.train.batch_size
    lr = args.lr or cfg.train.lr
    max_length = args.max_length or min(cfg.train.seq_len, cfg.model.max_seq_len)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.donor.model_id, trust_remote_code=cfg.donor.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = collect_texts(args)
    dataset = build_fixed_dataset(tokenizer, texts, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    eval_batch = next(iter(DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=False)))

    model = QTRMMultimodalModel(cfg.model).to(device)
    model.train()
    donor = None if args.no_donor else QwenDonorAdapter(cfg.donor)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and cfg.train.use_amp))

    print("=" * 72)
    print("QTRM tiny-overfit diagnostic")
    print(f"config={args.config}")
    print(f"device={device}, donor={'off' if donor is None else 'on'}")
    print(f"samples={len(dataset)}, steps={args.steps}, batch_size={batch_size}, lr={lr}, max_length={max_length}")
    print("=" * 72)

    initial_metrics, initial_diag = eval_fixed_batch(model.eval(), eval_batch, donor, device)
    model.train()
    print(
        "initial: "
        f"loss={float(initial_metrics['loss']):.4f} lm={float(initial_metrics['lm']):.4f} "
        f"rank={initial_diag['target_rank_mean']:.2f} top1={initial_diag['target_top1_acc']:.3f} "
        f"entropy={initial_diag['entropy_mean']:.3f}"
    )

    iterator = cycle(loader)
    pbar = tqdm(range(1, args.steps + 1))
    for step in pbar:
        input_ids, attention_mask = (x.to(device) for x in next(iterator))
        batch = {"input_ids": input_ids, "attention_mask": attention_mask}
        if donor is not None:
            batch.update(prepare_donor_batch(donor, batch))

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(device == "cuda" and cfg.train.use_amp), dtype=torch.bfloat16):
            loss, metrics, _ = qtrm_smoke_loss(model, **batch)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        if step % args.log_every == 0 or step == args.steps:
            eval_metrics, diag = eval_fixed_batch(model.eval(), eval_batch, donor, device)
            model.train()
            pbar.set_description(
                f"loss={float(eval_metrics['loss']):.3f} "
                f"lm={float(eval_metrics['lm']):.3f} "
                f"rank={diag['target_rank_mean']:.1f} "
                f"top1={diag['target_top1_acc']:.2f} "
                f"ent={diag['entropy_mean']:.2f}"
            )

    final_metrics, final_diag = eval_fixed_batch(model.eval(), eval_batch, donor, device)
    print(
        "final: "
        f"loss={float(final_metrics['loss']):.4f} lm={float(final_metrics['lm']):.4f} "
        f"rank={final_diag['target_rank_mean']:.2f} top1={final_diag['target_top1_acc']:.3f} "
        f"entropy={final_diag['entropy_mean']:.3f}"
    )
    print(
        "delta: "
        f"loss={float(initial_metrics['loss'] - final_metrics['loss']):.4f} "
        f"lm={float(initial_metrics['lm'] - final_metrics['lm']):.4f} "
        f"rank={initial_diag['target_rank_mean'] - final_diag['target_rank_mean']:.2f}"
    )

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict()}, out)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
