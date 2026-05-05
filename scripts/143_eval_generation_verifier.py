#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import torch

from qtrm_mm.config import load_config
from qtrm_mm.qtrm_model import QTRMMultimodalModel
from qtrm_mm.qwen_donor import QwenDonorAdapter


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Evaluate QTRM generation verifier heads on labeled JSONL rows.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--use-donor", action="store_true")
    return ap


def binary_metrics(
    *,
    probs: Iterable[float],
    targets: Iterable[float],
    threshold: float = 0.5,
) -> dict:
    tp = fp = tn = fn = 0
    for prob, target in zip(probs, targets):
        pred = float(prob) >= float(threshold)
        actual = float(target) >= 0.5
        if pred and actual:
            tp += 1
        elif pred and not actual:
            fp += 1
        elif not pred and actual:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / max(1, tp + fp + tn + fn)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def best_threshold_metrics(
    *,
    probs: Iterable[float],
    targets: Iterable[float],
) -> dict:
    probs_list = [float(prob) for prob in probs]
    targets_list = [float(target) for target in targets]
    if not probs_list:
        return {"threshold": 0.5, **binary_metrics(probs=[], targets=[])}
    rows = [
        {"threshold": threshold, **binary_metrics(probs=probs_list, targets=targets_list, threshold=threshold)}
        for threshold in sorted(set(probs_list), reverse=True)
    ]
    return max(
        rows,
        key=lambda row: (
            row["f1"],
            row["precision"],
            row["recall"],
            row["accuracy"],
            row["threshold"],
        ),
    )


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def select_device(requested: str, cfg_device: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return "cuda"
    return "cuda" if torch.cuda.is_available() and cfg_device in {"auto", "cuda"} else "cpu"


def load_model(config_path: str, checkpoint_path: str, device: str) -> QTRMMultimodalModel:
    cfg = load_config(config_path)
    model = QTRMMultimodalModel(cfg.model)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state.get("model", state), strict=False)
    return model.to(device).eval()


@torch.no_grad()
def run_eval(args: argparse.Namespace) -> dict:
    from transformers import AutoTokenizer

    cfg = load_config(args.config)
    device = select_device(args.device, cfg.train.device)
    if not cfg.donor.model_id:
        raise ValueError("donor.model_id is required for tokenizer loading")
    tokenizer = AutoTokenizer.from_pretrained(cfg.donor.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model(args.config, args.checkpoint, device)
    donor = QwenDonorAdapter(cfg.donor) if args.use_donor else None
    rows = load_jsonl(args.data_jsonl)

    records = []
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        texts = [str(row.get("text") or "") for row in batch_rows]
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=cfg.train.seq_len,
            return_tensors="pt",
        )
        enc = {key: value.to(device) for key, value in enc.items()}
        model_kwargs = {"attention_mask": enc["attention_mask"]}
        if donor is not None:
            donor_out = donor.encode_inputs(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                return_logits=False,
            )
            model_kwargs["text_states"] = donor_out["text_states"].detach().to(device)
        with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.bfloat16):
            outputs = model(enc["input_ids"], **model_kwargs)
        repeat_probs = outputs["generation_repeat_logits"].float().sigmoid().detach().cpu().tolist()
        stop_probs = outputs["generation_stop_logits"].float().sigmoid().detach().cpu().tolist()
        quality_probs = outputs["generation_quality_logits"].float().sigmoid().detach().cpu().tolist()
        for row, repeat_prob, stop_prob, quality_prob in zip(
            batch_rows,
            repeat_probs,
            stop_probs,
            quality_probs,
        ):
            records.append(
                {
                    "source_sample": row.get("source_sample"),
                    "candidate_id": row.get("candidate_id"),
                    "category": row.get("category"),
                    "text": row.get("text"),
                    "repeat_prob": float(repeat_prob),
                    "stop_prob": float(stop_prob),
                    "quality_prob": float(quality_prob),
                    "repeat_target": float(row.get("generation_verifier_repeat_target", 0.0)),
                    "stop_target": float(row.get("generation_verifier_stop_target", 0.0)),
                    "quality_target": float(row.get("generation_verifier_quality_target", 1.0)),
                }
            )

    summary = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "data_jsonl": args.data_jsonl,
        "threshold": args.threshold,
        "n": len(records),
        "repeat": binary_metrics(
            probs=[row["repeat_prob"] for row in records],
            targets=[row["repeat_target"] for row in records],
            threshold=args.threshold,
        ),
        "repeat_best_threshold": best_threshold_metrics(
            probs=[row["repeat_prob"] for row in records],
            targets=[row["repeat_target"] for row in records],
        ),
        "stop": binary_metrics(
            probs=[row["stop_prob"] for row in records],
            targets=[row["stop_target"] for row in records],
            threshold=args.threshold,
        ),
        "stop_best_threshold": best_threshold_metrics(
            probs=[row["stop_prob"] for row in records],
            targets=[row["stop_target"] for row in records],
        ),
        "quality": binary_metrics(
            probs=[row["quality_prob"] for row in records],
            targets=[row["quality_target"] for row in records],
            threshold=args.threshold,
        ),
        "quality_best_threshold": best_threshold_metrics(
            probs=[row["quality_prob"] for row in records],
            targets=[row["quality_target"] for row in records],
        ),
        "records": records,
    }
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = run_eval(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
