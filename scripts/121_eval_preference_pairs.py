#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from typing import Iterable

import torch

from wgram_lm.config import load_config
from wgram_lm.data.jsonl_dataset import JsonlTextVisionDataset, collate_jsonl
from wgram_lm.eval.preference import summarize_preference_records
from wgram_lm.losses import qtrm_smoke_loss
from wgram_lm.wgram_model import QTRMMultimodalModel
from wgram_lm.qwen_donor import QwenDonorAdapter
from wgram_lm.training.train import prepare_donor_batch, strip_training_only_batch_keys


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Evaluate chosen/rejected preference-pair margins.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-jsonl", nargs="+", required=True)
    ap.add_argument("--jsonl-out", default="runs/eval/preference_pairs_eval.jsonl")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--no-donor", action="store_true")
    ap.add_argument("--preference-beta", type=float, default=None)
    ap.add_argument("--preference-margin", type=float, default=None)
    return ap


def select_device(cfg_device: str, requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return "cuda"
    return "cuda" if torch.cuda.is_available() and cfg_device in {"auto", "cuda"} else "cpu"


def iter_preference_rows(paths: Iterable[str]) -> Iterable[dict]:
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
                if row.get("prompt") and row.get("chosen") and row.get("rejected"):
                    yield row


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


def row_weight(row: dict) -> float:
    return float(row.get("preference_weight", row.get("confidence", 1.0)))


def record_id(row: dict, index: int) -> str:
    for key in ("case_id", "id"):
        if row.get(key):
            return str(row[key])
    return f"preference-{index:06d}"


def eval_autocast_context(device: str, use_amp: bool):
    if use_amp and device == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.no_grad()
def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = load_config(args.config)
    device = select_device(cfg.train.device, args.device)
    max_length = int(args.max_length or cfg.train.seq_len)
    preference_beta = float(args.preference_beta if args.preference_beta is not None else cfg.train.preference_beta)
    preference_margin = float(
        args.preference_margin if args.preference_margin is not None else cfg.train.preference_margin
    )

    model = load_qtrm(args.config, args.checkpoint, device)
    donor = None if args.no_donor else QwenDonorAdapter(cfg.donor)
    dataset = JsonlTextVisionDataset(
        files=[],
        vocab_size=cfg.model.vocab_size,
        seq_len=max_length,
        visual_dim=cfg.model.visual_dim,
        max_visual_tokens=min(cfg.model.max_visual_tokens, 64),
        multimodal=False,
        tokenizer_model_id=cfg.donor.model_id,
        workspace_evidence_injection=cfg.train.workspace_evidence_injection,
    )

    out_path = Path(args.jsonl_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    use_donor_logits = bool(model.cfg.donor_logits_scale != 0.0)
    with out_path.open("w", encoding="utf-8") as out:
        for index, row in enumerate(iter_preference_rows(args.data_jsonl)):
            if args.max_samples is not None and index >= args.max_samples:
                break
            sample = dataset._make_sample(row)
            batch = collate_jsonl([sample])
            batch = {key: value.to(device) for key, value in batch.items()}
            model_batch = strip_training_only_batch_keys(batch)
            if donor is not None:
                model_batch.update(
                    prepare_donor_batch(
                        donor,
                        batch,
                        return_logits=use_donor_logits,
                    )
                )
            with eval_autocast_context(device, cfg.train.use_amp):
                _, metrics, _ = qtrm_smoke_loss(
                    model,
                    **model_batch,
                    jepa_weight=0.0,
                    aux_weight=0.0,
                    preference_weight=1.0,
                    preference_beta=preference_beta,
                    preference_margin=preference_margin,
                )
            chosen_logp = float(metrics["preference_chosen_logp"].detach().cpu().item())
            rejected_logp = float(metrics["preference_rejected_logp"].detach().cpu().item())
            margin_logp = chosen_logp - rejected_logp
            record = {
                "id": record_id(row, index),
                "case_id": row.get("case_id"),
                "category": row.get("category"),
                "task_family": row.get("task_family"),
                "mode": row.get("mode"),
                "chosen_logp": chosen_logp,
                "rejected_logp": rejected_logp,
                "margin_logp": margin_logp,
                "chosen_preferred": margin_logp > 0.0,
                "margin_passed": margin_logp >= preference_margin,
                "sample_weight": row_weight(row),
                "preference_loss": float(metrics["preference"].detach().cpu().item()),
            }
            records.append(record)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

        summary = summarize_preference_records(records, target_margin=preference_margin)
        out.write(
            json.dumps(
                {
                    "summary": summary,
                    "config": args.config,
                    "checkpoint": args.checkpoint,
                    "data_jsonl": args.data_jsonl,
                    "preference_beta": preference_beta,
                    "preference_margin": preference_margin,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    print(json.dumps({"wrote": str(out_path), "summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
