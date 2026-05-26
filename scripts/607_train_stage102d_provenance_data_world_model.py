#!/usr/bin/env python3
"""Train Stage102D provenance data-world model without answer labels.

This script learns a small energy model over provenance states:

  clean source/trust/support graph -> low energy
  corrupted graph                  -> high energy

It deliberately does not read yes/no answer fields.  The labels are generated
from self-supervised corruptions, so the model learns a "data sense" before it
is used as an answer-facing register.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STAGE102B = load_module(
    ROOT / "scripts" / "605_train_stage102b_provenance_graph_reasoner.py",
    "stage102d_stage102b_provenance_graph_reasoner",
)

CORRUPTIONS = ("source_id_conflict", "trust_edge_conflict", "support_conflict")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"no rows in JSONL: {path}")
    return rows


def verified_source_id(row: dict[str, Any]) -> str:
    explicit = row.get("verified_source")
    if explicit is not None:
        return str(explicit).upper()
    verified = STAGE102B.ledger_verified_map(row)
    for source_id, value in sorted(verified.items()):
        if float(value) >= 0.5:
            return str(source_id).upper()
    source_ids = STAGE102B.row_source_ids(row)
    return str(source_ids[0]).upper()


def clean_world_features(row: dict[str, Any], side: str) -> dict[str, Any]:
    base = STAGE102B.build_graph_features(row, side)
    source_ids = STAGE102B.row_source_ids(row)
    verified_id = verified_source_id(row)
    verified_index = int(source_ids.index(verified_id) if verified_id in source_ids else 0)
    return {
        "id": row.get("id"),
        "side": str(side),
        "corruption": "clean",
        "context_source_index": int(base["source_index"]),
        "context_verified_source_index": int(verified_index),
        "expected_source_verified": float(base["source_verified"]),
        "expected_claim_supported": float(base["claim_supported"]),
        "source_index": int(base["source_index"]),
        "verified_source_index": int(verified_index),
        "observed_source_verified": float(base["source_verified"]),
        "claim_supported": float(base["claim_supported"]),
        "is_clean": 1.0,
    }


def build_world_model_examples(row: dict[str, Any], side: str) -> list[dict[str, Any]]:
    clean = clean_world_features(row, side)
    source_ids = STAGE102B.row_source_ids(row)
    source_count = max(2, len(source_ids))

    source_conflict = dict(clean)
    source_conflict["corruption"] = "source_id_conflict"
    source_conflict["source_index"] = (int(clean["source_index"]) + 1) % source_count
    source_conflict["is_clean"] = 0.0

    trust_conflict = dict(clean)
    trust_conflict["corruption"] = "trust_edge_conflict"
    trust_conflict["observed_source_verified"] = 1.0 - float(clean["observed_source_verified"])
    trust_conflict["is_clean"] = 0.0

    support_conflict = dict(clean)
    support_conflict["corruption"] = "support_conflict"
    support_conflict["claim_supported"] = 1.0 - float(clean["claim_supported"])
    support_conflict["is_clean"] = 0.0

    return [clean, source_conflict, trust_conflict, support_conflict]


def build_world_model_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for row in rows:
        for side in ("original", "counterfactual"):
            prompt = str(row.get(f"{side}_prompt", ""))
            source = row.get(f"{side}_source")
            if not prompt or source is None:
                continue
            examples = build_world_model_examples(row, side)
            clean = examples[0]
            for corrupt in examples[1:]:
                pairs.append({"clean": clean, "corrupt": corrupt, "corruption": corrupt["corruption"]})
    if not pairs:
        raise ValueError("no world-model pairs were built")
    return pairs


class ProvenanceDataWorldModel(nn.Module):
    """Small energy model for label-free provenance consistency."""

    def __init__(self, d_model: int = 32, max_sources: int = 16, hidden_dim: int | None = None) -> None:
        super().__init__()
        width = int(hidden_dim or max(32, d_model * 2))
        self.source_embedding = nn.Embedding(int(max_sources), int(d_model))
        self.verified_source_embedding = nn.Embedding(int(max_sources), int(d_model))
        self.context_source_embedding = nn.Embedding(int(max_sources), int(d_model))
        self.context_verified_source_embedding = nn.Embedding(int(max_sources), int(d_model))
        self.scalar_proj = nn.Linear(4, int(d_model))
        self.encoder = nn.Sequential(
            nn.LayerNorm(int(d_model)),
            nn.Linear(int(d_model), width),
            nn.SiLU(),
            nn.Linear(width, int(d_model)),
            nn.LayerNorm(int(d_model)),
        )
        self.energy_head = nn.Sequential(
            nn.Linear(int(d_model), width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )

    def forward(
        self,
        examples: list[dict[str, Any]],
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not examples:
            raise ValueError("examples must not be empty")
        source_index = torch.tensor(
            [int(item["source_index"]) for item in examples],
            dtype=torch.long,
            device=device,
        ).clamp(0, self.source_embedding.num_embeddings - 1)
        verified_source_index = torch.tensor(
            [int(item["verified_source_index"]) for item in examples],
            dtype=torch.long,
            device=device,
        ).clamp(0, self.verified_source_embedding.num_embeddings - 1)
        context_source_index = torch.tensor(
            [int(item.get("context_source_index", item["source_index"])) for item in examples],
            dtype=torch.long,
            device=device,
        ).clamp(0, self.context_source_embedding.num_embeddings - 1)
        context_verified_source_index = torch.tensor(
            [
                int(item.get("context_verified_source_index", item["verified_source_index"]))
                for item in examples
            ],
            dtype=torch.long,
            device=device,
        ).clamp(0, self.context_verified_source_embedding.num_embeddings - 1)
        scalars = torch.tensor(
            [
                [
                    float(item["observed_source_verified"]),
                    float(item["claim_supported"]),
                    float(item.get("expected_source_verified", item["observed_source_verified"])),
                    float(item.get("expected_claim_supported", item["claim_supported"])),
                ]
                for item in examples
            ],
            dtype=torch.float32,
            device=device,
        )
        state = (
            self.source_embedding(source_index)
            + self.verified_source_embedding(verified_source_index)
            + self.context_source_embedding(context_source_index)
            + self.context_verified_source_embedding(context_verified_source_index)
            + self.scalar_proj(scalars)
        )
        latent = self.encoder(state)
        energy = F.softplus(self.energy_head(latent)).squeeze(-1)
        return energy, latent


def latent_spread_loss(latent: torch.Tensor, *, min_std: float) -> torch.Tensor:
    if latent.shape[0] <= 1:
        return latent.float().sum() * 0.0
    std = latent.float().std(dim=0, unbiased=False)
    return F.relu(float(min_std) - std).mean()


def energy_separation_loss(
    model: ProvenanceDataWorldModel,
    pairs: list[dict[str, Any]],
    *,
    device: torch.device,
    margin: float,
    clean_weight: float,
    spread_weight: float,
    min_latent_std: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    clean_examples = [pair["clean"] for pair in pairs]
    corrupt_examples = [pair["corrupt"] for pair in pairs]
    clean_energy, clean_latent = model(clean_examples, device=device)
    corrupt_energy, corrupt_latent = model(corrupt_examples, device=device)
    gap = corrupt_energy.float() - clean_energy.float()
    rank_loss = F.softplus(float(margin) - gap).mean()
    clean_loss = clean_energy.float().mean()
    spread_loss = latent_spread_loss(
        torch.cat([clean_latent, corrupt_latent], dim=0),
        min_std=float(min_latent_std),
    )
    loss = rank_loss + float(clean_weight) * clean_loss + float(spread_weight) * spread_loss
    metrics = {
        "loss": float(loss.detach().cpu().item()),
        "rank_loss": float(rank_loss.detach().cpu().item()),
        "clean_loss": float(clean_loss.detach().cpu().item()),
        "spread_loss": float(spread_loss.detach().cpu().item()),
        "pair_accuracy": float((gap.detach() > 0.0).float().mean().cpu().item()),
        "min_energy_gap": float(gap.detach().float().min().cpu().item()),
        "mean_energy_gap": float(gap.detach().float().mean().cpu().item()),
        "clean_mean_energy": float(clean_energy.detach().float().mean().cpu().item()),
        "corrupt_mean_energy": float(corrupt_energy.detach().float().mean().cpu().item()),
    }
    return loss, metrics


def batch_pairs_for_step(
    pairs: list[dict[str, Any]],
    *,
    step: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    if not pairs:
        raise ValueError("pairs must not be empty")
    start = (int(step) - 1) % len(pairs)
    size = max(1, int(batch_size))
    return [pairs[(start + offset) % len(pairs)] for offset in range(size)]


@torch.no_grad()
def evaluate_pairs(
    model: ProvenanceDataWorldModel,
    pairs: list[dict[str, Any]],
    *,
    split: str,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    all_gaps: list[float] = []
    all_clean: list[float] = []
    all_corrupt: list[float] = []
    by_corruption: dict[str, list[float]] = {}
    for start in range(0, len(pairs), max(1, int(batch_size))):
        batch = pairs[start : start + max(1, int(batch_size))]
        clean_examples = [pair["clean"] for pair in batch]
        corrupt_examples = [pair["corrupt"] for pair in batch]
        clean_energy, _clean_latent = model(clean_examples, device=device)
        corrupt_energy, _corrupt_latent = model(corrupt_examples, device=device)
        gaps = (corrupt_energy.float() - clean_energy.float()).detach().cpu().tolist()
        clean_values = clean_energy.float().detach().cpu().tolist()
        corrupt_values = corrupt_energy.float().detach().cpu().tolist()
        all_gaps.extend(float(item) for item in gaps)
        all_clean.extend(float(item) for item in clean_values)
        all_corrupt.extend(float(item) for item in corrupt_values)
        for pair, gap in zip(batch, gaps, strict=True):
            by_corruption.setdefault(str(pair["corruption"]), []).append(float(gap))
    pair_accuracy = sum(1 for gap in all_gaps if gap > 0.0) / float(len(all_gaps))
    corruption_summary = {
        key: {
            "rows": len(values),
            "pair_accuracy": sum(1 for value in values if value > 0.0) / float(len(values)),
            "min_energy_gap": float(min(values)),
            "mean_energy_gap": float(sum(values) / float(len(values))),
        }
        for key, values in sorted(by_corruption.items())
    }
    return {
        "split": str(split),
        "pairs": int(len(pairs)),
        "pair_accuracy": float(pair_accuracy),
        "min_energy_gap": float(min(all_gaps)),
        "mean_energy_gap": float(sum(all_gaps) / float(len(all_gaps))),
        "clean_mean_energy": float(sum(all_clean) / float(len(all_clean))),
        "corrupt_mean_energy": float(sum(all_corrupt) / float(len(all_corrupt))),
        "corruption_summary": corruption_summary,
        "accepted": bool(pair_accuracy == 1.0 and min(all_gaps) > 0.0),
    }


def save_checkpoint(
    path: Path,
    *,
    model: ProvenanceDataWorldModel,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    history: list[dict[str, Any]],
    eval_before: dict[str, Any],
    eval_after: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage102d_provenance_data_world_model": True,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args).copy(),
        "loss_history": history,
        "eval_before": eval_before,
        "eval_after": eval_after,
    }
    tmp = path.with_name(f".{path.name}.tmp.{int(time.time())}")
    try:
        torch.save(payload, tmp)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def run_train(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(str(args.device))
    train_rows = load_jsonl(Path(args.train_jsonl))
    eval_rows = load_jsonl(Path(args.eval_jsonl))
    if int(args.max_train_rows) > 0:
        train_rows = train_rows[: int(args.max_train_rows)]
    if int(args.max_eval_rows) > 0:
        eval_rows = eval_rows[: int(args.max_eval_rows)]
    train_pairs = build_world_model_pairs(train_rows)
    eval_pairs = build_world_model_pairs(eval_rows)
    model = ProvenanceDataWorldModel(
        d_model=int(args.d_model),
        max_sources=int(args.max_sources),
        hidden_dim=int(args.hidden_dim or max(32, int(args.d_model) * 2)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    eval_before = {
        "train": evaluate_pairs(
            model,
            train_pairs,
            split="train",
            device=device,
            batch_size=int(args.eval_batch_size),
        ),
        "heldout": evaluate_pairs(
            model,
            eval_pairs,
            split="heldout",
            device=device,
            batch_size=int(args.eval_batch_size),
        ),
    }

    history: list[dict[str, Any]] = []
    for step in range(1, int(args.steps) + 1):
        batch = batch_pairs_for_step(train_pairs, step=step, batch_size=int(args.batch_size))
        optimizer.zero_grad(set_to_none=True)
        model.train()
        loss, metrics = energy_separation_loss(
            model,
            batch,
            device=device,
            margin=float(args.margin),
            clean_weight=float(args.clean_weight),
            spread_weight=float(args.spread_weight),
            min_latent_std=float(args.min_latent_std),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
        optimizer.step()
        if step == 1 or step % int(args.log_every) == 0:
            metrics["step"] = int(step)
            print(json.dumps(metrics, ensure_ascii=False), flush=True)
            history.append(metrics)

    eval_after = {
        "train": evaluate_pairs(
            model,
            train_pairs,
            split="train",
            device=device,
            batch_size=int(args.eval_batch_size),
        ),
        "heldout": evaluate_pairs(
            model,
            eval_pairs,
            split="heldout",
            device=device,
            batch_size=int(args.eval_batch_size),
        ),
    }
    accepted = bool(eval_after["heldout"]["accepted"])
    report = {
        "decision": "stage102d_provenance_data_world_model",
        "accepted": accepted,
        "train_jsonl": str(args.train_jsonl),
        "eval_jsonl": str(args.eval_jsonl),
        "checkpoint_out": str(out_dir / "last_world_model.pt"),
        "train_pairs": int(len(train_pairs)),
        "eval_pairs": int(len(eval_pairs)),
        "steps": int(args.steps),
        "eval_before": eval_before,
        "eval_after": eval_after,
        "plain_language_read": (
            "Stage102D learns whether a provenance state smells clean or broken "
            "without reading final yes/no answer labels."
        ),
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    save_checkpoint(
        out_dir / "last_world_model.pt",
        model=model,
        optimizer=optimizer,
        args=args,
        history=history,
        eval_before=eval_before,
        eval_after=eval_after,
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", default="data/eval/stage102c_randomized_trust_ledger_train_probe.jsonl")
    parser.add_argument("--eval-jsonl", default="data/eval/stage102c_randomized_trust_ledger_heldout_probe.jsonl")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--clean-weight", type=float, default=0.1)
    parser.add_argument("--spread-weight", type=float, default=0.02)
    parser.add_argument("--min-latent-std", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument("--max-sources", type=int, default=16)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-eval-rows", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=40)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    report = run_train(build_arg_parser().parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
