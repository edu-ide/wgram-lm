#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def configure_source_pointer_model_from_args(cfg: Any, args: argparse.Namespace) -> None:
    if bool(args.token_numeric_source_slots):
        cfg.model.token_numeric_source_slot_embedding_enabled = True
        cfg.model.token_numeric_source_slot_vocab_size = int(
            args.token_numeric_source_slot_vocab_size
        )
        cfg.model.token_numeric_source_slot_max_slots = int(
            args.token_numeric_source_slot_max_slots
        )
        cfg.model.token_numeric_source_slot_gate_min = float(
            args.token_numeric_source_slot_gate_min
        )
        cfg.model.token_numeric_source_slot_predicate_feedback_enabled = bool(
            args.token_numeric_source_slot_predicate_feedback
        )
        cfg.model.token_numeric_source_slot_predicate_gate_min = float(
            args.token_numeric_source_slot_predicate_gate_min
        )
    if bool(args.core_source_position_binder):
        cfg.model.core_source_position_binder_enabled = True
        cfg.model.core_source_position_binder_gate_min = float(
            args.core_source_position_binder_gate_min
        )
        cfg.model.core_source_position_binder_state_gate_min = float(
            args.core_source_position_binder_state_gate_min
        )
        cfg.model.core_source_position_binder_state_straight_through = bool(
            args.core_source_position_binder_state_st
        )
        cfg.model.core_source_position_binder_source_slots_only = bool(
            args.core_source_position_binder_source_slots_only
        )
        cfg.model.core_source_position_binder_raw_source_slots_enabled = bool(
            args.core_source_position_binder_raw_source_slots
        )


def checkpoint_stack_paths(checkpoint: str | Path, *, load_state=None) -> list[str]:
    if load_state is None:
        import torch

        def load_state(path: Path):
            return torch.load(path, map_location="cpu", weights_only=False)

    paths: list[str] = []
    seen: set[str] = set()
    current = Path(checkpoint)
    while True:
        key = str(current)
        if key in seen:
            raise ValueError(f"checkpoint base chain cycle detected at {current}")
        seen.add(key)
        if not current.exists():
            raise FileNotFoundError(str(current))
        paths.append(str(current))
        state = load_state(current)
        base_checkpoint = ""
        if isinstance(state, dict):
            base_checkpoint = str(state.get("base_checkpoint") or "").strip()
        if not base_checkpoint:
            break
        current = Path(base_checkpoint)
    return paths


def materialize_checkpoint_stack(
    *,
    model: Any,
    checkpoint: str | Path,
    out: str | Path,
    map_location: str = "cpu",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import torch

    from qtrm_mm.training.train import load_initial_checkpoint

    checkpoint_path = Path(checkpoint)
    out_path = Path(out)
    stack = checkpoint_stack_paths(checkpoint_path)
    missing, unexpected = load_initial_checkpoint(
        model,
        str(checkpoint_path),
        map_location=map_location,
    )
    report = {
        "format": "qtrm_self_contained_checkpoint_v1",
        "source_checkpoint": str(checkpoint_path),
        "checkpoint_stack": stack,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "metadata": dict(metadata or {}),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "format": "qtrm_self_contained_checkpoint_v1",
            "materialized_from": str(checkpoint_path),
            "checkpoint_stack": stack,
            "materialization_report": report,
        },
        out_path,
    )
    report["out"] = str(out_path)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a trainable-delta QTRM checkpoint chain into one "
            "self-contained checkpoint with no base_checkpoint dependency."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--report", default="")
    parser.add_argument("--fail-on-unmatched-keys", action="store_true")
    parser.add_argument("--token-numeric-source-slots", action="store_true")
    parser.add_argument("--token-numeric-source-slot-vocab-size", type=int, default=128)
    parser.add_argument("--token-numeric-source-slot-max-slots", type=int, default=5)
    parser.add_argument("--token-numeric-source-slot-gate-min", type=float, default=0.0)
    parser.add_argument("--token-numeric-source-slot-predicate-feedback", action="store_true")
    parser.add_argument(
        "--token-numeric-source-slot-predicate-gate-min",
        type=float,
        default=0.0,
    )
    parser.add_argument("--core-source-position-binder", action="store_true")
    parser.add_argument("--core-source-position-binder-gate-min", type=float, default=0.0)
    parser.add_argument(
        "--core-source-position-binder-state-gate-min",
        type=float,
        default=0.0,
    )
    parser.add_argument("--core-source-position-binder-state-st", action="store_true")
    parser.add_argument(
        "--core-source-position-binder-source-slots-only",
        action="store_true",
    )
    parser.add_argument("--core-source-position-binder-raw-source-slots", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    from qtrm_mm.config import load_config
    from qtrm_mm.qtrm_model import QTRMMultimodalModel

    cfg = load_config(args.config)
    configure_source_pointer_model_from_args(cfg, args)
    model = QTRMMultimodalModel(cfg.model)
    if args.device == "cuda":
        model = model.to("cuda")
    report = materialize_checkpoint_stack(
        model=model,
        checkpoint=args.checkpoint,
        out=args.out,
        map_location=args.device,
        metadata={"config": str(args.config)},
    )
    if bool(args.fail_on_unmatched_keys) and (
        report["missing_keys"] or report["unexpected_keys"]
    ):
        raise RuntimeError(
            "materialized checkpoint has unmatched keys: "
            f"missing={report['missing_keys']} unexpected={report['unexpected_keys']}"
        )
    report_path = Path(args.report) if args.report else Path(args.out).with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    args = build_arg_parser().parse_args()
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
