#!/usr/bin/env python3
"""Materialize candidate choices exposed by a trained typed-pool selector.

Deprecated diagnostic scaffold: it materializes choices from a selector trained
over a hand-built typed pool. Final-only Stage59 runs must use a learned typed
working table directly in the answer path instead.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from wgram_lm.eval.general_answer_interface import (
    answer_aliases,
    normalize_answer_text,
    normalized_alias_set,
)


def _load_script(name: str, filename: str) -> Any:
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load helper script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[str(spec.name)] = module
    spec.loader.exec_module(module)
    return module


stage523 = _load_script("qtrm_stage523_for_529", "523_train_state_text_speaker.py")
stage528 = _load_script("qtrm_stage528_for_529", "528_train_candidate_pool_selector.py")


def target_choice_index(row: dict[str, Any]) -> int:
    aliases = set(normalized_alias_set(answer_aliases(row)))
    for index, choice in enumerate(row.get("choices") or []):
        if normalize_answer_text(choice) in aliases:
            return int(index)
    return -1


def collate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return rows


@torch.no_grad()
def materialize(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    payload = torch.load(str(args.selector_checkpoint), map_location=device)
    selector_args = payload.get("args") or {}
    rows = stage523.load_jsonl(args.input_jsonl, limit=int(args.limit))

    # Use the selector checkpoint's model contract unless explicitly overridden.
    for name, default in selector_args.items():
        if not hasattr(args, name):
            setattr(args, name, default)
    args.max_pool_candidates = int(args.max_pool_candidates or selector_args.get("max_pool_candidates", 16))
    args.max_candidate_chars = int(args.max_candidate_chars or selector_args.get("max_candidate_chars", 24))
    args.max_candidates = int(args.max_candidates or selector_args.get("max_candidates", 4))
    args.hidden_dim = int(args.hidden_dim or selector_args.get("hidden_dim", 0))

    wgram_model, tokenizer, load_stats = stage523.build_qtrm(args, device)
    allowed_chars = list(payload["allowed_chars"])
    selector = stage528.CandidatePoolSelector(
        d_state=int(wgram_model.d_state),
        vocab_size=len(allowed_chars),
        max_chars=int(args.max_candidate_chars),
        hidden_dim=int(args.hidden_dim) if int(args.hidden_dim) > 0 else None,
    ).to(device)
    selector.load_state_dict(payload["selector"], strict=True)
    selector.eval()
    wgram_model.eval()

    out_rows: list[dict[str, Any]] = []
    pool_oracle_hits = 0
    exposed_oracle_hits = 0
    usable = 0
    loader = DataLoader(rows, batch_size=int(args.batch_size), shuffle=False, collate_fn=collate_rows)
    for batch in loader:
        pools = [stage528.candidate_pool(row, max_pool_candidates=args.max_pool_candidates) for row in batch]
        context = stage523.thought_context_for_batch(
            wgram_model,
            tokenizer,
            batch,
            max_length=int(args.max_length),
            n_steps=int(args.n_steps),
            device=device,
            detach=True,
        )
        pool_ids, pool_mask = stage528.encode_string_table(
            pools,
            allowed_chars=allowed_chars,
            max_items=int(args.max_pool_candidates),
            max_chars=int(args.max_candidate_chars),
            device=device,
        )
        scores = selector(context["readout"], pool_ids, pool_mask)
        exposed = stage528.topk_candidates(scores, pools, k=int(args.max_candidates))
        for row, pool, choices in zip(batch, pools, exposed):
            aliases = set(normalized_alias_set(answer_aliases(row)))
            pool_has_gold = any(normalize_answer_text(candidate) in aliases for candidate in pool)
            exposed_has_gold = any(normalize_answer_text(candidate) in aliases for candidate in choices)
            out = dict(row)
            out["choices"] = choices
            out["candidate_source"] = "typed_pool_selector_exposed"
            out["pool_oracle_exact"] = bool(pool_has_gold)
            out["candidate_oracle_exact"] = bool(exposed_has_gold)
            out["candidate_target_index"] = target_choice_index(out)
            pool_oracle_hits += int(pool_has_gold)
            exposed_oracle_hits += int(exposed_has_gold)
            usable += int(out["candidate_target_index"] >= 0)
            out_rows.append(out)

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in out_rows),
        encoding="utf-8",
    )
    return {
        "input_jsonl": args.input_jsonl,
        "output_jsonl": args.output_jsonl,
        "selector_checkpoint": args.selector_checkpoint,
        "rows": len(out_rows),
        "pool_oracle_coverage": pool_oracle_hits / max(1, len(out_rows)),
        "exposed_oracle_coverage": exposed_oracle_hits / max(1, len(out_rows)),
        "usable_target_rows": usable,
        "load_stats": load_stats,
        "plain_language_read": (
            "This freezes the selector's answer table so the verifier can train on the exact distribution it will see."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-checkpoint", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--max-pool-candidates", type=int, default=0)
    parser.add_argument("--max-candidate-chars", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument("--checkpoint", default="/mnt/sdc1/tripleyoung/qtrm_eval/20260522_092733_LOCAL_STAGE54B_oracle_guard_mixedall_seed42/best_stochastic_oracle.pt")
    parser.add_argument("--qwen-model-id", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--core-impl", default="state_transition")
    parser.add_argument("--core-update", default="mlp")
    parser.add_argument("--answer-path", choices=("state_head", "lm_head"), default="lm_head")
    parser.add_argument("--workspace-pooling", default="sequence")
    parser.add_argument("--recurrent-readout-pooling", default="sharp_attention")
    parser.add_argument("--recurrent-readout-temperature", type=float, default=0.25)
    parser.add_argument("--n-steps", type=int, default=14)
    parser.add_argument("--state-update-schedule", default="nested")
    parser.add_argument("--stochastic-high-level-guidance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stochastic-high-level-scale", type=float, default=1.0)
    parser.add_argument("--stochastic-high-level-min-std", type=float, default=1e-4)
    parser.add_argument("--stochastic-high-level-max-std", type=float, default=1.0)
    parser.add_argument("--stochastic-high-level-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stochastic-posterior-guidance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stochastic-transition-mode", choices=("delta", "true_gram"), default="true_gram")
    parser.add_argument(
        "--allow-diagnostic-scaffold",
        action="store_true",
        help="Required because this script materializes choices from a hand-built heuristic-pool path.",
    )
    args = parser.parse_args()
    if not args.allow_diagnostic_scaffold:
        raise SystemExit(
            "scripts/529 is a deprecated diagnostic scaffold. "
            "Use --allow-diagnostic-scaffold only for audit/reproduction, not final-path experiments."
        )
    print(json.dumps(materialize(args), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
