#!/usr/bin/env python3
"""Evaluate Stage101W6 twins with a siamese counterfactual energy rule.

Instead of asking the LM head to directly emit A or B from a combined prompt,
score each world independently with the same answerability question:

  energy(world) = logp(" yes" | world prompt) - logp(" no" | world prompt)

The predicted answerable world is the world with higher energy. This tests
whether the bottleneck is the A/B comparison prompt or the underlying
answerability judgment.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch


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


def load_gd_lite_eval_module() -> Any:
    return load_module(ROOT / "scripts" / "567_eval_blt_generalization_dynamics_probe.py", "stage101w7_gd_lite")


def load_overthinking_module() -> Any:
    return load_module(ROOT / "scripts" / "576_eval_overthinking_noise_probe.py", "stage101w7_overthinking")


def opposite_world(world: str) -> str:
    if world == " A":
        return " B"
    if world == " B":
        return " A"
    raise ValueError(f"bad world {world!r}")


def world_answerability_prompt(pair: dict[str, Any], *, world_key: str) -> str:
    if world_key not in {"world_a", "world_b"}:
        raise ValueError(f"bad world_key {world_key!r}")
    return (
        f"Claim: {pair['source_claim']}\n"
        f"World: {pair[world_key]}\n"
        "Q: Can answer now? yes or no.\n"
        "A:"
    )


def unique_answerable_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        pair_id = str(row.get("twin_pair_id", ""))
        if not pair_id or pair_id in seen:
            continue
        step = str(row.get("stage101w6_chain_step", ""))
        task = str(row.get("task", ""))
        if step != "answerable_world" and not task.endswith("answerable_world_icl"):
            continue
        seen.add(pair_id)
        out.append(row)
    return out


def rows_from_pair_energy(
    pair: dict[str, Any],
    *,
    think_steps: int,
    energy_a: float,
    energy_b: float,
) -> list[dict[str, Any]]:
    predicted_answerable = " A" if float(energy_a) >= float(energy_b) else " B"
    predicted_blocked = opposite_world(predicted_answerable)
    target_answerable = str(pair["answerable_world"])
    target_blocked = str(pair["blocked_world"])
    target_energy = float(energy_a) if target_answerable == " A" else float(energy_b)
    other_energy = float(energy_b) if target_answerable == " A" else float(energy_a)
    margin = float(target_energy - other_energy)
    base = {
        "source": "stage101w7_siamese_counterfactual_energy",
        "think_steps": int(think_steps),
        "energy_a_yes_minus_no": float(energy_a),
        "energy_b_yes_minus_no": float(energy_b),
        "negative_answers": [opposite_world(target_answerable)],
        "negative_mean_logprobs": [other_energy],
        "skipped_reason": None,
    }
    answerable = {
        **base,
        "id": f"{pair['twin_pair_id']}_w7_answerable_world",
        "task": "stage101w7_siamese_answerable_world",
        "target_answer": target_answerable,
        "predicted_answer": predicted_answerable,
        "intelligence_mean_logprob": target_energy,
        "parrot_mean_logprob": other_energy,
        "normalized_margin": margin,
        "correct": bool(predicted_answerable == target_answerable),
    }
    blocked = {
        **base,
        "id": f"{pair['twin_pair_id']}_w7_blocked_world",
        "task": "stage101w7_siamese_blocked_world",
        "target_answer": target_blocked,
        "predicted_answer": predicted_blocked,
        "intelligence_mean_logprob": -other_energy,
        "parrot_mean_logprob": -target_energy,
        "normalized_margin": margin,
        "correct": bool(predicted_blocked == target_blocked),
        "negative_answers": [opposite_world(target_blocked)],
        "negative_mean_logprobs": [-target_energy],
    }
    return [answerable, blocked]


def score_world_energy(
    *,
    gd_lite: Any,
    model: Any,
    prompt: str,
    seq_len: int,
    byte_offset: int,
    device: torch.device,
    think_steps: int,
    amp_context: Any,
) -> dict[str, float]:
    yes = gd_lite.choice_logprob(
        model,
        prompt=prompt,
        answer=" yes",
        seq_len=seq_len,
        byte_offset=byte_offset,
        device=device,
        think_steps=int(think_steps),
        amp_context=amp_context,
    )
    no = gd_lite.choice_logprob(
        model,
        prompt=prompt,
        answer=" no",
        seq_len=seq_len,
        byte_offset=byte_offset,
        device=device,
        think_steps=int(think_steps),
        amp_context=amp_context,
    )
    return {
        "yes_mean_logprob": float(yes["mean_logprob"]),
        "no_mean_logprob": float(no["mean_logprob"]),
        "energy": float(yes["mean_logprob"]) - float(no["mean_logprob"]),
    }


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    gd_lite = load_gd_lite_eval_module()
    overthinking = load_overthinking_module()
    probe_rows = gd_lite.load_jsonl(Path(args.probe_jsonl))
    pairs = unique_answerable_pairs(probe_rows)
    if int(args.max_pairs) > 0:
        pairs = pairs[: int(args.max_pairs)]

    depth_probe = gd_lite.load_depth_probe_module()
    device = torch.device(str(args.device))
    trainer, _prefix, ckpt_args, loaded = depth_probe.load_checkpoint_model(
        checkpoint_path=Path(args.checkpoint),
        sampled_data=str(args.sampled_data),
        out_dir=str(Path(args.out).parent if str(args.out) else "local_eval/stage101w7_siamese_energy"),
        device=device,
        amp_dtype=str(args.amp_dtype),
    )
    model = loaded["model"]
    model.eval()
    amp_dtype = trainer.resolve_amp_dtype(str(args.amp_dtype))

    def make_amp_context() -> Any:
        if str(device.type) != "cuda":
            return nullcontext()
        return trainer.autocast_context(device, amp_dtype)

    tokenizer_info = dict(loaded.get("dataset_summary", {}).get("tokenizer_info") or {})
    byte_offset = int(args.byte_offset if int(args.byte_offset) >= 0 else tokenizer_info.get("byte_offset", 2))
    seq_len = int(args.seq_len or ckpt_args.seq_len)

    rows: list[dict[str, Any]] = []
    for depth in [int(value) for value in args.depths]:
        for pair in pairs:
            try:
                score_a = score_world_energy(
                    gd_lite=gd_lite,
                    model=model,
                    prompt=world_answerability_prompt(pair, world_key="world_a"),
                    seq_len=seq_len,
                    byte_offset=byte_offset,
                    device=device,
                    think_steps=int(depth),
                    amp_context=make_amp_context(),
                )
                score_b = score_world_energy(
                    gd_lite=gd_lite,
                    model=model,
                    prompt=world_answerability_prompt(pair, world_key="world_b"),
                    seq_len=seq_len,
                    byte_offset=byte_offset,
                    device=device,
                    think_steps=int(depth),
                    amp_context=make_amp_context(),
                )
                pair_rows = rows_from_pair_energy(
                    pair,
                    think_steps=int(depth),
                    energy_a=float(score_a["energy"]),
                    energy_b=float(score_b["energy"]),
                )
                for row in pair_rows:
                    row["world_a_yes_mean_logprob"] = float(score_a["yes_mean_logprob"])
                    row["world_a_no_mean_logprob"] = float(score_a["no_mean_logprob"])
                    row["world_b_yes_mean_logprob"] = float(score_b["yes_mean_logprob"])
                    row["world_b_no_mean_logprob"] = float(score_b["no_mean_logprob"])
                rows.extend(pair_rows)
            except Exception as exc:  # pragma: no cover - exercised by CLI failures.
                pair_id = str(pair.get("twin_pair_id", pair.get("id", "unknown")))
                for step in ["answerable_world", "blocked_world"]:
                    rows.append(
                        {
                            "id": f"{pair_id}_w7_{step}",
                            "task": f"stage101w7_siamese_{step}",
                            "source": "stage101w7_siamese_counterfactual_energy",
                            "think_steps": int(depth),
                            "normalized_margin": float("nan"),
                            "correct": False,
                            "skipped_reason": repr(exc),
                        }
                    )

    report = overthinking.build_overthinking_noise_report(
        rows=rows,
        depths=[int(value) for value in args.depths],
        checkpoint=str(args.checkpoint),
        probe_jsonl=str(args.probe_jsonl),
    )
    report.update(
        {
            "probe_type": "stage101w7_siamese_counterfactual_energy",
            "pair_count": len(pairs),
            "rows": rows,
            "plain_language_read": (
                "W7 evaluates each counterfactual world separately with the same "
                "yes/no judge, then chooses the world with higher yes-minus-no energy."
            ),
        }
    )
    if str(args.out):
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--probe-jsonl", default="data/eval/stage101w6_counterfactual_twin_heldout_probe.jsonl")
    parser.add_argument("--depths", nargs="+", type=int, default=[2, 4, 8, 16])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-dtype", default="bf16")
    parser.add_argument("--sampled-data", default="")
    parser.add_argument("--seq-len", type=int, default=0)
    parser.add_argument("--byte-offset", type=int, default=-1)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--out", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = run_eval(args)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
