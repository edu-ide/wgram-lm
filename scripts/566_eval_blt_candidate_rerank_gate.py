#!/usr/bin/env python3
"""Deprecated diagnostic for BLT PrefixLM candidate coverage/reranking.

Active evaluation policy as of 2026-05-31 is free-generation-only. This file is
kept so old reports remain readable, but it must not be used as a promotion
gate. Candidate coverage, oracle pass@K, self-consistency, and verifier-selected
answers are historical diagnostics only.

Old behavior, now forbidden for promotion, was:

* baseline = first generated candidate, no oracle help;
* oracle = whether any candidate contains the gold answer, an upper bound;
* self_consistency = majority answer among generated candidates;
* micro_math_verifier = deterministic verifier for the synthetic verified
  micro-math curriculum used in local 82M experiments.

The old micro verifier was intentionally narrow. Treat all pass@K/oracle/rerank
fields from historical reports as audit-only, never as model-quality scores.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import re
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_generation_gate_module() -> Any:
    path = ROOT / "scripts" / "565_eval_blt_generation_gate.py"
    spec = importlib.util.spec_from_file_location("blt_generation_gate_for_rerank", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_answer_interface_module() -> Any:
    path = ROOT / "src" / "wgram_lm" / "eval" / "general_answer_interface.py"
    spec = importlib.util.spec_from_file_location("general_answer_interface_for_rerank", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def solve_micro_math_instruction(text: str) -> str | None:
    """Return the answer for the local verified-math micro curriculum.

    This is deliberately template-bound.  It must not be mistaken for a general
    math solver or a proof of raw intelligence.
    """
    prompt = str(text)
    prompt = prompt.replace("<|im_start|>", " ").replace("<|im_end|>", " ")
    prompt = prompt.replace("<|object_ref_start|>", " ").replace("<|object_ref_end|>", " ")
    prompt = re.sub(r"\s+", " ", prompt).strip()

    match = re.search(r"What is (-?\d+) \+ (-?\d+)\?", prompt)
    if match:
        return str(int(match.group(1)) + int(match.group(2)))

    match = re.search(r"What is (-?\d+) - (-?\d+)\?", prompt)
    if match:
        return str(int(match.group(1)) - int(match.group(2)))

    match = re.search(r"What is (-?\d+) times (-?\d+)\?", prompt)
    if match:
        return str(int(match.group(1)) * int(match.group(2)))

    match = re.search(r"What is (-?\d+)/(-?\d+) \+ (-?\d+)/(-?\d+)\?", prompt)
    if match:
        left = Fraction(int(match.group(1)), int(match.group(2)))
        right = Fraction(int(match.group(3)), int(match.group(4)))
        return _format_fraction(left + right)

    match = re.search(r"Solve for x:\s*(-?\d+)x\s*([+-])\s*(\d+)\s*=\s*(-?\d+)", prompt)
    if match:
        coeff = int(match.group(1))
        bias = int(match.group(3)) * (1 if match.group(2) == "+" else -1)
        rhs = int(match.group(4))
        if coeff == 0:
            return None
        return _format_fraction(Fraction(rhs - bias, coeff))

    match = re.search(r"least common multiple of (\d+) and (\d+)", prompt, re.IGNORECASE)
    if match:
        a = int(match.group(1))
        b = int(match.group(2))
        return str(abs(a * b) // math.gcd(a, b))

    match = re.search(r"greatest common divisor of (\d+) and (\d+)", prompt, re.IGNORECASE)
    if match:
        return str(math.gcd(int(match.group(1)), int(match.group(2))))

    match = re.search(r"Compute binom\((\d+),\s*(\d+)\)", prompt)
    if match:
        return str(math.comb(int(match.group(1)), int(match.group(2))))

    match = re.search(
        r"A box has (\d+) bags with (\d+) marbles each, plus (\d+) extra marbles",
        prompt,
        re.IGNORECASE,
    )
    if match:
        return str(int(match.group(1)) * int(match.group(2)) + int(match.group(3)))

    return None


def normalized_answer(answer_interface: Any, text: Any) -> str:
    value = str(text)
    value = value.replace("<|box_end|>", " ").replace("<|im_end|>", " ")
    value = re.sub(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", r"\1/\2", value)
    return str(answer_interface.normalize_answer_text(value))


def choose_self_consistency(candidates: Sequence[str], answer_interface: Any) -> tuple[int, str]:
    normalized = [normalized_answer(answer_interface, candidate) for candidate in candidates]
    if not normalized:
        return 0, ""
    counts = Counter(normalized)
    best_value, best_count = counts.most_common(1)[0]
    for index, value in enumerate(normalized):
        if value == best_value:
            return int(index), str(best_value)
    return 0, str(best_value)


def _candidate_seeds(seed: int, row_index: int, candidate_count: int) -> list[int]:
    rng = random.Random((int(seed) * 1000003) ^ int(row_index))
    return [rng.randrange(0, 2**31 - 1) for _ in range(int(candidate_count))]


def evaluate_candidates(args: argparse.Namespace) -> dict[str, Any]:
    gate = load_generation_gate_module()
    answer_interface = load_answer_interface_module()
    depth_probe = gate.load_depth_probe_module()

    checkpoint_path = Path(args.checkpoint)
    sampled_data = str(args.sampled_data or gate.load_checkpoint_sampled_data(checkpoint_path))
    if not sampled_data:
        raise ValueError("--sampled-data is required when checkpoint args do not contain sampled_data")

    device = torch.device(str(args.device))
    _trainer, prefix, ckpt_args, loaded = depth_probe.load_checkpoint_model(
        checkpoint_path=checkpoint_path,
        sampled_data=sampled_data,
        out_dir=str(Path(args.out).parent if str(args.out) else "local_eval/blt_candidate_rerank_gate"),
        device=device,
        amp_dtype=str(args.amp_dtype),
    )
    model = loaded["model"]
    dataset = prefix.DataIOSampledPrefixLMDataset(
        sampled_data,
        seq_len=int(args.seq_len or ckpt_args.seq_len),
        epoch=int(args.epoch),
        target_only=True,
        max_rows=None,
        drop_overlength=True,
    )
    metadata = prefix.load_prefixlm_metadata(Path(sampled_data))
    tokenizer_info = dict(metadata.tokenizer_info or {})
    tokenizer = gate.load_optional_tokenizer(tokenizer_info)
    eoa_id = gate.resolve_eoa_id(tokenizer_info, tokenizer)
    think_steps = int(args.think_steps) if int(args.think_steps) > 0 else int(ckpt_args.train_think_steps)

    rows = min(int(args.max_rows), len(dataset))
    records: list[dict[str, Any]] = []
    baseline_hits = 0
    oracle_hits = 0
    self_consistency_hits = 0
    micro_verifier_hits = 0
    micro_verifier_covered = 0
    ended_with_eoa = 0
    repeated_token_loops = 0

    for index in range(rows):
        source_row = int(dataset.row_indices[int(index)])
        inst = dataset._slice_tokens(dataset.inst_start[source_row], dataset.inst_len[source_row])
        resp = dataset._slice_tokens(dataset.resp_start[source_row], dataset.resp_len[source_row])
        gold_ids = gate.gold_response_until_eoa(resp, int(eoa_id))
        instruction = gate.decode_ids(
            tokenizer,
            [int(token_id) for token_id in inst.astype(np.int64).tolist()],
            tokenizer_info,
        )
        gold = gate.decode_ids(tokenizer, gold_ids, tokenizer_info)
        gold_norm = normalized_answer(answer_interface, gold)
        micro_answer = solve_micro_math_instruction(instruction)
        micro_norm = normalized_answer(answer_interface, micro_answer) if micro_answer is not None else ""

        candidates: list[str] = []
        candidate_norms: list[str] = []
        candidate_ids: list[list[int]] = []
        for candidate_index, candidate_seed in enumerate(
            _candidate_seeds(int(args.seed), source_row, int(args.candidates))
        ):
            torch.manual_seed(int(candidate_seed))
            if device.type == "cuda":
                torch.cuda.manual_seed_all(int(candidate_seed))
            generated = gate.generate_one(
                model=model,
                prefix_ids=[int(token_id) for token_id in inst.astype(np.int64).tolist()],
                eoa_id=int(eoa_id),
                device=device,
                think_steps=int(think_steps),
                seq_len=int(args.seq_len or ckpt_args.seq_len),
                max_new_tokens=int(args.max_new_tokens),
                decode=str(args.generation_decode),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                repetition_penalty=float(args.repetition_penalty),
                repetition_window=int(args.repetition_window),
                frequency_penalty=float(args.frequency_penalty),
                no_repeat_ngram_size=int(args.no_repeat_ngram_size),
            )
            text = gate.decode_ids(tokenizer, generated, tokenizer_info)
            candidates.append(text)
            candidate_norms.append(normalized_answer(answer_interface, text))
            candidate_ids.append([int(token_id) for token_id in generated])
            if generated and generated[-1] == int(eoa_id):
                ended_with_eoa += 1
            if len(generated) >= 4 and Counter(generated).most_common(1)[0][1] / float(len(generated)) >= 0.8:
                repeated_token_loops += 1

        baseline_exact = bool(candidate_norms and candidate_norms[0] == gold_norm)
        oracle_index = next((i for i, value in enumerate(candidate_norms) if value == gold_norm), None)
        self_index, self_norm = choose_self_consistency(candidates, answer_interface)
        self_exact = bool(self_norm == gold_norm)
        micro_index = (
            next((i for i, value in enumerate(candidate_norms) if value == micro_norm), None)
            if micro_norm
            else None
        )
        micro_exact = bool(micro_index is not None and micro_norm == gold_norm)

        baseline_hits += int(baseline_exact)
        oracle_hits += int(oracle_index is not None)
        self_consistency_hits += int(self_exact)
        if micro_norm:
            micro_verifier_covered += 1
            micro_verifier_hits += int(micro_exact)

        if len(records) < int(args.keep_records):
            records.append(
                {
                    "row_index": int(source_row),
                    "instruction": instruction,
                    "gold": gold,
                    "gold_normalized": gold_norm,
                    "micro_math_answer": micro_answer,
                    "micro_math_normalized": micro_norm,
                    "candidate_count": int(len(candidates)),
                    "candidate_normalized": candidate_norms,
                    "candidates": candidates,
                    "baseline_exact": baseline_exact,
                    "oracle_exact": bool(oracle_index is not None),
                    "oracle_index": oracle_index,
                    "self_consistency_exact": self_exact,
                    "self_consistency_index": int(self_index),
                    "micro_math_verifier_exact": micro_exact,
                    "micro_math_verifier_index": micro_index,
                    "generated_ids": candidate_ids,
                }
            )

    candidate_total = rows * max(1, int(args.candidates))
    return {
        "gate_type": "blt_candidate_rerank_gate",
        "checkpoint": str(checkpoint_path),
        "sampled_data": str(sampled_data),
        "epoch": int(args.epoch),
        "rows": int(rows),
        "candidates_per_row": int(args.candidates),
        "decode": str(args.generation_decode),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "think_steps": int(think_steps),
        "baseline_accuracy": float(baseline_hits / max(1, rows)),
        "oracle_pass_at_k": float(oracle_hits / max(1, rows)),
        "self_consistency_accuracy": float(self_consistency_hits / max(1, rows)),
        "micro_math_verifier_accuracy": float(micro_verifier_hits / max(1, micro_verifier_covered)),
        "micro_math_verifier_coverage": float(micro_verifier_covered / max(1, rows)),
        "micro_math_verifier_covered_rows": int(micro_verifier_covered),
        "ended_with_eoa_fraction": float(ended_with_eoa / max(1, candidate_total)),
        "repeated_token_loop_fraction": float(repeated_token_loops / max(1, candidate_total)),
        "records": records,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sampled-data", default="")
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp-dtype", default="bf16")
    parser.add_argument("--seq-len", type=int, default=0)
    parser.add_argument("--think-steps", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=64)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--generation-decode", choices=("greedy", "sample"), default="sample")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--repetition-window", type=int, default=64)
    parser.add_argument("--frequency-penalty", type=float, default=0.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=260531)
    parser.add_argument("--keep-records", type=int, default=16)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    raise SystemExit(
        "scripts/566_eval_blt_candidate_rerank_gate.py is disabled for active "
        "evaluation. Use scripts/565_eval_blt_generation_gate.py; only free "
        "autoregressive generation reports may be used for promotion."
    )
    report = evaluate_candidates(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
