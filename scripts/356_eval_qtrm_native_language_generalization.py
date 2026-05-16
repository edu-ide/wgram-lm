#!/usr/bin/env python3
"""Evaluate a QTRM-native language checkpoint on heldout instruction prompts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch


def load_bootstrap_module():
    path = Path(__file__).with_name("354_train_qtrm_native_language_bootstrap.py")
    spec = importlib.util.spec_from_file_location("qtrm_native_language_bootstrap", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_bootstrap = load_bootstrap_module()
_text_probe = _bootstrap._text_probe


DEFAULT_GENERALIZATION_SEEDS = (
    "User: Why is checking a source important?\nAssistant: ||"
    "User: How do short sentences help readers?\nAssistant: ||"
    "User: 무엇이 답변을 믿을 만하게 만드나요?\nAssistant: ||"
    "User: How should a model respond when evidence is weak?\nAssistant: "
)


DEFAULT_GENERALIZATION_EXPECTATIONS = {
    "Why is checking a source important?": ["source", "trust", "evidence"],
    "How do short sentences help readers?": ["sentences", "readers", "clear"],
    "무엇이 답변을 믿을 만하게 만드나요?": ["답변", "근거", "출처"],
    "How should a model respond when evidence is weak?": ["evidence", "weak", "guess"],
}


def load_eval_suite_jsonl(path: str | Path) -> tuple[str, str]:
    """Load a fixed broad-unseen language suite.

    Expected JSONL rows:
      {"prompt": "...", "expected_keywords": ["..."]}

    For semantic variants, use one required meaning slot per inner list:
      {"prompt": "...", "expected_keyword_groups": [["date", "time"], ["source"]]}

    A row may also provide `seed_text` directly when the prompt should not use
    the default User/Assistant wrapper.
    """
    seeds: list[str] = []
    expectations: dict[str, list[str]] = {}
    for line_no, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid eval suite JSONL at {path}:{line_no}: {exc}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"eval suite row must be object at {path}:{line_no}")
        prompt = str(item.get("prompt", "")).strip()
        seed_text = str(item.get("seed_text", "")).strip()
        if not seed_text:
            if not prompt:
                raise ValueError(f"eval suite row missing prompt at {path}:{line_no}")
            seed_text = f"User: {prompt}\nAssistant:"
        key = prompt or seed_text
        keyword_groups = item.get("expected_keyword_groups")
        if keyword_groups is not None:
            if not isinstance(keyword_groups, list) or not all(
                isinstance(group, list)
                and all(isinstance(value, str) for value in group)
                for group in keyword_groups
            ):
                raise ValueError(
                    "eval suite row expected_keyword_groups must be list[list[str]] "
                    f"at {path}:{line_no}"
                )
            cleaned_keywords = [
                "|".join(value.strip() for value in group if value.strip())
                for group in keyword_groups
            ]
            cleaned_keywords = [keyword for keyword in cleaned_keywords if keyword]
        else:
            keywords = item.get("expected_keywords")
            if not isinstance(keywords, list) or not all(
                isinstance(value, str) for value in keywords
            ):
                raise ValueError(
                    f"eval suite row expected_keywords must be string list at {path}:{line_no}"
                )
            cleaned_keywords = [keyword.strip() for keyword in keywords if keyword.strip()]
        if not cleaned_keywords:
            raise ValueError(f"eval suite row has no non-empty expected keywords at {path}:{line_no}")
        seeds.append(seed_text)
        expectations[key] = cleaned_keywords
    if not seeds:
        raise ValueError(f"eval suite JSONL is empty: {path}")
    return "||".join(seeds), json.dumps(expectations, ensure_ascii=False)


def merged_checkpoint_args(checkpoint_args: dict[str, object], overrides: argparse.Namespace):
    parser = _bootstrap.build_arg_parser()
    args = parser.parse_args([])
    for key, value in checkpoint_args.items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.device = overrides.device
    args.out_dir = str(overrides.out_dir)
    args.eval_think_steps = int(overrides.eval_think_steps)
    args.max_new_chars = int(overrides.max_new_chars)
    args.repair_prompt_count = int(overrides.repair_prompt_count)
    args.repair_seed_texts = str(overrides.eval_seed_texts)
    args.repair_seed_expectations = str(overrides.eval_seed_expectations)
    args.min_on_policy_continuation_chars = int(overrides.min_on_policy_continuation_chars)
    args.min_on_policy_keyword_hits = int(overrides.min_on_policy_keyword_hits)
    args.min_on_policy_loop_check_lines = int(overrides.min_on_policy_loop_check_lines)
    args.min_on_policy_unique_line_fraction = float(overrides.min_on_policy_unique_line_fraction)
    args.max_on_policy_repeated_block_fraction = float(
        overrides.max_on_policy_repeated_block_fraction
    )
    args.max_on_policy_repeated_line_fraction = float(
        overrides.max_on_policy_repeated_line_fraction
    )
    if str(getattr(overrides, "eval_jsonl", "")):
        seed_texts, expectations = load_eval_suite_jsonl(str(overrides.eval_jsonl))
        args.repair_seed_texts = seed_texts
        args.repair_seed_expectations = expectations
    return args


def tokenizer_from_checkpoint(tokenizer_payload: dict[str, object], args):
    kind = str(tokenizer_payload.get("kind", ""))
    if kind == "byte_bpe":
        return _bootstrap.ByteBPETokenizerAdapter.from_payload(tokenizer_payload)
    if kind == "hf_compact":
        args.tokenizer_name = str(tokenizer_payload.get("name") or args.tokenizer_name)
        return _bootstrap.CompactHFTokenizerAdapter.from_payload(tokenizer_payload)
    if kind == "hf" or args.tokenizer_name:
        name = str(tokenizer_payload.get("name") or args.tokenizer_name)
        args.tokenizer_name = name
        return _bootstrap.HFTokenizerAdapter.from_name(name)
    chars = tuple(str(ch) for ch in tokenizer_payload.get("chars", ()))
    return _text_probe.CharTokenizer(chars=chars, char_to_id={ch: i for i, ch in enumerate(chars)})


@torch.no_grad()
def evaluate_checkpoint(args: argparse.Namespace) -> dict[str, object]:
    checkpoint = torch.load(str(args.checkpoint), map_location="cpu")
    ckpt_args = checkpoint.get("args", {})
    if not isinstance(ckpt_args, dict):
        ckpt_args = {}
    eval_args = merged_checkpoint_args(ckpt_args, args)
    tokenizer = tokenizer_from_checkpoint(checkpoint.get("tokenizer", {}), eval_args)
    device = torch.device(str(args.device))
    model = _text_probe.build_model(eval_args, vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _bootstrap.write_on_policy_candidates(
        model,
        tokenizer,
        eval_args,
        device=device,
        out_dir=out_dir,
    )
    reject_reasons = []
    reject_reasons.extend(_bootstrap.on_policy_loop_reject_reasons(eval_args, rows))
    reject_reasons.extend(_bootstrap.on_policy_answer_reject_reasons(eval_args, rows))
    if rows:
        (out_dir / "on_policy_candidates.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    report = {
        "status": "complete",
        "target_level": "QTRM-native language generalization gate",
        "checkpoint": str(args.checkpoint),
        "decision": "accepted_language_generalization" if not reject_reasons else "rejected",
        "accepted": not reject_reasons,
        "reject_reasons": sorted(set(reject_reasons)),
        "eval": {
            "eval_think_steps": int(eval_args.eval_think_steps),
            "max_new_chars": int(eval_args.max_new_chars),
            "repair_seed_texts": str(eval_args.repair_seed_texts),
            "repair_seed_expectations": str(eval_args.repair_seed_expectations),
            "min_on_policy_continuation_chars": int(
                eval_args.min_on_policy_continuation_chars
            ),
            "min_on_policy_keyword_hits": int(eval_args.min_on_policy_keyword_hits),
        },
        "tokenizer": _bootstrap.tokenizer_report_payload(checkpoint.get("tokenizer", {})),
        "on_policy_candidates": {
            "count": len(rows),
            "path": str(out_dir / "on_policy_candidates.jsonl") if rows else "",
            "line_loop_metrics": [row.get("line_loop_metrics", {}) for row in rows],
            "answer_surface_metrics": [
                row.get("answer_surface_metrics", {}) for row in rows
            ],
            "semantic_relevance_metrics": [
                row.get("semantic_relevance_metrics", {}) for row in rows
            ],
        },
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a QTRM-native language checkpoint on heldout prompts."
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "local_eval/qtrm_native_language_bootstrap_qwen_tokenizer_semantic_eos_"
            "repair_s1200_20260515/last.pt"
        ),
    )
    parser.add_argument("--out-dir", default="local_eval/qtrm_native_language_generalization_gate")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-think-steps", type=int, default=4)
    parser.add_argument("--max-new-chars", type=int, default=180)
    parser.add_argument("--repair-prompt-count", type=int, default=8)
    parser.add_argument("--eval-jsonl", default="")
    parser.add_argument("--eval-seed-texts", default=DEFAULT_GENERALIZATION_SEEDS)
    parser.add_argument(
        "--eval-seed-expectations",
        default=json.dumps(DEFAULT_GENERALIZATION_EXPECTATIONS, ensure_ascii=False),
    )
    parser.add_argument("--min-on-policy-loop-check-lines", type=int, default=4)
    parser.add_argument("--min-on-policy-continuation-chars", type=int, default=16)
    parser.add_argument("--min-on-policy-keyword-hits", type=int, default=2)
    parser.add_argument("--min-on-policy-unique-line-fraction", type=float, default=0.55)
    parser.add_argument("--max-on-policy-repeated-block-fraction", type=float, default=0.24)
    parser.add_argument("--max-on-policy-repeated-line-fraction", type=float, default=0.30)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = evaluate_checkpoint(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if bool(report["accepted"]) else 1)


if __name__ == "__main__":
    main()
