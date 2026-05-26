#!/usr/bin/env python3
"""Evaluate Stage102G free-form provenance front-end.

Stage102F removed row-field dependency for the fixed Stage102C prompt template.
Stage102G checks the next boundary: can the same provenance cards be recovered
from paraphrased/free-form prompt text?

This remains a deterministic front-end gate, not a learned universal reader.
Its purpose is to make the data contract explicit before training a learned
text-to-provenance model.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any


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


STAGE102F = load_module(
    ROOT / "scripts" / "609_eval_stage102f_prompt_provenance_frontend.py",
    "stage102g_stage102f_prompt_provenance_frontend",
)
STAGE102B = STAGE102F.STAGE102B
STAGE102D = STAGE102F.STAGE102D


TRUST_PATTERNS = (
    re.compile(r"\b(S\d+)\s*=\s*(verified|unverified)\b", re.IGNORECASE),
    re.compile(r"\b(S\d+)\s+is\s+(verified|unverified)\b", re.IGNORECASE),
)
CLAIM_PATTERNS = (
    re.compile(r"^Claim:\s*(?P<claim>.+?)\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(
        r"claim\s+(?:under review|being checked)\s+is:\s*(?P<claim>.+?)\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(
        r"claim\s+being\s+checked:\s*(?P<claim>.+?)\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
)
SOURCE_PATTERNS = (
    re.compile(r"^Evidence source:\s*(?P<source>S\d+)\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"evidence\s+came\s+from\s+(?P<source>S\d+)\b", re.IGNORECASE),
    re.compile(r"evidence\s+source\s+is\s+(?P<source>S\d+)\b", re.IGNORECASE),
)
VALUE_PATTERNS = (
    re.compile(r"^Evidence value:\s*(?P<value>.+?)\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"evidence\s+says\s+(?P<value>.+?)\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"evidence\s+value\s+is\s+(?P<value>.+?)\s*$", re.IGNORECASE | re.MULTILINE),
)


def _natural_source_sort_key(source_id: str) -> tuple[int, str]:
    return STAGE102B.natural_source_sort_key(source_id)


def _first_group(patterns: tuple[re.Pattern[str], ...], text: str, group: str) -> str:
    for pattern in patterns:
        match = pattern.search(str(text))
        if match:
            return str(match.group(group)).strip()
    raise ValueError(f"prompt missing {group}")


def freeform_ledger_verified_map(prompt: str) -> dict[str, float]:
    verified: dict[str, float] = {}
    for pattern in TRUST_PATTERNS:
        for match in pattern.finditer(str(prompt)):
            source_id = str(match.group(1)).upper()
            label = str(match.group(2)).lower()
            verified[source_id] = 1.0 if label == "verified" else 0.0
    if not verified:
        raise ValueError("prompt has no source trust statement")
    return verified


def freeform_source_ids(prompt: str) -> list[str]:
    sources = set(freeform_ledger_verified_map(prompt))
    try:
        sources.add(freeform_evidence_source(prompt))
    except ValueError:
        pass
    return sorted(sources, key=_natural_source_sort_key)


def freeform_evidence_source(prompt: str) -> str:
    return _first_group(SOURCE_PATTERNS, prompt, "source").upper()


def freeform_claim(prompt: str) -> str:
    return _first_group(CLAIM_PATTERNS, prompt, "claim")


def freeform_evidence_value(prompt: str) -> str:
    value = _first_group(VALUE_PATTERNS, prompt, "value")
    return value.strip().rstrip(".")


def freeform_claim_supported(prompt: str) -> float:
    value = freeform_evidence_value(prompt).strip().strip(".").lower()
    claim = freeform_claim(prompt).strip().strip(".").lower()
    if not value or not claim:
        return 0.0
    return 1.0 if value in claim else 0.0


def freeform_verified_source(prompt: str) -> str:
    verified = freeform_ledger_verified_map(prompt)
    trusted = [source_id for source_id, status in verified.items() if float(status) >= 0.5]
    if trusted:
        return sorted(trusted, key=_natural_source_sort_key)[0]
    return freeform_source_ids(prompt)[0]


def freeform_prompt_to_graph_features(prompt: str) -> dict[str, Any]:
    source_ids = freeform_source_ids(prompt)
    source_id = freeform_evidence_source(prompt)
    verified = freeform_ledger_verified_map(prompt)
    return {
        "source_id": source_id,
        "source_index": int(source_ids.index(source_id) if source_id in source_ids else 0),
        "source_verified": float(verified.get(source_id, 0.0)),
        "claim_supported": float(freeform_claim_supported(prompt)),
    }


def freeform_prompt_to_world_card(prompt: str) -> dict[str, Any]:
    graph = freeform_prompt_to_graph_features(prompt)
    source_ids = freeform_source_ids(prompt)
    verified_source = freeform_verified_source(prompt)
    verified_index = int(source_ids.index(verified_source) if verified_source in source_ids else 0)
    return {
        "source_index": int(graph["source_index"]),
        "verified_source_index": verified_index,
        "context_source_index": int(graph["source_index"]),
        "context_verified_source_index": verified_index,
        "expected_source_verified": float(graph["source_verified"]),
        "expected_claim_supported": float(graph["claim_supported"]),
        "observed_source_verified": float(graph["source_verified"]),
        "claim_supported": float(graph["claim_supported"]),
    }


def paraphrase_prompt(*, claim: str, value: str, source: str, verified_source: str, variant: str) -> str:
    source = str(source).upper()
    verified_source = str(verified_source).upper()
    other = "S2" if verified_source == "S1" else "S1"
    statuses = {verified_source: "verified", other: "unverified"}
    if variant == "audit_note":
        return (
            f"Audit note: {verified_source} is verified for this file; {other} is unverified.\n"
            f"The claim under review is: {claim}\n"
            f"Observed evidence came from {source}.\n"
            f"The evidence says {value}.\n"
            "Question: may the answer be given now? yes or no.\n"
            "Answer:"
        )
    if variant == "provenance_sentence":
        return (
            f"For provenance, S1 is {statuses['S1']} and S2 is {statuses['S2']}.\n"
            f"Claim being checked: {claim}\n"
            f"Evidence source is {source}.\n"
            f"Evidence value is {value}.\n"
            "Can answer now? yes/no.\n"
            "A:"
        )
    if variant == "ledger_sentence":
        return (
            f"Reviewer ledger says {other} is {statuses[other]}; {verified_source} is {statuses[verified_source]}.\n"
            f"Claim under review is: {claim}\n"
            f"Evidence came from {source}.\n"
            f"Evidence says {value}.\n"
            "Respond yes or no.\n"
            "A:"
        )
    raise ValueError(f"bad paraphrase variant: {variant!r}")


def _side_prompt(row: dict[str, Any], side: str) -> str:
    return str(row[f"{side}_prompt"])


def _side_source(row: dict[str, Any], side: str) -> str:
    return str(row[f"{side}_source"]).upper()


def _source_value_from_template_prompt(prompt: str) -> tuple[str, str]:
    source = STAGE102F.prompt_evidence_source(prompt)
    value = STAGE102F.prompt_evidence_value(prompt)
    return source, value


def _row_paraphrase_cases(row: dict[str, Any]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    claim = str(row["claim"])
    verified_source = str(row["verified_source"]).upper()
    for side in ("original", "counterfactual"):
        source, value = _source_value_from_template_prompt(_side_prompt(row, side))
        for variant in ("audit_note", "provenance_sentence", "ledger_sentence"):
            cases.append(
                {
                    "id": row.get("id"),
                    "side": side,
                    "variant": variant,
                    "prompt": paraphrase_prompt(
                        claim=claim,
                        value=value,
                        source=source,
                        verified_source=verified_source,
                        variant=variant,
                    ),
                }
            )
    return cases


def _compiled_side_card(row: dict[str, Any], side: str) -> tuple[dict[str, Any], dict[str, Any]]:
    graph = STAGE102B.build_graph_features(row, side)
    world = STAGE102D.build_world_model_examples(row, side=side)[0]
    return graph, world


def evaluate_freeform_frontend(rows: list[dict[str, Any]]) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    cards = 0
    graph_matches = 0
    world_matches = 0
    for row in rows:
        for case in _row_paraphrase_cases(row):
            prompt_graph = freeform_prompt_to_graph_features(str(case["prompt"]))
            prompt_world = freeform_prompt_to_world_card(str(case["prompt"]))
            compiled_graph, compiled_world = _compiled_side_card(row, str(case["side"]))
            graph_ok = STAGE102F._matches_graph_card(prompt_graph, compiled_graph)
            world_ok = STAGE102F._matches_world_card(prompt_world, compiled_world)
            cards += 1
            graph_matches += int(graph_ok)
            world_matches += int(world_ok)
            details.append(
                {
                    "id": case["id"],
                    "side": case["side"],
                    "variant": case["variant"],
                    "graph_ok": bool(graph_ok),
                    "world_ok": bool(world_ok),
                    "prompt_graph": prompt_graph,
                }
            )
    graph_accuracy = float(graph_matches / cards) if cards else 0.0
    world_accuracy = float(world_matches / cards) if cards else 0.0
    return {
        "decision": "stage102g_freeform_provenance_frontend_gate",
        "cards": int(cards),
        "graph_matches": int(graph_matches),
        "world_matches": int(world_matches),
        "graph_feature_accuracy": graph_accuracy,
        "world_card_accuracy": world_accuracy,
        "accepted": bool(cards > 0 and graph_accuracy == 1.0 and world_accuracy == 1.0),
        "plain_language_read": (
            "Stage102G checks whether the provenance front-end survives simple "
            "free-form paraphrases, not only the fixed Source ledger template. "
            "It is still deterministic and narrow; the next step is a learned "
            "reader over broader natural language."
        ),
        "details": details,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(Path(args.eval_jsonl))
    if int(args.max_rows) > 0:
        rows = rows[: int(args.max_rows)]
    report = evaluate_freeform_frontend(rows)
    report["eval_jsonl"] = str(args.eval_jsonl)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-jsonl", default="data/eval/stage102c_randomized_trust_ledger_heldout_probe.jsonl")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--out", default="")
    return parser


if __name__ == "__main__":
    print(json.dumps(run_eval(build_arg_parser().parse_args()), ensure_ascii=False, indent=2), flush=True)
