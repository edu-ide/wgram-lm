#!/usr/bin/env python3
"""Evaluate Stage102F prompt-only provenance front-end.

Stage102B/C proved that a provenance graph register can causally steer the
answer path, but its features were compiled from probe row fields.  Stage102F is
the first bridge away from that shortcut: recover the graph/world cards from
the visible prompt text only, then compare them to the compiled Stage102B/D
contracts.

This script is a contract gate, not a final learned parser.  It proves the
front-end boundary explicitly: ordinary prompt text in, provenance cards out,
no answer fields required.
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


STAGE102B = load_module(
    ROOT / "scripts" / "605_train_stage102b_provenance_graph_reasoner.py",
    "stage102f_stage102b_provenance_graph_reasoner",
)
STAGE102D = load_module(
    ROOT / "scripts" / "607_train_stage102d_provenance_data_world_model.py",
    "stage102f_stage102d_provenance_data_world_model",
)


SOURCE_RE = re.compile(r"\b(S\d+)\s*=\s*(verified|unverified)\b", re.IGNORECASE)
CLAIM_RE = re.compile(r"^Claim:\s*(?P<claim>.+?)\s*$", re.IGNORECASE | re.MULTILINE)
EVIDENCE_SOURCE_RE = re.compile(
    r"^Evidence source:\s*(?P<source>S\d+)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
EVIDENCE_VALUE_RE = re.compile(
    r"^Evidence value:\s*(?P<value>.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _natural_source_sort_key(source_id: str) -> tuple[int, str]:
    return STAGE102B.natural_source_sort_key(source_id)


def prompt_source_ids(prompt: str) -> list[str]:
    sources = {str(match.group(1)).upper() for match in SOURCE_RE.finditer(str(prompt))}
    evidence_match = EVIDENCE_SOURCE_RE.search(str(prompt))
    if evidence_match:
        sources.add(str(evidence_match.group("source")).upper())
    if not sources:
        raise ValueError("prompt has no source ledger or evidence source")
    return sorted(sources, key=_natural_source_sort_key)


def prompt_ledger_verified_map(prompt: str) -> dict[str, float]:
    verified: dict[str, float] = {}
    for match in SOURCE_RE.finditer(str(prompt)):
        source_id = str(match.group(1)).upper()
        label = str(match.group(2)).lower()
        verified[source_id] = 1.0 if label == "verified" else 0.0
    if not verified:
        raise ValueError("prompt has no source trust ledger")
    for source_id in prompt_source_ids(prompt):
        verified.setdefault(source_id, 0.0)
    return verified


def prompt_evidence_source(prompt: str) -> str:
    match = EVIDENCE_SOURCE_RE.search(str(prompt))
    if not match:
        raise ValueError("prompt has no evidence source")
    return str(match.group("source")).upper()


def prompt_claim(prompt: str) -> str:
    match = CLAIM_RE.search(str(prompt))
    if not match:
        raise ValueError("prompt has no claim")
    return str(match.group("claim")).strip()


def prompt_evidence_value(prompt: str) -> str:
    match = EVIDENCE_VALUE_RE.search(str(prompt))
    if not match:
        raise ValueError("prompt has no evidence value")
    return str(match.group("value")).strip()


def prompt_claim_supported(prompt: str) -> float:
    value = prompt_evidence_value(prompt).strip().strip(".").lower()
    claim = prompt_claim(prompt).strip().strip(".").lower()
    if not value or not claim:
        return 0.0
    return 1.0 if value in claim else 0.0


def verified_source_from_prompt(prompt: str) -> str:
    verified = prompt_ledger_verified_map(prompt)
    trusted = [source_id for source_id, is_verified in verified.items() if float(is_verified) >= 0.5]
    if not trusted:
        return prompt_source_ids(prompt)[0]
    return sorted(trusted, key=_natural_source_sort_key)[0]


def prompt_to_graph_features(prompt: str) -> dict[str, Any]:
    source_ids = prompt_source_ids(prompt)
    source_id = prompt_evidence_source(prompt)
    verified = prompt_ledger_verified_map(prompt)
    return {
        "source_id": source_id,
        "source_index": int(source_ids.index(source_id) if source_id in source_ids else 0),
        "source_verified": float(verified.get(source_id, 0.0)),
        "claim_supported": float(prompt_claim_supported(prompt)),
    }


def prompt_to_world_card(prompt: str) -> dict[str, Any]:
    graph = prompt_to_graph_features(prompt)
    source_ids = prompt_source_ids(prompt)
    verified_source = verified_source_from_prompt(prompt)
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


def _compiled_side_for_prompt(row: dict[str, Any], prompt_key: str) -> str:
    if prompt_key == "original_prompt":
        return "original"
    if prompt_key == "counterfactual_prompt":
        return "counterfactual"
    raise ValueError(f"bad prompt key: {prompt_key!r}")


def _matches_graph_card(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        int(left["source_index"]) == int(right["source_index"])
        and abs(float(left["source_verified"]) - float(right["source_verified"])) < 1e-6
        and abs(float(left["claim_supported"]) - float(right["claim_supported"])) < 1e-6
    )


def _matches_world_card(left: dict[str, Any], right: dict[str, Any]) -> bool:
    keys = (
        "source_index",
        "verified_source_index",
        "context_source_index",
        "context_verified_source_index",
        "expected_source_verified",
        "expected_claim_supported",
        "observed_source_verified",
        "claim_supported",
    )
    for key in keys:
        if key.endswith("_index"):
            if int(left[key]) != int(right[key]):
                return False
        elif abs(float(left[key]) - float(right[key])) >= 1e-6:
            return False
    return True


def _compiled_world_clean_card(row: dict[str, Any], side: str) -> dict[str, Any]:
    return STAGE102D.build_world_model_examples(row, side=side)[0]


def evaluate_prompt_frontend(rows: list[dict[str, Any]]) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    graph_matches = 0
    world_matches = 0
    cards = 0
    for row in rows:
        for prompt_key in ("original_prompt", "counterfactual_prompt"):
            prompt = str(row.get(prompt_key, ""))
            if not prompt:
                continue
            side = _compiled_side_for_prompt(row, prompt_key)
            prompt_graph = prompt_to_graph_features(prompt)
            compiled_graph = STAGE102B.build_graph_features(row, side)
            prompt_world = prompt_to_world_card(prompt)
            compiled_world = _compiled_world_clean_card(row, side)
            graph_ok = _matches_graph_card(prompt_graph, compiled_graph)
            world_ok = _matches_world_card(prompt_world, compiled_world)
            graph_matches += int(graph_ok)
            world_matches += int(world_ok)
            cards += 1
            details.append(
                {
                    "id": row.get("id"),
                    "side": side,
                    "graph_ok": bool(graph_ok),
                    "world_ok": bool(world_ok),
                    "prompt_graph": prompt_graph,
                    "compiled_graph": compiled_graph,
                    "prompt_world": prompt_world,
                    "compiled_world": {
                        key: compiled_world[key]
                        for key in (
                            "source_index",
                            "verified_source_index",
                            "context_source_index",
                            "context_verified_source_index",
                            "expected_source_verified",
                            "expected_claim_supported",
                            "observed_source_verified",
                            "claim_supported",
                        )
                    },
                }
            )
    graph_accuracy = float(graph_matches / cards) if cards else 0.0
    world_accuracy = float(world_matches / cards) if cards else 0.0
    return {
        "decision": "stage102f_prompt_only_provenance_frontend_gate",
        "cards": int(cards),
        "graph_matches": int(graph_matches),
        "world_matches": int(world_matches),
        "graph_feature_accuracy": graph_accuracy,
        "world_card_accuracy": world_accuracy,
        "accepted": bool(cards > 0 and graph_accuracy == 1.0 and world_accuracy == 1.0),
        "plain_language_read": (
            "Stage102F removes the answer-sheet shortcut for the provenance front-end: "
            "the visible prompt alone reconstructs the source/trust/support cards used "
            "by the graph reasoner and data-world model.  This is still a template "
            "front-end, not a learned universal reader."
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
    report = evaluate_prompt_frontend(rows)
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
