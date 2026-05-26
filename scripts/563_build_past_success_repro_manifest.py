#!/usr/bin/env python3
"""Build a reproducibility manifest for past successful small-run gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _last_history_eval(payload: dict[str, Any], *, path: Path) -> dict[str, Any]:
    history = payload.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError(f"summary has no non-empty history list: {path}")
    last = history[-1]
    if not isinstance(last, dict) or not isinstance(last.get("eval"), dict):
        raise ValueError(f"summary history[-1] has no eval object: {path}")
    return last["eval"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(name: str, path: Path) -> dict[str, Any]:
    exists = path.exists()
    return {
        "name": name,
        "path": str(path),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else 0,
        "sha256": _sha256(path) if exists else "",
    }


def _metadata(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _args(payload: dict[str, Any]) -> dict[str, Any]:
    args = _metadata(payload).get("args")
    return args if isinstance(args, dict) else {}


def _has_code_commit(payload: dict[str, Any]) -> bool:
    metadata = _metadata(payload)
    args = _args(payload)
    for source in (metadata, args):
        for key in ("code_commit", "git_commit", "commit", "code_hash", "git_hash"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return True
    return False


def _has_eval_rows(summary_path: Path, payload: dict[str, Any]) -> bool:
    args = _args(payload)
    for key in ("eval_rows", "eval_jsonl", "eval_cases_path", "materialized_eval_rows"):
        raw_value = args.get(key)
        if not isinstance(raw_value, str) or not raw_value.strip():
            continue
        value = Path(raw_value)
        if not value.is_absolute():
            value = summary_path.parent / value
        if value.exists():
            return True
    return any(summary_path.parent.glob("*.jsonl"))


def _settings(args: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "seed",
        "eval_seed",
        "eval_count",
        "samples",
        "candidate_topk_per_sample",
        "eval_depths",
        "train_surface_mode",
        "eval_surface_mode",
        "stochastic_high_level_eval",
        "stochastic_transition_mode",
        "stochastic_posterior_guidance",
        "qwen_model_id",
    ]
    return {key: args.get(key) for key in keys if key in args}


def _metrics(eval_payload: dict[str, Any]) -> dict[str, Any]:
    selected = eval_payload.get("mean_selected_accuracy_oracle_depth")
    oracle = eval_payload.get("mean_oracle_accuracy")
    packed = eval_payload.get("mean_packed_register_answer_accuracy_oracle_depth")
    return {
        "selected_accuracy": float(selected) if selected is not None else None,
        "oracle_accuracy": float(oracle) if oracle is not None else None,
        "packed_register_answer_accuracy": float(packed) if packed is not None else None,
    }


def build_stage_repro_row(summary_path: Path | str, *, label: str) -> dict[str, Any]:
    summary_path = Path(summary_path)
    payload = _load_json(summary_path)
    args = _args(payload)
    checkpoint = Path(str(args.get("checkpoint", "")))
    extractor_checkpoint = Path(str(args.get("extractor_checkpoint", "")))
    artifacts = [
        _artifact("summary", summary_path),
        _artifact("generator_checkpoint", checkpoint),
        _artifact("extractor_checkpoint", extractor_checkpoint),
    ]

    blocking_gaps: list[str] = []
    missing = [artifact["name"] for artifact in artifacts if not artifact["exists"]]
    if missing:
        blocking_gaps.append(f"missing required artifact(s): {', '.join(missing)}")
    if not _has_code_commit(payload):
        blocking_gaps.append("no immutable code commit/hash recorded")
    if not _has_eval_rows(summary_path, payload):
        blocking_gaps.append("no materialized eval JSONL rows stored beside summary")
    if args.get("stochastic_high_level_eval"):
        blocking_gaps.append("stochastic eval enabled without deterministic replay proof")
    blocking_gaps.append("no one-command rerun script with tolerance compare recorded")

    can_replay = not missing
    fully_sealed = can_replay and not blocking_gaps
    if fully_sealed:
        status = "sealed"
    elif can_replay:
        status = "replayable_not_sealed"
    else:
        status = "not_replayable"

    return {
        "label": str(label),
        "summary_path": str(summary_path),
        "reproducibility_status": status,
        "can_replay": can_replay,
        "fully_sealed": fully_sealed,
        "metrics": _metrics(_last_history_eval(payload, path=summary_path)),
        "settings": _settings(args),
        "artifacts": artifacts,
        "blocking_gaps": blocking_gaps,
        "plain_language_verdict": (
            "Replayable from the surviving artifacts, but not sealed for exact/paper-grade reproduction."
            if can_replay
            else "Not replayable until the missing required artifacts are restored."
        ),
    }


def build_reproducibility_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    can_replay_any = any(bool(row.get("can_replay")) for row in rows)
    all_fully_sealed = bool(rows) and all(bool(row.get("fully_sealed")) for row in rows)
    if all_fully_sealed:
        overall_status = "sealed"
    elif can_replay_any:
        overall_status = "replayable_not_sealed"
    else:
        overall_status = "not_replayable"
    return {
        "manifest_type": "past_success_reproducibility_manifest",
        "overall_status": overall_status,
        "can_replay_any": can_replay_any,
        "all_fully_sealed": all_fully_sealed,
        "plain_korean_verdict": (
            "다시 돌려볼 수는 있지만, 아직 논문식 완전 재현 패키지는 아니다."
            if overall_status == "replayable_not_sealed"
            else (
                "재현 패키지가 봉인되어 있다."
                if overall_status == "sealed"
                else "필수 아티팩트가 빠져 있어 지금 상태로는 재현 가능하다고 말하기 어렵다."
            )
        ),
        "rows": rows,
    }


def render_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# Past-Success Reproducibility Manifest",
        "",
        manifest["plain_korean_verdict"],
        "",
        f"Overall status: `{manifest.get('overall_status', '')}`",
        "",
        "| Run | Status | Can Replay | Fully Sealed | Selected | Oracle | Blocking Gaps |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in manifest.get("rows", []):
        metrics = row.get("metrics", {})
        lines.append(
            "| {label} | {status} | {can_replay} | {fully_sealed} | {selected} | {oracle} | {gaps} |".format(
                label=row.get("label", ""),
                status=row.get("reproducibility_status", ""),
                can_replay=row.get("can_replay", ""),
                fully_sealed=row.get("fully_sealed", ""),
                selected=metrics.get("selected_accuracy", ""),
                oracle=metrics.get("oracle_accuracy", ""),
                gaps="; ".join(row.get("blocking_gaps", [])),
            )
        )
    lines.extend(
        [
            "",
            "## Artifact Hashes",
            "",
            "| Run | Artifact | Exists | SHA256 | Path |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in manifest.get("rows", []):
        for artifact in row.get("artifacts", []):
            lines.append(
                "| {label} | {name} | {exists} | `{sha256}` | `{path}` |".format(
                    label=row.get("label", ""),
                    name=artifact.get("name", ""),
                    exists=artifact.get("exists", ""),
                    sha256=artifact.get("sha256", ""),
                    path=artifact.get("path", ""),
                )
            )
    lines.append("")
    return "\n".join(lines)


def _parse_label_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    label, raw_path = value.split("=", 1)
    if not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    return label.strip(), Path(raw_path.strip())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-md", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rows = [
        build_stage_repro_row(path, label=label)
        for label, path in (_parse_label_path(value) for value in args.stage)
    ]
    manifest = build_reproducibility_manifest(rows)
    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = render_markdown(manifest)
    if args.out_md:
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(markdown, encoding="utf-8")
    print(markdown, flush=True)


if __name__ == "__main__":
    main()
