from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping


def now_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def today_text() -> str:
    return datetime.now().astimezone().date().isoformat()


def resolve_history_path(
    path: str | Path | None = "auto",
    *,
    root: str | Path = "runs",
    kind: str = "generations",
    date_text: str | None = None,
) -> Path | None:
    if path is None:
        return None
    path_text = str(path).strip()
    if not path_text or path_text.lower() in {"none", "off", "false", "0"}:
        return None
    if path_text == "auto":
        return Path(root) / "history" / kind / f"{date_text or today_text()}.jsonl"
    return Path(path_text)


def json_safe_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): json_safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_value(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def infer_failure_type(*, hit: bool | None, completion: str) -> str:
    if hit is True:
        return "good"
    text = str(completion or "")
    words = text.split()
    if words:
        max_run = 1
        cur_run = 1
        for prev, cur in zip(words, words[1:]):
            if prev == cur:
                cur_run += 1
                max_run = max(max_run, cur_run)
            else:
                cur_run = 1
        if max_run >= 4:
            return "repetition"
    if hit is False:
        return "miss"
    return "unlabeled"


def append_jsonl(path: str | Path | None, row: Mapping[str, Any]) -> dict[str, Any]:
    out = resolve_history_path(path)
    if out is None:
        return dict(row)
    out.parent.mkdir(parents=True, exist_ok=True)
    safe_row = json_safe_value(dict(row))
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(safe_row, ensure_ascii=False) + "\n")
    return safe_row


def build_generation_history_row(
    *,
    source: str,
    checkpoint: str,
    config: str,
    prompt: str,
    output: str,
    mode: str,
    completion: str | None = None,
    hit: bool | None = None,
    answer_channel: str | None = None,
    answer_channel_meta: Mapping[str, Any] | None = None,
    retrieved_evidence: list[Mapping[str, Any]] | None = None,
    metadata: Mapping[str, Any] | None = None,
    timestamp: str | None = None,
    case_id: str | None = None,
) -> dict[str, Any]:
    completion_text = str(completion if completion is not None else output)
    return {
        "timestamp": timestamp or now_timestamp(),
        "source": str(source),
        "case_id": case_id,
        "checkpoint": str(checkpoint),
        "config": str(config),
        "mode": str(mode),
        "prompt": str(prompt),
        "output": str(output),
        "completion": completion_text,
        "hit": hit,
        "failure_type": infer_failure_type(hit=hit, completion=completion_text),
        "answer_channel": answer_channel,
        "answer_channel_meta": dict(answer_channel_meta or {}),
        "retrieved_evidence": list(retrieved_evidence or []),
        "metadata": dict(metadata or {}),
    }


def append_generation_history(
    path: str | Path | None,
    **kwargs: Any,
) -> dict[str, Any]:
    row = build_generation_history_row(**kwargs)
    return append_jsonl(path, row)


def eval_record_to_history_row(
    record: Mapping[str, Any],
    *,
    checkpoint: str,
    config: str,
    source: str = "memory_eval",
    timestamp: str | None = None,
) -> dict[str, Any]:
    retrieved_sources = list(record.get("retrieved_sources") or [])
    retrieved_roles = list(record.get("retrieved_roles") or [])
    retrieved_scores = list(record.get("retrieved_retrieval_scores") or [])
    retrieved_evidence = []
    for idx, source_name in enumerate(retrieved_sources):
        retrieved_evidence.append(
            {
                "source": source_name,
                "role": retrieved_roles[idx] if idx < len(retrieved_roles) else None,
                "score": retrieved_scores[idx] if idx < len(retrieved_scores) else None,
            }
        )
    return build_generation_history_row(
        source=source,
        checkpoint=checkpoint,
        config=config,
        prompt=str(record.get("question", "")),
        output=str(record.get("full_text") or record.get("raw_completion") or record.get("completion", "")),
        mode=str(record.get("mode", "")),
        completion=str(record.get("completion", "")),
        hit=record.get("hit"),
        answer_channel=record.get("answer_channel"),
        answer_channel_meta=record.get("answer_channel_meta") or {},
        retrieved_evidence=retrieved_evidence,
        metadata={
            "category": record.get("category"),
            "task_family": record.get("task_family"),
            "expected_unknown": record.get("expected_unknown"),
            "completion_token_count": record.get("completion_token_count"),
            "retrieved_target": record.get("retrieved_target"),
            "match_type": record.get("match_type"),
            "audit_reasons": record.get("audit_reasons"),
        },
        timestamp=timestamp,
        case_id=str(record.get("id", "")),
    )
