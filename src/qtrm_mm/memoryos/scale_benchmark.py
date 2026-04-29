from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

from .scale_plan import build_memory_scale_plan


def parse_token_count(value: str) -> int:
    text = value.strip().replace("_", "")
    if not text:
        raise ValueError("empty token target")
    suffix = text[-1].casefold()
    multiplier = 1
    if suffix == "k":
        multiplier = 1_000
        text = text[:-1]
    elif suffix == "m":
        multiplier = 1_000_000
        text = text[:-1]
    elif suffix == "b":
        multiplier = 1_000_000_000
        text = text[:-1]
    number = float(text)
    if number <= 0:
        raise ValueError("token target must be positive")
    return int(number * multiplier)


def parse_token_targets(value: str | Iterable[int]) -> list[int]:
    if isinstance(value, str):
        targets = [parse_token_count(part) for part in value.split(",") if part.strip()]
    else:
        targets = [int(item) for item in value]
    if not targets:
        raise ValueError("at least one token target is required")
    if any(target <= 0 for target in targets):
        raise ValueError("token targets must be positive")
    return targets


def token_target_label(total_tokens: int) -> str:
    if total_tokens % 1_000_000_000 == 0:
        return f"{total_tokens // 1_000_000_000}B"
    if total_tokens % 1_000_000 == 0:
        return f"{total_tokens // 1_000_000}M"
    if total_tokens % 1_000 == 0:
        return f"{total_tokens // 1_000}K"
    return str(total_tokens)


def build_scale_benchmark_records(
    token_targets: str | Iterable[int] = (1_000_000, 10_000_000),
    *,
    chunk_tokens: int = 512,
    overlap_tokens: int = 64,
    embedding_dim: int = 640,
    embedding_dtype_bits: int = 32,
    available_ram_gib: float = 64,
    available_vram_gib: float = 24,
) -> list[dict]:
    created_at = datetime.now(timezone.utc).isoformat()
    records = []
    for total_tokens in parse_token_targets(token_targets):
        plan = build_memory_scale_plan(
            total_tokens=total_tokens,
            chunk_tokens=chunk_tokens,
            overlap_tokens=overlap_tokens,
            embedding_dim=embedding_dim,
            embedding_dtype_bits=embedding_dtype_bits,
            available_ram_gib=available_ram_gib,
            available_vram_gib=available_vram_gib,
        )
        records.append(
            {
                "kind": "memoryos_scale_plan",
                "created_at": created_at,
                "target_label": token_target_label(total_tokens),
                "plan": plan.to_dict(),
            }
        )
    return records
