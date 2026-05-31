from __future__ import annotations
from pathlib import Path
from typing import Iterable


def iter_text_files(root: str | Path):
    root = Path(root)
    for p in root.rglob("*"):
        if p.suffix.lower() in {".txt", ".md", ".rst", ".py", ".json", ".yaml", ".yml"}:
            yield p


def chunk_text(text: str, chunk_chars: int = 4000, overlap: int = 400):
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_chars)
        yield text[start:end]
        if end == n:
            break
        start = max(end - overlap, start + 1)
