from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil


BYTES_PER_GIB = 1024**3
DEFAULT_CHUNK_TOKENS = 512
DEFAULT_OVERLAP_TOKENS = 64
DEFAULT_HARRIER_270M_DIM = 640


@dataclass(frozen=True)
class MemoryScalePlan:
    total_tokens: int
    chunk_tokens: int
    overlap_tokens: int
    estimated_chunks: int
    embedding_dim: int
    embedding_dtype_bits: int
    embedding_gib: float
    text_gib: float
    available_ram_gib: float
    available_vram_gib: float
    build_backend: str
    serving_pattern: str
    needs_latent_memory_layer: bool
    notes: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_chunk_count(
    total_tokens: int,
    *,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> int:
    if total_tokens < 0:
        raise ValueError("total_tokens must be non-negative")
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be non-negative")
    if overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens must be smaller than chunk_tokens")
    if total_tokens == 0:
        return 0

    stride = chunk_tokens - overlap_tokens
    if total_tokens <= chunk_tokens:
        return 1
    return 1 + ceil((total_tokens - chunk_tokens) / stride)


def bytes_to_gib(num_bytes: float) -> float:
    return float(num_bytes) / BYTES_PER_GIB


def estimate_embedding_gib(
    estimated_chunks: int,
    *,
    embedding_dim: int = DEFAULT_HARRIER_270M_DIM,
    dtype_bits: int = 32,
) -> float:
    if estimated_chunks < 0:
        raise ValueError("estimated_chunks must be non-negative")
    if embedding_dim <= 0:
        raise ValueError("embedding_dim must be positive")
    if dtype_bits <= 0 or dtype_bits % 8 != 0:
        raise ValueError("dtype_bits must be a positive multiple of 8")
    return bytes_to_gib(estimated_chunks * embedding_dim * (dtype_bits // 8))


def estimate_text_gib(total_tokens: int, *, bytes_per_token: float = 4.0) -> float:
    if total_tokens < 0:
        raise ValueError("total_tokens must be non-negative")
    if bytes_per_token <= 0:
        raise ValueError("bytes_per_token must be positive")
    return bytes_to_gib(total_tokens * bytes_per_token)


def choose_build_backend(estimated_chunks: int) -> str:
    if estimated_chunks < 0:
        raise ValueError("estimated_chunks must be non-negative")
    if estimated_chunks <= 50_000:
        return "faiss_flat"
    return "faiss_hnsw"


def build_memory_scale_plan(
    *,
    total_tokens: int,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    embedding_dim: int = DEFAULT_HARRIER_270M_DIM,
    embedding_dtype_bits: int = 32,
    available_ram_gib: float = 64,
    available_vram_gib: float = 24,
) -> MemoryScalePlan:
    estimated_chunks = estimate_chunk_count(
        total_tokens,
        chunk_tokens=chunk_tokens,
        overlap_tokens=overlap_tokens,
    )
    embedding_gib = estimate_embedding_gib(
        estimated_chunks,
        embedding_dim=embedding_dim,
        dtype_bits=embedding_dtype_bits,
    )
    text_gib = estimate_text_gib(total_tokens)
    build_backend = choose_build_backend(estimated_chunks)

    notes = [
        "Large MemoryOS pools should be treated as external memory, not a direct prompt.",
        "Use dense retrieval plus reranking first; only selected evidence should enter the model context.",
    ]
    needs_latent_memory_layer = total_tokens >= 1_000_000
    serving_pattern = "retrieve-rerank-compress"

    if estimated_chunks >= 200_000:
        notes.append("Shard records/index files before long production runs to keep rebuilds and audits tractable.")
    if embedding_gib > available_ram_gib * 0.5:
        notes.append("Embedding storage is large relative to RAM; prefer sharding or compressed/vector-DB storage.")
    if available_vram_gib <= 24 and total_tokens >= 100_000_000:
        notes.append("A 24GB GPU should keep full memory outside VRAM and fetch only top-k evidence/KV blocks.")
    if needs_latent_memory_layer:
        notes.append("MSA-style latent memory or sparse memory attention is needed for end-to-end memory use.")

    return MemoryScalePlan(
        total_tokens=total_tokens,
        chunk_tokens=chunk_tokens,
        overlap_tokens=overlap_tokens,
        estimated_chunks=estimated_chunks,
        embedding_dim=embedding_dim,
        embedding_dtype_bits=embedding_dtype_bits,
        embedding_gib=round(embedding_gib, 6),
        text_gib=round(text_gib, 6),
        available_ram_gib=float(available_ram_gib),
        available_vram_gib=float(available_vram_gib),
        build_backend=build_backend,
        serving_pattern=serving_pattern,
        needs_latent_memory_layer=needs_latent_memory_layer,
        notes=tuple(notes),
    )


def format_plan_summary(plan: MemoryScalePlan) -> str:
    lines = [
        "MemoryOS scale plan",
        f"- total_tokens: {plan.total_tokens:,}",
        f"- chunk_tokens: {plan.chunk_tokens:,}",
        f"- overlap_tokens: {plan.overlap_tokens:,}",
        f"- estimated_chunks: {plan.estimated_chunks:,}",
        f"- embedding_storage: {plan.embedding_gib:.3f} GiB",
        f"- text_storage: {plan.text_gib:.3f} GiB",
        f"- build_backend: {plan.build_backend}",
        f"- serving_pattern: {plan.serving_pattern}",
        f"- needs_latent_memory_layer: {plan.needs_latent_memory_layer}",
    ]
    lines.extend(f"- note: {note}" for note in plan.notes)
    return "\n".join(lines)
