from __future__ import annotations

from typing import Iterable

import torch


MEMORY_EFFICIENT_OPTIMIZERS = (
    "auto",
    "adamw",
    "adamw8bit",
    "paged_adamw8bit",
    "ademamix8bit",
    "paged_ademamix8bit",
    "galore_adamw",
    "galore_adamw8bit",
)


def _import_bitsandbytes_adamw8bit():
    try:
        import bitsandbytes as bnb
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError(
            "bitsandbytes is required for adamw8bit/paged_adamw8bit"
        ) from exc
    return bnb.optim.AdamW8bit, bnb.optim.PagedAdamW8bit


def _import_bitsandbytes_ademamix8bit():
    try:
        import bitsandbytes as bnb
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError(
            "bitsandbytes is required for ademamix8bit/paged_ademamix8bit"
        ) from exc
    return bnb.optim.AdEMAMix8bit, bnb.optim.PagedAdEMAMix8bit


def _import_galore():
    try:
        from galore_torch import GaLoreAdamW, GaLoreAdamW8bit
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError(
            "galore-torch is required for galore_adamw/galore_adamw8bit"
        ) from exc
    return GaLoreAdamW, GaLoreAdamW8bit


def _named_trainable_parameters(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    return [(name, param) for name, param in model.named_parameters() if param.requires_grad]


def _split_galore_param_groups(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    *,
    weight_decay: float,
    rank: int,
    update_proj_gap: int,
    scale: float,
    proj_type: str,
    min_dim: int,
    include_embeddings: bool,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    galore_params: list[torch.nn.Parameter] = []
    regular_params: list[torch.nn.Parameter] = []
    galore_names: list[str] = []
    regular_names: list[str] = []
    rank = max(1, int(rank))
    min_dim = max(1, int(min_dim))
    for name, param in named_parameters:
        shape = tuple(int(dim) for dim in param.shape)
        is_embedding_or_head = "embed" in name or "lm_head" in name
        use_galore = (
            param.ndim == 2
            and min(shape) >= rank
            and max(shape) >= min_dim
            and (bool(include_embeddings) or not is_embedding_or_head)
        )
        if use_galore:
            galore_params.append(param)
            galore_names.append(name)
        else:
            regular_params.append(param)
            regular_names.append(name)

    groups: list[dict[str, object]] = []
    if regular_params:
        groups.append({"params": regular_params, "weight_decay": float(weight_decay)})
    if galore_params:
        groups.append(
            {
                "params": galore_params,
                "weight_decay": float(weight_decay),
                "rank": rank,
                "update_proj_gap": max(1, int(update_proj_gap)),
                "scale": float(scale),
                "proj_type": str(proj_type),
            }
        )
    report = {
        "galore_parameter_count": int(sum(param.numel() for param in galore_params)),
        "regular_parameter_count": int(sum(param.numel() for param in regular_params)),
        "galore_tensor_count": len(galore_params),
        "regular_tensor_count": len(regular_params),
        "galore_names_preview": galore_names[:16],
        "regular_names_preview": regular_names[:16],
        "rank": rank,
        "update_proj_gap": max(1, int(update_proj_gap)),
        "scale": float(scale),
        "proj_type": str(proj_type),
        "min_dim": min_dim,
        "include_embeddings": bool(include_embeddings),
    }
    return groups, report


def resolve_optimizer_name(requested: str, *, device: torch.device) -> str:
    name = str(requested or "adamw").lower()
    if name != "auto":
        if name not in MEMORY_EFFICIENT_OPTIMIZERS:
            raise ValueError(f"unknown optimizer: {requested}")
        return name
    if device.type == "cuda":
        try:
            _import_galore()
            return "galore_adamw8bit"
        except RuntimeError:
            try:
                _import_bitsandbytes_adamw8bit()
                return "adamw8bit"
            except RuntimeError:
                return "adamw"
    return "adamw"


def build_memory_efficient_optimizer(
    model: torch.nn.Module,
    *,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    device: torch.device,
    beta1: float = 0.9,
    beta2: float = 0.999,
    extra_named_parameters: Iterable[tuple[str, torch.nn.Parameter]] | None = None,
    galore_rank: int = 128,
    galore_update_proj_gap: int = 200,
    galore_scale: float = 0.25,
    galore_proj_type: str = "std",
    galore_min_dim: int = 128,
    galore_include_embeddings: bool = False,
) -> tuple[torch.optim.Optimizer, dict[str, object]]:
    """Build a full-parameter optimizer with optional low-memory state.

    GaLore is applied to large 2D weight tensors only; all remaining trainable
    parameters are still updated by the same optimizer. This keeps the training
    path full-parameter while reducing optimizer-state memory where it matters.
    """

    resolved = resolve_optimizer_name(str(optimizer_name), device=device)
    named = _named_trainable_parameters(model)
    if extra_named_parameters is not None:
        named.extend(
            (str(name), param)
            for name, param in extra_named_parameters
            if param.requires_grad
        )
    total_params = int(sum(param.numel() for _, param in named))
    if not named:
        raise ValueError("no trainable parameters selected")
    report: dict[str, object] = {
        "requested": str(optimizer_name),
        "resolved": resolved,
        "trainable_parameter_count": total_params,
        "beta1": float(beta1),
        "beta2": float(beta2),
    }
    if resolved == "adamw":
        optimizer = torch.optim.AdamW(
            [param for _, param in named],
            lr=float(lr),
            betas=(float(beta1), float(beta2)),
            weight_decay=float(weight_decay),
        )
        return optimizer, report
    if resolved in {"adamw8bit", "paged_adamw8bit"}:
        AdamW8bit, PagedAdamW8bit = _import_bitsandbytes_adamw8bit()
        cls = PagedAdamW8bit if resolved == "paged_adamw8bit" else AdamW8bit
        optimizer = cls(
            [param for _, param in named],
            lr=float(lr),
            betas=(float(beta1), float(beta2)),
            weight_decay=float(weight_decay),
        )
        return optimizer, report
    if resolved in {"ademamix8bit", "paged_ademamix8bit"}:
        AdEMAMix8bit, PagedAdEMAMix8bit = _import_bitsandbytes_ademamix8bit()
        cls = PagedAdEMAMix8bit if resolved == "paged_ademamix8bit" else AdEMAMix8bit
        optimizer = cls(
            [param for _, param in named],
            lr=float(lr),
            betas=(float(beta1), float(beta2), 0.9999),
            weight_decay=float(weight_decay),
        )
        report["beta3"] = 0.9999
        return optimizer, report
    if resolved in {"galore_adamw", "galore_adamw8bit"}:
        GaLoreAdamW, GaLoreAdamW8bit = _import_galore()
        groups, galore_report = _split_galore_param_groups(
            named,
            weight_decay=float(weight_decay),
            rank=int(galore_rank),
            update_proj_gap=int(galore_update_proj_gap),
            scale=float(galore_scale),
            proj_type=str(galore_proj_type),
            min_dim=int(galore_min_dim),
            include_embeddings=bool(galore_include_embeddings),
        )
        cls = GaLoreAdamW8bit if resolved == "galore_adamw8bit" else GaLoreAdamW
        kwargs: dict[str, object] = {
            "lr": float(lr),
            "betas": (float(beta1), float(beta2)),
            "weight_decay": float(weight_decay),
        }
        optimizer = cls(groups, **kwargs)
        report.update(galore_report)
        return optimizer, report
    raise AssertionError(f"unhandled optimizer: {resolved}")
