from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import Counter, defaultdict
from typing import Iterable, Mapping, Sequence

import torch


REQUIRED_SAE_KEYS = ("W_enc", "W_dec", "b_enc", "b_dec")


@dataclass(frozen=True)
class QwenScopeSAE:
    """Qwen-Scope TopK SAE tensors for one donor residual-stream layer."""

    layer: int
    W_enc: torch.Tensor
    W_dec: torch.Tensor
    b_enc: torch.Tensor
    b_dec: torch.Tensor

    @property
    def d_sae(self) -> int:
        return int(self.W_enc.shape[0])

    @property
    def d_model(self) -> int:
        return int(self.W_enc.shape[1])


@dataclass(frozen=True)
class QwenScopeTopKFeatures:
    indices: torch.Tensor
    values: torch.Tensor
    pre_activation_shape: tuple[int, ...]


def load_qwen_scope_sae_file(
    path: str | Path,
    *,
    layer: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> QwenScopeSAE:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Qwen-Scope SAE file must contain a tensor dict")
    missing = [key for key in REQUIRED_SAE_KEYS if key not in payload]
    if missing:
        raise ValueError(f"Qwen-Scope SAE file is missing keys: {missing}")
    tensors = {
        key: _as_tensor(payload[key], key=key, device=device, dtype=dtype)
        for key in REQUIRED_SAE_KEYS
    }
    _validate_sae_shapes(tensors)
    return QwenScopeSAE(layer=int(layer), **tensors)


def load_qwen_scope_sae_from_hub(
    repo_id: str,
    *,
    layer: int,
    filename: str | None = None,
    cache_dir: str | Path | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> QwenScopeSAE:
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise RuntimeError("huggingface_hub is required to download Qwen-Scope SAE files") from exc

    resolved = hf_hub_download(
        repo_id=repo_id,
        filename=filename or f"layer{int(layer)}.sae.pt",
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    return load_qwen_scope_sae_file(resolved, layer=layer, device=device, dtype=dtype)


def extract_topk_features(
    sae: QwenScopeSAE,
    residual: torch.Tensor,
    *,
    top_k: int = 100,
) -> QwenScopeTopKFeatures:
    """Project donor residual states into Qwen-Scope SAE feature space.

    Returns only top-k feature ids and values, avoiding a dense sparse-activation
    tensor for large SAE widths.
    """

    if residual.shape[-1] != sae.d_model:
        raise ValueError(
            f"residual hidden size {residual.shape[-1]} does not match SAE d_model {sae.d_model}"
        )
    top_k = min(max(int(top_k), 1), sae.d_sae)
    residual_f = residual.to(device=sae.W_enc.device, dtype=sae.W_enc.dtype)
    pre_acts = residual_f @ sae.W_enc.T + sae.b_enc
    values, indices = pre_acts.topk(top_k, dim=-1)
    return QwenScopeTopKFeatures(
        indices=indices.detach().cpu(),
        values=values.detach().cpu(),
        pre_activation_shape=tuple(int(x) for x in pre_acts.shape),
    )


def qwen_scope_feature_records(
    sae: QwenScopeSAE,
    residual: torch.Tensor,
    *,
    prompts: Sequence[str],
    token_ids: torch.Tensor | None = None,
    token_texts: Sequence[Sequence[str]] | None = None,
    attention_mask: torch.Tensor | None = None,
    top_k: int = 100,
    token_position: int | str = -1,
) -> list[dict]:
    features = extract_topk_features(sae, residual, top_k=top_k)
    b, s = residual.shape[:2]
    if len(prompts) != b:
        raise ValueError("prompts length must match residual batch size")
    if token_position == "last_nonpad":
        if attention_mask is None:
            raise ValueError("attention_mask is required for token_position='last_nonpad'")
        if attention_mask.shape != (b, s):
            raise ValueError("attention_mask must match residual batch/sequence shape")
        valid_counts = attention_mask.to(dtype=torch.long).sum(dim=1)
        token_indices = (valid_counts - 1).clamp_min(0).detach().cpu().tolist()
    else:
        token_index = int(token_position)
        if token_index < 0:
            token_index = s + token_index
        if token_index < 0 or token_index >= s:
            raise ValueError("token_position is outside the residual sequence")
        token_indices = [token_index for _ in range(b)]

    rows = []
    for batch_idx, prompt in enumerate(prompts):
        token_index = int(token_indices[batch_idx])
        token_id = None
        if token_ids is not None:
            token_id = int(token_ids[batch_idx, token_index].detach().cpu().item())
        token = None
        if token_texts is not None:
            token = str(token_texts[batch_idx][token_index])
        rows.append(
            {
                "prompt_index": int(batch_idx),
                "prompt": str(prompt),
                "layer": int(sae.layer),
                "token_index": int(token_index),
                "token_id": token_id,
                "token": token,
                "feature_ids": [
                    int(x)
                    for x in features.indices[batch_idx, token_index].tolist()
                ],
                "feature_values": [
                    float(x)
                    for x in features.values[batch_idx, token_index].tolist()
                ],
            }
        )
    return rows


def decode_token_texts(tokenizer, input_ids: torch.Tensor) -> list[list[str]]:
    out = []
    for row in input_ids.detach().cpu().tolist():
        out.append(
            [
                tokenizer.decode([int(token_id)], skip_special_tokens=False)
                for token_id in row
            ]
        )
    return out


def compare_qwen_scope_feature_groups(
    records: Sequence[dict],
    *,
    normal_prompt_indices: Iterable[int],
    repeat_prompt_indices: Iterable[int],
    feature_limit: int = 20,
    top_output: int = 15,
) -> dict:
    normal_set = {int(idx) for idx in normal_prompt_indices}
    repeat_set = {int(idx) for idx in repeat_prompt_indices}
    layers = sorted({int(row["layer"]) for row in records})
    summary = {
        "normal_prompt_indices": sorted(normal_set),
        "repeat_prompt_indices": sorted(repeat_set),
        "feature_limit": int(feature_limit),
        "layers": {},
    }
    for layer in layers:
        layer_rows = [row for row in records if int(row["layer"]) == layer]
        normal = [row for row in layer_rows if int(row["prompt_index"]) in normal_set]
        repeat = [row for row in layer_rows if int(row["prompt_index"]) in repeat_set]
        normal_stats = _qwen_scope_group_stats(normal, feature_limit=feature_limit)
        repeat_stats = _qwen_scope_group_stats(repeat, feature_limit=feature_limit)
        diff = _feature_group_deltas(
            normal_stats,
            repeat_stats,
            normal_count=max(1, len(normal)),
            repeat_count=max(1, len(repeat)),
        )
        repeat_enriched = sorted(
            diff,
            key=lambda item: (item["freq_delta"], item["value_delta"]),
            reverse=True,
        )[:top_output]
        normal_enriched = sorted(
            diff,
            key=lambda item: (
                item["normal_freq"] - item["repeat_freq"],
                item["normal_avg_value"] - item["repeat_avg_value"],
            ),
            reverse=True,
        )[:top_output]
        summary["layers"][str(layer)] = {
            "normal_tokens": normal_stats["tokens"],
            "repeat_tokens": repeat_stats["tokens"],
            "normal_top1": normal_stats["top1"],
            "repeat_top1": repeat_stats["top1"],
            "normal_common_top_features": normal_stats["common_top_features"],
            "repeat_common_top_features": repeat_stats["common_top_features"],
            "repeat_enriched_features": repeat_enriched,
            "normal_enriched_features": normal_enriched,
            "repeat_shared_not_normal_features": [
                item
                for item in repeat_enriched
                if item["repeat_freq"] >= (2.0 / 3.0) and item["normal_freq"] == 0.0
            ],
        }
    return summary


def score_qwen_scope_candidate_features(
    records: Sequence[dict],
    *,
    candidate_features_by_layer: Mapping[int, Iterable[int]],
    feature_limit: int | None = None,
) -> list[dict]:
    """Score prompt-level hits for known Qwen-Scope diagnostic candidates.

    This is a correlation probe, not a causal steering mechanism. It only reads
    top-k SAE feature records and reports whether chosen layer-specific feature
    ids appear in each prompt's logged features.
    """

    candidates = {
        int(layer): {int(fid) for fid in feature_ids}
        for layer, feature_ids in candidate_features_by_layer.items()
    }
    prompt_rows: dict[int, dict] = {}
    for row in records:
        layer = int(row["layer"])
        if layer not in candidates:
            continue
        prompt_index = int(row["prompt_index"])
        prompt_score = prompt_rows.setdefault(
            prompt_index,
            {
                "prompt_index": prompt_index,
                "prompt": row.get("prompt"),
                "total_hit_count": 0,
                "total_value_sum": 0.0,
                "total_value_max": 0.0,
                "layers": {},
            },
        )
        if prompt_score.get("prompt") is None and row.get("prompt") is not None:
            prompt_score["prompt"] = row.get("prompt")
        layer_candidates = candidates[layer]
        feature_ids = [int(fid) for fid in row.get("feature_ids", [])]
        feature_values = [float(value) for value in row.get("feature_values", [])]
        if feature_limit is not None:
            limit = max(0, int(feature_limit))
            feature_ids = feature_ids[:limit]
            feature_values = feature_values[:limit]
        hits = []
        for rank, (feature_id, value) in enumerate(zip(feature_ids, feature_values), start=1):
            if feature_id in layer_candidates:
                hits.append({"feature_id": feature_id, "rank": rank, "value": value})
        value_sum = round(float(sum(hit["value"] for hit in hits)), 12)
        value_max = max((float(hit["value"]) for hit in hits), default=0.0)
        prompt_score["total_hit_count"] += len(hits)
        prompt_score["total_value_sum"] = round(
            float(prompt_score["total_value_sum"] + value_sum),
            12,
        )
        prompt_score["total_value_max"] = float(max(prompt_score["total_value_max"], value_max))
        prompt_score["layers"][str(layer)] = {
            "candidate_features": sorted(layer_candidates),
            "hit_count": len(hits),
            "value_sum": value_sum,
            "value_max": value_max,
            "hits": hits,
        }
    return [prompt_rows[idx] for idx in sorted(prompt_rows)]


def summarize_qwen_scope_repeat_score_thresholds(
    scores: Sequence[dict],
    *,
    score_field: str = "total_value_sum",
    label_field: str | None = "repeat_label",
    positive_label: str = "repeat",
    repeat_rate_field: str = "repeated_2gram_rate",
    repeat_rate_threshold: float | None = None,
) -> dict:
    rows = []
    label_counts: Counter[str] = Counter()
    for row in scores:
        if score_field not in row or row[score_field] is None:
            continue
        label = None
        if label_field is not None and row.get(label_field) is not None:
            label = str(row[label_field])
        elif repeat_rate_threshold is not None and row.get(repeat_rate_field) is not None:
            label = positive_label if float(row[repeat_rate_field]) >= repeat_rate_threshold else "normal"
        if label is None:
            continue
        score = float(row[score_field])
        rows.append(
            {
                "prompt_index": int(row["prompt_index"]),
                "score": score,
                "label": label,
                "is_positive": label == positive_label,
            }
        )
        label_counts[label] += 1
    thresholds = sorted({row["score"] for row in rows}, reverse=True)
    threshold_rows = [
        _score_threshold_metrics(rows, threshold=threshold)
        for threshold in thresholds
    ]
    best = None
    if threshold_rows:
        best = max(
            threshold_rows,
            key=lambda item: (
                item["f1"],
                item["precision"],
                item["recall"],
                item["accuracy"],
                item["threshold"],
            ),
        )
    return {
        "score_field": score_field,
        "label_field": label_field,
        "positive_label": positive_label,
        "repeat_rate_field": repeat_rate_field,
        "repeat_rate_threshold": repeat_rate_threshold,
        "n": len(rows),
        "label_counts": dict(sorted(label_counts.items())),
        "best_threshold": best,
        "thresholds": threshold_rows,
    }


def _score_threshold_metrics(rows: Sequence[dict], *, threshold: float) -> dict:
    tp = fp = tn = fn = 0
    for row in rows:
        pred = float(row["score"]) >= float(threshold)
        actual = bool(row["is_positive"])
        if pred and actual:
            tp += 1
        elif pred and not actual:
            fp += 1
        elif not pred and actual:
            fn += 1
        else:
            tn += 1
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = float((tp + tn) / max(1, len(rows)))
    return {
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def _qwen_scope_group_stats(records: Sequence[dict], *, feature_limit: int) -> dict:
    counts: Counter[int] = Counter()
    values: defaultdict[int, list[float]] = defaultdict(list)
    tokens = []
    top1 = []
    top_sets = []
    for row in records:
        token = row.get("token")
        tokens.append(token)
        feature_ids = [int(fid) for fid in row.get("feature_ids", [])]
        feature_values = [float(val) for val in row.get("feature_values", [])]
        if feature_ids and feature_values:
            top1.append(
                {
                    "feature_id": feature_ids[0],
                    "feature_value": feature_values[0],
                    "prompt_index": int(row["prompt_index"]),
                    "token": token,
                }
            )
        top_features = feature_ids[:feature_limit]
        if top_features:
            top_sets.append(set(top_features))
        for fid, val in zip(top_features, feature_values[:feature_limit]):
            counts[fid] += 1
            values[fid].append(val)
    common = sorted(set.intersection(*top_sets)) if top_sets else []
    return {
        "counts": counts,
        "values": values,
        "tokens": tokens,
        "top1": top1,
        "common_top_features": common,
    }


def _feature_group_deltas(
    normal_stats: dict,
    repeat_stats: dict,
    *,
    normal_count: int,
    repeat_count: int,
) -> list[dict]:
    normal_counts = normal_stats["counts"]
    repeat_counts = repeat_stats["counts"]
    normal_values = normal_stats["values"]
    repeat_values = repeat_stats["values"]
    all_features = set(normal_counts) | set(repeat_counts)
    rows = []
    for fid in all_features:
        normal_avg = _avg(normal_values.get(fid, []))
        repeat_avg = _avg(repeat_values.get(fid, []))
        normal_freq = float(normal_counts[fid]) / float(normal_count)
        repeat_freq = float(repeat_counts[fid]) / float(repeat_count)
        rows.append(
            {
                "feature_id": int(fid),
                "repeat_freq": repeat_freq,
                "normal_freq": normal_freq,
                "freq_delta": repeat_freq - normal_freq,
                "repeat_avg_value": repeat_avg,
                "normal_avg_value": normal_avg,
                "value_delta": repeat_avg - normal_avg,
            }
        )
    return rows


def _avg(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _as_tensor(
    value,
    *,
    key: str,
    device: str | torch.device,
    dtype: torch.dtype | None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{key} must be a torch.Tensor")
    if dtype is None:
        return value.to(device=device)
    return value.to(device=device, dtype=dtype)


def _validate_sae_shapes(tensors: dict[str, torch.Tensor]) -> None:
    W_enc = tensors["W_enc"]
    W_dec = tensors["W_dec"]
    b_enc = tensors["b_enc"]
    b_dec = tensors["b_dec"]
    if W_enc.ndim != 2 or W_dec.ndim != 2:
        raise ValueError("W_enc and W_dec must be rank-2 tensors")
    if b_enc.ndim != 1 or b_dec.ndim != 1:
        raise ValueError("b_enc and b_dec must be rank-1 tensors")
    d_sae, d_model = W_enc.shape
    if W_dec.shape != (d_model, d_sae):
        raise ValueError("W_dec must have shape [d_model, d_sae]")
    if b_enc.shape != (d_sae,):
        raise ValueError("b_enc must have shape [d_sae]")
    if b_dec.shape != (d_model,):
        raise ValueError("b_dec must have shape [d_model]")
