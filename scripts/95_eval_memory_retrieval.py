#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from qtrm_mm.config import load_config
from qtrm_mm.eval.memory_retrieval import (
    answer_hit,
    build_case_prompt,
    case_task_family,
    expected_unknown_case,
    filter_results_for_case,
    expand_linked_evidence_results,
    load_cases,
    select_evidence_results,
    summarize_records,
    target_retrieval_stats,
    target_retrieved,
)
from qtrm_mm.qtrm_model import QTRMMultimodalModel
from qtrm_mm.qwen_donor import QwenDonorAdapter


DEFAULT_MODES = [
    "donor_only_with_evidence",
    "qtrm_residual_with_evidence",
    "qtrm_workspace_off_with_evidence",
    "qtrm_core_off_with_evidence",
    "qtrm_coda_off_with_evidence",
    "qtrm_residual_head_off_with_evidence",
    "qtrm_donor_hidden_off_with_evidence",
    "qtrm_workspace_only_with_evidence",
    "donor_only_no_evidence",
    "qtrm_residual_no_evidence",
]


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Probe whether QTRM residual generation uses provided MemoryOS evidence."
    )
    ap.add_argument("--config", default="configs/qwen35_2b_4090_donor_residual_s010_1000.yaml")
    ap.add_argument("--checkpoint", default="runs/qwen35_2b_4090_donor_residual_s010_1000/last.pt")
    ap.add_argument("--cases", default="data/eval/memory_retrieval_probe.jsonl")
    ap.add_argument("--mode", action="append", default=None, help="Mode to run. Can be repeated.")
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--memory-max-chars", type=int, default=2000)
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument(
        "--evidence-mode",
        default="target",
        choices=["target", "all", "lexical", "memoryos"],
        help=(
            "target=gold evidence only, all=target+distractors in file order, "
            "lexical=rank target+distractors by query overlap, memoryos=retrieve from --memory-index."
        ),
    )
    ap.add_argument("--retrieval-top-k", type=int, default=3)
    ap.add_argument(
        "--memory-link-expansion",
        type=int,
        default=0,
        help="Append up to N case-scoped records named by already selected evidence.",
    )
    ap.add_argument("--memory-index", default=None)
    ap.add_argument("--memory-model-id", default=None)
    ap.add_argument("--memory-backend", default=None)
    ap.add_argument("--hnsw-ef-search", type=int, default=None)
    ap.add_argument("--retrieve-top-n", type=int, default=None)
    ap.add_argument("--rerank-backend", default="none", choices=["none", "lexical", "cross_encoder"])
    ap.add_argument("--reranker-model-id", default="Qwen/Qwen3-Reranker-0.6B")
    ap.add_argument("--reranker-device", default=None)
    ap.add_argument(
        "--memoryos-global",
        action="store_true",
        help="Do not case-filter MemoryOS retrieval results. Useful for real global-memory tests.",
    )
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--qtrm-logits-scale", type=float, default=None)
    ap.add_argument("--donor-logits-scale", type=float, default=1.0)
    ap.add_argument(
        "--core-halt-mode",
        default="config",
        choices=["config", "enabled", "disabled"],
        help=(
            "Control recursive-core early halt during eval. config preserves the "
            "checkpoint config default, enabled forces early halt, disabled forces full depth."
        ),
    )
    ap.add_argument("--no-logit-shift", action="store_true")
    ap.add_argument("--jsonl-out", default="runs/eval/memory_retrieval_probe.jsonl")
    ap.add_argument("--print-completions", action="store_true")
    return ap


def resolve_qtrm_scale(config_scale: float, override: float | None) -> float:
    return float(config_scale if override is None else override)


def select_device(cfg_device: str, requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return "cuda"
    return "cuda" if torch.cuda.is_available() and cfg_device in {"auto", "cuda"} else "cpu"


def load_qtrm(config_path: str, checkpoint_path: str, device: str) -> QTRMMultimodalModel:
    cfg = load_config(config_path)
    model = QTRMMultimodalModel(cfg.model)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state.get("model", state), strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)}")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)}")
    return model.to(device).eval()


def prepare_inputs(tokenizer, text: str, max_length: int, device: str) -> dict[str, torch.Tensor]:
    original_side = getattr(tokenizer, "truncation_side", "right")
    tokenizer.truncation_side = "left"
    try:
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
            add_special_tokens=True,
        )
    finally:
        tokenizer.truncation_side = original_side
    return {k: v.to(device) for k, v in enc.items()}


@torch.no_grad()
def donor_kwargs(
    donor: QwenDonorAdapter,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: str,
) -> dict[str, torch.Tensor]:
    encoded = donor.encode_inputs(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_logits=True,
    )
    out: dict[str, torch.Tensor] = {
        "text_states": encoded["text_states"].to(device),
        "donor_logits": encoded["logits"].to(device),
    }
    if encoded.get("visual_features") is not None:
        out["visual_features"] = encoded["visual_features"].to(device)
    return out


def mode_settings(mode: str, *, qtrm_scale: float, donor_scale: float) -> tuple[bool, float, float]:
    if mode.endswith("_with_evidence"):
        include_evidence = True
    elif mode.endswith("_no_evidence"):
        include_evidence = False
    else:
        raise ValueError(f"mode must end with _with_evidence or _no_evidence: {mode}")

    if mode.startswith("donor_only_"):
        return include_evidence, 0.0, donor_scale
    if (
        mode.startswith("qtrm_residual_")
        or mode.startswith("qtrm_workspace_off_")
        or mode.startswith("qtrm_core_off_")
        or mode.startswith("qtrm_coda_off_")
        or mode.startswith("qtrm_residual_head_off_")
        or mode.startswith("qtrm_donor_hidden_off_")
        or mode.startswith("qtrm_workspace_only_")
    ):
        return include_evidence, qtrm_scale, donor_scale
    raise ValueError(f"unknown mode: {mode}")


def mode_forward_kwargs(mode: str, *, core_halt_mode: str = "config") -> dict[str, bool]:
    kwargs = {
        "disable_workspace": mode.startswith("qtrm_workspace_off_"),
        "disable_core": mode.startswith("qtrm_core_off_"),
    }
    if (
        mode.startswith("qtrm_coda_off_")
        or mode.startswith("qtrm_residual_head_off_")
        or mode.startswith("qtrm_donor_hidden_off_")
        or mode.startswith("qtrm_workspace_only_")
    ):
        kwargs.update(
            {
                "disable_coda": mode.startswith("qtrm_coda_off_"),
                "disable_qtrm_residual": mode.startswith("qtrm_residual_head_off_"),
                "disable_donor_context": mode.startswith("qtrm_donor_hidden_off_"),
                "workspace_only_context": mode.startswith("qtrm_workspace_only_"),
            }
        )
    if core_halt_mode == "enabled":
        kwargs["enable_core_halt"] = True
    elif core_halt_mode == "disabled":
        kwargs["enable_core_halt"] = False
    elif core_halt_mode != "config":
        raise ValueError(f"unknown core_halt_mode: {core_halt_mode}")
    return kwargs


def core_halt_telemetry(outputs: dict[str, torch.Tensor], *, core_halt_mode: str) -> dict[str, Any]:
    q_halt = outputs.get("core_q_halt_logits")
    q_continue = outputs.get("core_q_continue_logits")
    core_steps = outputs.get("core_steps")
    core_halted = outputs.get("core_halted")

    record: dict[str, Any] = {
        "mode": core_halt_mode,
        "core_steps": None,
        "core_halted": None,
        "q_halt_steps": 0,
        "q_halt_last_mean": None,
        "q_continue_steps": 0,
        "q_continue_last_mean": None,
    }
    if core_steps is not None:
        record["core_steps"] = core_steps.detach().cpu().tolist()
    if core_halted is not None:
        record["core_halted"] = core_halted.detach().cpu().tolist()
    if q_halt is not None and q_halt.numel() > 0:
        record["q_halt_steps"] = int(q_halt.shape[1]) if q_halt.ndim >= 2 else int(q_halt.numel())
        q_halt_last = q_halt[:, -1] if q_halt.ndim >= 2 else q_halt[-1:]
        record["q_halt_last_mean"] = float(q_halt_last.float().mean().detach().cpu().item())
    if q_continue is not None and q_continue.numel() > 0:
        record["q_continue_steps"] = int(q_continue.shape[1]) if q_continue.ndim >= 2 else int(q_continue.numel())
        q_continue_last = q_continue[:, -1] if q_continue.ndim >= 2 else q_continue[-1:]
        record["q_continue_last_mean"] = float(q_continue_last.float().mean().detach().cpu().item())
    return record


@torch.no_grad()
def greedy_completion(
    model: QTRMMultimodalModel,
    donor: QwenDonorAdapter,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    device: str,
    max_new_tokens: int,
    forward_kwargs: dict[str, bool] | None = None,
) -> tuple[str, str, list[int]]:
    generated = input_ids[0].detach().cpu().tolist()
    prompt_len = len(generated)
    forward_kwargs = forward_kwargs or {}

    for _ in range(max_new_tokens):
        cur_ids = torch.tensor([generated], dtype=torch.long, device=device)
        cur_mask = torch.ones_like(cur_ids)
        extra = donor_kwargs(donor, cur_ids, cur_mask, device)
        with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.bfloat16):
            outputs = model(cur_ids, attention_mask=cur_mask, **extra, **forward_kwargs)
        next_id = int(outputs["logits"][0, -1].float().argmax(dim=-1).detach().cpu().item())
        if tokenizer.eos_token_id is not None and next_id == tokenizer.eos_token_id:
            break
        generated.append(next_id)

    completion_ids = generated[prompt_len:]
    completion = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    full_text = tokenizer.decode(generated, skip_special_tokens=True)
    return completion, full_text, completion_ids


@torch.no_grad()
def first_step_logit_shift(
    model: QTRMMultimodalModel,
    donor: QwenDonorAdapter,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    device: str,
    qtrm_scale: float,
    donor_scale: float,
    top_k: int = 5,
    forward_kwargs: dict[str, bool] | None = None,
) -> dict[str, Any]:
    original_qtrm_scale = float(model.cfg.qtrm_logits_scale)
    original_donor_scale = float(model.cfg.donor_logits_scale)
    extra = donor_kwargs(donor, input_ids, attention_mask, device)
    forward_kwargs = forward_kwargs or {}

    try:
        model.cfg.donor_logits_scale = donor_scale
        model.cfg.qtrm_logits_scale = 0.0
        with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.bfloat16):
            donor_only = model(
                input_ids,
                attention_mask=attention_mask,
                **extra,
                **forward_kwargs,
            )["logits"][0, -1].float()

        model.cfg.qtrm_logits_scale = qtrm_scale
        with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.bfloat16):
            residual = model(
                input_ids,
                attention_mask=attention_mask,
                **extra,
                **forward_kwargs,
            )["logits"][0, -1].float()
    finally:
        model.cfg.qtrm_logits_scale = original_qtrm_scale
        model.cfg.donor_logits_scale = original_donor_scale

    delta = residual - donor_only
    k = min(top_k, delta.numel())
    top_abs = torch.topk(delta.abs(), k=k)
    donor_top = int(donor_only.argmax(dim=-1).detach().cpu().item())
    residual_top = int(residual.argmax(dim=-1).detach().cpu().item())
    return {
        "argmax_changed": donor_top != residual_top,
        "donor_top_id": donor_top,
        "donor_top_token": tokenizer.decode([donor_top], skip_special_tokens=False),
        "residual_top_id": residual_top,
        "residual_top_token": tokenizer.decode([residual_top], skip_special_tokens=False),
        "max_abs_delta": float(delta.abs().max().detach().cpu().item()),
        "mean_abs_delta": float(delta.abs().mean().detach().cpu().item()),
        "l2_delta": float(torch.linalg.vector_norm(delta).detach().cpu().item()),
        "top_abs_delta": [
            {
                "token_id": int(idx.detach().cpu().item()),
                "token": tokenizer.decode([int(idx.detach().cpu().item())], skip_special_tokens=False),
                "delta": float(delta[int(idx)].detach().cpu().item()),
            }
            for idx in top_abs.indices
        ],
    }


@torch.no_grad()
def prompt_core_halt_telemetry(
    model: QTRMMultimodalModel,
    donor: QwenDonorAdapter,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    device: str,
    forward_kwargs: dict[str, bool] | None = None,
    core_halt_mode: str = "config",
) -> dict[str, Any]:
    extra = donor_kwargs(donor, input_ids, attention_mask, device)
    forward_kwargs = forward_kwargs or {}
    with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.bfloat16):
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            **extra,
            **forward_kwargs,
        )
    return core_halt_telemetry(outputs, core_halt_mode=core_halt_mode)


def evaluate_case(
    *,
    case: dict[str, Any],
    mode: str,
    model: QTRMMultimodalModel,
    donor: QwenDonorAdapter,
    tokenizer,
    device: str,
    max_length: int,
    max_new_tokens: int,
    memory_max_chars: int,
    evidence_mode: str,
    retrieval_top_k: int,
    memory_link_expansion: int,
    memory_index: str | None,
    memory_model_id: str | None,
    memory_backend: str | None,
    hnsw_ef_search: int | None,
    retrieve_top_n: int | None,
    rerank_backend: str,
    reranker_model_id: str,
    reranker_device: str | None,
    memoryos_case_filter: bool,
    base_qtrm_scale: float,
    donor_logits_scale: float,
    measure_logit_shift: bool,
    core_halt_mode: str,
) -> dict[str, Any]:
    include_evidence, qtrm_scale, donor_scale = mode_settings(
        mode,
        qtrm_scale=base_qtrm_scale,
        donor_scale=donor_logits_scale,
    )
    forward_kwargs = mode_forward_kwargs(mode, core_halt_mode=core_halt_mode)
    model.cfg.qtrm_logits_scale = qtrm_scale
    model.cfg.donor_logits_scale = donor_scale

    evidence_results = []
    if include_evidence:
        if evidence_mode == "memoryos":
            if not memory_index:
                raise ValueError("--memory-index is required when --evidence-mode memoryos")
            from qtrm_mm.memoryos.retrieve import retrieve

            raw_results = retrieve(
                memory_index,
                str(case.get("question", "")),
                top_k=retrieve_top_n or max(retrieval_top_k * 8, retrieval_top_k),
                model_id=memory_model_id,
                backend=memory_backend,
                hnsw_ef_search=hnsw_ef_search,
                rerank_backend=rerank_backend,
                reranker_model_id=reranker_model_id,
                rerank_top_k=(retrieve_top_n or max(retrieval_top_k * 8, retrieval_top_k))
                if memoryos_case_filter
                else retrieval_top_k,
                reranker_device=reranker_device,
            )
            if memoryos_case_filter:
                case_candidates = filter_results_for_case(
                    raw_results,
                    case_id=str(case.get("id", "")),
                    top_k=retrieve_top_n or max(retrieval_top_k * 8, retrieval_top_k),
                )
                evidence_results = case_candidates[:retrieval_top_k]
                evidence_results = expand_linked_evidence_results(
                    evidence_results,
                    case_candidates,
                    max_extra=memory_link_expansion,
                )
            else:
                evidence_results = raw_results[:retrieval_top_k]
                evidence_results = expand_linked_evidence_results(
                    evidence_results,
                    raw_results,
                    max_extra=memory_link_expansion,
                )
        else:
            evidence_results = select_evidence_results(
                case,
                evidence_mode=evidence_mode,
                top_k=retrieval_top_k,
            )
    prompt = build_case_prompt(
        case,
        include_evidence=include_evidence,
        evidence_results=evidence_results,
        max_evidence_chars=memory_max_chars,
    )
    inputs = prepare_inputs(tokenizer, prompt, max_length, device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
    logit_shift = first_step_logit_shift(
        model,
        donor,
        tokenizer,
        input_ids,
        attention_mask,
        device=device,
        qtrm_scale=base_qtrm_scale,
        donor_scale=donor_logits_scale,
        forward_kwargs=forward_kwargs,
    ) if measure_logit_shift else None
    core_halt = prompt_core_halt_telemetry(
        model,
        donor,
        input_ids,
        attention_mask,
        device=device,
        forward_kwargs=forward_kwargs,
        core_halt_mode=core_halt_mode,
    )
    completion, full_text, completion_ids = greedy_completion(
        model,
        donor,
        tokenizer,
        input_ids,
        attention_mask,
        device=device,
        max_new_tokens=max_new_tokens,
        forward_kwargs=forward_kwargs,
    )
    hit = answer_hit(completion, case["answer_aliases"])
    retrieval_stats = (
        target_retrieval_stats(case, evidence_results)
        if include_evidence
        else {
            "target_count": 0,
            "retrieved_target_count": 0,
            "retrieved_target": False,
            "all_targets_retrieved": False,
            "target_recall": 0.0,
        }
    )
    return {
        "id": case["id"],
        "category": case.get("category", "uncategorized"),
        "task_family": case_task_family(case),
        "expected_unknown": expected_unknown_case(case),
        "mode": mode,
        "hit": hit,
        "answer_aliases": case["answer_aliases"],
        "completion": completion,
        "completion_token_count": len(completion_ids),
        "prompt_token_count": int(input_ids.shape[1]),
        "include_evidence": include_evidence,
        "evidence_mode": evidence_mode if include_evidence else "none",
        "retrieved_target": retrieval_stats["retrieved_target"],
        "target_count": retrieval_stats["target_count"],
        "retrieved_target_count": retrieval_stats["retrieved_target_count"],
        "all_targets_retrieved": retrieval_stats["all_targets_retrieved"],
        "target_recall": retrieval_stats["target_recall"],
        "retrieved_roles": [rec.get("evidence_role", "unknown") for _, rec in evidence_results],
        "retrieved_sources": [rec.get("source", "?") for _, rec in evidence_results],
        "retrieved_rerank_backend": [rec.get("rerank_backend", "none") for _, rec in evidence_results],
        "retrieved_rerank_scores": [rec.get("rerank_score") for _, rec in evidence_results],
        "retrieved_retrieval_scores": [rec.get("retrieval_score", score) for score, rec in evidence_results],
        "qtrm_logits_scale": qtrm_scale,
        "donor_logits_scale": donor_scale,
        "forward_ablation": forward_kwargs,
        "core_halt": core_halt,
        "first_step_logit_shift": logit_shift,
        "full_text": full_text,
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = load_config(args.config)
    if not cfg.donor.model_id:
        raise SystemExit("donor.model_id is required")
    device = select_device(cfg.train.device, args.device)
    max_length = args.max_length or cfg.train.seq_len
    modes = args.mode or DEFAULT_MODES
    cases = load_cases(args.cases)
    if args.max_cases is not None:
        cases = cases[: args.max_cases]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.donor.model_id,
        trust_remote_code=cfg.donor.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("=" * 72)
    print("QTRM Memory Retrieval Probe")
    print(f"config={args.config}")
    print(f"checkpoint={args.checkpoint}")
    print(f"cases={args.cases} count={len(cases)}")
    print(
        f"device={device} max_length={max_length} max_new_tokens={args.max_new_tokens} "
        f"evidence_mode={args.evidence_mode} retrieval_top_k={args.retrieval_top_k} "
        f"retrieve_top_n={args.retrieve_top_n} rerank_backend={args.rerank_backend}"
    )
    print("=" * 72)

    model = load_qtrm(args.config, args.checkpoint, device)
    donor = QwenDonorAdapter(cfg.donor)
    base_qtrm_scale = resolve_qtrm_scale(cfg.model.qtrm_logits_scale, args.qtrm_logits_scale)

    records = []
    for case in cases:
        for mode in modes:
            record = evaluate_case(
                case=case,
                mode=mode,
                model=model,
                donor=donor,
                tokenizer=tokenizer,
                device=device,
                max_length=max_length,
                max_new_tokens=args.max_new_tokens,
                memory_max_chars=args.memory_max_chars,
                evidence_mode=args.evidence_mode,
                retrieval_top_k=args.retrieval_top_k,
                memory_link_expansion=args.memory_link_expansion,
                memory_index=args.memory_index,
                memory_model_id=args.memory_model_id,
                memory_backend=args.memory_backend,
                hnsw_ef_search=args.hnsw_ef_search,
                retrieve_top_n=args.retrieve_top_n,
                rerank_backend=args.rerank_backend,
                reranker_model_id=args.reranker_model_id,
                reranker_device=args.reranker_device,
                memoryos_case_filter=not args.memoryos_global,
                base_qtrm_scale=base_qtrm_scale,
                donor_logits_scale=args.donor_logits_scale,
                measure_logit_shift=not args.no_logit_shift,
                core_halt_mode=args.core_halt_mode,
            )
            records.append(record)
            status = "hit" if record["hit"] else "miss"
            retrieval_status = "retrieved" if record["retrieved_target"] else "no-target"
            shift = record.get("first_step_logit_shift") or {}
            shift_text = f" delta={shift.get('max_abs_delta', 0.0):.3f}" if shift else ""
            core_halt = record.get("core_halt") or {}
            core_steps = core_halt.get("core_steps")
            halt_text = f" core_steps={core_steps}" if core_steps is not None else ""
            print(
                f"{status:4s} {record['mode']:28s} {record['id']:22s} "
                f"{retrieval_status:10s}{shift_text}{halt_text} -> {record['completion']!r}"
            )

    summary = summarize_records(records)
    print("\nSummary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.jsonl_out:
        out_path = Path(args.jsonl_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for record in records:
                if not args.print_completions:
                    record = {k: v for k, v in record.items() if k != "full_text"}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.write(json.dumps({"summary": summary}, ensure_ascii=False) + "\n")
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
