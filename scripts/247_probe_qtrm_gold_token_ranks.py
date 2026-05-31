#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


def _load_raw_eval_module():
    path = Path(__file__).with_name("192_eval_raw_intelligence.py")
    spec = importlib.util.spec_from_file_location("qtrm_raw_eval_192", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Probe gold next-token ranks under QTRM generation prefixes. "
            "This diagnoses whether generation fails because the correct token "
            "is far from the top of the LM distribution or because decoding/"
            "verification is selecting the wrong near-top option."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--base-checkpoint",
        default=None,
        help="Optional full/base checkpoint to load before the checkpoint delta.",
    )
    parser.add_argument("--cases", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-cases", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--skip-leading-whitespace-targets",
        action="store_true",
        help=(
            "Probe the same stripped-answer target sequence used by "
            "--causal-prefix-skip-leading-whitespace-targets in training."
        ),
    )
    parser.add_argument(
        "--append-eos-target",
        action="store_true",
        help="Append tokenizer.eos_token_id to the probed target sequence.",
    )
    parser.add_argument(
        "--mode",
        action="append",
        default=None,
        help="Eval mode. Can be repeated. Defaults to donor/core/depth modes.",
    )
    parser.add_argument("--qtrm-logits-scale", type=float, default=None)
    parser.add_argument("--donor-logits-scale", type=float, default=None)
    return parser


def _gold_answer(case: dict[str, Any]) -> str:
    for key in ("answer", "chosen", "canonical_answer"):
        value = str(case.get(key) or "").strip()
        if value:
            return value
    aliases = case.get("answer_aliases") or []
    for alias in aliases:
        value = str(alias).strip()
        if value:
            return value
    raise ValueError(f"case has no gold answer: {case.get('id')}")


def _target_token_ids(
    tokenizer,
    answer: str,
    *,
    skip_leading_whitespace_targets: bool = False,
    append_eos_target: bool = False,
) -> list[int]:
    if bool(skip_leading_whitespace_targets):
        token_ids = tokenizer.encode(str(answer).strip(), add_special_tokens=False)
    else:
        token_ids = tokenizer.encode(f" {answer}", add_special_tokens=False)
    if not token_ids:
        token_ids = tokenizer.encode(answer, add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"answer has no tokens: {answer!r}")
    out = [int(token_id) for token_id in token_ids]
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if bool(append_eos_target) and eos_token_id is not None:
        eos_id = int(eos_token_id)
        if eos_id >= 0 and (not out or out[-1] != eos_id):
            out.append(eos_id)
    return out


def _first_content_token_index(tokens: list[str]) -> int:
    for index, token in enumerate(tokens):
        if str(token).strip():
            return index
    return 0


def _target_rank_stats(logits, target_id: int) -> dict[str, Any]:
    import torch

    target_logit = logits[int(target_id)]
    strict_rank = int((logits > target_logit).sum().detach().cpu().item()) + 1
    tie_mask = torch.isclose(
        logits,
        target_logit,
        rtol=0.0,
        atol=1e-7,
    )
    tie_count = int(tie_mask.sum().detach().cpu().item())
    max_logit = logits.max()
    top_tie_count = int(
        torch.isclose(logits, max_logit, rtol=0.0, atol=1e-7)
        .sum()
        .detach()
        .cpu()
        .item()
    )
    return {
        "strict_rank": strict_rank,
        "tie_count": tie_count,
        "top_tie_count": top_tie_count,
        "unique_top1": bool(strict_rank == 1 and tie_count == 1),
        "target_logit": float(target_logit.detach().cpu().item()),
        "max_logit": float(max_logit.detach().cpu().item()),
        "target_minus_top_logit": float(
            (target_logit - max_logit).detach().cpu().item()
        ),
    }


def _top_tokens(tokenizer, logits, *, top_k: int) -> list[dict[str, Any]]:
    import torch

    k = min(max(1, int(top_k)), int(logits.shape[-1]))
    values, indices = torch.topk(logits, k=k)
    out = []
    for value, token_id in zip(values.detach().cpu().tolist(), indices.detach().cpu().tolist()):
        out.append(
            {
                "token_id": int(token_id),
                "token": tokenizer.decode([int(token_id)]),
                "logit": float(value),
            }
        )
    return out


def _select_position_top_tokens(
    top_tokens_by_position: list[list[dict[str, Any]]],
    *,
    content_index: int,
) -> dict[str, list[dict[str, Any]] | None]:
    first_top_tokens = top_tokens_by_position[0] if top_tokens_by_position else None
    content_first_top_tokens = (
        top_tokens_by_position[int(content_index)]
        if 0 <= int(content_index) < len(top_tokens_by_position)
        else None
    )
    return {
        "first_top_tokens": first_top_tokens,
        "content_first_top_tokens": content_first_top_tokens,
    }


def _model_disable_kwargs_from_runtime(runtime: dict[str, Any]) -> dict[str, bool]:
    return {
        "zero_core_trajectory": bool(runtime.get("zero_core_trajectory", False)),
        "disable_core": bool(runtime.get("disable_core", False)),
        "disable_qtrm_residual": bool(runtime.get("disable_qtrm_residual", False)),
        "disable_qtrm_residual_gate": bool(
            runtime.get("disable_qtrm_residual_gate", False)
        ),
        "disable_transition_state": bool(
            runtime.get("disable_transition_state", False)
        ),
        "disable_token_numeric_source_slots": bool(
            runtime.get("disable_token_numeric_source_slots", False)
        ),
        "disable_core_source_position_binder": bool(
            runtime.get("disable_core_source_position_binder", False)
        ),
        "disable_core_primitive_role_value_executor": bool(
            runtime.get("disable_core_primitive_role_value_executor", False)
        ),
        "disable_core_role_value_answer_bridge": bool(
            runtime.get("disable_core_role_value_answer_bridge", False)
        ),
        "disable_core_role_value_answer_final_binder": bool(
            runtime.get("disable_core_role_value_answer_final_binder", False)
        ),
        "disable_core_role_value_vocab_renderer": bool(
            runtime.get("disable_core_role_value_vocab_renderer", False)
        ),
        "disable_answer_state_loop_recurrent": bool(
            runtime.get("disable_answer_state_loop_recurrent", False)
        ),
        "disable_typed_algorithmic_value_state_answer_bridge": bool(
            runtime.get(
                "disable_typed_algorithmic_value_state_answer_bridge",
                False,
            )
        ),
        "disable_answer_state_loop_selective_context": bool(
            runtime.get("disable_answer_state_loop_selective_context", False)
        ),
        "disable_answer_state_loop_finality_selector": bool(
            runtime.get("disable_answer_state_loop_finality_selector", False)
        ),
        "disable_answer_state_loop_finality_gate": bool(
            runtime.get("disable_answer_state_loop_finality_gate", False)
        ),
        "disable_answer_state_loop_halt_gate": bool(
            runtime.get("disable_answer_state_loop_halt_gate", False)
        ),
        "disable_answer_state_loop_hidden_bridge": bool(
            runtime.get("disable_answer_state_loop_hidden_bridge", False)
        ),
        "disable_answer_state_loop_next_token_decoder": bool(
            runtime.get("disable_answer_state_loop_next_token_decoder", False)
        ),
        "disable_answer_state_loop_free_transformer_latent": bool(
            runtime.get("disable_answer_state_loop_free_transformer_latent", False)
        ),
        "disable_transition_state_joint_answer_bridge": bool(
            runtime.get("disable_transition_state_joint_answer_bridge", False)
        ),
        "disable_transition_state_final_answer_binder": bool(
            runtime.get("disable_transition_state_final_answer_binder", False)
        ),
    }


def _checkpoint_model_state(state: Any) -> Any:
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def _load_checkpoint_stack(model, *, checkpoint: str, base_checkpoint: str | None, device: str) -> None:
    import torch

    if base_checkpoint:
        base_state = torch.load(base_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(_checkpoint_model_state(base_state), strict=False)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(_checkpoint_model_state(state), strict=False)


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoTokenizer

    from wgram_lm.config import load_config
    from wgram_lm.wgram_model import QTRMMultimodalModel
    from wgram_lm.qwen_donor import QwenDonorAdapter

    raw_eval = _load_raw_eval_module()

    cfg = load_config(args.config)
    device = raw_eval._select_device(cfg.train.device, args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.donor.model_id,
        trust_remote_code=cfg.donor.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = QTRMMultimodalModel(cfg.model)
    _load_checkpoint_stack(
        model,
        checkpoint=args.checkpoint,
        base_checkpoint=args.base_checkpoint,
        device=device,
    )
    model = model.to(device).eval()

    donor = QwenDonorAdapter(cfg.donor)
    cases = raw_eval.load_cases(args.cases, max_cases=args.max_cases)
    modes = args.mode or [
        "donor_only_no_evidence",
        "qtrm_core_off_no_evidence",
        "qtrm_core_steps_1_no_evidence",
        "qtrm_core_steps_4_no_evidence",
        "qtrm_core_steps_8_no_evidence",
    ]

    records: list[dict[str, Any]] = []
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f, torch.no_grad():
        for mode in modes:
            runtime = raw_eval.mode_runtime(mode)
            old_outer_steps = int(model.cfg.outer_steps)
            old_qtrm_scale = float(model.cfg.qtrm_logits_scale)
            old_donor_scale = float(model.cfg.donor_logits_scale)
            model.cfg.qtrm_logits_scale = (
                float(runtime["qtrm_logits_scale"])
                if runtime["qtrm_logits_scale"] is not None
                else float(args.qtrm_logits_scale)
                if args.qtrm_logits_scale is not None
                else old_qtrm_scale
            )
            model.cfg.donor_logits_scale = (
                float(runtime["donor_logits_scale"])
                if runtime["donor_logits_scale"] is not None
                else float(args.donor_logits_scale)
                if args.donor_logits_scale is not None
                else old_donor_scale
            )
            if runtime.get("core_steps_override") is not None:
                model.cfg.outer_steps = int(runtime["core_steps_override"])
            try:
                for case in cases:
                    prompt = str(case.get("prompt") or case.get("question") or "")
                    answer = _gold_answer(case)
                    target_ids = _target_token_ids(
                        tokenizer,
                        answer,
                        skip_leading_whitespace_targets=bool(
                            args.skip_leading_whitespace_targets
                        ),
                        append_eos_target=bool(args.append_eos_target),
                    )
                    prompt_inputs = raw_eval._prepare_inputs(
                        tokenizer,
                        prompt,
                        args.max_length,
                        device,
                    )
                    prompt_ids = prompt_inputs["input_ids"][0].detach().cpu().tolist()
                    ranks: list[int] = []
                    tie_counts: list[int] = []
                    top_tie_counts: list[int] = []
                    unique_top1: list[bool] = []
                    target_tokens: list[str] = []
                    target_logits: list[float] = []
                    target_minus_top_logits: list[float] = []
                    top_tokens_by_position: list[list[dict[str, Any]]] = []
                    for pos, target_id in enumerate(target_ids):
                        prefix_ids = prompt_ids + target_ids[:pos]
                        input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
                        attention_mask = torch.ones_like(input_ids)
                        (
                            source_slot_ids,
                            source_slot_token_ids,
                            source_slot_mask,
                        ) = raw_eval._token_numeric_source_slots_for_prompt_prefix(
                            tokenizer,
                            case,
                            prompt,
                            max_length=args.max_length,
                            device=device,
                            enabled=bool(
                                cfg.model.token_numeric_source_slot_embedding_enabled
                            ),
                            value_vocab_size=int(
                                cfg.model.token_numeric_source_slot_vocab_size
                            ),
                            max_slots=int(cfg.model.token_numeric_source_slot_max_slots),
                        )
                        extra = raw_eval._donor_kwargs(
                            donor,
                            input_ids,
                            attention_mask,
                            device,
                            return_logits=bool(model.cfg.donor_logits_scale != 0.0),
                        )
                        with torch.amp.autocast(
                            "cuda",
                            enabled=(device == "cuda"),
                            dtype=torch.bfloat16,
                        ):
                            outputs = model(
                                input_ids,
                                attention_mask=attention_mask,
                                token_numeric_source_slot_ids=source_slot_ids,
                                token_numeric_source_slot_token_ids=source_slot_token_ids,
                                token_numeric_source_slot_mask=source_slot_mask,
                                **extra,
                                **_model_disable_kwargs_from_runtime(runtime),
                                enable_core_halt=raw_eval._runtime_enable_core_halt(runtime),
                            )
                        logits = outputs["logits"][0, -1].float()
                        rank_stats = _target_rank_stats(logits, int(target_id))
                        rank = int(rank_stats["strict_rank"])
                        ranks.append(rank)
                        tie_counts.append(int(rank_stats["tie_count"]))
                        top_tie_counts.append(int(rank_stats["top_tie_count"]))
                        unique_top1.append(bool(rank_stats["unique_top1"]))
                        target_logits.append(float(rank_stats["target_logit"]))
                        target_minus_top_logits.append(
                            float(rank_stats["target_minus_top_logit"])
                        )
                        target_tokens.append(tokenizer.decode([int(target_id)]))
                        top_tokens_by_position.append(
                            _top_tokens(tokenizer, logits, top_k=args.top_k)
                        )
                    content_index = _first_content_token_index(target_tokens)
                    selected_top_tokens = _select_position_top_tokens(
                        top_tokens_by_position,
                        content_index=content_index,
                    )
                    record = {
                        "id": case.get("id"),
                        "mode": mode,
                        "answer": answer,
                        "target_token_ids": target_ids,
                        "target_tokens": target_tokens,
                        "ranks": ranks,
                        "target_tie_counts": tie_counts,
                        "top_tie_counts": top_tie_counts,
                        "target_logits": target_logits,
                        "target_minus_top_logits": target_minus_top_logits,
                        "unique_top1": unique_top1,
                        "first_rank": ranks[0],
                        "first_unique_top1": unique_top1[0],
                        "first_target_logit": target_logits[0],
                        "first_target_minus_top_logit": target_minus_top_logits[0],
                        "content_first_index": content_index,
                        "content_first_rank": ranks[content_index],
                        "content_first_unique_top1": unique_top1[content_index],
                        "content_first_target_logit": target_logits[content_index],
                        "content_first_target_minus_top_logit": target_minus_top_logits[
                            content_index
                        ],
                        "max_rank": max(ranks),
                        "all_rank_1": all(rank == 1 for rank in ranks),
                        "all_unique_top1": all(unique_top1),
                        "all_rank_le_5": all(rank <= 5 for rank in ranks),
                        "all_rank_le_10": all(rank <= 10 for rank in ranks),
                        **selected_top_tokens,
                    }
                    records.append(record)
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
            finally:
                model.cfg.outer_steps = old_outer_steps
                model.cfg.qtrm_logits_scale = old_qtrm_scale
                model.cfg.donor_logits_scale = old_donor_scale
    return records


def main() -> None:
    args = build_arg_parser().parse_args()
    records = run(args)
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_mode.setdefault(str(record["mode"]), []).append(record)
    print(f"wrote {len(records)} records to {args.out}")
    for mode, rows in sorted(by_mode.items()):
        first_at_1 = sum(1 for row in rows if int(row["first_rank"]) == 1)
        first_unique_at_1 = sum(1 for row in rows if bool(row["first_unique_top1"]))
        content_first_at_1 = sum(1 for row in rows if int(row["content_first_rank"]) == 1)
        content_first_unique_at_1 = sum(
            1 for row in rows if bool(row["content_first_unique_top1"])
        )
        all_at_1 = sum(1 for row in rows if bool(row["all_rank_1"]))
        all_unique_at_1 = sum(1 for row in rows if bool(row["all_unique_top1"]))
        all_le_10 = sum(1 for row in rows if bool(row["all_rank_le_10"]))
        first_mean = sum(float(row["first_rank"]) for row in rows) / max(1, len(rows))
        content_first_mean = sum(float(row["content_first_rank"]) for row in rows) / max(
            1, len(rows)
        )
        max_mean = sum(float(row["max_rank"]) for row in rows) / max(1, len(rows))
        print(
            f"{mode}: first@1={first_at_1}/{len(rows)} "
            f"first_unique@1={first_unique_at_1}/{len(rows)} "
            f"content_first@1={content_first_at_1}/{len(rows)} "
            f"content_first_unique@1={content_first_unique_at_1}/{len(rows)} "
            f"all@1={all_at_1}/{len(rows)} "
            f"all_unique@1={all_unique_at_1}/{len(rows)} "
            f"all<=10={all_le_10}/{len(rows)} "
            f"first_rank_mean={first_mean:.2f} "
            f"content_first_rank_mean={content_first_mean:.2f} "
            f"max_rank_mean={max_mean:.2f}"
        )


if __name__ == "__main__":
    main()
