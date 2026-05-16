#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

KNOWN_CHECKPOINT_SHA256 = {
    "local_eval/research_gate_runner/primitive_field_heads_delta_codec_s90_lr5e4_seed11/last.pt": (
        "9a9204a9b01001713772294afcf30ae5753b0e3cd3877adabb83918caf52747d"
    ),
}


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
            "Probe whether QTRM source-position logits point to the source "
            "slots that a pointer/copy renderer must copy. This is a "
            "diagnostic for the L4 lexicalization bottleneck, not a hidden "
            "answer solver."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-checkpoint", default=None)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-cases", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--token-numeric-source-slots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mirror the L4 source-pointer runner by feeding numeric source slots.",
    )
    parser.add_argument("--token-numeric-source-slot-vocab-size", type=int, default=128)
    parser.add_argument("--token-numeric-source-slot-max-slots", type=int, default=5)
    parser.add_argument("--token-numeric-source-slot-gate-min", type=float, default=1.0)
    parser.add_argument(
        "--token-numeric-source-slot-predicate-feedback",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--token-numeric-source-slot-predicate-gate-min",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--core-source-position-binder",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--core-source-position-binder-gate-min", type=float, default=1.0)
    parser.add_argument(
        "--core-source-position-binder-state-gate-min",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--core-source-position-binder-state-st",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--core-source-position-binder-source-slots-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Probe source-slot pointer logits instead of arbitrary prompt positions.",
    )
    parser.add_argument(
        "--core-source-position-binder-raw-source-slots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use raw source-slot embeddings for the source-position binder.",
    )
    parser.add_argument(
        "--mode",
        action="append",
        default=None,
        help="Eval mode. Can be repeated. Defaults to full/source-slot-off/binder-off.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    return parser


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_jsonl(path: str | Path, *, max_cases: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if int(max_cases) > 0 and len(rows) >= int(max_cases):
                break
    return rows


def resolve_checkpoint_path(path: str | Path, *, root: Path) -> Path:
    checkpoint = Path(path)
    if checkpoint.is_absolute():
        return checkpoint
    return root / checkpoint


def missing_checkpoint_base_chain(
    checkpoint: str | Path,
    *,
    root: Path,
    load_state=None,
) -> list[str]:
    return [
        str(issue["path"])
        for issue in checkpoint_base_chain_issues(
            checkpoint,
            root=root,
            load_state=load_state,
        )
        if issue["issue"] == "missing"
    ]


def _known_checkpoint_sha256(path: Path, *, root: Path) -> str:
    candidates = [str(path)]
    try:
        candidates.append(str(path.relative_to(root)))
    except ValueError:
        pass
    for candidate in candidates:
        expected = KNOWN_CHECKPOINT_SHA256.get(candidate)
        if expected:
            return expected
    return ""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_base_chain_issues(
    checkpoint: str | Path,
    *,
    root: Path,
    load_state=None,
) -> list[dict[str, str]]:
    if load_state is None:
        import torch

        def load_state(path: Path):
            return torch.load(path, map_location="cpu", weights_only=False)

    issues: list[dict[str, str]] = []
    seen: set[str] = set()
    current = Path(checkpoint)
    while True:
        resolved = resolve_checkpoint_path(current, root=root)
        resolved_key = str(resolved)
        if resolved_key in seen:
            break
        seen.add(resolved_key)
        if not resolved.exists():
            issues.append({"issue": "missing", "path": resolved_key})
            break
        expected_sha256 = _known_checkpoint_sha256(resolved, root=root)
        if expected_sha256:
            actual_sha256 = _sha256_file(resolved)
            if actual_sha256 != expected_sha256:
                issues.append(
                    {
                        "issue": "sha256_mismatch",
                        "path": resolved_key,
                        "expected_sha256": expected_sha256,
                        "actual_sha256": actual_sha256,
                    }
                )
                break
        state = load_state(resolved)
        base_checkpoint = ""
        if isinstance(state, dict):
            base_checkpoint = str(state.get("base_checkpoint") or "").strip()
        if not base_checkpoint:
            break
        current = Path(base_checkpoint)
    return issues


def requested_checkpoint_chain_issues(args: argparse.Namespace, *, root: Path) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    for checkpoint in (args.base_checkpoint, args.checkpoint):
        if not checkpoint:
            continue
        for issue in checkpoint_base_chain_issues(checkpoint, root=root):
            if issue not in issues:
                issues.append(issue)
    return issues


def checkpoint_chain_missing_report(
    args: argparse.Namespace,
    *,
    issues: list[dict[str, str]],
) -> dict[str, Any]:
    missing = [issue["path"] for issue in issues if issue["issue"] == "missing"]
    decision = (
        "checkpoint_chain_missing"
        if missing
        else "checkpoint_chain_sha256_mismatch"
    )
    report = {
        "decision": decision,
        "accepted": False,
        "target_level": "L2 diagnostic: QTRM source-position logits replace oracle positions",
        "major_bottleneck": "checkpoint hygiene: trainable-delta base chain missing",
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "base_checkpoint": args.base_checkpoint,
        "cases": str(args.cases),
        "missing_base_checkpoints": missing,
        "checkpoint_chain_issues": issues,
        "next_action": (
            "restore the exact checkpoint by sha256 or materialize a "
            "self-contained checkpoint after replayed gate acceptance"
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def normalize_answer(value: Any) -> str:
    return str(value or "").strip()


def gold_answer(row: dict[str, Any]) -> str:
    for key in ("answer", "chosen", "canonical_answer"):
        value = normalize_answer(row.get(key))
        if value:
            return value
    for alias in row.get("answer_aliases") or []:
        value = normalize_answer(alias)
        if value:
            return value
    return ""


def oracle_source_positions(row: dict[str, Any]) -> list[int]:
    values = [int(value) for value in row.get("input_list") or []]
    return [index for index, value in enumerate(values) if value % 2 == 0]


def oracle_source_classes(row: dict[str, Any]) -> list[int]:
    raw = row.get("source_even_position_signature")
    if raw:
        return [int(value) for value in raw]
    return [int(position) + 1 for position in oracle_source_positions(row)]


def copy_answer_from_positions(row: dict[str, Any], positions: list[int]) -> str:
    values = [int(value) for value in row.get("input_list") or []]
    copied = [
        values[int(position)]
        for position in positions
        if 0 <= int(position) < len(values)
    ]
    if not copied:
        return "EMPTY"
    return ",".join(str(value) for value in copied)


def copy_answer_from_source_classes(row: dict[str, Any], classes: list[int]) -> str:
    positions = [int(value) - 1 for value in classes if int(value) > 0]
    return copy_answer_from_positions(row, positions)


def select_depth_logits(logits, *, batch_index: int = 0, depth_index: int = -1):
    if logits.ndim != 4:
        raise ValueError("source-position logits must have shape [batch, depth, roles, positions]")
    batch = int(batch_index)
    depth = int(depth_index)
    if depth < 0:
        depth = int(logits.shape[1]) + depth
    return logits[batch, depth]


def source_copy_answer_role_offset(role_count: int) -> int:
    roles = max(0, int(role_count))
    answer_role_count = max(1, (roles - 2) // 2)
    if roles >= answer_role_count * 2 + 2:
        return answer_role_count
    return 0


def select_renderer_copy_logits(
    *,
    source_position_prompt_logits,
    core_role_value_state_logits,
    core_primitive_role_value_state_logits=None,
    prefer_primitive: bool = False,
):
    if (
        bool(prefer_primitive)
        and source_position_prompt_logits is not None
        and source_position_prompt_logits.ndim == 4
        and source_position_prompt_logits.numel() != 0
        and core_primitive_role_value_state_logits is not None
        and core_primitive_role_value_state_logits.ndim == 4
        and int(core_primitive_role_value_state_logits.shape[1]) > 0
        and int(core_primitive_role_value_state_logits.shape[0])
        == int(source_position_prompt_logits.shape[0])
        and int(core_primitive_role_value_state_logits.shape[2])
        == int(source_position_prompt_logits.shape[2])
        and int(core_primitive_role_value_state_logits.shape[-1]) > 0
    ):
        return (
            core_primitive_role_value_state_logits[:, -1:, :, :],
            "core_primitive_role_value_state_logits",
        )
    if (
        source_position_prompt_logits is not None
        and source_position_prompt_logits.ndim == 4
        and source_position_prompt_logits.numel() != 0
        and core_role_value_state_logits is not None
        and core_role_value_state_logits.ndim == 4
        and int(core_role_value_state_logits.shape[1]) > 0
        and int(core_role_value_state_logits.shape[0])
        == int(source_position_prompt_logits.shape[0])
        and int(core_role_value_state_logits.shape[2])
        == int(source_position_prompt_logits.shape[2])
        and int(core_role_value_state_logits.shape[-1]) > 0
    ):
        return core_role_value_state_logits[:, -1:, :, :], "core_role_value_state_logits"
    return source_position_prompt_logits, "source_position_prompt_logits"


def apply_source_pointer_defaults(cfg, args: argparse.Namespace) -> None:
    cfg.model.token_numeric_source_slot_embedding_enabled = bool(
        args.token_numeric_source_slots
    )
    cfg.model.token_numeric_source_slot_vocab_size = int(
        args.token_numeric_source_slot_vocab_size
    )
    cfg.model.token_numeric_source_slot_max_slots = int(
        args.token_numeric_source_slot_max_slots
    )
    cfg.model.token_numeric_source_slot_gate_min = float(
        args.token_numeric_source_slot_gate_min
    )
    cfg.model.token_numeric_source_slot_predicate_feedback_enabled = bool(
        args.token_numeric_source_slot_predicate_feedback
    )
    cfg.model.token_numeric_source_slot_predicate_gate_min = float(
        args.token_numeric_source_slot_predicate_gate_min
    )
    cfg.model.core_source_position_binder_enabled = bool(
        args.core_source_position_binder
    )
    cfg.model.core_source_position_binder_gate_min = float(
        args.core_source_position_binder_gate_min
    )
    cfg.model.core_source_position_binder_state_gate_min = float(
        args.core_source_position_binder_state_gate_min
    )
    cfg.model.core_source_position_binder_state_straight_through = bool(
        args.core_source_position_binder_state_st
    )
    cfg.model.core_source_position_binder_source_slots_only = bool(
        args.core_source_position_binder_source_slots_only
    )
    cfg.model.core_source_position_binder_raw_source_slots_enabled = bool(
        args.core_source_position_binder_raw_source_slots
    )


def _position_rank_stats(role_logits, target_position: int) -> dict[str, Any]:
    import torch

    target = int(target_position)
    if target < 0 or target >= int(role_logits.shape[-1]):
        return {
            "rank": None,
            "tie_count": 0,
            "unique_top1": False,
            "target_logit": None,
            "max_logit": float(role_logits.max().detach().cpu().item())
            if role_logits.numel()
            else None,
            "target_minus_top_logit": None,
        }
    target_logit = role_logits[target]
    strict_rank = int((role_logits > target_logit).sum().detach().cpu().item()) + 1
    tie_count = int(
        torch.isclose(role_logits, target_logit, rtol=0.0, atol=1e-7)
        .sum()
        .detach()
        .cpu()
        .item()
    )
    max_logit = role_logits.max()
    return {
        "rank": strict_rank,
        "tie_count": tie_count,
        "unique_top1": bool(strict_rank == 1 and tie_count == 1),
        "target_logit": float(target_logit.detach().cpu().item()),
        "max_logit": float(max_logit.detach().cpu().item()),
        "target_minus_top_logit": float(
            (target_logit - max_logit).detach().cpu().item()
        ),
    }


def _top_positions(role_logits, *, top_k: int) -> list[dict[str, Any]]:
    import torch

    if role_logits.numel() == 0:
        return []
    k = min(max(1, int(top_k)), int(role_logits.shape[-1]))
    values, indices = torch.topk(role_logits, k=k)
    return [
        {"position": int(index), "logit": float(value)}
        for value, index in zip(
            values.detach().cpu().tolist(),
            indices.detach().cpu().tolist(),
        )
    ]


def summarize_pointer_logits(
    role_position_logits,
    *,
    oracle_positions: list[int],
    role_offset: int = 0,
    valid_position_count: int | None = None,
    top_k: int = 3,
) -> dict[str, Any]:
    role_count = int(role_position_logits.shape[0]) if role_position_logits.ndim == 2 else 0
    position_count = (
        int(role_position_logits.shape[1]) if role_position_logits.ndim == 2 else 0
    )
    valid_count = (
        position_count
        if valid_position_count is None
        else max(0, min(int(valid_position_count), position_count))
    )
    offset = max(0, min(int(role_offset), role_count))
    selected_count = min(len(oracle_positions), max(0, role_count - offset))
    predicted_positions: list[int] = []
    valid_predicted_positions: list[int] = []
    roles: list[dict[str, Any]] = []
    selected_correct = 0
    valid_selected_correct = 0
    invalid_top_position_count = 0
    for role_index in range(selected_count):
        actual_role_index = offset + role_index
        role_logits = role_position_logits[actual_role_index].float()
        predicted = (
            int(role_logits.argmax().detach().cpu().item())
            if position_count > 0
            else -1
        )
        valid_logits = role_logits[:valid_count]
        valid_predicted = (
            int(valid_logits.argmax().detach().cpu().item())
            if valid_count > 0
            else -1
        )
        target = int(oracle_positions[role_index])
        stats = _position_rank_stats(role_logits, target)
        valid_stats = _position_rank_stats(valid_logits, target)
        correct = predicted == target
        valid_correct = valid_predicted == target
        selected_correct += int(correct)
        valid_selected_correct += int(valid_correct)
        invalid_top_position_count += int(predicted >= valid_count)
        predicted_positions.append(predicted)
        valid_predicted_positions.append(valid_predicted)
        roles.append(
            {
                "role": actual_role_index,
                "target_position": target,
                "predicted_position": predicted,
                "valid_predicted_position": valid_predicted,
                "correct": correct,
                "valid_correct": valid_correct,
                "top_positions": _top_positions(role_logits, top_k=top_k),
                "valid_top_positions": _top_positions(valid_logits, top_k=top_k),
                "valid_rank": valid_stats["rank"],
                "valid_unique_top1": valid_stats["unique_top1"],
                "valid_target_minus_top_logit": valid_stats[
                    "target_minus_top_logit"
                ],
                **stats,
            }
        )
    exact = selected_count == len(oracle_positions) and all(
        bool(role["correct"]) for role in roles
    )
    valid_exact = selected_count == len(oracle_positions) and all(
        bool(role["valid_correct"]) for role in roles
    )
    accuracy = float(selected_correct) / float(len(oracle_positions)) if oracle_positions else 1.0
    valid_accuracy = (
        float(valid_selected_correct) / float(len(oracle_positions))
        if oracle_positions
        else 1.0
    )
    return {
        "role_count": role_count,
        "role_offset": offset,
        "position_count": position_count,
        "valid_position_count": valid_count,
        "selected_role_count": len(oracle_positions),
        "selected_role_evaluated": selected_count,
        "selected_role_correct": selected_correct,
        "selected_role_accuracy": accuracy,
        "selected_role_top1_exact": bool(exact),
        "valid_selected_role_correct": valid_selected_correct,
        "valid_selected_role_accuracy": valid_accuracy,
        "valid_selected_role_top1_exact": bool(valid_exact),
        "invalid_top_position_count": invalid_top_position_count,
        "predicted_positions": predicted_positions,
        "valid_predicted_positions": valid_predicted_positions,
        "roles": roles,
    }


def _load_checkpoint_stack(model, *, checkpoint: str, base_checkpoint: str | None, device: str) -> None:
    from qtrm_mm.training.train import load_initial_checkpoint

    if base_checkpoint:
        load_initial_checkpoint(model, base_checkpoint, map_location=device)
    load_initial_checkpoint(model, checkpoint, map_location=device)


def _model_disable_kwargs_from_runtime(runtime: dict[str, Any]) -> dict[str, bool]:
    return {
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
        "disable_answer_state_loop_next_token_decoder": bool(
            runtime.get("disable_answer_state_loop_next_token_decoder", False)
        ),
    }


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_mode.setdefault(str(record["mode"]), []).append(record)
    for mode, rows in by_mode.items():
        n = max(1, len(rows))
        out[mode] = {
            "rows": len(rows),
            "pointer_exact": sum(
                int(bool(row["pointer"]["selected_role_top1_exact"]))
                for row in rows
            ),
            "pointer_exact_accuracy": sum(
                int(bool(row["pointer"]["selected_role_top1_exact"]))
                for row in rows
            )
            / n,
            "pointer_role_accuracy": sum(
                float(row["pointer"]["selected_role_accuracy"]) for row in rows
            )
            / n,
            "copy_answer_exact": sum(int(bool(row["copy_answer_exact"])) for row in rows),
            "copy_answer_accuracy": sum(
                int(bool(row["copy_answer_exact"])) for row in rows
            )
            / n,
            "valid_pointer_exact": sum(
                int(bool(row["pointer"].get("valid_selected_role_top1_exact")))
                for row in rows
            ),
            "valid_pointer_exact_accuracy": sum(
                int(bool(row["pointer"].get("valid_selected_role_top1_exact")))
                for row in rows
            )
            / n,
            "valid_pointer_role_accuracy": sum(
                float(row["pointer"].get("valid_selected_role_accuracy", 0.0))
                for row in rows
            )
            / n,
            "valid_copy_answer_exact": sum(
                int(bool(row["valid_copy_answer_exact"])) for row in rows
            ),
            "valid_copy_answer_accuracy": sum(
                int(bool(row["valid_copy_answer_exact"])) for row in rows
            )
            / n,
            "empty_logits": sum(int(bool(row["empty_logits"])) for row in rows),
        }
    return out


def _build_decision(summary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    full = summary.get("qtrm_core_steps_8_no_evidence") or {}
    slot_off = summary.get(
        "qtrm_core_steps_8_token_numeric_source_slots_off_no_evidence"
    ) or {}
    binder_off = summary.get(
        "qtrm_core_steps_8_core_source_position_binder_off_no_evidence"
    ) or {}
    full_acc = float(full.get("copy_answer_accuracy", 0.0))
    full_valid_acc = float(full.get("valid_copy_answer_accuracy", 0.0))
    slot_off_acc = float(slot_off.get("copy_answer_accuracy", 0.0))
    binder_off_acc = float(binder_off.get("copy_answer_accuracy", 0.0))
    accepted = (
        full_acc >= 0.95
        and full_acc - slot_off_acc >= 0.50
        and full_acc - binder_off_acc >= 0.50
    )
    if not accepted and full_valid_acc >= 0.95 and full_acc < 0.95:
        decision = "rejected_invalid_source_position_leakage"
        next_action = (
            "mask source-position logits to valid source-slot width before the "
            "copy softmax, then rerun the L4 LM path gate"
        )
    elif not accepted:
        decision = "rejected_source_position_logits_probe"
        next_action = (
            "train or redesign answer-time source-position binder before more "
            "L4 LM tuning"
        )
    else:
        decision = "accepted_l2_source_position_logits_probe"
        next_action = "repair copy renderer/donor fusion and rerun the L4 LM path gate"
    return {
        "decision": decision,
        "accepted": accepted,
        "target_level": "L2 diagnostic: QTRM source-position logits replace oracle positions",
        "major_bottleneck": "latent/source-position state to token lexicalization",
        "prior_principle": "pointer-generator / copy attention causal binding",
        "full_copy_answer_accuracy": full_acc,
        "full_valid_copy_answer_accuracy": full_valid_acc,
        "source_slot_off_copy_answer_accuracy": slot_off_acc,
        "source_binder_off_copy_answer_accuracy": binder_off_acc,
        "next_action": next_action,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    issues = requested_checkpoint_chain_issues(args, root=repo_root())
    if issues:
        return checkpoint_chain_missing_report(args, issues=issues)

    import torch
    from transformers import AutoTokenizer

    from qtrm_mm.config import load_config
    from qtrm_mm.qtrm_model import QTRMMultimodalModel
    from qtrm_mm.qwen_donor import QwenDonorAdapter

    raw_eval = _load_raw_eval_module()
    cfg = load_config(args.config)
    apply_source_pointer_defaults(cfg, args)
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
    cases = load_jsonl(args.cases, max_cases=args.max_cases)
    modes = args.mode or [
        "qtrm_core_steps_8_no_evidence",
        "qtrm_core_steps_8_token_numeric_source_slots_off_no_evidence",
        "qtrm_core_steps_8_core_source_position_binder_off_no_evidence",
    ]

    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for mode in modes:
            runtime = raw_eval.mode_runtime(mode)
            old_outer_steps = int(model.cfg.outer_steps)
            if runtime.get("core_steps_override") is not None:
                model.cfg.outer_steps = int(runtime["core_steps_override"])
            try:
                for case in cases:
                    prompt = str(case.get("prompt") or case.get("question") or "")
                    inputs = raw_eval._prepare_inputs(
                        tokenizer,
                        prompt,
                        args.max_length,
                        device,
                    )
                    input_ids = inputs["input_ids"]
                    attention_mask = inputs["attention_mask"]
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
                    prompt_logits = outputs.get("core_source_position_prompt_logits")
                    core_logits = outputs.get("core_role_value_state_logits")
                    primitive_logits = outputs.get(
                        "core_primitive_role_value_state_logits"
                    )
                    raw_logits, copy_logits_source = select_renderer_copy_logits(
                        source_position_prompt_logits=prompt_logits,
                        core_role_value_state_logits=core_logits,
                        core_primitive_role_value_state_logits=primitive_logits,
                        prefer_primitive=bool(
                            cfg.model.core_role_value_state_vocab_renderer_source_copy_from_primitive_enabled
                        ),
                    )
                    empty_logits = (
                        raw_logits is None
                        or raw_logits.ndim != 4
                        or raw_logits.numel() == 0
                        or int(raw_logits.shape[-1]) == 0
                    )
                    source_classes = oracle_source_classes(case)
                    if empty_logits:
                        pointer = {
                            "role_count": 0,
                            "position_count": 0,
                            "valid_position_count": 0,
                            "selected_role_count": len(source_classes),
                            "selected_role_evaluated": 0,
                            "selected_role_correct": 0,
                            "selected_role_accuracy": 0.0
                            if source_classes
                            else 1.0,
                            "selected_role_top1_exact": False,
                            "valid_selected_role_correct": 0,
                            "valid_selected_role_accuracy": 0.0
                            if source_classes
                            else 1.0,
                            "valid_selected_role_top1_exact": False,
                            "invalid_top_position_count": 0,
                            "predicted_positions": [],
                            "valid_predicted_positions": [],
                            "roles": [],
                        }
                    else:
                        pointer_logits = select_depth_logits(
                            raw_logits.float(),
                            batch_index=0,
                            depth_index=-1,
                        )
                        role_offset = source_copy_answer_role_offset(
                            int(pointer_logits.shape[0])
                        )
                        pointer = summarize_pointer_logits(
                            pointer_logits,
                            oracle_positions=source_classes,
                            role_offset=role_offset,
                            valid_position_count=len(case.get("input_list") or []) + 1,
                            top_k=args.top_k,
                        )
                    predicted_answer = copy_answer_from_source_classes(
                        case,
                        [int(position) for position in pointer["predicted_positions"]],
                    )
                    valid_predicted_answer = copy_answer_from_source_classes(
                        case,
                        [
                            int(position)
                            for position in pointer["valid_predicted_positions"]
                        ],
                    )
                    target = gold_answer(case)
                    records.append(
                        {
                            "id": case.get("id"),
                            "mode": mode,
                            "target": target,
                            "oracle_zero_based_positions": oracle_source_positions(case),
                            "oracle_source_classes": source_classes,
                            "predicted_source_classes": pointer["predicted_positions"],
                            "valid_predicted_source_classes": pointer[
                                "valid_predicted_positions"
                            ],
                            "predicted_copy_answer": predicted_answer,
                            "valid_predicted_copy_answer": valid_predicted_answer,
                            "copy_answer_exact": normalize_answer(predicted_answer)
                            == normalize_answer(target),
                            "valid_copy_answer_exact": normalize_answer(
                                valid_predicted_answer
                            )
                            == normalize_answer(target),
                            "empty_logits": bool(empty_logits),
                            "source_copy_logits_source": copy_logits_source,
                            "source_copy_answer_role_offset": (
                                pointer.get("role_offset") if not empty_logits else None
                            ),
                            "source_position_prompt_logits_shape": (
                                list(prompt_logits.shape)
                                if prompt_logits is not None
                                else None
                            ),
                            "core_role_value_state_logits_shape": (
                                list(core_logits.shape)
                                if core_logits is not None
                                else None
                            ),
                            "core_primitive_role_value_state_logits_shape": (
                                list(primitive_logits.shape)
                                if primitive_logits is not None
                                else None
                            ),
                            "selected_source_copy_logits_shape": (
                                list(raw_logits.shape)
                                if raw_logits is not None
                                else None
                            ),
                            "pointer": pointer,
                        }
                    )
            finally:
                model.cfg.outer_steps = old_outer_steps

    summary = _summarize_records(records)
    decision = _build_decision(summary)
    report = {
        **decision,
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "base_checkpoint": args.base_checkpoint,
        "cases": str(args.cases),
        "rows": len(cases),
        "modes": modes,
        "summary": summary,
        "records": records,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    report = run(build_arg_parser().parse_args())
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "records"},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if bool(report.get("accepted")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
