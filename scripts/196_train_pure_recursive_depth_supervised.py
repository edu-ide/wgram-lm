#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("evidence"):
                raise ValueError(f"{path}:{line_no}: depth-supervised rows must not include evidence")
            if not row.get("prompt"):
                raise ValueError(f"{path}:{line_no}: missing prompt")
            answer = row.get("chosen") or row.get("answer")
            if not answer:
                raise ValueError(f"{path}:{line_no}: missing chosen/answer")
            rows.append(row)
    if not rows:
        raise ValueError(f"no training rows in {path}")
    return rows


def answer_first_token_id(tokenizer: Any, answer: str) -> int:
    token_ids = answer_token_ids(tokenizer, answer)
    return int(token_ids[0])


def answer_token_ids(tokenizer: Any, answer: str) -> list[int]:
    token_ids = tokenizer.encode(f" {answer}", add_special_tokens=False)
    if not token_ids:
        token_ids = tokenizer.encode(str(answer), add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"answer produced no tokens: {answer!r}")
    return [int(token_id) for token_id in token_ids]


def target_for_core_steps(
    row: dict[str, Any],
    core_steps: int,
    *,
    target_mode: str = "staged",
) -> str:
    final_answer = str(row.get("chosen") or row.get("answer"))
    if str(target_mode).lower() == "final":
        return final_answer
    if str(target_mode).lower() != "staged":
        raise ValueError("target_mode must be 'staged' or 'final'")
    depth_targets = row.get("depth_targets")
    if isinstance(depth_targets, dict):
        target = depth_targets.get(str(int(core_steps)))
        if target:
            return str(target)
    return final_answer


def _row_temporal_spatial_context(row: dict[str, Any], *, device: str):
    value = row.get("temporal_spatial_context")
    if value is None:
        return None
    import torch

    tensor = torch.tensor(value, dtype=torch.float32, device=device)
    if tensor.ndim == 1:
        return tensor.view(1, -1)
    if tensor.ndim == 2:
        return tensor.unsqueeze(0)
    if tensor.ndim == 3 and int(tensor.shape[0]) == 1:
        return tensor
    raise ValueError(
        "temporal_spatial_context must be a vector, token list, or single-batch token list"
    )


def parse_depth_steps(value: str | Iterable[int]) -> list[int]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        steps = [int(part) for part in parts]
    else:
        steps = [int(part) for part in value]
    if not steps:
        raise ValueError("at least one depth step is required")
    if any(step <= 0 for step in steps):
        raise ValueError("depth steps must be positive")
    return steps


def scheduled_row_and_core_steps(
    step: int,
    *,
    row_count: int,
    depth_steps: list[int],
    row_indices: list[int] | None = None,
) -> tuple[int, int]:
    effective_count = len(row_indices) if row_indices is not None else int(row_count)
    if effective_count <= 0:
        raise ValueError("row_count must be positive")
    if not depth_steps:
        raise ValueError("depth_steps must not be empty")
    depth_index = int(step) % len(depth_steps)
    curriculum_index = (int(step) // len(depth_steps)) % effective_count
    row_index = int(row_indices[curriculum_index]) if row_indices is not None else curriculum_index
    return row_index, int(depth_steps[depth_index])


def parse_family_repeat_spec(value: str | None) -> dict[str, int]:
    repeats: dict[str, int] = {}
    if not value:
        return repeats
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError("family repeat entries must look like family=repeat")
        family, repeat_text = item.split("=", 1)
        family = family.strip()
        if not family:
            raise ValueError("family repeat entry has empty family name")
        repeat = int(repeat_text.strip())
        if repeat <= 0:
            raise ValueError("family repeat must be positive")
        repeats[family] = repeat
    return repeats


def build_curriculum_indices(
    rows: list[dict[str, Any]],
    family_repeats: dict[str, int] | None,
) -> list[int]:
    repeats = family_repeats or {}
    indices: list[int] = []
    for idx, row in enumerate(rows):
        family = str(row.get("task_family") or row.get("category") or "")
        repeat = max(1, int(repeats.get(family, 1)))
        indices.extend([idx] * repeat)
    if not indices:
        raise ValueError("curriculum has no rows")
    return indices


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train QTRM recursive core depth logits on prompt-only pure reasoning rows. "
            "This avoids MemoryOS/retrieval shortcuts and gives the core a direct "
            "per-depth answer pressure."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-jsonl", required=True)
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument(
        "--allow-random-init",
        action="store_true",
        help=(
            "Explicitly start from a freshly initialized QTRM model when no "
            "--init-checkpoint is provided. This is for checkpoint-loss recovery "
            "and matched random-init baselines only."
        ),
    )
    parser.add_argument("--tokenizer-model-id", default="Qwen/Qwen3.5-2B-Base")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--depth-steps", default="1,2,4,8")
    parser.add_argument("--target-mode", choices=["staged", "final"], default="staged")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--final-logit-ce-weight", type=float, default=1.0)
    parser.add_argument("--depth-final-ce-weight", type=float, default=1.0)
    parser.add_argument("--all-depth-ce-weight", type=float, default=0.0)
    parser.add_argument("--progress-margin-weight", type=float, default=0.25)
    parser.add_argument("--progress-margin", type=float, default=0.10)
    parser.add_argument(
        "--family-repeat",
        default="",
        help="Comma-separated hard-family repeat spec, for example list_transform=4,boolean_logic=2.",
    )
    parser.add_argument("--choice-margin-weight", type=float, default=0.0)
    parser.add_argument("--choice-margin", type=float, default=0.10)
    parser.add_argument(
        "--choice-margin-mode",
        choices=["first_token", "sequence"],
        default="first_token",
        help=(
            "Preference margin contract. first_token matches legacy training; "
            "sequence applies the margin to each available causal-prefix answer "
            "token, aligning better with mean forced-choice eval scoring."
        ),
    )
    parser.add_argument("--temporal-spatial-context-contrast-weight", type=float, default=0.0)
    parser.add_argument("--temporal-spatial-context-contrast-margin", type=float, default=0.10)
    parser.add_argument("--transition-state-contrast-weight", type=float, default=0.0)
    parser.add_argument("--transition-state-contrast-margin", type=float, default=0.10)
    parser.add_argument("--transition-state-ce-weight", type=float, default=0.0)
    parser.add_argument("--transition-state-code-ce-weight", type=float, default=0.0)
    parser.add_argument("--transition-state-finality-ce-weight", type=float, default=0.0)
    parser.add_argument("--primitive-transition-operation-ce-weight", type=float, default=0.0)
    parser.add_argument(
        "--causal-prefix-supervision",
        action="store_true",
        help=(
            "Train answer logits from prompt/prefix-only inputs. This prevents "
            "workspace/core paths from seeing future answer tokens during depth supervision."
        ),
    )
    parser.add_argument(
        "--causal-prefix-max-target-tokens",
        type=int,
        default=1,
        help=(
            "When causal-prefix supervision is enabled, train up to this many "
            "answer tokens by appending the previous answer-token prefix only."
        ),
    )
    parser.add_argument(
        "--causal-prefix-later-token-weight",
        type=float,
        default=1.0,
        help=(
            "Loss weight for causal-prefix answer tokens after the first. "
            "The first answer token always keeps weight 1.0."
        ),
    )
    parser.add_argument(
        "--teacher-checkpoint",
        default="",
        help="Optional frozen QTRM checkpoint used to preserve first-token recursive depth logits.",
    )
    parser.add_argument("--teacher-first-token-depth-kl-weight", type=float, default=0.0)
    parser.add_argument("--teacher-depth-kl-temperature", type=float, default=1.0)
    parser.add_argument(
        "--core-world-model-weight",
        type=float,
        default=None,
        help=(
            "Optional LeWM-style auxiliary loss weight over recursive core states. "
            "Defaults to train.loss_core_world_model_weight from the config."
        ),
    )
    parser.add_argument(
        "--staged-internal-first-token-ce-weight",
        type=float,
        default=0.0,
        help=(
            "Optional semantic transition pressure: at labelled internal core "
            "depths, train the first answer-token logit toward depth_targets[depth]."
        ),
    )
    parser.add_argument(
        "--staged-internal-sequence-ce-weight",
        type=float,
        default=0.0,
        help=(
            "Optional value-bearing state pressure: at labelled internal core "
            "depths, train depth readout logits toward the full token sequence "
            "from depth_targets[depth]."
        ),
    )
    parser.add_argument(
        "--staged-internal-sequence-max-target-tokens",
        type=int,
        default=6,
        help="Maximum target tokens used by staged internal sequence CE.",
    )
    parser.add_argument(
        "--noise-warmup-steps",
        type=int,
        default=0,
        help=(
            "Optional random-token/random-label warm-up before real training. "
            "This calibrates only QTRM trainable modules; the Qwen donor remains frozen."
        ),
    )
    parser.add_argument("--noise-warmup-seq-len", type=int, default=32)
    parser.add_argument("--noise-warmup-batch-size", type=int, default=1)
    parser.add_argument("--noise-warmup-core-steps", type=int, default=2)
    parser.add_argument(
        "--noise-warmup-target-vocab-size",
        type=int,
        default=0,
        help="Optional target-token range for random labels. 0 means full model vocab.",
    )
    parser.add_argument("--noise-warmup-final-ce-weight", type=float, default=1.0)
    parser.add_argument("--noise-warmup-depth-ce-weight", type=float, default=1.0)
    parser.add_argument(
        "--noise-warmup-uniform-weight",
        type=float,
        default=0.0,
        help=(
            "Optional uncertainty-calibration loss for random-token warm-up. "
            "When non-zero, pushes final/depth logits toward high-entropy "
            "uniform predictions on random inputs instead of overconfident guesses."
        ),
    )
    parser.add_argument("--log-every", type=int, default=10)
    return parser


def validate_init_checkpoint_args(init_checkpoint: str, *, allow_random_init: bool) -> str:
    if str(init_checkpoint or "").strip():
        return "checkpoint"
    if bool(allow_random_init):
        return "random_init"
    raise ValueError(
        "--init-checkpoint is required unless --allow-random-init is explicitly set"
    )


def _prepare_prompt(tokenizer: Any, prompt: str, *, max_length: int, device: str):
    import torch

    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=True,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)
    return input_ids, attention_mask


def _prepare_prompt_answer(
    tokenizer: Any,
    prompt: str,
    answer: str,
    *,
    max_length: int,
    device: str,
):
    import torch

    prompt_enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=True,
    )
    full_text = f"{prompt} {answer}"
    enc = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=True,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)
    prompt_len = int(prompt_enc["input_ids"].shape[1])
    full_len = int(input_ids.shape[1])
    if full_len <= prompt_len:
        raise ValueError(
            f"answer tokens were truncated or missing for prompt={prompt!r} answer={answer!r}"
        )
    target_start = prompt_len
    target_end = full_len
    target_ids = input_ids[:, target_start:target_end]
    return input_ids, attention_mask, target_ids, target_start, target_end


def _prepare_causal_prefix_answer(
    tokenizer: Any,
    prompt: str,
    answer: str,
    *,
    max_length: int,
    device: str,
):
    return _prepare_causal_prefix_answer_examples(
        tokenizer,
        prompt,
        answer,
        max_length=max_length,
        device=device,
        max_target_tokens=1,
    )[0]


def _prepare_causal_prefix_answer_examples(
    tokenizer: Any,
    prompt: str,
    answer: str,
    *,
    max_length: int,
    device: str,
    max_target_tokens: int,
):
    import torch

    if int(max_target_tokens) <= 0:
        raise ValueError("max_target_tokens must be positive")

    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=True,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)
    answer_ids = tokenizer.encode(f" {answer}", add_special_tokens=False)
    if not answer_ids:
        answer_ids = tokenizer.encode(str(answer), add_special_tokens=False)
    if not answer_ids:
        raise ValueError(f"answer produced no tokens: {answer!r}")

    examples = []
    target_count = min(len(answer_ids), int(max_target_tokens))
    for target_index in range(target_count):
        prefix_ids = [int(token_id) for token_id in answer_ids[:target_index]]
        example_input_ids = input_ids
        example_attention_mask = attention_mask
        if prefix_ids:
            prefix_tensor = input_ids.new_tensor([prefix_ids])
            prefix_mask = attention_mask.new_ones((attention_mask.shape[0], len(prefix_ids)))
            example_input_ids = torch.cat([input_ids, prefix_tensor], dim=1)
            example_attention_mask = torch.cat([attention_mask, prefix_mask], dim=1)
        if int(example_input_ids.shape[1]) > int(max_length):
            break
        target_start = int(example_input_ids.shape[1])
        target_ids = input_ids.new_tensor([[int(answer_ids[target_index])]])
        examples.append(
            (
                example_input_ids,
                example_attention_mask,
                target_ids,
                target_start,
                target_start + 1,
            )
        )
    if not examples:
        raise ValueError(
            f"causal-prefix answer produced no target examples: prompt={prompt!r} answer={answer!r}"
        )
    return examples


def _causal_prefix_example_loss_weight(example_index: int, later_token_weight: float) -> float:
    if float(later_token_weight) < 0.0:
        raise ValueError("later_token_weight must be non-negative")
    if int(example_index) == 0:
        return 1.0
    return float(later_token_weight)


def _should_apply_teacher_first_token_depth_kl(example_index: int, weight: float) -> bool:
    return int(example_index) == 0 and float(weight) > 0.0


def _target_log_probs(depth_logits, target_ids):
    import torch

    log_probs = depth_logits.float().log_softmax(dim=-1)
    gather_index = target_ids[:, None, None].expand(-1, depth_logits.shape[1], 1)
    return log_probs.gather(dim=-1, index=gather_index).squeeze(-1)


def depth_supervision_loss(
    depth_logits,
    final_logits,
    target_ids,
    *,
    final_logit_ce_weight: float,
    depth_final_ce_weight: float = 1.0,
    all_depth_ce_weight: float,
    progress_margin_weight: float,
    progress_margin: float,
):
    import torch
    import torch.nn.functional as F

    if depth_logits.ndim != 3:
        raise ValueError("depth_logits must have shape [batch, steps, vocab]")
    if depth_logits.shape[1] == 0:
        raise ValueError("recursive core produced no depth logits")
    final_ce = F.cross_entropy(depth_logits[:, -1, :].float(), target_ids)
    final_path_ce = F.cross_entropy(final_logits.float(), target_ids)
    all_depth_ce = final_ce.new_zeros(())
    if float(all_depth_ce_weight) != 0.0:
        repeated_targets = target_ids[:, None].expand(-1, depth_logits.shape[1]).reshape(-1)
        all_depth_ce = F.cross_entropy(
            depth_logits.float().reshape(-1, depth_logits.shape[-1]),
            repeated_targets,
        )
    progress = final_ce.new_zeros(())
    if depth_logits.shape[1] > 1 and float(progress_margin_weight) != 0.0:
        target_logps = _target_log_probs(depth_logits, target_ids)
        final_logp = target_logps[:, -1:]
        progress = F.relu(float(progress_margin) + target_logps[:, :-1] - final_logp).mean()
    loss = (
        float(final_logit_ce_weight) * final_path_ce
        + float(depth_final_ce_weight) * final_ce
        + float(all_depth_ce_weight) * all_depth_ce
        + float(progress_margin_weight) * progress
    )
    with torch.no_grad():
        pred = depth_logits[:, -1, :].argmax(dim=-1)
        acc = (pred == target_ids).float().mean()
        final_path_pred = final_logits.argmax(dim=-1)
        final_path_acc = (final_path_pred == target_ids).float().mean()
        target_logps = _target_log_probs(depth_logits, target_ids)
        depth_delta = (
            target_logps[:, -1] - target_logps[:, 0]
            if depth_logits.shape[1] > 1
            else target_logps[:, -1] * 0.0
        )
    return loss, {
        "depth_final_ce": final_ce.detach(),
        "final_path_ce": final_path_ce.detach(),
        "depth_all_ce": all_depth_ce.detach(),
        "depth_progress": progress.detach(),
        "depth_final_acc": acc.detach(),
        "final_path_acc": final_path_acc.detach(),
        "depth_target_logp_delta": depth_delta.mean().detach(),
    }


def _token_sequence_cross_entropy(logits, target_ids):
    import torch.nn.functional as F

    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, tokens, vocab]")
    if target_ids.ndim != 2:
        raise ValueError("target_ids must have shape [batch, tokens]")
    if logits.shape[:2] != target_ids.shape:
        raise ValueError("logits token dimension must match target_ids")
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        target_ids.reshape(-1),
    )


def _sequence_target_log_probs(depth_text_logits, target_ids):
    log_probs = depth_text_logits.float().log_softmax(dim=-1)
    gather_index = target_ids[:, None, :, None].expand(
        -1,
        depth_text_logits.shape[1],
        -1,
        1,
    )
    return log_probs.gather(dim=-1, index=gather_index).squeeze(-1).mean(dim=-1)


def depth_text_logit_distillation_loss(
    student_depth_text_logits,
    teacher_depth_text_logits,
    *,
    temperature: float,
):
    import torch.nn.functional as F

    if student_depth_text_logits.shape != teacher_depth_text_logits.shape:
        raise ValueError("student and teacher depth text logits must have matching shapes")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    vocab = student_depth_text_logits.shape[-1]
    student = student_depth_text_logits.float().reshape(-1, vocab) / float(temperature)
    teacher = teacher_depth_text_logits.detach().float().reshape(-1, vocab) / float(temperature)
    return F.kl_div(
        student.log_softmax(dim=-1),
        teacher.softmax(dim=-1),
        reduction="batchmean",
    ) * (float(temperature) ** 2)


def depth_sequence_supervision_loss(
    depth_text_logits,
    final_text_logits,
    target_ids,
    *,
    final_logit_ce_weight: float,
    depth_final_ce_weight: float = 1.0,
    all_depth_ce_weight: float,
    progress_margin_weight: float,
    progress_margin: float,
):
    import torch

    if depth_text_logits.ndim != 4:
        raise ValueError("depth_text_logits must have shape [batch, steps, tokens, vocab]")
    if depth_text_logits.shape[1] == 0:
        raise ValueError("recursive core produced no depth text logits")
    if depth_text_logits.shape[2] != target_ids.shape[1]:
        raise ValueError("depth text logits must align to target token count")

    final_ce = _token_sequence_cross_entropy(depth_text_logits[:, -1, :, :], target_ids)
    final_path_ce = _token_sequence_cross_entropy(final_text_logits, target_ids)
    all_depth_ce = final_ce.new_zeros(())
    if float(all_depth_ce_weight) != 0.0:
        repeated_targets = target_ids[:, None, :].expand(
            -1,
            depth_text_logits.shape[1],
            -1,
        )
        all_depth_ce = _token_sequence_cross_entropy(
            depth_text_logits.reshape(
                -1,
                depth_text_logits.shape[2],
                depth_text_logits.shape[-1],
            ),
            repeated_targets.reshape(-1, target_ids.shape[1]),
        )
    progress = final_ce.new_zeros(())
    if depth_text_logits.shape[1] > 1 and float(progress_margin_weight) != 0.0:
        target_logps = _sequence_target_log_probs(depth_text_logits, target_ids)
        final_logp = target_logps[:, -1:]
        progress = torch.relu(
            float(progress_margin) + target_logps[:, :-1] - final_logp
        ).mean()
    loss = (
        float(final_logit_ce_weight) * final_path_ce
        + float(depth_final_ce_weight) * final_ce
        + float(all_depth_ce_weight) * all_depth_ce
        + float(progress_margin_weight) * progress
    )
    with torch.no_grad():
        pred = depth_text_logits[:, -1, :, :].argmax(dim=-1)
        acc = (pred == target_ids).float().mean()
        final_path_pred = final_text_logits.argmax(dim=-1)
        final_path_acc = (final_path_pred == target_ids).float().mean()
        target_logps = _sequence_target_log_probs(depth_text_logits, target_ids)
        depth_delta = (
            target_logps[:, -1] - target_logps[:, 0]
            if depth_text_logits.shape[1] > 1
            else target_logps[:, -1] * 0.0
        )
    return loss, {
        "depth_final_ce": final_ce.detach(),
        "final_path_ce": final_path_ce.detach(),
        "depth_all_ce": all_depth_ce.detach(),
        "depth_progress": progress.detach(),
        "depth_final_acc": acc.detach(),
        "final_path_acc": final_path_acc.detach(),
        "depth_target_logp_delta": depth_delta.mean().detach(),
    }


def context_ablation_contrastive_loss(
    context_depth_text_logits,
    context_off_depth_text_logits,
    target_ids,
    *,
    margin: float,
):
    import torch

    if context_depth_text_logits.shape != context_off_depth_text_logits.shape:
        raise ValueError("context-on and context-off depth logits must have matching shapes")
    on_logp = _sequence_target_log_probs(context_depth_text_logits, target_ids)[:, -1]
    off_logp = _sequence_target_log_probs(context_off_depth_text_logits, target_ids)[:, -1]
    delta = on_logp - off_logp
    loss = torch.relu(float(margin) - delta).mean()
    return loss, {
        "context_contrast": loss.detach(),
        "context_contrast_target_logp_delta": delta.mean().detach(),
    }


def transition_state_ablation_contrastive_loss(
    transition_depth_text_logits,
    transition_off_depth_text_logits,
    target_ids,
    *,
    margin: float,
):
    import torch

    if transition_depth_text_logits.shape != transition_off_depth_text_logits.shape:
        raise ValueError(
            "transition-state-on and transition-state-off depth logits must have matching shapes"
        )
    on_logp = _sequence_target_log_probs(transition_depth_text_logits, target_ids)[:, -1]
    off_logp = _sequence_target_log_probs(
        transition_off_depth_text_logits,
        target_ids,
    )[:, -1].detach()
    delta = on_logp - off_logp
    loss = torch.relu(float(margin) - delta).mean()
    return loss, {
        "transition_state_contrast": loss.detach(),
        "transition_state_contrast_target_logp_delta": delta.mean().detach(),
    }


def staged_internal_first_token_targets(
    tokenizer: Any,
    row: dict[str, Any],
    *,
    num_depths: int,
    device: str,
    target_mode: str = "staged",
):
    import torch

    if int(num_depths) <= 0:
        raise ValueError("num_depths must be positive")
    target_ids: list[int] = []
    depth_targets = row.get("depth_targets") if str(target_mode).lower() == "staged" else None
    for depth in range(1, int(num_depths) + 1):
        target_text: str | None = None
        if isinstance(depth_targets, dict):
            exact = depth_targets.get(str(depth))
            if exact:
                target_text = str(exact)
        elif str(target_mode).lower() == "final":
            target_text = target_for_core_steps(row, depth, target_mode="final")
        if target_text is None:
            target_ids.append(-100)
            continue
        target_ids.append(answer_first_token_id(tokenizer, target_text))
    return torch.tensor([target_ids], dtype=torch.long, device=device)


def staged_internal_first_token_ce_loss(depth_text_logits, staged_target_ids):
    import torch
    import torch.nn.functional as F

    if depth_text_logits.ndim != 4:
        raise ValueError("depth_text_logits must have shape [batch, steps, tokens, vocab]")
    if depth_text_logits.shape[2] < 1:
        raise ValueError("staged internal CE needs at least one answer-token logit")
    targets = staged_target_ids.to(device=depth_text_logits.device, dtype=torch.long)
    if targets.shape != depth_text_logits.shape[:2]:
        raise ValueError("staged_target_ids must have shape [batch, steps]")
    mask = targets >= 0
    if not bool(mask.any()):
        zero = depth_text_logits.sum() * 0.0
        return zero, {
            "staged_internal_first_token_ce": zero.detach(),
            "staged_internal_first_token_acc": zero.detach(),
            "staged_internal_first_token_samples": zero.detach(),
        }
    logits = depth_text_logits[:, :, 0, :].float()
    loss = F.cross_entropy(logits[mask], targets[mask])
    pred = logits.detach().argmax(dim=-1)
    acc = (pred[mask] == targets[mask]).float().mean()
    samples = depth_text_logits.detach().new_tensor(float(int(mask.sum().item())))
    return loss, {
        "staged_internal_first_token_ce": loss.detach(),
        "staged_internal_first_token_acc": acc.detach(),
        "staged_internal_first_token_samples": samples,
    }


def staged_internal_sequence_targets(
    tokenizer: Any,
    row: dict[str, Any],
    *,
    num_depths: int,
    max_target_tokens: int,
    device: str,
    target_mode: str = "staged",
):
    import torch

    if int(num_depths) <= 0:
        raise ValueError("num_depths must be positive")
    if int(max_target_tokens) <= 0:
        raise ValueError("max_target_tokens must be positive")
    targets: list[list[int]] = []
    depth_targets = row.get("depth_targets") if str(target_mode).lower() == "staged" else None
    for depth in range(1, int(num_depths) + 1):
        target_text: str | None = None
        if isinstance(depth_targets, dict):
            exact = depth_targets.get(str(depth))
            if exact:
                target_text = str(exact)
        elif str(target_mode).lower() == "final":
            target_text = target_for_core_steps(row, depth, target_mode="final")
        if target_text is None:
            targets.append([-100] * int(max_target_tokens))
            continue
        token_ids = answer_token_ids(tokenizer, target_text)[: int(max_target_tokens)]
        padded = [int(token_id) for token_id in token_ids]
        padded.extend([-100] * (int(max_target_tokens) - len(padded)))
        targets.append(padded)
    return torch.tensor([targets], dtype=torch.long, device=device)


def staged_internal_sequence_ce_loss(depth_text_logits, staged_sequence_target_ids):
    import torch
    import torch.nn.functional as F

    if depth_text_logits.ndim != 4:
        raise ValueError("depth_text_logits must have shape [batch, steps, tokens, vocab]")
    targets = staged_sequence_target_ids.to(
        device=depth_text_logits.device,
        dtype=torch.long,
    )
    if targets.ndim != 3:
        raise ValueError("staged_sequence_target_ids must have shape [batch, steps, tokens]")
    if targets.shape[:2] != depth_text_logits.shape[:2]:
        raise ValueError("staged sequence targets must match batch and depth dimensions")
    token_count = min(int(targets.shape[2]), int(depth_text_logits.shape[2]))
    if token_count <= 0:
        zero = depth_text_logits.sum() * 0.0
        return zero, {
            "staged_internal_sequence_ce": zero.detach(),
            "staged_internal_sequence_acc": zero.detach(),
            "staged_internal_sequence_samples": zero.detach(),
        }
    targets = targets[:, :, :token_count]
    logits = depth_text_logits[:, :, :token_count, :].float()
    mask = targets >= 0
    if not bool(mask.any()):
        zero = depth_text_logits.sum() * 0.0
        return zero, {
            "staged_internal_sequence_ce": zero.detach(),
            "staged_internal_sequence_acc": zero.detach(),
            "staged_internal_sequence_samples": zero.detach(),
        }
    loss = F.cross_entropy(logits[mask], targets[mask])
    pred = logits.detach().argmax(dim=-1)
    acc = (pred[mask] == targets[mask]).float().mean()
    samples = depth_text_logits.detach().new_tensor(float(int(mask.sum().item())))
    return loss, {
        "staged_internal_sequence_ce": loss.detach(),
        "staged_internal_sequence_acc": acc.detach(),
        "staged_internal_sequence_samples": samples,
    }


def transition_state_first_token_ce_loss(transition_state_text_logits, staged_target_ids):
    import torch
    import torch.nn.functional as F

    if transition_state_text_logits.ndim != 3:
        raise ValueError(
            "transition_state_text_logits must have shape [batch, steps, vocab]"
        )
    targets = staged_target_ids.to(
        device=transition_state_text_logits.device,
        dtype=torch.long,
    )
    if targets.shape != transition_state_text_logits.shape[:2]:
        raise ValueError("staged_target_ids must have shape [batch, steps]")
    mask = targets >= 0
    if not bool(mask.any()):
        zero = transition_state_text_logits.sum() * 0.0
        return zero, {
            "transition_state_first_token_ce": zero.detach(),
            "transition_state_first_token_acc": zero.detach(),
            "transition_state_first_token_samples": zero.detach(),
        }
    logits = transition_state_text_logits.float()
    loss = F.cross_entropy(logits[mask], targets[mask])
    pred = logits.detach().argmax(dim=-1)
    acc = (pred[mask] == targets[mask]).float().mean()
    samples = transition_state_text_logits.detach().new_tensor(
        float(int(mask.sum().item()))
    )
    return loss, {
        "transition_state_first_token_ce": loss.detach(),
        "transition_state_first_token_acc": acc.detach(),
        "transition_state_first_token_samples": samples,
    }


def transition_state_code_targets(
    tokenizer: Any,
    row: dict[str, Any],
    *,
    num_depths: int,
    codebook_size: int,
    device: str,
    target_mode: str = "staged",
):
    import torch

    if int(codebook_size) <= 0:
        raise ValueError("codebook_size must be positive")
    explicit_codes = row.get("transition_state_codes")
    if isinstance(explicit_codes, dict) and explicit_codes:
        targets: list[int] = []
        for depth in range(1, int(num_depths) + 1):
            raw_code = explicit_codes.get(str(depth))
            if raw_code is None:
                targets.append(-100)
                continue
            code = int(raw_code)
            if code < 0 or code >= int(codebook_size):
                raise ValueError(
                    "transition_state_codes must be within the transition-state codebook"
                )
            targets.append(code)
        return torch.tensor([targets], dtype=torch.long, device=device)
    first_token_targets = staged_internal_first_token_targets(
        tokenizer,
        row,
        num_depths=num_depths,
        device=device,
        target_mode=target_mode,
    )
    targets = first_token_targets.clone()
    mask = targets >= 0
    targets[mask] = targets[mask] % int(codebook_size)
    return targets


def transition_state_code_ce_loss(transition_state_code_logits, code_targets):
    import torch
    import torch.nn.functional as F

    if transition_state_code_logits.ndim != 3:
        raise ValueError(
            "transition_state_code_logits must have shape [batch, steps, codebook]"
        )
    targets = code_targets.to(
        device=transition_state_code_logits.device,
        dtype=torch.long,
    )
    if targets.shape != transition_state_code_logits.shape[:2]:
        raise ValueError("code_targets must have shape [batch, steps]")
    mask = targets >= 0
    if not bool(mask.any()):
        zero = transition_state_code_logits.sum() * 0.0
        return zero, {
            "transition_state_code_ce": zero.detach(),
            "transition_state_code_acc": zero.detach(),
            "transition_state_code_samples": zero.detach(),
        }
    logits = transition_state_code_logits.float()
    loss = F.cross_entropy(logits[mask], targets[mask])
    pred = logits.detach().argmax(dim=-1)
    acc = (pred[mask] == targets[mask]).float().mean()
    samples = transition_state_code_logits.detach().new_tensor(
        float(int(mask.sum().item()))
    )
    return loss, {
        "transition_state_code_ce": loss.detach(),
        "transition_state_code_acc": acc.detach(),
        "transition_state_code_samples": samples,
    }


def transition_state_finality_targets(
    row: dict[str, Any],
    *,
    num_depths: int,
    device: str,
):
    import torch

    targets = [-100.0] * int(num_depths)
    explicit_targets = row.get("transition_finality_targets")
    if not isinstance(explicit_targets, dict):
        return torch.tensor([targets], dtype=torch.float32, device=device)
    for depth in range(1, int(num_depths) + 1):
        raw_target = explicit_targets.get(str(depth))
        if raw_target is None:
            continue
        value = float(raw_target)
        if value not in {0.0, 1.0}:
            raise ValueError("transition_finality_targets must be binary")
        targets[depth - 1] = value
    return torch.tensor([targets], dtype=torch.float32, device=device)


def transition_state_finality_bce_loss(finality_logits, finality_targets):
    import torch
    import torch.nn.functional as F

    if finality_logits.ndim != 2:
        raise ValueError("finality_logits must have shape [batch, steps]")
    targets = finality_targets.to(device=finality_logits.device, dtype=torch.float32)
    if targets.shape != finality_logits.shape:
        raise ValueError("finality_targets must have shape [batch, steps]")
    mask = targets >= 0.0
    if not bool(mask.any()):
        zero = finality_logits.sum() * 0.0
        return zero, {
            "transition_state_finality_bce": zero.detach(),
            "transition_state_finality_acc": zero.detach(),
            "transition_state_finality_samples": zero.detach(),
        }
    logits = finality_logits.float()
    loss = F.binary_cross_entropy_with_logits(logits[mask], targets[mask])
    pred = (torch.sigmoid(logits.detach()) >= 0.5).float()
    acc = (pred[mask] == targets[mask]).float().mean()
    samples = finality_logits.detach().new_tensor(float(int(mask.sum().item())))
    return loss, {
        "transition_state_finality_bce": loss.detach(),
        "transition_state_finality_acc": acc.detach(),
        "transition_state_finality_samples": samples,
    }


PRIMITIVE_TRANSITION_OPERATION_ORDER = (
    "add_operands",
    "multiply_sum",
    "subtract_offset",
    "hold_final",
    "filter_even",
    "double_filtered",
    "first_mapping",
    "second_mapping",
    "not_q",
    "and_with_p",
    "or_with_r",
)


def primitive_transition_operation_id_map(num_operations: int) -> dict[str, int]:
    if int(num_operations) < len(PRIMITIVE_TRANSITION_OPERATION_ORDER):
        raise ValueError(
            "primitive transition operation head is smaller than the canonical operation set"
        )
    return {
        operation: index
        for index, operation in enumerate(PRIMITIVE_TRANSITION_OPERATION_ORDER)
    }


def primitive_transition_operation_targets(
    row: dict[str, Any],
    *,
    num_steps: int,
    operation_to_id: dict[str, int],
    device: str,
):
    import torch

    targets = [-100] * int(num_steps)
    solver_trace = row.get("solver_trace")
    if not isinstance(solver_trace, list):
        return torch.tensor([targets], dtype=torch.long, device=device)
    for index, step in enumerate(solver_trace[: int(num_steps)]):
        if not isinstance(step, dict):
            continue
        operation = str(step.get("operation") or "")
        if operation not in operation_to_id:
            raise ValueError(f"unknown primitive transition operation: {operation}")
        targets[index] = int(operation_to_id[operation])
    return torch.tensor([targets], dtype=torch.long, device=device)


def primitive_transition_operation_ce_loss(operation_logits, operation_targets):
    import torch
    import torch.nn.functional as F

    if operation_logits.ndim != 3:
        raise ValueError("operation_logits must have shape [batch, steps, operations]")
    targets = operation_targets.to(device=operation_logits.device, dtype=torch.long)
    if targets.shape != operation_logits.shape[:2]:
        raise ValueError("operation_targets must have shape [batch, steps]")
    mask = targets >= 0
    if not bool(mask.any()):
        zero = operation_logits.sum() * 0.0
        return zero, {
            "primitive_transition_operation_ce": zero.detach(),
            "primitive_transition_operation_acc": zero.detach(),
            "primitive_transition_operation_samples": zero.detach(),
        }
    logits = operation_logits.float()
    loss = F.cross_entropy(logits[mask], targets[mask])
    pred = logits.detach().argmax(dim=-1)
    acc = (pred[mask] == targets[mask]).float().mean()
    samples = operation_logits.detach().new_tensor(float(int(mask.sum().item())))
    return loss, {
        "primitive_transition_operation_ce": loss.detach(),
        "primitive_transition_operation_acc": acc.detach(),
        "primitive_transition_operation_samples": samples,
    }


def build_random_noise_warmup_batch(
    *,
    vocab_size: int,
    seq_len: int,
    batch_size: int,
    device: str,
    target_vocab_size: int | None = None,
    generator: Any | None = None,
):
    import torch

    if int(vocab_size) <= 0:
        raise ValueError("vocab_size must be positive")
    if int(seq_len) <= 0:
        raise ValueError("seq_len must be positive")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    target_limit = int(target_vocab_size or vocab_size)
    if target_limit <= 0 or target_limit > int(vocab_size):
        raise ValueError("target_vocab_size must be in [1, vocab_size]")
    input_ids = torch.randint(
        low=0,
        high=int(vocab_size),
        size=(int(batch_size), int(seq_len)),
        generator=generator,
        device=device,
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids)
    target_ids = torch.randint(
        low=0,
        high=target_limit,
        size=(int(batch_size),),
        generator=generator,
        device=device,
        dtype=torch.long,
    )
    return input_ids, attention_mask, target_ids


def random_noise_warmup_loss(
    final_logits,
    depth_text_logits,
    target_ids,
    *,
    final_ce_weight: float,
    depth_ce_weight: float,
    uniform_weight: float = 0.0,
):
    import torch
    import torch.nn.functional as F

    if final_logits.ndim != 2:
        raise ValueError("final_logits must have shape [batch, vocab]")
    if depth_text_logits.ndim != 4:
        raise ValueError("depth_text_logits must have shape [batch, steps, tokens, vocab]")
    if depth_text_logits.shape[2] < 1:
        raise ValueError("depth_text_logits must include at least one token position")
    targets = target_ids.to(device=final_logits.device, dtype=torch.long)
    if targets.ndim != 1 or targets.shape[0] != final_logits.shape[0]:
        raise ValueError("target_ids must have shape [batch]")

    final_ce = F.cross_entropy(final_logits.float(), targets)
    depth_logits = depth_text_logits[:, :, 0, :].float()
    repeated_targets = targets[:, None].expand(-1, depth_logits.shape[1]).reshape(-1)
    depth_ce = F.cross_entropy(
        depth_logits.reshape(-1, depth_logits.shape[-1]),
        repeated_targets,
    )
    final_uniform_ce = -final_logits.float().log_softmax(dim=-1).mean(dim=-1).mean()
    depth_uniform_ce = -depth_logits.log_softmax(dim=-1).mean(dim=-1).mean()
    uniform_loss = 0.5 * (final_uniform_ce + depth_uniform_ce)
    loss = (
        float(final_ce_weight) * final_ce
        + float(depth_ce_weight) * depth_ce
        + float(uniform_weight) * uniform_loss
    )
    with torch.no_grad():
        final_acc = (final_logits.argmax(dim=-1) == targets).float().mean()
        depth_acc = (depth_logits.argmax(dim=-1) == targets[:, None]).float().mean()
    return loss, {
        "noise_warmup_final_ce": final_ce.detach(),
        "noise_warmup_depth_ce": depth_ce.detach(),
        "noise_warmup_final_uniform_ce": final_uniform_ce.detach(),
        "noise_warmup_depth_uniform_ce": depth_uniform_ce.detach(),
        "noise_warmup_final_acc": final_acc.detach(),
        "noise_warmup_depth_acc": depth_acc.detach(),
    }


def depth_choice_margin_loss(
    depth_text_logits,
    final_text_logits,
    chosen_first_token_ids,
    rejected_first_token_ids,
    *,
    margin: float,
    all_depth_weight: float,
    final_weight: float,
):
    import torch

    if depth_text_logits.ndim != 4:
        raise ValueError("depth_text_logits must have shape [batch, steps, tokens, vocab]")
    if final_text_logits.ndim != 3:
        raise ValueError("final_text_logits must have shape [batch, tokens, vocab]")
    if depth_text_logits.shape[2] < 1 or final_text_logits.shape[1] < 1:
        raise ValueError("choice margin needs at least one answer-token logit")

    chosen = chosen_first_token_ids.to(device=depth_text_logits.device, dtype=torch.long)
    rejected = rejected_first_token_ids.to(device=depth_text_logits.device, dtype=torch.long)
    if chosen.ndim == 0:
        chosen = chosen[None]
    if rejected.ndim == 0:
        rejected = rejected[None]
    if chosen.shape != rejected.shape:
        raise ValueError("chosen and rejected token ids must have matching shapes")

    depth_log_probs = depth_text_logits[:, :, 0, :].float().log_softmax(dim=-1)
    chosen_depth = depth_log_probs.gather(
        dim=-1,
        index=chosen[:, None, None].expand(-1, depth_log_probs.shape[1], 1),
    ).squeeze(-1)
    rejected_depth = depth_log_probs.gather(
        dim=-1,
        index=rejected[:, None, None].expand(-1, depth_log_probs.shape[1], 1),
    ).squeeze(-1)
    all_depth_margin = torch.relu(float(margin) + rejected_depth - chosen_depth).mean()

    final_log_probs = final_text_logits[:, 0, :].float().log_softmax(dim=-1)
    chosen_final = final_log_probs.gather(dim=-1, index=chosen[:, None]).squeeze(-1)
    rejected_final = final_log_probs.gather(dim=-1, index=rejected[:, None]).squeeze(-1)
    final_margin = torch.relu(float(margin) + rejected_final - chosen_final).mean()

    loss = float(all_depth_weight) * all_depth_margin + float(final_weight) * final_margin
    return loss, {
        "choice_margin_all_depth": all_depth_margin.detach(),
        "choice_margin_final_path": final_margin.detach(),
    }


def _masked_mean_log_probs(logits, target_ids):
    import torch

    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, tokens, vocab]")
    if target_ids.ndim != 2:
        raise ValueError("target_ids must have shape [batch, tokens]")
    if logits.shape[:2] != target_ids.shape:
        raise ValueError("logits token dimension must match target ids")
    mask = target_ids >= 0
    if not bool(mask.any()):
        raise ValueError("at least one target token is required")
    safe_targets = target_ids.clamp_min(0)
    log_probs = logits.float().log_softmax(dim=-1)
    token_log_probs = log_probs.gather(
        dim=-1,
        index=safe_targets[:, :, None],
    ).squeeze(-1)
    token_log_probs = token_log_probs.masked_fill(~mask, 0.0)
    lengths = mask.sum(dim=-1).clamp_min(1)
    return token_log_probs.sum(dim=-1) / lengths


def depth_choice_sequence_margin_loss(
    depth_text_logits,
    final_text_logits,
    chosen_target_ids,
    rejected_target_ids,
    *,
    margin: float,
    all_depth_weight: float,
    final_weight: float,
):
    import torch

    if depth_text_logits.ndim != 4:
        raise ValueError("depth_text_logits must have shape [batch, steps, tokens, vocab]")
    if final_text_logits.ndim != 3:
        raise ValueError("final_text_logits must have shape [batch, tokens, vocab]")
    chosen = chosen_target_ids.to(device=depth_text_logits.device, dtype=torch.long)
    rejected = rejected_target_ids.to(device=depth_text_logits.device, dtype=torch.long)
    if chosen.ndim == 1:
        chosen = chosen[None, :]
    if rejected.ndim == 1:
        rejected = rejected[None, :]
    if chosen.shape != rejected.shape:
        raise ValueError("chosen and rejected target ids must have matching shapes")
    if depth_text_logits.shape[2] != chosen.shape[1]:
        raise ValueError("depth text logits token dimension must match target ids")
    if final_text_logits.shape[1] != chosen.shape[1]:
        raise ValueError("final text logits token dimension must match target ids")

    depth_chosen = []
    depth_rejected = []
    for step_index in range(depth_text_logits.shape[1]):
        depth_chosen.append(
            _masked_mean_log_probs(depth_text_logits[:, step_index, :, :], chosen)
        )
        depth_rejected.append(
            _masked_mean_log_probs(depth_text_logits[:, step_index, :, :], rejected)
        )
    chosen_depth = torch.stack(depth_chosen, dim=1)
    rejected_depth = torch.stack(depth_rejected, dim=1)
    all_depth_margin = torch.relu(float(margin) + rejected_depth - chosen_depth).mean()

    chosen_final = _masked_mean_log_probs(final_text_logits, chosen)
    rejected_final = _masked_mean_log_probs(final_text_logits, rejected)
    final_margin = torch.relu(float(margin) + rejected_final - chosen_final).mean()

    loss = float(all_depth_weight) * all_depth_margin + float(final_weight) * final_margin
    return loss, {
        "choice_sequence_margin_all_depth": all_depth_margin.detach(),
        "choice_sequence_margin_final_path": final_margin.detach(),
    }


def main() -> None:
    args = build_arg_parser().parse_args()

    import torch
    from tqdm import tqdm
    from transformers import AutoTokenizer

    from qtrm_mm.config import load_config
    from qtrm_mm.losses import jepa_world_model_loss
    from qtrm_mm.qtrm_model import QTRMMultimodalModel
    from qtrm_mm.qwen_donor import QwenDonorAdapter
    from qtrm_mm.training.train import (
        build_core_world_model_actions,
        configure_trainable_parameters,
        load_initial_checkpoint,
    )

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() and cfg.train.device in {"auto", "cuda"} else "cpu"
    rows = load_rows(args.data_jsonl)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_model_id,
        trust_remote_code=cfg.donor.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = QTRMMultimodalModel(cfg.model).to(device)
    init_mode = validate_init_checkpoint_args(
        args.init_checkpoint,
        allow_random_init=bool(args.allow_random_init),
    )
    if init_mode == "checkpoint":
        missing, unexpected = load_initial_checkpoint(
            model,
            args.init_checkpoint,
            map_location=device,
        )
        if missing:
            print(f"[init] missing keys: {len(missing)}")
        if unexpected:
            print(f"[init] unexpected keys: {len(unexpected)}")
    else:
        print("[init] random QTRM initialization (--allow-random-init)")
    teacher_model = None
    if float(args.teacher_first_token_depth_kl_weight) != 0.0:
        if not args.teacher_checkpoint:
            raise ValueError("--teacher-checkpoint is required when teacher depth KL weight is non-zero")
        teacher_model = QTRMMultimodalModel(cfg.model).to(device)
        teacher_missing, teacher_unexpected = load_initial_checkpoint(
            teacher_model,
            args.teacher_checkpoint,
            map_location=device,
        )
        if teacher_missing:
            print(f"[teacher] missing keys: {len(teacher_missing)}")
        if teacher_unexpected:
            print(f"[teacher] unexpected keys: {len(teacher_unexpected)}")
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad_(False)
    donor = QwenDonorAdapter(cfg.donor)
    trainable_names = configure_trainable_parameters(model, cfg.train.trainable_param_policy)
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise ValueError("no trainable parameters selected")
    print(
        f"[trainable] policy={cfg.train.trainable_param_policy} "
        f"params={sum(p.numel() for p in params):,} tensors={len(trainable_names)}"
    )
    opt = torch.optim.AdamW(
        params,
        lr=float(args.lr if args.lr is not None else cfg.train.lr),
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.train.use_amp and device == "cuda"))
    steps = int(args.steps if args.steps is not None else cfg.train.steps)
    depth_steps = parse_depth_steps(args.depth_steps)
    family_repeats = parse_family_repeat_spec(args.family_repeat)
    curriculum_indices = build_curriculum_indices(rows, family_repeats)
    if family_repeats:
        print(f"[curriculum] family_repeat={family_repeats} effective_rows={len(curriculum_indices)}")
    max_length = int(args.max_length or cfg.train.seq_len)
    core_world_model_weight = (
        float(args.core_world_model_weight)
        if args.core_world_model_weight is not None
        else float(getattr(cfg.train, "loss_core_world_model_weight", 0.0))
    )
    if core_world_model_weight != 0.0 and not bool(cfg.model.core_world_model_enabled):
        raise ValueError(
            "core world-model loss requires model.core_world_model_enabled=true"
        )
    if (
        float(args.transition_state_contrast_weight) != 0.0
        and not (
            bool(cfg.model.transition_state_enabled)
            or bool(cfg.model.transition_state_code_enabled)
        )
    ):
        raise ValueError(
            "transition-state contrast requires model.transition_state_enabled=true "
            "or model.transition_state_code_enabled=true"
        )
    if (
        float(args.transition_state_ce_weight) != 0.0
        and not bool(cfg.model.transition_state_enabled)
    ):
        raise ValueError(
            "transition-state CE requires model.transition_state_enabled=true"
        )
    if (
        float(args.transition_state_code_ce_weight) != 0.0
        and not bool(cfg.model.transition_state_code_enabled)
    ):
        raise ValueError(
            "transition-state code CE requires model.transition_state_code_enabled=true"
        )
    if (
        float(args.transition_state_finality_ce_weight) != 0.0
        and not bool(cfg.model.transition_state_finality_enabled)
    ):
        raise ValueError(
            "transition-state finality CE requires model.transition_state_finality_enabled=true"
        )
    if (
        float(args.primitive_transition_operation_ce_weight) != 0.0
        and not bool(cfg.model.primitive_transition_enabled)
    ):
        raise ValueError(
            "primitive transition operation CE requires model.primitive_transition_enabled=true"
        )
    primitive_operation_to_id = primitive_transition_operation_id_map(
        int(cfg.model.primitive_transition_num_operations)
    ) if float(args.primitive_transition_operation_ce_weight) != 0.0 else {}
    out_dir = Path(args.out_dir or cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.train()

    noise_warmup_steps = int(args.noise_warmup_steps)
    if noise_warmup_steps < 0:
        raise ValueError("--noise-warmup-steps must be non-negative")
    if noise_warmup_steps > 0:
        vocab_size = int(getattr(cfg.model, "vocab_size", 0) or len(tokenizer))
        target_vocab_size = int(args.noise_warmup_target_vocab_size or vocab_size)
        print(
            "[noise_warmup] "
            f"steps={noise_warmup_steps} seq_len={int(args.noise_warmup_seq_len)} "
            f"batch={int(args.noise_warmup_batch_size)} core_steps={int(args.noise_warmup_core_steps)} "
            f"target_vocab_size={target_vocab_size}"
        )
        warmup_pbar = tqdm(range(noise_warmup_steps), desc="noise_warmup")
        for warmup_step in warmup_pbar:
            input_ids, attention_mask, target_ids = build_random_noise_warmup_batch(
                vocab_size=vocab_size,
                seq_len=int(args.noise_warmup_seq_len),
                batch_size=int(args.noise_warmup_batch_size),
                device=device,
                target_vocab_size=target_vocab_size,
            )
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                "cuda",
                enabled=(cfg.train.use_amp and device == "cuda"),
                dtype=torch.bfloat16,
            ):
                with torch.no_grad():
                    donor_out = donor.encode_inputs(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_logits=False,
                    )
                old_outer_steps = int(model.cfg.outer_steps)
                model.cfg.outer_steps = int(args.noise_warmup_core_steps)
                core_world_model_actions = None
                if bool(model.cfg.core_world_model_enabled):
                    core_world_model_actions = build_core_world_model_actions(
                        {"input_ids": input_ids},
                        num_steps=int(args.noise_warmup_core_steps),
                        num_actions=int(model.cfg.num_actions),
                        device=device,
                    )
                try:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        text_states=donor_out["text_states"].detach().to(device),
                        core_world_model_actions=core_world_model_actions,
                        return_core_depth_logits=True,
                        return_core_depth_text_logits=True,
                    )
                finally:
                    model.cfg.outer_steps = old_outer_steps
                offset = outputs["logits"].shape[1] - input_ids.shape[1]
                final_logits = outputs["logits"][
                    :,
                    offset + input_ids.shape[1] - 1,
                    :,
                ]
                depth_text_logits = outputs["core_depth_text_logits"][
                    :,
                    :,
                    input_ids.shape[1] - 1 : input_ids.shape[1],
                    :,
                ]
                loss, metrics = random_noise_warmup_loss(
                    final_logits,
                    depth_text_logits,
                    target_ids,
                    final_ce_weight=args.noise_warmup_final_ce_weight,
                    depth_ce_weight=args.noise_warmup_depth_ce_weight,
                    uniform_weight=args.noise_warmup_uniform_weight,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            if warmup_step % max(1, int(args.log_every)) == 0:
                warmup_pbar.set_description(
                    " ".join(
                        f"{key}={float(value):.4f}"
                        for key, value in {"noise_loss": loss.detach(), **metrics}.items()
                    )
                )

    pbar = tqdm(range(steps))
    for step in pbar:
        row_index, core_steps = scheduled_row_and_core_steps(
            step,
            row_count=len(rows),
            depth_steps=depth_steps,
            row_indices=curriculum_indices,
        )
        row = rows[row_index]
        prompt = str(row["prompt"])
        answer = target_for_core_steps(row, core_steps, target_mode=args.target_mode)
        temporal_spatial_context = _row_temporal_spatial_context(row, device=device)
        if bool(args.causal_prefix_supervision):
            train_examples = _prepare_causal_prefix_answer_examples(
                tokenizer,
                prompt,
                answer,
                max_length=max_length,
                device=device,
                max_target_tokens=args.causal_prefix_max_target_tokens,
            )
        else:
            train_examples = [
                _prepare_prompt_answer(
                    tokenizer,
                    prompt,
                    answer,
                    max_length=max_length,
                    device=device,
                )
            ]
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(cfg.train.use_amp and device == "cuda"), dtype=torch.bfloat16):
            losses = []
            loss_weights = []
            metric_sums = {}
            metric_counts = {}

            def add_metrics(new_metrics):
                for key, value in new_metrics.items():
                    detached = value.detach()
                    if key not in metric_sums:
                        metric_sums[key] = detached.new_zeros(())
                        metric_counts[key] = 0
                    metric_sums[key] = metric_sums[key] + detached
                    metric_counts[key] += 1

            for example_index, (
                input_ids,
                attention_mask,
                target_ids,
                target_start,
                target_end,
            ) in enumerate(train_examples):
                donor_out = donor.encode_inputs(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_logits=False,
                )
                old_outer_steps = int(model.cfg.outer_steps)
                model.cfg.outer_steps = int(core_steps)
                core_world_model_actions = None
                if bool(model.cfg.core_world_model_enabled):
                    core_world_model_actions = build_core_world_model_actions(
                        {"input_ids": input_ids},
                        num_steps=int(core_steps),
                        num_actions=int(model.cfg.num_actions),
                        device=device,
                    )
                context_off_depth_text_logits = None
                context_off_outputs = None
                transition_state_off_depth_text_logits = None
                transition_state_off_outputs = None
                try:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        text_states=donor_out["text_states"].detach().to(device),
                        core_world_model_actions=core_world_model_actions,
                        temporal_spatial_context=temporal_spatial_context,
                        return_core_depth_logits=True,
                        return_core_depth_text_logits=True,
                    )
                    if (
                        float(args.temporal_spatial_context_contrast_weight) != 0.0
                        and temporal_spatial_context is not None
                        and example_index == 0
                    ):
                        context_off_outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            text_states=donor_out["text_states"].detach().to(device),
                            core_world_model_actions=core_world_model_actions,
                            temporal_spatial_context=temporal_spatial_context,
                            disable_temporal_spatial_context=True,
                            return_core_depth_logits=True,
                            return_core_depth_text_logits=True,
                        )
                    if (
                        float(args.transition_state_contrast_weight) != 0.0
                        and example_index == 0
                    ):
                        transition_state_off_outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            text_states=donor_out["text_states"].detach().to(device),
                            core_world_model_actions=core_world_model_actions,
                            temporal_spatial_context=temporal_spatial_context,
                            disable_transition_state=True,
                            return_core_depth_logits=True,
                            return_core_depth_text_logits=True,
                        )
                finally:
                    model.cfg.outer_steps = old_outer_steps
                offset = outputs["logits"].shape[1] - input_ids.shape[1]
                final_text_logits = outputs["logits"][
                    :,
                    offset + target_start - 1 : offset + target_end - 1,
                    :,
                ]
                depth_text_logits = outputs["core_depth_text_logits"][
                    :,
                    :,
                    target_start - 1 : target_end - 1,
                    :,
                ]
                if context_off_depth_text_logits is None and (
                    float(args.temporal_spatial_context_contrast_weight) != 0.0
                    and temporal_spatial_context is not None
                    and example_index == 0
                ):
                    if context_off_outputs is None:
                        raise ValueError("context contrast requested but context-off output is missing")
                    context_off_depth_text_logits = context_off_outputs["core_depth_text_logits"][
                        :,
                        :,
                        target_start - 1 : target_end - 1,
                        :,
                    ]
                if transition_state_off_depth_text_logits is None and (
                    float(args.transition_state_contrast_weight) != 0.0
                    and example_index == 0
                ):
                    if transition_state_off_outputs is None:
                        raise ValueError(
                            "transition-state contrast requested but state-off output is missing"
                        )
                    transition_state_off_depth_text_logits = transition_state_off_outputs[
                        "core_depth_text_logits"
                    ][
                        :,
                        :,
                        target_start - 1 : target_end - 1,
                        :,
                    ]
                teacher_depth_text_logits = None
                if _should_apply_teacher_first_token_depth_kl(
                    example_index,
                    args.teacher_first_token_depth_kl_weight,
                ):
                    if teacher_model is None:
                        raise ValueError("teacher model is not loaded")
                    teacher_old_outer_steps = int(teacher_model.cfg.outer_steps)
                    teacher_model.cfg.outer_steps = int(core_steps)
                    teacher_temporal_spatial_context = temporal_spatial_context
                    try:
                        with torch.no_grad():
                            teacher_outputs = teacher_model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                text_states=donor_out["text_states"].detach().to(device),
                                temporal_spatial_context=teacher_temporal_spatial_context,
                                return_core_depth_logits=True,
                                return_core_depth_text_logits=True,
                            )
                    finally:
                        teacher_model.cfg.outer_steps = teacher_old_outer_steps
                    teacher_offset = teacher_outputs["logits"].shape[1] - input_ids.shape[1]
                    if teacher_offset != offset:
                        raise ValueError("teacher and student logit offsets must match")
                    teacher_depth_text_logits = teacher_outputs["core_depth_text_logits"][
                        :,
                        :,
                        target_start - 1 : target_end - 1,
                        :,
                    ]
                example_loss, example_metrics = depth_sequence_supervision_loss(
                    depth_text_logits,
                    final_text_logits,
                    target_ids,
                    final_logit_ce_weight=args.final_logit_ce_weight,
                    depth_final_ce_weight=args.depth_final_ce_weight,
                    all_depth_ce_weight=args.all_depth_ce_weight,
                    progress_margin_weight=args.progress_margin_weight,
                    progress_margin=args.progress_margin,
                )
                if context_off_depth_text_logits is not None:
                    context_contrast, context_metrics = context_ablation_contrastive_loss(
                        depth_text_logits,
                        context_off_depth_text_logits,
                        target_ids,
                        margin=args.temporal_spatial_context_contrast_margin,
                    )
                    example_loss = (
                        example_loss
                        + float(args.temporal_spatial_context_contrast_weight)
                        * context_contrast
                    )
                    example_metrics.update(context_metrics)
                if transition_state_off_depth_text_logits is not None:
                    transition_contrast, transition_metrics = (
                        transition_state_ablation_contrastive_loss(
                            depth_text_logits,
                            transition_state_off_depth_text_logits,
                            target_ids,
                            margin=args.transition_state_contrast_margin,
                        )
                    )
                    example_loss = (
                        example_loss
                        + float(args.transition_state_contrast_weight)
                        * transition_contrast
                    )
                    example_metrics.update(transition_metrics)
                if (
                    float(args.staged_internal_first_token_ce_weight) != 0.0
                    and example_index == 0
                ):
                    staged_target_ids = staged_internal_first_token_targets(
                        tokenizer,
                        row,
                        num_depths=int(depth_text_logits.shape[1]),
                        device=device,
                        target_mode=args.target_mode,
                    )
                    staged_ce, staged_metrics = staged_internal_first_token_ce_loss(
                        depth_text_logits,
                        staged_target_ids,
                    )
                    example_loss = (
                        example_loss
                        + float(args.staged_internal_first_token_ce_weight) * staged_ce
                    )
                    example_metrics.update(staged_metrics)
                if (
                    float(args.staged_internal_sequence_ce_weight) != 0.0
                    and example_index == 0
                ):
                    staged_sequence_target_ids = staged_internal_sequence_targets(
                        tokenizer,
                        row,
                        num_depths=int(depth_text_logits.shape[1]),
                        max_target_tokens=min(
                            int(args.staged_internal_sequence_max_target_tokens),
                            int(depth_text_logits.shape[2]),
                        ),
                        device=device,
                        target_mode=args.target_mode,
                    )
                    staged_sequence_ce, staged_sequence_metrics = (
                        staged_internal_sequence_ce_loss(
                            depth_text_logits,
                            staged_sequence_target_ids,
                        )
                    )
                    example_loss = (
                        example_loss
                        + float(args.staged_internal_sequence_ce_weight)
                        * staged_sequence_ce
                    )
                    example_metrics.update(staged_sequence_metrics)
                if float(args.transition_state_ce_weight) != 0.0 and example_index == 0:
                    transition_targets = staged_internal_first_token_targets(
                        tokenizer,
                        row,
                        num_depths=int(outputs["transition_state_text_logits"].shape[1]),
                        device=device,
                        target_mode=args.target_mode,
                    )
                    transition_ce, transition_metrics = transition_state_first_token_ce_loss(
                        outputs["transition_state_text_logits"],
                        transition_targets,
                    )
                    example_loss = (
                        example_loss
                        + float(args.transition_state_ce_weight) * transition_ce
                    )
                    example_metrics.update(transition_metrics)
                if (
                    float(args.transition_state_code_ce_weight) != 0.0
                    and example_index == 0
                ):
                    code_logits = outputs["transition_state_code_logits"]
                    code_targets = transition_state_code_targets(
                        tokenizer,
                        row,
                        num_depths=int(code_logits.shape[1]),
                        codebook_size=int(code_logits.shape[-1]),
                        device=device,
                        target_mode=args.target_mode,
                    )
                    code_ce, code_metrics = transition_state_code_ce_loss(
                        code_logits,
                        code_targets,
                    )
                    example_loss = (
                        example_loss
                        + float(args.transition_state_code_ce_weight) * code_ce
                    )
                    example_metrics.update(code_metrics)
                if (
                    float(args.transition_state_finality_ce_weight) != 0.0
                    and example_index == 0
                ):
                    finality_logits = outputs["transition_state_finality_logits"]
                    finality_targets = transition_state_finality_targets(
                        row,
                        num_depths=int(finality_logits.shape[1]),
                        device=device,
                    )
                    finality_ce, finality_metrics = transition_state_finality_bce_loss(
                        finality_logits,
                        finality_targets,
                    )
                    example_loss = (
                        example_loss
                        + float(args.transition_state_finality_ce_weight) * finality_ce
                    )
                    example_metrics.update(finality_metrics)
                if (
                    float(args.primitive_transition_operation_ce_weight) != 0.0
                    and example_index == 0
                ):
                    operation_logits = outputs["primitive_transition_operation_logits"]
                    operation_targets = primitive_transition_operation_targets(
                        row,
                        num_steps=int(operation_logits.shape[1]),
                        operation_to_id=primitive_operation_to_id,
                        device=device,
                    )
                    operation_ce, operation_metrics = primitive_transition_operation_ce_loss(
                        operation_logits,
                        operation_targets,
                    )
                    example_loss = (
                        example_loss
                        + float(args.primitive_transition_operation_ce_weight)
                        * operation_ce
                    )
                    example_metrics.update(operation_metrics)
                if core_world_model_weight != 0.0:
                    core_world_model_loss = jepa_world_model_loss(
                        outputs["core_world_model_pred"],
                        outputs["core_world_model_target"],
                        outputs["core_world_model_mask"],
                        latents=outputs.get("core_world_model_latents"),
                        latent_mask=outputs.get("core_world_model_latent_mask"),
                        sigreg=getattr(model, "core_world_model_sigreg", None),
                        sigreg_weight=float(model.cfg.core_world_model_sigreg_weight),
                    )
                    example_loss = (
                        example_loss + core_world_model_weight * core_world_model_loss
                    )
                    example_metrics["core_world_model"] = core_world_model_loss.detach()
                if teacher_depth_text_logits is not None:
                    teacher_depth_kl = depth_text_logit_distillation_loss(
                        depth_text_logits,
                        teacher_depth_text_logits,
                        temperature=args.teacher_depth_kl_temperature,
                    )
                    example_loss = (
                        example_loss
                        + float(args.teacher_first_token_depth_kl_weight) * teacher_depth_kl
                    )
                    example_metrics["teacher_first_token_depth_kl"] = teacher_depth_kl.detach()
                if (
                    float(args.choice_margin_weight) != 0.0
                    and row.get("rejected")
                ):
                    if str(args.choice_margin_mode) == "sequence":
                        rejected_ids = answer_token_ids(tokenizer, str(row["rejected"]))
                        if example_index < len(rejected_ids):
                            rejected_target_ids = input_ids.new_tensor(
                                [[int(rejected_ids[example_index])]]
                            )
                            margin_loss, margin_metrics = depth_choice_sequence_margin_loss(
                                depth_text_logits,
                                final_text_logits,
                                target_ids,
                                rejected_target_ids,
                                margin=args.choice_margin,
                                all_depth_weight=1.0,
                                final_weight=1.0,
                            )
                            example_loss = (
                                example_loss
                                + float(args.choice_margin_weight) * margin_loss
                            )
                            example_metrics.update(margin_metrics)
                    elif example_index == 0:
                        chosen_first = input_ids.new_tensor(
                            [
                                answer_first_token_id(
                                    tokenizer,
                                    str(row.get("chosen") or row.get("answer")),
                                )
                            ]
                        )
                        rejected_first = input_ids.new_tensor(
                            [answer_first_token_id(tokenizer, str(row["rejected"]))]
                        )
                        margin_loss, margin_metrics = depth_choice_margin_loss(
                            depth_text_logits,
                            final_text_logits,
                            chosen_first,
                            rejected_first,
                            margin=args.choice_margin,
                            all_depth_weight=1.0,
                            final_weight=1.0,
                        )
                        example_loss = (
                            example_loss
                            + float(args.choice_margin_weight) * margin_loss
                        )
                        example_metrics.update(margin_metrics)
                losses.append(example_loss)
                loss_weights.append(
                    _causal_prefix_example_loss_weight(
                        example_index,
                        args.causal_prefix_later_token_weight
                        if bool(args.causal_prefix_supervision)
                        else 1.0,
                    )
                )
                add_metrics(example_metrics)
            weighted_loss = losses[0] * float(loss_weights[0])
            for extra_loss, extra_weight in zip(losses[1:], loss_weights[1:]):
                weighted_loss = weighted_loss + extra_loss * float(extra_weight)
            loss = weighted_loss / max(1.0, float(sum(loss_weights)))
            metrics = {
                key: metric_sums[key] / float(metric_counts[key])
                for key in metric_sums
            }
            metrics["causal_prefix_examples"] = loss.detach().new_tensor(float(len(train_examples)))
            metrics["causal_prefix_later_token_weight"] = loss.detach().new_tensor(
                float(args.causal_prefix_later_token_weight)
                if bool(args.causal_prefix_supervision)
                else 1.0
            )
            metrics["core_world_model_weight"] = loss.detach().new_tensor(core_world_model_weight)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        if step % max(1, int(args.log_every)) == 0:
            metrics["core_steps"] = loss.detach().new_tensor(float(core_steps))
            pbar.set_description(
                " ".join(
                    f"{key}={float(value):.4f}"
                    for key, value in {
                        "loss": loss.detach(),
                        **metrics,
                    }.items()
                )
            )

    torch.save({"model": model.state_dict()}, out_dir / "last.pt")
    print(f"saved {out_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
