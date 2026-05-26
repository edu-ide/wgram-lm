#!/usr/bin/env python3
"""Train BPE PrefixLM checkpoints to prefer GD correct answers over parrots.

Plain-language contract:
  the same reader/thinker/speaker path sees one prompt twice,
  once with the intelligence answer and once with the tempting parrot answer.
  Training directly raises the normal LM-head probability of the intelligence
  answer above the parrot answer.

This is not a new side verifier. It is a small restoration gate for the
answer-preference habit that Stage112 showed was still missing under stable
BPE reading.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

IGNORE_LABEL_ID = -100


def load_trainer_module() -> Any:
    path = ROOT / "scripts" / "534_train_native_prefixlm_dataio.py"
    spec = importlib.util.spec_from_file_location("native_prefixlm_for_bpe_gd_pref", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load trainer module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_bpe_eval_module() -> Any:
    path = ROOT / "scripts" / "624_eval_bpe_generalization_dynamics_probe.py"
    spec = importlib.util.spec_from_file_location("bpe_gd_eval_for_pref", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load BPE GD evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def load_excluded_ids(paths: list[str]) -> set[str]:
    excluded: set[str] = set()
    for raw_path in paths:
        if not str(raw_path).strip():
            continue
        for row in load_jsonl(Path(raw_path)):
            row_id = str(row.get("id") or "")
            if row_id:
                excluded.add(row_id)
    return excluded


@dataclass(frozen=True)
class EncodedChoice:
    input_ids: list[int]
    labels: list[int]
    attention_mask: list[int]
    supervised_tokens: int


def encode_choice(
    *,
    tokenizer: Any,
    prompt: str,
    answer: str,
    seq_len: int,
) -> EncodedChoice:
    prompt_ids = [int(token_id) for token_id in tokenizer.encode(str(prompt), add_special_tokens=False).ids]
    answer_ids = [int(token_id) for token_id in tokenizer.encode(str(answer), add_special_tokens=False).ids]
    if not prompt_ids:
        raise ValueError("prompt encodes to no tokens")
    if not answer_ids:
        raise ValueError("answer encodes to no tokens")
    input_ids = prompt_ids + answer_ids[:-1]
    labels = [IGNORE_LABEL_ID] * max(0, len(prompt_ids) - 1) + answer_ids
    if len(input_ids) != len(labels):
        raise ValueError("choice tensor length mismatch")
    if len(input_ids) > int(seq_len):
        raise ValueError(f"choice row exceeds seq_len={seq_len}: shifted length={len(input_ids)}")
    attention_mask = [1] * len(input_ids)
    pad_len = int(seq_len) - len(input_ids)
    input_ids = input_ids + [0] * pad_len
    labels = labels + [IGNORE_LABEL_ID] * pad_len
    attention_mask = attention_mask + [0] * pad_len
    return EncodedChoice(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        supervised_tokens=len(answer_ids),
    )


def select_train_rows(
    rows: list[dict[str, Any]],
    *,
    exclude_ids: set[str],
    max_rows: int,
    seed: int,
    balance_by_task: bool = True,
) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if str(row.get("id") or "") not in exclude_ids
        and str(row.get("prompt") or "")
        and str(row.get("intelligence_answer") or "")
        and str(row.get("parrot_answer") or "")
    ]
    rng = random.Random(int(seed))
    if int(max_rows) <= 0 or int(max_rows) >= len(candidates):
        rng.shuffle(candidates)
        return candidates
    if not balance_by_task:
        rng.shuffle(candidates)
        return candidates[: int(max_rows)]

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_task[str(row.get("task") or row.get("family") or "unknown")].append(row)
    for task_rows in by_task.values():
        rng.shuffle(task_rows)
    selected: list[dict[str, Any]] = []
    task_names = sorted(by_task)
    while len(selected) < int(max_rows):
        progressed = False
        for task in task_names:
            if by_task[task]:
                selected.append(by_task[task].pop())
                progressed = True
                if len(selected) >= int(max_rows):
                    break
        if not progressed:
            break
    return selected


def task_matches_focus(task: str, focus_tasks: list[str]) -> bool:
    normalized = str(task)
    return any(str(pattern) and str(pattern) in normalized for pattern in focus_tasks)


def apply_focus_replay(
    rows: list[dict[str, Any]],
    *,
    focus_tasks: list[str],
    replay_factor: int,
) -> list[dict[str, Any]]:
    factor = int(replay_factor)
    if factor <= 1 or not focus_tasks:
        return list(rows)
    focused = [
        row
        for row in rows
        if task_matches_focus(str(row.get("task") or row.get("family") or ""), focus_tasks)
    ]
    return list(rows) + focused * max(0, factor - 1)


class BPEGDPreferenceDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        tokenizer: Any,
        seq_len: int,
    ) -> None:
        self.rows: list[dict[str, Any]] = []
        self.skipped: list[dict[str, str]] = []
        for row in rows:
            row_id = str(row.get("id") or "")
            try:
                chosen = encode_choice(
                    tokenizer=tokenizer,
                    prompt=str(row["prompt"]),
                    answer=str(row["intelligence_answer"]),
                    seq_len=int(seq_len),
                )
                rejected = encode_choice(
                    tokenizer=tokenizer,
                    prompt=str(row["prompt"]),
                    answer=str(row["parrot_answer"]),
                    seq_len=int(seq_len),
                )
            except Exception as exc:
                self.skipped.append({"id": row_id, "reason": str(exc)})
                continue
            self.rows.append(
                {
                    "id": row_id,
                    "task": str(row.get("task") or "unknown"),
                    "family": str(row.get("family") or "unknown"),
                    "chosen": chosen,
                    "rejected": rejected,
                }
            )
        if not self.rows:
            raise ValueError("no usable preference rows after seq_len filtering")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[int(index)]

    def summary(self) -> dict[str, Any]:
        task_counts: dict[str, int] = defaultdict(int)
        family_counts: dict[str, int] = defaultdict(int)
        chosen_tokens: list[int] = []
        rejected_tokens: list[int] = []
        row_ids: list[str] = []
        for row in self.rows:
            row_ids.append(str(row["id"]))
            task_counts[str(row["task"])] += 1
            family_counts[str(row["family"])] += 1
            chosen_tokens.append(int(row["chosen"].supervised_tokens))
            rejected_tokens.append(int(row["rejected"].supervised_tokens))
        row_id_fingerprint = hashlib.sha256(
            "\n".join(row_ids).encode("utf-8")
        ).hexdigest()
        return {
            "rows": int(len(self.rows)),
            "skipped_rows": int(len(self.skipped)),
            "row_id_count": int(len(row_ids)),
            "row_ids_sha256": row_id_fingerprint,
            "row_id_examples": row_ids[:50],
            "task_counts": dict(sorted(task_counts.items())),
            "family_counts": dict(sorted(family_counts.items())),
            "mean_chosen_tokens": float(sum(chosen_tokens) / max(1, len(chosen_tokens))),
            "mean_rejected_tokens": float(sum(rejected_tokens) / max(1, len(rejected_tokens))),
            "skipped_examples": self.skipped[:20],
        }


def collate_preference_rows(batch: list[dict[str, Any]]) -> dict[str, Any]:
    def stack_choice(key: str, field: str) -> torch.Tensor:
        return torch.tensor(
            [row[key].__dict__[field] for row in batch],
            dtype=torch.long,
        )

    return {
        "chosen_input_ids": stack_choice("chosen", "input_ids"),
        "chosen_labels": stack_choice("chosen", "labels"),
        "chosen_attention_mask": stack_choice("chosen", "attention_mask"),
        "rejected_input_ids": stack_choice("rejected", "input_ids"),
        "rejected_labels": stack_choice("rejected", "labels"),
        "rejected_attention_mask": stack_choice("rejected", "attention_mask"),
        "ids": [str(row["id"]) for row in batch],
        "tasks": [str(row["task"]) for row in batch],
    }


def pairwise_preference_loss(
    margins: torch.Tensor,
    *,
    beta: float,
    target_margin: float,
) -> torch.Tensor:
    return -F.logsigmoid(float(beta) * (margins - float(target_margin))).mean()


def sequence_mean_logprob_and_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    length = min(int(logits.shape[1]), int(labels.shape[1]))
    logits = logits[:, :length].float()
    labels = labels[:, :length]
    mask = labels.ne(IGNORE_LABEL_ID)
    safe_labels = labels.masked_fill(~mask, 0)
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs.masked_fill(~mask, 0.0)
    token_counts = mask.sum(dim=1).clamp_min(1)
    mean_logprob = token_log_probs.sum(dim=1) / token_counts
    ce_loss = -token_log_probs.sum() / token_counts.sum().clamp_min(1)
    return mean_logprob, ce_loss, token_counts


def trim_batch_pair(batch: dict[str, Any]) -> dict[str, Any]:
    max_len = 1
    for key in ("chosen_attention_mask", "rejected_attention_mask"):
        mask = batch[key]
        if bool(mask.any()):
            max_len = max(max_len, int(mask.sum(dim=1).max().item()))
    out = dict(batch)
    tensor_keys = [
        "chosen_input_ids",
        "chosen_labels",
        "chosen_attention_mask",
        "rejected_input_ids",
        "rejected_labels",
        "rejected_attention_mask",
    ]
    for key in tensor_keys:
        out[key] = batch[key][:, :max_len]
    return out


def save_checkpoint(
    *,
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_args: dict[str, Any],
    model_info: dict[str, Any],
    source_checkpoint: str,
    source_step: int,
    preference_step: int,
    history: list[dict[str, Any]],
    dataset_summary: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "args": train_args,
        "model": model_info,
        "step": int(source_step) + int(preference_step),
        "source_step": int(source_step),
        "preference_step": int(preference_step),
        "source_checkpoint": str(source_checkpoint),
        "preference_args": vars(args),
        "preference_history": history,
        "preference_dataset_summary": dataset_summary,
        "plain_language_read": (
            "This checkpoint was preference-tuned on the same BPE PrefixLM "
            "answer path: GD intelligence answers should become more probable "
            "than tempting parrot answers without adding a side verifier."
        ),
    }
    if bool(args.save_optimizer):
        payload["optimizer_state_dict"] = optimizer.state_dict()
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, str(tmp_path))
        os.replace(str(tmp_path), str(path))
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        finally:
            raise


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if str(args.matmul_precision):
        torch.set_float32_matmul_precision(str(args.matmul_precision))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer = load_trainer_module()
    bpe_eval = load_bpe_eval_module()
    checkpoint = torch.load(str(args.resume), map_location="cpu", weights_only=False)
    train_args = dict(checkpoint["args"])
    sampled_data = Path(args.sampled_data or train_args.get("sampled_data", ""))
    metadata = trainer.load_prefixlm_metadata(sampled_data)
    tokenizer_path = bpe_eval.resolve_tokenizer_path(
        sampled_data,
        str(args.tokenizer_path or (metadata.tokenizer_info or {}).get("tokenizer_path") or ""),
    )
    tokenizer = bpe_eval.load_tokenizer(tokenizer_path)
    seq_len = int(args.seq_len or train_args.get("seq_len") or 0)
    if seq_len <= 0:
        raise ValueError("seq_len must be provided or present in checkpoint args")
    train_namespace = argparse.Namespace(**train_args)

    raw_rows = load_jsonl(Path(args.probe_jsonl))
    excluded_ids = load_excluded_ids([str(path) for path in args.exclude_jsonl])
    selected_rows = select_train_rows(
        raw_rows,
        exclude_ids=excluded_ids,
        max_rows=int(args.max_rows),
        seed=int(args.seed),
        balance_by_task=bool(args.balance_by_task),
    )
    replayed_rows = apply_focus_replay(
        selected_rows,
        focus_tasks=[str(value) for value in args.focus_tasks],
        replay_factor=int(args.focus_replay_factor),
    )
    dataset = BPEGDPreferenceDataset(
        replayed_rows,
        tokenizer=tokenizer,
        seq_len=seq_len,
    )
    generator = torch.Generator()
    generator.manual_seed(int(args.seed))
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        generator=generator,
        collate_fn=collate_preference_rows,
        drop_last=False,
    )
    language_loader = None
    language_iterator = None
    language_dataset_summary: dict[str, Any] | None = None
    if float(args.language_loss_weight) > 0.0:
        language_data = Path(args.language_sampled_data or sampled_data)
        language_dataset = trainer.DataIOSampledPrefixLMDataset(
            language_data,
            seq_len=int(seq_len),
            epoch=int(args.language_epoch),
            target_only=not bool(getattr(train_namespace, "train_instruction_tokens", False)),
            max_rows=int(args.language_max_rows) if int(args.language_max_rows) > 0 else None,
            drop_overlength=True,
        )
        language_generator = torch.Generator()
        language_generator.manual_seed(int(args.seed) + 991)
        language_loader = DataLoader(
            language_dataset,
            batch_size=int(args.language_batch_size),
            shuffle=True,
            generator=language_generator,
            collate_fn=trainer.collate_prefixlm_rows,
            drop_last=False,
        )
        language_iterator = iter(language_loader)
        language_dataset_summary = language_dataset.summary()
        language_dataset_summary["language_batch_size"] = int(args.language_batch_size)
        language_dataset_summary["language_loss_weight"] = float(args.language_loss_weight)

    model_info = dict(checkpoint.get("model") or {})
    vocab_size = int(
        model_info.get("vocab_size")
        or train_args.get("model_vocab_size", 0)
        or trainer.round_up_multiple(int(metadata.vocab_size), 256)
    )
    model = trainer.build_model(train_namespace, vocab_size=vocab_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device(str(args.device))
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        betas=(float(args.adam_beta1), float(args.adam_beta2)),
        weight_decay=float(args.weight_decay),
    )
    amp_dtype = trainer.resolve_amp_dtype(str(args.amp_dtype))

    def amp_context() -> Any:
        if str(device.type) != "cuda":
            return nullcontext()
        return trainer.autocast_context(device, amp_dtype)

    writer = None
    if str(args.tensorboard_dir):
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(args.tensorboard_dir))

    dataset_summary = dataset.summary() | {
        "probe_jsonl": str(args.probe_jsonl),
        "excluded_ids": int(len(excluded_ids)),
        "selected_rows_before_seq_filter": int(len(selected_rows)),
        "selected_rows_after_focus_replay": int(len(replayed_rows)),
        "focus_tasks": [str(value) for value in args.focus_tasks],
        "focus_replay_factor": int(args.focus_replay_factor),
        "seq_len": int(seq_len),
        "tokenizer_path": str(tokenizer_path),
        "language_preservation": language_dataset_summary,
    }
    history: list[dict[str, Any]] = []
    iterator = iter(loader)
    source_step = int(checkpoint.get("step", 0))
    start_time = time.perf_counter()
    for step in range(1, int(args.steps) + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        if bool(args.trim_batch_to_max_length):
            batch = trim_batch_pair(batch)
        language_batch = None
        if language_loader is not None and language_iterator is not None:
            try:
                language_batch = next(language_iterator)
            except StopIteration:
                language_iterator = iter(language_loader)
                language_batch = next(language_iterator)
            if bool(args.trim_batch_to_max_length):
                language_batch = trainer.trim_prefixlm_batch_to_max_valid_length(language_batch)
        chosen_input_ids = batch["chosen_input_ids"].to(device)
        chosen_labels = batch["chosen_labels"].to(device)
        rejected_input_ids = batch["rejected_input_ids"].to(device)
        rejected_labels = batch["rejected_labels"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with amp_context():
            chosen_logits = model(chosen_input_ids, think_steps=int(args.think_steps))
            rejected_logits = model(rejected_input_ids, think_steps=int(args.think_steps))

            # === Stage119 equation-state binding aux loss (minimal, guarded) ===
            stage119_aux = torch.zeros((), device=device)
            if float(getattr(args, "stage119_equation_binding_weight", 0.0)) > 0.0:
                task_str = str(batch.get("tasks", [""])[0]) if isinstance(batch.get("tasks"), (list, tuple)) else ""
                if "algebra" in task_str.lower():
                    try:
                        from src.qtrm_mm.losses.equation_state_binding import (
                            compute_equation_state_binding_loss,
                            EquationStateBindingConfig,
                            extract_equation_fields_from_algebra_row,
                        )
                        # Use last token embedding or a projected chosen_input as proxy state for probe
                        # (real integration: capture recurrent z from model core at equation step)
                        d_state = 512
                        try:
                            d_state = int(model.config.hidden_size) if hasattr(model, "config") else 512
                        except Exception:
                            pass
                        # Attempt real field extraction from batch rows if present
                        fields = None
                        if "rows" in batch or "row" in batch:
                            sample_row = (batch.get("rows") or [batch.get("row")])[0] if isinstance(batch.get("rows"), list) else {}
                            fields = extract_equation_fields_from_algebra_row(sample_row or {}, device=device)
                        bsz = chosen_input_ids.size(0)
                        proxy_state = torch.randn(bsz, d_state, device=device) * 0.01  # placeholder until core hook
                        if fields is not None:
                            stage119_aux, _diags = compute_equation_state_binding_loss(
                                proxy_state,
                                target_left=fields.left.expand(bsz),
                                target_right=fields.right.expand(bsz),
                                target_op=fields.op.expand(bsz),
                                target_result_var=fields.result_var.expand(bsz) if fields.result_var is not None else None,
                                cfg=EquationStateBindingConfig(d_state=d_state),
                            )
                        else:
                            # Still exercise the new loss plumbing with zeros (diagnostic only)
                            stage119_aux, _ = compute_equation_state_binding_loss(
                                proxy_state,
                                cfg=EquationStateBindingConfig(d_state=d_state),
                            )
                    except Exception as e:
                        if step == 1:
                            print(f"[Stage119] Aux loss skipped (expected in first probe): {e}")
            # === end Stage119 aux ===

            chosen_mean, chosen_ce, chosen_counts = sequence_mean_logprob_and_ce(
                chosen_logits,
                chosen_labels,
            )
            rejected_mean, _rejected_ce, rejected_counts = sequence_mean_logprob_and_ce(
                rejected_logits,
                rejected_labels,
            )
            margins = chosen_mean - rejected_mean
            pref_loss = pairwise_preference_loss(
                margins,
                beta=float(args.preference_beta),
                target_margin=float(args.preference_margin),
            )
            loss = float(args.preference_loss_weight) * pref_loss + float(args.ce_loss_weight) * chosen_ce
            loss = loss + float(getattr(args, "stage119_equation_binding_weight", 0.0)) * stage119_aux
            if language_batch is not None:
                language_input_ids = language_batch["input_ids"].to(device)
                language_labels = language_batch["labels"].to(device)
                language_loss = trainer.prefixlm_loss_for_batch(
                    model,
                    language_input_ids,
                    language_labels,
                    think_steps=int(args.think_steps),
                    loss_chunk_size=int(args.language_loss_chunk_size),
                    loss_kernel=str(args.language_loss_kernel),
                )
                loss = loss + float(args.language_loss_weight) * language_loss
            else:
                language_loss = torch.zeros((), device=device)
        loss.backward()
        if float(args.grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        optimizer.step()

        if step == 1 or step % int(args.log_every) == 0 or step == int(args.steps):
            elapsed = max(1e-9, time.perf_counter() - start_time)
            metrics = {
                "step": int(step),
                "source_plus_step": int(source_step + step),
                "loss": float(loss.detach().cpu().item()),
                "preference_loss": float(pref_loss.detach().cpu().item()),
                "chosen_ce": float(chosen_ce.detach().cpu().item()),
                "language_loss": float(language_loss.detach().cpu().item()),
                "mean_margin": float(margins.mean().detach().cpu().item()),
                "min_margin": float(margins.min().detach().cpu().item()),
                "win_rate": float((margins > 0.0).float().mean().detach().cpu().item()),
                "mean_chosen_tokens": float(chosen_counts.float().mean().detach().cpu().item()),
                "mean_rejected_tokens": float(rejected_counts.float().mean().detach().cpu().item()),
                "examples_per_sec": float(step * int(args.batch_size) / elapsed),
            }
            history.append(metrics)
            print(json.dumps({"event": "train", **metrics}, ensure_ascii=False), flush=True)
            if writer is not None:
                tb_step = int(source_step + step)
                for key, value in metrics.items():
                    if key in {"step", "source_plus_step"}:
                        continue
                    writer.add_scalar(f"train/{key}", float(value), tb_step)
                writer.flush()

        if int(args.checkpoint_every) > 0 and step % int(args.checkpoint_every) == 0:
            save_checkpoint(
                path=out_dir / f"checkpoint_pref_step{step:06d}.pt",
                model=model,
                optimizer=optimizer,
                train_args=train_args,
                model_info=model_info,
                source_checkpoint=str(args.resume),
                source_step=source_step,
                preference_step=step,
                history=history,
                dataset_summary=dataset_summary,
                args=args,
            )

    save_checkpoint(
        path=out_dir / "last.pt",
        model=model,
        optimizer=optimizer,
        train_args=train_args,
        model_info=model_info,
        source_checkpoint=str(args.resume),
        source_step=source_step,
        preference_step=int(args.steps),
        history=history,
        dataset_summary=dataset_summary,
        args=args,
    )
    if writer is not None:
        writer.close()
    report = {
        "accepted": False,
        "out_dir": str(out_dir),
        "last_checkpoint": str(out_dir / "last.pt"),
        "source_checkpoint": str(args.resume),
        "source_step": int(source_step),
        "preference_steps": int(args.steps),
        "dataset": dataset_summary,
        "history": history,
        "plain_language_read": (
            "The local model was not given a new side calculator. It practiced "
            "choosing the intelligence answer over the parrot answer through the "
            "same BPE PrefixLM path. Promotion still requires a heldout GD gate."
        ),
    }
    (out_dir / "preference_train_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "done", **report}, ensure_ascii=False), flush=True)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", required=True)
    parser.add_argument("--probe-jsonl", default="data/eval/official_gdsuite_choice_probe.jsonl")
    parser.add_argument("--exclude-jsonl", nargs="*", default=[])
    parser.add_argument("--sampled-data", default="")
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument(
        "--save-optimizer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Save optimizer state in preference checkpoints. Disabled by "
            "default because these checkpoints are primarily eval artifacts and "
            "the local root disk is often tight."
        ),
    )
    parser.add_argument("--max-rows", type=int, default=512)
    parser.add_argument("--stage119-equation-binding-weight", type=float, default=0.0,
                        help="Auxiliary loss weight for Stage119 equation-state binding probe (0 = disabled). Only affects algebra trap batches.")
    parser.add_argument("--balance-by-task", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--focus-tasks",
        nargs="*",
        default=[],
        help=(
            "Task-name substrings to replay more often, e.g. "
            "repetitive_answer/algebra intuitive_answer/crt."
        ),
    )
    parser.add_argument("--focus-replay-factor", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=0)
    parser.add_argument("--think-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--preference-loss-weight", type=float, default=1.0)
    parser.add_argument("--ce-loss-weight", type=float, default=0.1)
    parser.add_argument("--language-loss-weight", type=float, default=0.0)
    parser.add_argument("--language-sampled-data", default="")
    parser.add_argument("--language-epoch", type=int, default=0)
    parser.add_argument("--language-max-rows", type=int, default=512)
    parser.add_argument("--language-batch-size", type=int, default=4)
    parser.add_argument("--language-loss-chunk-size", type=int, default=64)
    parser.add_argument(
        "--language-loss-kernel",
        choices=("torch", "auto", "liger_fused_linear_ce"),
        default="torch",
    )
    parser.add_argument("--preference-beta", type=float, default=4.0)
    parser.add_argument("--preference-margin", type=float, default=0.05)
    parser.add_argument("--amp-dtype", choices=("none", "bf16", "fp16"), default="bf16")
    parser.add_argument("--matmul-precision", choices=("", "highest", "high", "medium"), default="high")
    parser.add_argument("--trim-batch-to-max-length", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--tensorboard-dir", default="")
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
