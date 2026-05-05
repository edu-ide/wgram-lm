from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .qwen35_full_msa import (
    Qwen35FullMsaForCausalLM,
    copy_qwen35_text_weights_into_full_msa,
)


def build_tiny_healing_models(*, seed: int = 0) -> tuple[nn.Module, Qwen35FullMsaForCausalLM]:
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    source_cfg = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        layer_types=["linear_attention", "full_attention"],
        max_position_embeddings=128,
        pad_token_id=0,
    )
    target_cfg = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        layer_types=["sparse", "sparse"],
        max_position_embeddings=128,
        pad_token_id=0,
    )
    target_cfg.qtrm_original_layer_types = ["linear_attention", "full_attention"]
    target_cfg.qtrm_full_msa_fork = True
    target_cfg.msa_config = {
        "top_k_docs": 1,
        "pooling_kernel_size": 2,
        "head_reduce_method": "mean",
        "query_reduce_method": "max",
        "chunk_reduce_method": "max",
        "decouple_router": True,
        "aux_loss_method": "INFONCE",
    }

    torch.manual_seed(seed)
    source = Qwen3_5ForCausalLM(source_cfg)
    torch.manual_seed(seed + 1)
    target = Qwen35FullMsaForCausalLM(target_cfg)
    copy_qwen35_text_weights_into_full_msa(source, target)
    return source, target


def freeze_for_stage1_healing(model: Qwen35FullMsaForCausalLM) -> dict[str, int]:
    """Freeze stable copied backbone weights and train only MSA attention/router.

    Stage 1 is intentionally conservative: embeddings, MLPs, norms, and lm_head
    remain fixed while replaced/seeded MSA attention learns not to destroy the
    donor language distribution.
    """

    trainable_param_count = 0
    frozen_param_count = 0
    for name, param in model.named_parameters():
        trainable = ".self_attn." in name
        param.requires_grad_(trainable)
        if trainable:
            trainable_param_count += param.numel()
        else:
            frozen_param_count += param.numel()
    return {
        "trainable_param_count": trainable_param_count,
        "frozen_param_count": frozen_param_count,
    }


def synthetic_doc_batch(
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    num_docs: int = 2,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    if seq_len < num_docs * 2 + 2:
        raise ValueError("seq_len must leave room for document and query tokens")
    generator = torch.Generator(device="cpu").manual_seed(batch_size * 1000 + seq_len + vocab_size)
    input_ids = torch.randint(
        3,
        vocab_size,
        (batch_size, seq_len),
        generator=generator,
        dtype=torch.long,
    ).to(device)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)
    doc_ids = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
    doc_span = max(1, (seq_len - 2) // num_docs)
    cursor = 0
    for doc_id in range(1, num_docs + 1):
        end = min(cursor + doc_span, seq_len - 2)
        if cursor < end:
            doc_ids[:, cursor:end] = doc_id
        cursor = end
    labels = input_ids.clone()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "doc_ids": doc_ids,
        "labels": labels,
    }


def qwen35_full_msa_healing_loss(
    teacher: nn.Module,
    student: Qwen35FullMsaForCausalLM,
    batch: dict[str, torch.Tensor],
    *,
    lm_weight: float = 1.0,
    donor_kl_weight: float = 1.0,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    student_out = student(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
        doc_ids=batch.get("doc_ids"),
        labels=batch.get("labels"),
    )
    with torch.no_grad():
        teacher_out = teacher(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
        )

    lm_loss = student_out.loss
    if lm_loss is None:
        lm_loss = F.cross_entropy(
            student_out.logits[:, :-1, :].contiguous().view(-1, student_out.logits.shape[-1]),
            batch["labels"][:, 1:].contiguous().view(-1),
            ignore_index=-100,
        )
    donor_kl = _next_token_kl(
        student_logits=student_out.logits,
        teacher_logits=teacher_out.logits,
        temperature=temperature,
    )
    loss = float(lm_weight) * lm_loss + float(donor_kl_weight) * donor_kl
    return loss, {
        "loss": float(loss.detach().cpu().item()),
        "lm_loss": float(lm_loss.detach().cpu().item()),
        "donor_kl": float(donor_kl.detach().cpu().item()),
    }


def run_tiny_healing_smoke(
    *,
    out_dir: str | Path,
    steps: int = 2,
    batch_size: int = 2,
    seq_len: int = 8,
    lr: float = 1e-3,
    lm_weight: float = 1.0,
    donor_kl_weight: float = 1.0,
    temperature: float = 1.0,
    seed: int = 0,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    teacher, student = build_tiny_healing_models(seed=seed)
    teacher.eval()
    student.train()
    freeze_summary = freeze_for_stage1_healing(student)
    initial_trainable = {
        name: param.detach().clone()
        for name, param in student.named_parameters()
        if param.requires_grad
    }
    optimizer = torch.optim.AdamW(
        [param for param in student.parameters() if param.requires_grad],
        lr=lr,
    )

    metrics_history = []
    for _ in range(steps):
        batch = synthetic_doc_batch(batch_size=batch_size, seq_len=seq_len, vocab_size=student.config.vocab_size)
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = qwen35_full_msa_healing_loss(
            teacher,
            student,
            batch,
            lm_weight=lm_weight,
            donor_kl_weight=donor_kl_weight,
            temperature=temperature,
        )
        loss.backward()
        optimizer.step()
        metrics_history.append(metrics)

    updated_trainable_l1 = 0.0
    for name, initial in initial_trainable.items():
        current = dict(student.named_parameters())[name].detach()
        updated_trainable_l1 += float((current - initial).abs().sum().cpu().item())

    report = {
        "mode": "tiny_smoke",
        "steps": steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "lr": lr,
        "lm_weight": lm_weight,
        "donor_kl_weight": donor_kl_weight,
        "temperature": temperature,
        "freeze_summary": freeze_summary,
        "updated_trainable_l1": updated_trainable_l1,
        "metrics_history": metrics_history,
        "final_metrics": metrics_history[-1] if metrics_history else {},
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "healing_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _next_token_kl(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "student/teacher logits must have the same shape, got "
            f"{tuple(student_logits.shape)} and {tuple(teacher_logits.shape)}"
        )
    temp = float(temperature)
    if temp <= 0:
        raise ValueError("temperature must be positive")
    student = student_logits[:, :-1, :] / temp
    teacher = teacher_logits[:, :-1, :] / temp
    return F.kl_div(
        F.log_softmax(student, dim=-1),
        F.softmax(teacher, dim=-1),
        reduction="batchmean",
    ) * (temp * temp)
