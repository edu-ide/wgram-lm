#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
import torch.nn.functional as F


def _load_raw_eval_module():
    path = Path(__file__).with_name("192_eval_raw_intelligence.py")
    spec = importlib.util.spec_from_file_location("qtrm_raw_eval_192", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CoreSoftPrefixAdapter(nn.Module):
    """Map QTRM core state to virtual donor embedding tokens."""

    def __init__(
        self,
        *,
        core_dim: int,
        donor_dim: int,
        prefix_tokens: int,
        rank: int = 64,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.prefix_tokens = max(1, int(prefix_tokens))
        self.donor_dim = int(donor_dim)
        self.scale = float(scale)
        self.norm = nn.LayerNorm(int(core_dim), elementwise_affine=False)
        self.down = nn.Linear(int(core_dim), max(1, int(rank)), bias=False)
        self.up = nn.Linear(max(1, int(rank)), self.prefix_tokens * self.donor_dim, bias=False)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.up.weight, mean=0.0, std=0.002)

    def forward(self, core_hidden: torch.Tensor) -> torch.Tensor:
        x = self.norm(core_hidden.float())
        out = self.up(F.gelu(self.down(x))) * self.scale
        return out.view(core_hidden.shape[0], self.prefix_tokens, self.donor_dim)


class StateConditionedSoftPrefixAdapter(nn.Module):
    """Map QTRM core state plus explicit value-state features to donor prefix tokens."""

    def __init__(
        self,
        *,
        core_dim: int,
        state_dim: int,
        donor_dim: int,
        prefix_tokens: int,
        rank: int = 64,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.prefix_tokens = max(1, int(prefix_tokens))
        self.donor_dim = int(donor_dim)
        self.state_dim = max(0, int(state_dim))
        in_dim = int(core_dim) + self.state_dim
        self.scale = float(scale)
        self.norm = nn.LayerNorm(in_dim, elementwise_affine=False)
        self.down = nn.Linear(in_dim, max(1, int(rank)), bias=False)
        self.up = nn.Linear(max(1, int(rank)), self.prefix_tokens * self.donor_dim, bias=False)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.up.weight, mean=0.0, std=0.002)

    def forward(self, core_hidden: torch.Tensor, state_features: torch.Tensor) -> torch.Tensor:
        if self.state_dim > 0:
            state = state_features.float()
            if state.ndim == 1:
                state = state.view(1, -1)
            if int(state.shape[-1]) != self.state_dim:
                raise ValueError(
                    f"state feature dim mismatch: got {int(state.shape[-1])}, "
                    f"expected {self.state_dim}"
                )
            x = torch.cat([core_hidden.float(), state], dim=-1)
        else:
            x = core_hidden.float()
        x = self.norm(x)
        out = self.up(F.gelu(self.down(x))) * self.scale
        return out.view(core_hidden.shape[0], self.prefix_tokens, self.donor_dim)


def find_input_embeddings(model: nn.Module) -> nn.Module:
    getter = getattr(model, "get_input_embeddings", None)
    if callable(getter):
        embeds = getter()
        if embeds is not None:
            return embeds
    for path in ("embed_tokens", "model.embed_tokens", "language_model.embed_tokens"):
        cur: Any = model
        ok = True
        for part in path.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok and isinstance(cur, nn.Module):
            return cur
    raise RuntimeError("could not locate donor input embeddings")


def _gold_answer(case: dict[str, Any]) -> str:
    for key in ("answer", "chosen", "canonical_answer"):
        value = str(case.get(key) or "").strip()
        if value:
            return value
    for alias in case.get("answer_aliases") or []:
        value = str(alias).strip()
        if value:
            return value
    raise ValueError(f"case has no answer: {case.get('id')}")


def _answer_token_ids(
    tokenizer,
    answer: str,
    *,
    max_target_tokens: int,
    append_eos_target: bool = False,
) -> list[int]:
    ids = tokenizer.encode(f" {answer}", add_special_tokens=False)
    if not ids:
        ids = tokenizer.encode(str(answer), add_special_tokens=False)
    ids = [int(token_id) for token_id in ids]
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if append_eos_target and eos_id is not None:
        if max_target_tokens > 0:
            ids = ids[: max(1, int(max_target_tokens) - 1)]
        ids.append(int(eos_id))
    elif max_target_tokens > 0:
        ids = ids[: int(max_target_tokens)]
    if not ids:
        raise ValueError(f"answer has no token ids: {answer!r}")
    return ids


def _last_core_hidden(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    for key in ("core_loop_readout_hidden", "answer_state_loop_hidden", "answer_bottleneck_hidden"):
        value = outputs.get(key)
        if value is not None and value.numel() != 0:
            return value[:, -1, :].detach()
    raise RuntimeError("QTRM output has no usable core/readout hidden state")


def _empty_state_features() -> torch.Tensor:
    return torch.empty(0, dtype=torch.float32)


def _state_features_from_outputs(
    outputs: dict[str, torch.Tensor],
    *,
    key: str,
    mode: str,
) -> torch.Tensor:
    if not key:
        return _empty_state_features()
    keys = [part.strip() for part in str(key).split(",") if part.strip()]
    if len(keys) > 1:
        parts = [
            _state_features_from_outputs(outputs, key=part, mode=mode)
            for part in keys
        ]
        parts = [part for part in parts if int(part.numel()) > 0]
        if not parts:
            return _empty_state_features()
        return torch.cat(parts, dim=0)
    value = outputs.get(key)
    if value is None or not torch.is_tensor(value) or value.numel() == 0:
        return _empty_state_features()
    x = value.detach().float()
    if x.ndim >= 3:
        x = x[:, -1]
    elif x.ndim == 2:
        x = x[:, -1:]
    if mode == "softmax":
        if x.ndim >= 2 and int(x.shape[-1]) > 1:
            x = F.softmax(x, dim=-1)
    elif mode == "argmax_onehot":
        if x.ndim < 2 or int(x.shape[-1]) <= 1:
            x = torch.zeros_like(x)
        else:
            x = F.one_hot(x.argmax(dim=-1), num_classes=int(x.shape[-1])).float()
    elif mode == "logits":
        flat = x.reshape(x.shape[0], -1)
        x = (flat - flat.mean(dim=-1, keepdim=True)) / flat.std(dim=-1, keepdim=True).clamp_min(1e-6)
    else:
        raise ValueError(f"unknown state feature mode: {mode}")
    return x.reshape(x.shape[0], -1)[0].cpu()


def _runtime_kwargs(runtime: dict[str, Any]) -> dict[str, Any]:
    raw_eval = _load_raw_eval_module()
    return {
        "disable_core": bool(runtime.get("disable_core", False)),
        "enable_core_halt": raw_eval._runtime_enable_core_halt(runtime),
        "disable_qtrm_residual": bool(runtime.get("disable_qtrm_residual", False)),
        "disable_qtrm_residual_gate": bool(runtime.get("disable_qtrm_residual_gate", False)),
        "disable_transition_state": bool(runtime.get("disable_transition_state", False)),
        "disable_core_role_value_answer_bridge": bool(
            runtime.get("disable_core_role_value_answer_bridge", False)
        ),
        "disable_answer_state_loop_recurrent": bool(
            runtime.get("disable_answer_state_loop_recurrent", False)
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
        "disable_answer_state_loop_talker": bool(
            runtime.get("disable_answer_state_loop_talker", False)
        ),
    }


@torch.no_grad()
def build_rows(
    *,
    cases: list[dict[str, Any]],
    tokenizer,
    donor,
    qtrm_model,
    device: str,
    core_steps: int,
    max_length: int,
    max_target_tokens: int,
    append_eos_target: bool = False,
    state_logits_key: str = "",
    state_feature_mode: str = "softmax",
) -> list[dict[str, Any]]:
    raw_eval = _load_raw_eval_module()
    old_outer_steps = int(qtrm_model.cfg.outer_steps)
    qtrm_model.cfg.outer_steps = int(core_steps)
    runtime = raw_eval.mode_runtime(f"qtrm_core_steps_{int(core_steps)}_no_evidence")
    rows: list[dict[str, Any]] = []
    try:
        for case in cases:
            prompt = str(case.get("prompt") or case.get("question") or "")
            prompt_ids = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False,
                add_special_tokens=True,
            )["input_ids"][0].detach().cpu()
            input_ids = prompt_ids.view(1, -1).to(device)
            attention_mask = torch.ones_like(input_ids)
            donor_out = donor.encode_inputs(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_logits=False,
            )
            with torch.amp.autocast(
                "cuda",
                enabled=(device == "cuda"),
                dtype=torch.bfloat16,
            ):
                outputs = qtrm_model(
                    input_ids,
                    attention_mask=attention_mask,
                    text_states=donor_out["text_states"].to(device),
                    **_runtime_kwargs(runtime),
                )
            answer = _gold_answer(case)
            rows.append(
                {
                    "case": case,
                    "prompt_ids": prompt_ids.long(),
                    "target_ids": torch.tensor(
                        _answer_token_ids(
                            tokenizer,
                            answer,
                            max_target_tokens=max_target_tokens,
                            append_eos_target=append_eos_target,
                        ),
                        dtype=torch.long,
                    ),
                    "core_hidden": _last_core_hidden(outputs)[0].float().cpu(),
                    "state_features": _state_features_from_outputs(
                        outputs,
                        key=state_logits_key,
                        mode=state_feature_mode,
                    ),
                    "answer": answer,
                }
            )
    finally:
        qtrm_model.cfg.outer_steps = old_outer_steps
    return rows


def _token_embeddings(embed: nn.Module, token_ids: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        return embed(token_ids.to(device)).detach()


def _adapter_prefix(
    *,
    adapter: nn.Module,
    core_hidden: torch.Tensor,
    state_features: torch.Tensor,
    core_off: bool = False,
    state_off: bool = False,
) -> torch.Tensor:
    if core_off:
        core_hidden = torch.zeros_like(core_hidden)
    if state_features.ndim == 1:
        state_features = state_features.view(1, -1)
    if state_off:
        state_features = torch.zeros_like(state_features)
    if isinstance(adapter, StateConditionedSoftPrefixAdapter):
        return adapter(core_hidden, state_features)
    return adapter(core_hidden)


def soft_prefix_logits(
    *,
    donor_model: nn.Module,
    input_embed: nn.Module,
    adapter: nn.Module,
    row: dict[str, Any],
    device: torch.device,
    core_off: bool = False,
    state_off: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_ids = row["prompt_ids"].to(device)
    target_ids = row["target_ids"].to(device)
    core_hidden = row["core_hidden"].view(1, -1).to(device)
    state_features = row.get("state_features", _empty_state_features()).view(1, -1).to(device)
    soft = _adapter_prefix(
        adapter=adapter,
        core_hidden=core_hidden,
        state_features=state_features,
        core_off=core_off,
        state_off=state_off,
    )
    prompt_embeds = _token_embeddings(input_embed, prompt_ids.view(1, -1), device=device)
    if target_ids.numel() > 1:
        prefix_answer_ids = target_ids[:-1].view(1, -1)
        answer_embeds = _token_embeddings(input_embed, prefix_answer_ids, device=device)
        inputs_embeds = torch.cat([prompt_embeds, soft, answer_embeds], dim=1)
    else:
        inputs_embeds = torch.cat([prompt_embeds, soft], dim=1)
    attention_mask = torch.ones(
        inputs_embeds.shape[:2],
        dtype=torch.long,
        device=inputs_embeds.device,
    )
    out = donor_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=False,
    )
    start = int(prompt_embeds.shape[1] + soft.shape[1] - 1)
    positions = torch.arange(start, start + int(target_ids.numel()), device=device)
    return out.logits[:, positions, :], target_ids.view(1, -1)


def soft_prefix_next_logits(
    *,
    donor_model: nn.Module,
    input_embed: nn.Module,
    adapter: nn.Module,
    row: dict[str, Any],
    answer_prefix_ids: torch.Tensor,
    target_id: torch.Tensor,
    device: torch.device,
    core_off: bool = False,
    state_off: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_ids = row["prompt_ids"].to(device)
    core_hidden = row["core_hidden"].view(1, -1).to(device)
    state_features = row.get("state_features", _empty_state_features()).view(1, -1).to(device)
    soft = _adapter_prefix(
        adapter=adapter,
        core_hidden=core_hidden,
        state_features=state_features,
        core_off=core_off,
        state_off=state_off,
    )
    prompt_embeds = _token_embeddings(input_embed, prompt_ids.view(1, -1), device=device)
    if int(answer_prefix_ids.numel()) > 0:
        answer_embeds = _token_embeddings(input_embed, answer_prefix_ids.view(1, -1), device=device)
        inputs_embeds = torch.cat([prompt_embeds, soft, answer_embeds], dim=1)
    else:
        inputs_embeds = torch.cat([prompt_embeds, soft], dim=1)
    attention_mask = torch.ones(
        inputs_embeds.shape[:2],
        dtype=torch.long,
        device=inputs_embeds.device,
    )
    out = donor_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=False,
    )
    return out.logits[:, -1, :], target_id.view(1).to(device)


def rollout_answer_prefix(
    *,
    donor_model: nn.Module,
    input_embed: nn.Module,
    adapter: nn.Module,
    row: dict[str, Any],
    device: torch.device,
    prefix_len: int,
    suppressed_token_ids: Iterable[int] = (),
) -> torch.Tensor:
    generated: list[int] = []
    for _ in range(max(0, int(prefix_len))):
        with torch.no_grad(), torch.amp.autocast(
            "cuda",
            enabled=(device.type == "cuda"),
            dtype=torch.bfloat16,
        ):
            logits, _ = soft_prefix_next_logits(
                donor_model=donor_model,
                input_embed=input_embed,
                adapter=adapter,
                row=row,
                answer_prefix_ids=torch.tensor(generated, dtype=torch.long, device=device),
                target_id=torch.tensor(0, dtype=torch.long, device=device),
                device=device,
                core_off=False,
            )
        valid_ids = [
            int(token_id)
            for token_id in suppressed_token_ids
            if 0 <= int(token_id) < int(logits.shape[-1])
        ]
        if valid_ids:
            logits[:, valid_ids] = -1.0e9
        generated.append(int(logits.argmax(dim=-1).detach().cpu().item()))
    return torch.tensor(generated, dtype=torch.long, device=device)


def _use_scheduled_sampling(
    *,
    step: int,
    pos: int,
    scheduled_sampling_prob: float,
    warmup_steps: int = 0,
) -> bool:
    if float(scheduled_sampling_prob) <= 0.0 or int(pos) <= 0:
        return False
    if int(step) <= int(warmup_steps):
        return False
    schedule_step = int(step) - int(warmup_steps)
    return ((schedule_step % 1000) / 1000.0) < float(scheduled_sampling_prob)


def train_adapter(
    *,
    donor_model: nn.Module,
    input_embed: nn.Module,
    adapter: nn.Module,
    rows: list[dict[str, Any]],
    device: torch.device,
    steps: int,
    lr: float,
    log_every: int,
    scheduled_sampling_prob: float = 0.0,
    scheduled_sampling_warmup_steps: int = 0,
    suppressed_token_ids: Iterable[int] = (),
) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("no train rows")
    adapter.train()
    donor_model.eval()
    opt = torch.optim.AdamW(adapter.parameters(), lr=float(lr), weight_decay=0.01)
    history: list[dict[str, Any]] = []
    for step in range(1, int(steps) + 1):
        row = rows[(step - 1) % len(rows)]
        target_ids = row["target_ids"].to(device)
        pos = (step - 1) % max(1, int(target_ids.numel()))
        use_rollout = _use_scheduled_sampling(
            step=step,
            pos=pos,
            scheduled_sampling_prob=scheduled_sampling_prob,
            warmup_steps=scheduled_sampling_warmup_steps,
        )
        answer_prefix = None
        if use_rollout:
            answer_prefix = rollout_answer_prefix(
                donor_model=donor_model,
                input_embed=input_embed,
                adapter=adapter,
                row=row,
                device=device,
                prefix_len=pos,
                suppressed_token_ids=suppressed_token_ids,
            )
        with torch.amp.autocast(
            "cuda",
            enabled=(device.type == "cuda"),
            dtype=torch.bfloat16,
        ):
            if use_rollout:
                with torch.enable_grad():
                    logits, target = soft_prefix_next_logits(
                        donor_model=donor_model,
                        input_embed=input_embed,
                        adapter=adapter,
                        row=row,
                        answer_prefix_ids=answer_prefix,
                        target_id=target_ids[pos],
                        device=device,
                        core_off=False,
                    )
            else:
                logits, target = soft_prefix_logits(
                    donor_model=donor_model,
                    input_embed=input_embed,
                    adapter=adapter,
                    row=row,
                    device=device,
                    core_off=False,
                )
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), target.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
        opt.step()
        if step == 1 or step % int(log_every) == 0 or step == int(steps):
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                acc = (pred == target).float().mean().item()
            item = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "token_acc": float(acc),
                "use_rollout": bool(use_rollout),
            }
            history.append(item)
            print(f"step={step} loss={item['loss']:.4f} token_acc={item['token_acc']:.3f}")
    return history


@torch.no_grad()
def teacher_forced_metrics(
    *,
    donor_model: nn.Module,
    input_embed: nn.Module,
    adapter: nn.Module,
    rows: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    out: dict[str, Any] = {"count": len(rows)}
    modes = [
        ("soft_full", False, False),
        ("soft_core_off", True, False),
        ("soft_state_off", False, True),
    ]
    for name, core_off, state_off in modes:
        correct = 0
        total = 0
        for row in rows:
            with torch.amp.autocast(
                "cuda",
                enabled=(device.type == "cuda"),
                dtype=torch.bfloat16,
            ):
                logits, target = soft_prefix_logits(
                    donor_model=donor_model,
                    input_embed=input_embed,
                    adapter=adapter,
                    row=row,
                    device=device,
                    core_off=core_off,
                    state_off=state_off,
                )
            pred = logits.argmax(dim=-1)
            correct += int((pred == target).sum().detach().cpu().item())
            total += int(target.numel())
        out[f"{name}_token_acc"] = float(correct / max(1, total))
    return out


@torch.no_grad()
def generate_one(
    *,
    case: dict[str, Any],
    tokenizer,
    donor_model: nn.Module,
    input_embed: nn.Module,
    adapter: nn.Module,
    core_hidden: torch.Tensor,
    state_features: torch.Tensor,
    mode: str,
    device: torch.device,
    max_length: int,
    max_new_tokens: int,
    suppressed_token_ids: Iterable[int],
) -> tuple[str, int]:
    prompt = str(case.get("prompt") or case.get("question") or "")
    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=True,
    )["input_ids"][0].detach().cpu().tolist()
    generated: list[int] = []
    if mode == "donor_only_no_evidence":
        soft = None
    else:
        core = core_hidden.view(1, -1).to(device)
        state = state_features.view(1, -1).to(device)
        soft = _adapter_prefix(
            adapter=adapter,
            core_hidden=core,
            state_features=state,
            core_off=(mode == "soft_core_off_no_evidence"),
            state_off=(mode == "soft_state_off_no_evidence"),
        )
    for _ in range(int(max_new_tokens)):
        token_ids = torch.tensor([prompt_ids + generated], dtype=torch.long, device=device)
        embeds = _token_embeddings(input_embed, token_ids, device=device)
        if soft is not None:
            prompt_len = len(prompt_ids)
            prompt_embeds = embeds[:, :prompt_len, :]
            gen_embeds = embeds[:, prompt_len:, :]
            inputs_embeds = torch.cat([prompt_embeds, soft, gen_embeds], dim=1)
        else:
            inputs_embeds = embeds
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
        with torch.amp.autocast(
            "cuda",
            enabled=(device.type == "cuda"),
            dtype=torch.bfloat16,
        ):
            logits = donor_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits[:, -1, :]
        valid_ids = [
            int(token_id)
            for token_id in suppressed_token_ids
            if 0 <= int(token_id) < int(logits.shape[-1])
        ]
        if valid_ids:
            logits[:, valid_ids] = -1.0e9
        next_id = int(logits.argmax(dim=-1).detach().cpu().item())
        generated.append(next_id)
        if tokenizer.eos_token_id is not None and next_id == int(tokenizer.eos_token_id):
            break
    return tokenizer.decode(generated, skip_special_tokens=True).strip(), len(generated)


@torch.no_grad()
def generation_metrics(
    *,
    rows: list[dict[str, Any]],
    tokenizer,
    donor_model: nn.Module,
    input_embed: nn.Module,
    adapter: nn.Module,
    device: torch.device,
    max_length: int,
    max_new_tokens: int,
    suppress_visible_reasoning_tokens: bool,
    out_jsonl: Path,
) -> dict[str, Any]:
    raw_eval = _load_raw_eval_module()
    suppressed = raw_eval._visible_reasoning_token_ids(
        tokenizer,
        enabled=bool(suppress_visible_reasoning_tokens),
    )
    modes = [
        "donor_only_no_evidence",
        "soft_core_off_no_evidence",
        "soft_state_off_no_evidence",
        "soft_full_no_evidence",
    ]
    summary: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    for mode in modes:
        hits = 0
        exact = 0
        runtime = {
            "mode": mode,
            "disable_core": mode in {"donor_only_no_evidence", "soft_core_off_no_evidence"},
            "disable_state_features": mode == "soft_state_off_no_evidence",
            "memoryos_used": False,
            "retrieval_used": False,
        }
        for row in rows:
            completion, generated_tokens = generate_one(
                case=row["case"],
                tokenizer=tokenizer,
                donor_model=donor_model,
                input_embed=input_embed,
                adapter=adapter,
                core_hidden=row["core_hidden"],
                state_features=row.get("state_features", _empty_state_features()),
                mode=mode,
                device=device,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                suppressed_token_ids=suppressed,
            )
            record = raw_eval.score_case_record(
                row["case"],
                mode=mode,
                completion=completion,
                runtime=runtime,
                generated_tokens=generated_tokens,
            )
            records.append(record)
            hits += int(bool(record["hit"]))
            exact += int(bool(record["exact_match"] or record["normalized_exact"]))
        summary[mode] = {
            "hits": hits,
            "exact": exact,
            "total": len(rows),
            "accuracy": float(hits / max(1, len(rows))),
            "exact_accuracy": float(exact / max(1, len(rows))),
        }
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a QTRM-core-conditioned donor soft-prefix falsifier.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-cases", required=True)
    parser.add_argument("--eval-cases", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-train-cases", type=int, default=16)
    parser.add_argument("--max-eval-cases", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-target-tokens", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--append-eos-target",
        action="store_true",
        help="Append donor EOS to answer targets so the renderer learns to stop after the answer.",
    )
    parser.add_argument("--core-steps", type=int, default=8)
    parser.add_argument("--prefix-tokens", type=int, default=4)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--scale", type=float, default=4.0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--scheduled-sampling-prob", type=float, default=0.0)
    parser.add_argument("--scheduled-sampling-warmup-steps", type=int, default=0)
    parser.add_argument(
        "--state-logits-key",
        default="",
        help=(
            "Optional QTRM output logits key or comma-separated keys to flatten into "
            "explicit state features (for example core_role_value_state_logits)."
        ),
    )
    parser.add_argument(
        "--state-feature-mode",
        choices=["softmax", "argmax_onehot", "logits"],
        default="softmax",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--suppress-visible-reasoning-tokens", action="store_true")
    return parser


def main() -> None:
    from transformers import AutoTokenizer

    from qtrm_mm.config import load_config
    from qtrm_mm.qtrm_model import QTRMMultimodalModel
    from qtrm_mm.qwen_donor import QwenDonorAdapter

    args = build_arg_parser().parse_args()
    raw_eval = _load_raw_eval_module()
    cfg = load_config(args.config)
    device_name = raw_eval._select_device(cfg.train.device, args.device)
    device = torch.device(device_name)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.donor.model_id,
        trust_remote_code=cfg.donor.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    qtrm_model = QTRMMultimodalModel(cfg.model).to(device_name)
    state = torch.load(args.checkpoint, map_location=device_name, weights_only=False)
    qtrm_model.load_state_dict(state.get("model", state), strict=False)
    qtrm_model.eval()
    for param in qtrm_model.parameters():
        param.requires_grad_(False)

    donor = QwenDonorAdapter(cfg.donor)
    donor.model.eval()
    for param in donor.model.parameters():
        param.requires_grad_(False)
    input_embed = find_input_embeddings(donor.model)
    embed_param = next(input_embed.parameters(), None)
    adapter_device = embed_param.device if embed_param is not None else device

    train_cases = raw_eval.load_cases(args.train_cases, max_cases=args.max_train_cases)
    eval_cases = raw_eval.load_cases(args.eval_cases, max_cases=args.max_eval_cases)
    print("[gate] target=L0 soft-prefix falsifier bottleneck=donor-internal-renderer")
    print("[features] train")
    train_rows = build_rows(
        cases=train_cases,
        tokenizer=tokenizer,
        donor=donor,
        qtrm_model=qtrm_model,
        device=device_name,
        core_steps=int(args.core_steps),
        max_length=int(args.max_length),
        max_target_tokens=int(args.max_target_tokens),
        append_eos_target=bool(args.append_eos_target),
        state_logits_key=str(args.state_logits_key),
        state_feature_mode=str(args.state_feature_mode),
    )
    print("[features] eval")
    eval_rows = build_rows(
        cases=eval_cases,
        tokenizer=tokenizer,
        donor=donor,
        qtrm_model=qtrm_model,
        device=device_name,
        core_steps=int(args.core_steps),
        max_length=int(args.max_length),
        max_target_tokens=int(args.max_target_tokens),
        append_eos_target=bool(args.append_eos_target),
        state_logits_key=str(args.state_logits_key),
        state_feature_mode=str(args.state_feature_mode),
    )
    donor_dim = int(input_embed.weight.shape[-1])
    core_dim = int(train_rows[0]["core_hidden"].numel())
    state_dim = int(train_rows[0].get("state_features", _empty_state_features()).numel())
    for row in train_rows + eval_rows:
        row_state_dim = int(row.get("state_features", _empty_state_features()).numel())
        if row_state_dim != state_dim:
            raise ValueError(f"inconsistent state feature dim: got {row_state_dim}, expected {state_dim}")
    if state_dim > 0:
        adapter = StateConditionedSoftPrefixAdapter(
            core_dim=core_dim,
            state_dim=state_dim,
            donor_dim=donor_dim,
            prefix_tokens=int(args.prefix_tokens),
            rank=int(args.rank),
            scale=float(args.scale),
        ).to(adapter_device)
    else:
        adapter = CoreSoftPrefixAdapter(
            core_dim=core_dim,
            donor_dim=donor_dim,
            prefix_tokens=int(args.prefix_tokens),
            rank=int(args.rank),
            scale=float(args.scale),
        ).to(adapter_device)
    history = train_adapter(
        donor_model=donor.model,
        input_embed=input_embed,
        adapter=adapter,
        rows=train_rows,
        device=adapter_device,
        steps=int(args.steps),
        lr=float(args.lr),
        log_every=int(args.log_every),
        scheduled_sampling_prob=float(args.scheduled_sampling_prob),
        scheduled_sampling_warmup_steps=int(args.scheduled_sampling_warmup_steps),
        suppressed_token_ids=raw_eval._visible_reasoning_token_ids(
            tokenizer,
            enabled=bool(args.suppress_visible_reasoning_tokens),
        ),
    )
    tf_metrics = teacher_forced_metrics(
        donor_model=donor.model,
        input_embed=input_embed,
        adapter=adapter,
        rows=eval_rows,
        device=adapter_device,
    )
    generation = generation_metrics(
        rows=eval_rows,
        tokenizer=tokenizer,
        donor_model=donor.model,
        input_embed=input_embed,
        adapter=adapter,
        device=adapter_device,
        max_length=int(args.max_length),
        max_new_tokens=int(args.max_new_tokens),
        suppress_visible_reasoning_tokens=bool(args.suppress_visible_reasoning_tokens),
        out_jsonl=out_dir / "generation.jsonl",
    )
    full = generation["soft_full_no_evidence"]["accuracy"]
    donor_acc = generation["donor_only_no_evidence"]["accuracy"]
    core_off = generation["soft_core_off_no_evidence"]["accuracy"]
    state_off = generation["soft_state_off_no_evidence"]["accuracy"]
    accepted = bool(full > donor_acc and full > core_off and (state_dim == 0 or full > state_off))
    report = {
        "decision": (
            "accepted_l1_state_conditioned_soft_prefix"
            if accepted and state_dim > 0
            else ("accepted_l1_soft_prefix" if accepted else "rejected")
        ),
        "accepted": accepted,
        "target_level": "L0/L1 donor-internal-renderer falsifier",
        "major_bottleneck": "latent-core-to-autoregressive-text",
        "adapter": {
            "core_dim": core_dim,
            "state_dim": state_dim,
            "state_logits_key": str(args.state_logits_key),
            "state_feature_mode": str(args.state_feature_mode),
            "donor_dim": donor_dim,
            "prefix_tokens": int(args.prefix_tokens),
            "rank": int(args.rank),
            "scale": float(args.scale),
        },
        "teacher_forced": tf_metrics,
        "generation": generation,
        "history": history,
        "suppress_visible_reasoning_tokens": bool(args.suppress_visible_reasoning_tokens),
        "scheduled_sampling_prob": float(args.scheduled_sampling_prob),
        "scheduled_sampling_warmup_steps": int(args.scheduled_sampling_warmup_steps),
        "append_eos_target": bool(args.append_eos_target),
        "next_action": (
            "broaden held-out gate and add core_off/delta_off perturbations"
            if accepted
            else (
                "state-conditioned soft-prefix failed; redesign explicit value-state codec"
                if state_dim > 0
                else "soft-prefix did not solve renderer; implement true donor residual layer hook or reset data/model"
            )
        ),
    }
    torch.save({"adapter": adapter.state_dict(), "report": report}, out_dir / "adapter.pt")
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
