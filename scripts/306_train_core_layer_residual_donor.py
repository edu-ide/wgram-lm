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


def _load_script_module(filename: str, module_name: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CoreLayerResidualAdapter(nn.Module):
    """Low-rank residual from QTRM core state into one frozen donor layer."""

    def __init__(
        self,
        *,
        core_dim: int,
        donor_dim: int,
        rank: int = 64,
        scale: float = 1.0,
        gate_init_bias: float = -1.0,
    ) -> None:
        super().__init__()
        self.scale = float(scale)
        self.norm = nn.LayerNorm(int(core_dim), elementwise_affine=False)
        self.down = nn.Linear(int(core_dim), max(1, int(rank)), bias=False)
        self.up = nn.Linear(max(1, int(rank)), int(donor_dim), bias=False)
        self.gate = nn.Linear(int(core_dim), 1)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.up.weight, mean=0.0, std=0.002)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, float(gate_init_bias))

    def forward(self, core_hidden: torch.Tensor) -> torch.Tensor:
        x = self.norm(core_hidden.float())
        delta = self.up(F.gelu(self.down(x)))
        gate = torch.sigmoid(self.gate(x))
        return delta * gate * self.scale


def find_layers(model: nn.Module) -> list[nn.Module]:
    for path in (
        "model.layers",
        "model.language_model.layers",
        "language_model.model.layers",
        "language_model.layers",
        "base_model.model.layers",
        "transformer.h",
        "gpt_neox.layers",
    ):
        cur: Any = model
        ok = True
        for part in path.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok and isinstance(cur, (nn.ModuleList, list, tuple)):
            return list(cur)
    raise RuntimeError("could not locate donor transformer layers")


def _target_positions(prompt_len: int, target_len: int) -> list[int]:
    if int(target_len) <= 0:
        return []
    start = max(0, int(prompt_len) - 1)
    return list(range(start, start + int(target_len)))


def _patch_layer_output(
    output: Any,
    *,
    positions: Iterable[int],
    delta: torch.Tensor,
) -> Any:
    hidden = output[0] if isinstance(output, tuple) else output
    if not torch.is_tensor(hidden) or hidden.ndim != 3:
        return output
    pos = [int(item) for item in positions if 0 <= int(item) < int(hidden.shape[1])]
    if not pos:
        return output
    patch = delta.to(device=hidden.device, dtype=hidden.dtype).view(hidden.shape[0], 1, -1)
    patched = hidden.clone()
    patched[:, pos, :] = patched[:, pos, :] + patch
    if isinstance(output, tuple):
        return (patched, *output[1:])
    return patched


def layer_residual_logits(
    *,
    donor_model: nn.Module,
    input_embed: nn.Module,
    layer: nn.Module,
    adapter: CoreLayerResidualAdapter,
    row: dict[str, Any],
    device: torch.device,
    core_off: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    helper = _load_script_module("304_train_core_soft_prefix_donor.py", "core_soft_prefix_helper_306")
    prompt_ids_cpu = row["prompt_ids"].detach().cpu()
    target_ids = row["target_ids"].to(device)
    core_hidden = row["core_hidden"].view(1, -1).to(device)
    if core_off:
        core_hidden = torch.zeros_like(core_hidden)
    if target_ids.numel() > 1:
        input_ids = torch.cat([prompt_ids_cpu, target_ids[:-1].detach().cpu()]).view(1, -1)
    else:
        input_ids = prompt_ids_cpu.view(1, -1)
    inputs_embeds = helper._token_embeddings(input_embed, input_ids, device=device)
    positions = _target_positions(prompt_len=int(prompt_ids_cpu.numel()), target_len=int(target_ids.numel()))
    delta = adapter(core_hidden)

    def hook(_module, _inputs, output):
        return _patch_layer_output(output, positions=positions, delta=delta)

    handle = layer.register_forward_hook(hook)
    try:
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
        out = donor_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False)
    finally:
        handle.remove()
    return out.logits[:, positions, :], target_ids.view(1, -1)


def train_adapter(
    *,
    donor_model: nn.Module,
    input_embed: nn.Module,
    layer: nn.Module,
    adapter: CoreLayerResidualAdapter,
    rows: list[dict[str, Any]],
    device: torch.device,
    steps: int,
    lr: float,
    log_every: int,
) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("no train rows")
    adapter.train()
    donor_model.eval()
    opt = torch.optim.AdamW(adapter.parameters(), lr=float(lr), weight_decay=0.01)
    history: list[dict[str, Any]] = []
    for step in range(1, int(steps) + 1):
        row = rows[(step - 1) % len(rows)]
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.bfloat16):
            logits, target = layer_residual_logits(
                donor_model=donor_model,
                input_embed=input_embed,
                layer=layer,
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
        if step == 1 or (int(log_every) > 0 and step % int(log_every) == 0) or step == int(steps):
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                acc = (pred == target).float().mean().item()
            item = {"step": step, "loss": float(loss.detach().cpu()), "token_acc": float(acc)}
            history.append(item)
            print(f"step={step} loss={item['loss']:.4f} token_acc={item['token_acc']:.3f}")
    return history


@torch.no_grad()
def teacher_forced_metrics(
    *,
    donor_model: nn.Module,
    input_embed: nn.Module,
    layer: nn.Module,
    adapter: CoreLayerResidualAdapter,
    rows: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    out: dict[str, Any] = {"count": len(rows)}
    for name, core_off in (("layer_full", False), ("layer_core_off", True)):
        correct = 0
        total = 0
        for row in rows:
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.bfloat16):
                logits, target = layer_residual_logits(
                    donor_model=donor_model,
                    input_embed=input_embed,
                    layer=layer,
                    adapter=adapter,
                    row=row,
                    device=device,
                    core_off=core_off,
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
    layer: nn.Module,
    adapter: CoreLayerResidualAdapter,
    core_hidden: torch.Tensor,
    mode: str,
    device: torch.device,
    max_length: int,
    max_new_tokens: int,
    suppressed_token_ids: Iterable[int],
) -> tuple[str, int]:
    helper = _load_script_module("304_train_core_soft_prefix_donor.py", "core_soft_prefix_helper_gen_306")
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
    for _ in range(int(max_new_tokens)):
        token_ids = torch.tensor([prompt_ids + generated], dtype=torch.long, device=device)
        inputs_embeds = helper._token_embeddings(input_embed, token_ids, device=device)
        delta = None
        if mode != "donor_only_no_evidence":
            core = core_hidden.view(1, -1).to(device)
            if mode == "layer_core_off_no_evidence":
                core = torch.zeros_like(core)
            delta = adapter(core)

        def hook(_module, _inputs, output):
            if delta is None:
                return output
            return _patch_layer_output(output, positions=[inputs_embeds.shape[1] - 1], delta=delta)

        handle = layer.register_forward_hook(hook)
        try:
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.bfloat16):
                logits = donor_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).logits[:, -1, :]
        finally:
            handle.remove()
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
    layer: nn.Module,
    adapter: CoreLayerResidualAdapter,
    device: torch.device,
    max_length: int,
    max_new_tokens: int,
    suppress_visible_reasoning_tokens: bool,
    out_jsonl: Path,
) -> dict[str, Any]:
    raw_eval = _load_script_module("192_eval_raw_intelligence.py", "raw_eval_306")
    suppressed = raw_eval._visible_reasoning_token_ids(
        tokenizer,
        enabled=bool(suppress_visible_reasoning_tokens),
    )
    modes = ["donor_only_no_evidence", "layer_core_off_no_evidence", "layer_full_no_evidence"]
    summary: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    for mode in modes:
        hits = 0
        exact = 0
        runtime = {
            "mode": mode,
            "disable_core": mode != "layer_full_no_evidence",
            "memoryos_used": False,
            "retrieval_used": False,
        }
        for row in rows:
            completion, generated_tokens = generate_one(
                case=row["case"],
                tokenizer=tokenizer,
                donor_model=donor_model,
                input_embed=input_embed,
                layer=layer,
                adapter=adapter,
                core_hidden=row["core_hidden"],
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
    parser = argparse.ArgumentParser(description="Train a QTRM core residual hook into one frozen donor layer.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-cases", required=True)
    parser.add_argument("--eval-cases", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--layer-index", type=int, default=-4)
    parser.add_argument("--max-train-cases", type=int, default=16)
    parser.add_argument("--max-eval-cases", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-target-tokens", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--core-steps", type=int, default=4)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--scale", type=float, default=4.0)
    parser.add_argument("--gate-init-bias", type=float, default=-1.0)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--suppress-visible-reasoning-tokens", action="store_true")
    return parser


def main() -> None:
    from transformers import AutoTokenizer

    from qtrm_mm.config import load_config
    from qtrm_mm.qtrm_model import QTRMMultimodalModel
    from qtrm_mm.qwen_donor import QwenDonorAdapter

    helper = _load_script_module("304_train_core_soft_prefix_donor.py", "core_soft_prefix_helper_main_306")
    raw_eval = _load_script_module("192_eval_raw_intelligence.py", "raw_eval_main_306")
    args = build_arg_parser().parse_args()
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
    input_embed = helper.find_input_embeddings(donor.model)
    embed_param = next(input_embed.parameters(), None)
    adapter_device = embed_param.device if embed_param is not None else device
    layers = find_layers(donor.model)
    layer_index = int(args.layer_index)
    if layer_index < 0:
        layer_index = len(layers) + layer_index
    if not 0 <= layer_index < len(layers):
        raise ValueError(f"layer index out of range: {args.layer_index}; layers={len(layers)}")
    layer = layers[layer_index]

    train_cases = raw_eval.load_cases(args.train_cases, max_cases=args.max_train_cases)
    eval_cases = raw_eval.load_cases(args.eval_cases, max_cases=args.max_eval_cases)
    print(
        "[gate] target=L0/L1 layer-residual falsifier "
        f"bottleneck=donor-internal-renderer layer={layer_index}/{len(layers)}"
    )
    print("[features] train")
    train_rows = helper.build_rows(
        cases=train_cases,
        tokenizer=tokenizer,
        donor=donor,
        qtrm_model=qtrm_model,
        device=device_name,
        core_steps=int(args.core_steps),
        max_length=int(args.max_length),
        max_target_tokens=int(args.max_target_tokens),
    )
    print("[features] eval")
    eval_rows = helper.build_rows(
        cases=eval_cases,
        tokenizer=tokenizer,
        donor=donor,
        qtrm_model=qtrm_model,
        device=device_name,
        core_steps=int(args.core_steps),
        max_length=int(args.max_length),
        max_target_tokens=int(args.max_target_tokens),
    )
    donor_dim = int(input_embed.weight.shape[-1])
    core_dim = int(train_rows[0]["core_hidden"].numel())
    adapter = CoreLayerResidualAdapter(
        core_dim=core_dim,
        donor_dim=donor_dim,
        rank=int(args.rank),
        scale=float(args.scale),
        gate_init_bias=float(args.gate_init_bias),
    ).to(adapter_device)
    history = train_adapter(
        donor_model=donor.model,
        input_embed=input_embed,
        layer=layer,
        adapter=adapter,
        rows=train_rows,
        device=adapter_device,
        steps=int(args.steps),
        lr=float(args.lr),
        log_every=int(args.log_every),
    )
    tf_metrics = teacher_forced_metrics(
        donor_model=donor.model,
        input_embed=input_embed,
        layer=layer,
        adapter=adapter,
        rows=eval_rows,
        device=adapter_device,
    )
    generation = generation_metrics(
        rows=eval_rows,
        tokenizer=tokenizer,
        donor_model=donor.model,
        input_embed=input_embed,
        layer=layer,
        adapter=adapter,
        device=adapter_device,
        max_length=int(args.max_length),
        max_new_tokens=int(args.max_new_tokens),
        suppress_visible_reasoning_tokens=bool(args.suppress_visible_reasoning_tokens),
        out_jsonl=out_dir / "generation.jsonl",
    )
    full = generation["layer_full_no_evidence"]["accuracy"]
    donor_acc = generation["donor_only_no_evidence"]["accuracy"]
    off = generation["layer_core_off_no_evidence"]["accuracy"]
    accepted = bool(full > donor_acc and full > off)
    report = {
        "decision": "accepted_l1_layer_residual" if accepted else "rejected",
        "accepted": accepted,
        "target_level": "L0/L1 donor-layer-residual renderer falsifier",
        "major_bottleneck": "latent-core-to-autoregressive-text",
        "adapter": {
            "core_dim": core_dim,
            "donor_dim": donor_dim,
            "layer_index": layer_index,
            "layer_count": len(layers),
            "rank": int(args.rank),
            "scale": float(args.scale),
            "gate_init_bias": float(args.gate_init_bias),
        },
        "teacher_forced": tf_metrics,
        "generation": generation,
        "history": history,
        "suppress_visible_reasoning_tokens": bool(args.suppress_visible_reasoning_tokens),
        "next_action": (
            "broaden held-out gate and sweep nearby layer/scale only after exact accuracy improves"
            if accepted
            else "layer residual did not beat core_off/donor; reset renderer target or inspect layer/token positions"
        ),
    }
    torch.save({"adapter": adapter.state_dict(), "report": report}, out_dir / "adapter.pt")
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
