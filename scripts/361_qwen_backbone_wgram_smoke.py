#!/usr/bin/env python3
"""Smoke-test Qwen-backbone QTRM insertion.

This keeps the pretrained Qwen3.5 text backbone as the model body and inserts a
gated QTRM recurrent residual into the same LM-logit path. It is a runtime
equivalence and causality check, not a training script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from wgram_lm.qwen_backbone_wgram import QwenBackboneQTRM


def _dtype(name: str) -> torch.dtype:
    value = str(name).lower()
    if value in {"float16", "fp16"}:
        return torch.float16
    if value in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if value in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _max_abs_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().float() - b.detach().float()).abs().max().cpu())


def _parse_int_list(value: str) -> tuple[int, ...] | None:
    text = str(value).strip()
    if text == "":
        return None
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def _load_ouro_model(args: argparse.Namespace, *, dtype: torch.dtype, device: torch.device):
    if str(args.core_impl) != "ouro_weight_wrapped":
        return None
    layer_indices = _parse_int_list(str(args.ouro_core_layer_indices))
    if bool(args.ouro_partial_safetensors):
        from wgram_lm.ouro_partial import build_partial_ouro_model_from_safetensors

        return build_partial_ouro_model_from_safetensors(
            str(args.ouro_model_id),
            layer_indices=layer_indices or (18,),
            dtype=dtype,
            device=device,
        )
    try:
        from transformers import AutoModelForCausalLM
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("transformers is required to load Ouro") from exc
    model = AutoModelForCausalLM.from_pretrained(
        str(args.ouro_model_id),
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    return model.to(device)


@torch.no_grad()
def run_smoke(args: argparse.Namespace) -> dict[str, object]:
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("transformers is required") from exc

    device = torch.device(str(args.device))
    dtype = _dtype(str(args.dtype))
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_id),
        trust_remote_code=True,
    )
    ouro_model = _load_ouro_model(args, dtype=dtype, device=device)
    model = QwenBackboneQTRM.from_pretrained(
        str(args.model_id),
        dtype=dtype,
        device=device,
        max_seq_len=int(args.max_seq_len),
        freeze_qwen=not bool(args.train_qwen),
        core_gate_init=float(args.core_gate_init),
        residual_scale=float(args.residual_scale),
        core_impl=str(args.core_impl),
        mandatory_core=bool(args.mandatory_core),
        qwen_core_layer_indices=_parse_int_list(str(args.qwen_core_layer_indices)),
        ouro_model=ouro_model,
        ouro_core_layer_indices=_parse_int_list(str(args.ouro_core_layer_indices)),
        core_adapter_dim=int(args.core_adapter_dim),
        n_core_layers=int(args.n_core_layers),
        h_cycles=int(args.h_cycles),
        l_cycles=int(args.l_cycles),
        outer_steps=int(args.outer_steps),
        delta_backend=str(args.delta_backend),
        strict_backends=bool(args.strict_backends),
        core_causal=not bool(args.non_causal_core),
    ).to(device)
    model.eval()

    encoded = tokenizer(
        str(args.prompt),
        return_tensors="pt",
        truncation=True,
        max_length=int(args.max_seq_len),
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    base = model(
        input_ids,
        attention_mask=attention_mask,
        force_core_off=True,
    ).logits
    hidden_path_off = model(
        input_ids,
        attention_mask=attention_mask,
        core_gate_override=0.0,
    ).logits
    core_on = model(
        input_ids,
        attention_mask=attention_mask,
        core_gate_override=float(args.core_gate_on),
    ).logits
    normal_path = model(
        input_ids,
        attention_mask=attention_mask,
    ).logits

    report = model.report()
    core_layer_indices = list(getattr(model.core, "layer_indices", []))
    qwen_core_layers = core_layer_indices if str(args.core_impl) in {
        "qwen_layer_wrapped",
        "ouro_shared_qwen_layer",
    } else []
    ouro_core_layers = core_layer_indices if str(args.core_impl) == "ouro_weight_wrapped" else []
    result = {
        "status": "complete",
        "model_id": str(args.model_id),
        "prompt": str(args.prompt),
        "device": str(device),
        "dtype": str(args.dtype),
        "input_shape": list(input_ids.shape),
        "report": report.__dict__,
        "core_config": {
            "core_impl": str(getattr(model, "core_impl", "unknown")),
            "mandatory_core": bool(getattr(model, "mandatory_core", False)),
            "core_causal": bool(getattr(model.core_cfg, "core_causal", False)),
            "n_core_layers": int(model.core_cfg.n_core_layers),
            "h_cycles": int(model.core_cfg.h_cycles),
            "l_cycles": int(model.core_cfg.l_cycles),
            "outer_steps": int(model.core_cfg.outer_steps),
            "attn_every": int(model.core_cfg.attn_every),
            "delta_backend": str(model.core_cfg.delta_backend),
            "core_layer_indices": core_layer_indices,
            "qwen_core_layer_indices": qwen_core_layers,
            "ouro_core_layer_indices": ouro_core_layers,
            "ouro_model_id": str(args.ouro_model_id) if ouro_model is not None else "",
            "qwen_core_shared_stack": bool(getattr(model.core, "shared_stack", False)),
            "core_adapter_dim": int(args.core_adapter_dim),
        },
        "max_abs_delta_base_vs_hidden_path_gate0": _max_abs_delta(base, hidden_path_off),
        "max_abs_delta_base_vs_core_on": _max_abs_delta(base, core_on),
        "max_abs_delta_base_vs_normal_path": _max_abs_delta(base, normal_path),
        "core_gate_on": float(args.core_gate_on),
        "normal_core_gate": float(model.normal_core_gate_value()),
        "accepted_equivalence": _max_abs_delta(base, hidden_path_off)
        <= float(args.max_equivalence_delta),
        "accepted_core_causality": _max_abs_delta(base, core_on)
        >= float(args.min_core_on_delta),
        "runtime_donor": False,
    }
    result["accepted"] = bool(
        result["accepted_equivalence"] and result["accepted_core_causality"]
    )
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-2B-Base")
    parser.add_argument("--out-dir", default="local_eval/qwen_backbone_wgram_smoke")
    parser.add_argument("--prompt", default="User: Why should evidence be checked?\nAssistant:")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--n-core-layers", type=int, default=1)
    parser.add_argument("--h-cycles", type=int, default=1)
    parser.add_argument("--l-cycles", type=int, default=1)
    parser.add_argument("--outer-steps", type=int, default=1)
    parser.add_argument(
        "--core-impl",
        choices=[
            "qtrm_block_stack",
            "qwen_layer_wrapped",
            "ouro_shared_qwen_layer",
            "ouro_weight_wrapped",
        ],
        default="qwen_layer_wrapped",
    )
    parser.add_argument("--qwen-core-layer-indices", default="3")
    parser.add_argument("--ouro-model-id", default="ByteDance/Ouro-2.6B-Thinking")
    parser.add_argument("--ouro-core-layer-indices", default="")
    parser.add_argument("--ouro-partial-safetensors", action="store_true")
    parser.add_argument("--core-adapter-dim", type=int, default=0)
    parser.add_argument("--delta-backend", default="fla_gated_delta")
    parser.add_argument("--strict-backends", action="store_true")
    parser.add_argument("--non-causal-core", action="store_true")
    parser.add_argument("--train-qwen", action="store_true")
    parser.add_argument("--mandatory-core", action="store_true")
    parser.add_argument("--core-gate-init", type=float, default=-8.0)
    parser.add_argument("--core-gate-on", type=float, default=0.125)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--max-equivalence-delta", type=float, default=5e-3)
    parser.add_argument("--min-core-on-delta", type=float, default=1e-5)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_smoke(args)
    (out_dir / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["accepted"] else 1)


if __name__ == "__main__":
    main()
