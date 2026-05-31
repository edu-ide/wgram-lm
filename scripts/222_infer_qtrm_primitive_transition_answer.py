#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Sequence

from wgram_lm.agentic.solver_state_machine import (
    answer_from_primitive_operations,
    operation_names_from_logits,
)


def build_runtime_row(prompt: str) -> dict[str, Any]:
    return {"prompt": str(prompt)}


def runtime_report_from_operations(
    row: dict[str, Any],
    operations: Sequence[str],
) -> dict[str, Any]:
    answer_info = answer_from_primitive_operations(row, operations)
    return {
        "prompt": row.get("prompt") or row.get("question") or "",
        "predicted_operations": [str(operation) for operation in operations],
        "executed_operations": answer_info["executed_operations"],
        "states": answer_info["states"],
        "answer": answer_info["answer"],
        "records": answer_info["records"],
    }


def _load_depth_train_module():
    path = Path(__file__).with_name("196_train_pure_recursive_depth_supervised.py")
    spec = importlib.util.spec_from_file_location("depth_supervised_train", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    spec.loader.exec_module(module)
    return module


def predict_operations_for_prompt(
    *,
    prompt: str,
    config_path: str,
    checkpoint_path: str,
    tokenizer_model_id: str,
    max_length: int | None,
    core_steps: int,
    disable_core: bool = False,
) -> list[str]:
    import torch
    from transformers import AutoTokenizer

    from wgram_lm.config import load_config
    from wgram_lm.wgram_model import QTRMMultimodalModel
    from wgram_lm.qwen_donor import QwenDonorAdapter
    from wgram_lm.training.train import load_initial_checkpoint

    depth_train = _load_depth_train_module()
    cfg = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = QTRMMultimodalModel(cfg.model).to(device)
    load_initial_checkpoint(model, checkpoint_path, map_location=device)
    model.eval()
    donor = QwenDonorAdapter(cfg.donor)
    operation_to_id = depth_train.primitive_transition_operation_id_map(
        int(cfg.model.primitive_transition_num_operations)
    )
    id_to_operation = {idx: operation for operation, idx in operation_to_id.items()}
    seq_len = int(max_length or cfg.train.seq_len)
    with torch.no_grad():
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=seq_len,
        ).to(device)
        donor_out = donor.encode_inputs(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            return_logits=False,
        )
        old_steps = model.cfg.outer_steps
        model.cfg.outer_steps = int(core_steps)
        try:
            with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.bfloat16):
                outputs = model(
                    enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    text_states=donor_out["text_states"].detach().to(device),
                    disable_core=bool(disable_core),
                )
        finally:
            model.cfg.outer_steps = old_steps
    return operation_names_from_logits(
        outputs["primitive_transition_operation_logits"],
        id_to_operation,
    )


def infer_primitive_answer(
    *,
    prompt: str,
    config_path: str,
    checkpoint_path: str,
    tokenizer_model_id: str,
    max_length: int | None,
    core_steps: int,
    disable_core: bool = False,
) -> dict[str, Any]:
    operations = predict_operations_for_prompt(
        prompt=prompt,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        tokenizer_model_id=tokenizer_model_id,
        max_length=max_length,
        core_steps=core_steps,
        disable_core=disable_core,
    )
    report = runtime_report_from_operations(build_runtime_row(prompt), operations)
    report["checkpoint"] = checkpoint_path
    report["config"] = config_path
    report["core_steps"] = int(core_steps)
    report["disable_core"] = bool(disable_core)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Infer a primitive-transition answer from QTRM operation logits."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out-json", default="")
    parser.add_argument("--tokenizer-model-id", default="Qwen/Qwen3.5-2B-Base")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--core-steps", type=int, default=4)
    parser.add_argument("--disable-core", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = infer_primitive_answer(
        prompt=args.prompt,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        tokenizer_model_id=args.tokenizer_model_id,
        max_length=args.max_length,
        core_steps=args.core_steps,
        disable_core=args.disable_core,
    )
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
