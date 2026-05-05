from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script():
    path = Path("scripts/136_qwen_scope_probe.py")
    spec = importlib.util.spec_from_file_location("qwen_scope_probe_script", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_qwen_scope_probe_defaults_match_qwen35_2b_base() -> None:
    module = _load_script()

    args = module.build_arg_parser().parse_args(["--prompt", "hello"])

    assert args.model_id == "Qwen/Qwen3.5-2B-Base"
    assert args.sae_repo == "Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_100"
    assert args.layer == [23]
    assert args.top_k == 20
    assert args.load_in_4bit is False


def test_qwen_scope_probe_accepts_multiple_layers_and_prompts() -> None:
    module = _load_script()

    args = module.build_arg_parser().parse_args(
        [
            "--layer",
            "0",
            "--layer",
            "12",
            "--prompt",
            "first",
            "--prompt",
            "second",
            "--load-in-4bit",
            "--out",
            "runs/qwen_scope/test.jsonl",
        ]
    )

    assert args.layer == [0, 12]
    assert args.prompt == ["first", "second"]
    assert args.load_in_4bit is True
    assert args.out == "runs/qwen_scope/test.jsonl"
