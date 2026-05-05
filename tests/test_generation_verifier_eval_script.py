from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script():
    path = Path("scripts/143_eval_generation_verifier.py")
    spec = importlib.util.spec_from_file_location("generation_verifier_eval_script", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_generation_verifier_binary_metrics() -> None:
    module = _load_script()

    metrics = module.binary_metrics(
        probs=[0.9, 0.8, 0.2, 0.1],
        targets=[1.0, 0.0, 1.0, 0.0],
        threshold=0.5,
    )

    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fn"] == 1
    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5


def test_generation_verifier_best_threshold_metrics() -> None:
    module = _load_script()

    best = module.best_threshold_metrics(
        probs=[0.9, 0.8, 0.2, 0.1],
        targets=[1.0, 1.0, 0.0, 0.0],
    )

    assert best["threshold"] == 0.8
    assert best["f1"] == 1.0
    assert best["accuracy"] == 1.0


def test_generation_verifier_eval_parser_defaults() -> None:
    module = _load_script()

    args = module.build_arg_parser().parse_args(
        [
            "--config",
            "cfg.yaml",
            "--checkpoint",
            "last.pt",
            "--data-jsonl",
            "data.jsonl",
            "--out",
            "summary.json",
        ]
    )

    assert args.batch_size == 4
    assert args.threshold == 0.5
    assert args.device == "auto"
