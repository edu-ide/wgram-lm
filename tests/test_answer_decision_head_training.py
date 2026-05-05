from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_script():
    path = Path("scripts/161_train_answer_decision_head.py")
    spec = importlib.util.spec_from_file_location("answer_decision_head", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(
    *,
    record_id: str,
    completion: str,
    aliases: list[str],
    expected_unknown: bool,
    hit: bool,
    missing: float,
    no_answer: float,
) -> dict:
    return {
        "id": record_id,
        "mode": "qtrm_residual_with_evidence",
        "question": f"Question {record_id}",
        "completion": completion,
        "raw_completion": completion,
        "answer_aliases": aliases,
        "expected_unknown": expected_unknown,
        "hit": hit,
        "task_family": "abstention" if expected_unknown else "multi_hop",
        "completion_token_count": 4,
        "prompt_token_count": 64,
        "first_step_logit_shift": {"max_abs_delta": 0.5},
        "latent_gates": {
            "workspace_update_gate_mean": 0.1,
            "workspace_update_gate_last_mean": 0.1,
            "core_context_gate_mean": 0.2,
            "core_context_gate_last_mean": 0.2,
        },
        "answer_channel_meta": {
            "status": "span",
            "selected_score": 12.0,
            "selected_token_ids": [1, 2, 3],
            "no_answer_prob": no_answer,
            "truth_gate": {
                "allow": True,
                "block_reasons": [],
                "support_prob": 0.7,
                "causal_prob": 0.7,
                "refute_prob": 0.1,
                "missing_prob": missing,
            },
        },
    }


def test_label_block_improves_only_when_unknown_block_fixes_record() -> None:
    module = _load_script()
    false_positive = _record(
        record_id="fp",
        completion="Answer: fake",
        aliases=["UNKNOWN"],
        expected_unknown=True,
        hit=False,
        missing=0.8,
        no_answer=0.8,
    )
    positive_wrong = _record(
        record_id="wrong",
        completion="Answer: fake",
        aliases=["gold"],
        expected_unknown=False,
        hit=False,
        missing=0.8,
        no_answer=0.8,
    )

    assert module.label_block_improves(false_positive) == 1
    assert module.label_block_improves(positive_wrong) == 0


def test_train_head_learns_synthetic_missing_signal() -> None:
    module = _load_script()
    records = []
    for idx in range(12):
        records.append(
            _record(
                record_id=f"fp-{idx}",
                completion="Answer: fake",
                aliases=["UNKNOWN"],
                expected_unknown=True,
                hit=False,
                missing=0.8,
                no_answer=0.8,
            )
        )
        records.append(
            _record(
                record_id=f"pos-{idx}",
                completion="Answer: gold",
                aliases=["gold"],
                expected_unknown=False,
                hit=True,
                missing=0.1,
                no_answer=0.1,
            )
        )
    examples = module.build_examples(records)
    model = module.train_head(
        examples,
        epochs=120,
        lr=3e-3,
        hidden_dim=16,
        dropout=0.0,
        seed=7,
    )
    probs = module.block_probabilities(model, examples)
    threshold, metrics = module.select_threshold(examples, probs)

    assert 0.05 <= threshold <= 0.95
    assert metrics["accuracy"] == 1.0
    assert metrics["false_positive"] == 0


def test_main_writes_model_and_report(tmp_path: Path) -> None:
    module = _load_script()
    rows = []
    for idx in range(8):
        rows.append(
            _record(
                record_id=f"fp-{idx}",
                completion="Answer: fake",
                aliases=["UNKNOWN"],
                expected_unknown=True,
                hit=False,
                missing=0.8,
                no_answer=0.8,
            )
        )
        rows.append(
            _record(
                record_id=f"pos-{idx}",
                completion="Answer: gold",
                aliases=["gold"],
                expected_unknown=False,
                hit=True,
                missing=0.1,
                no_answer=0.1,
            )
        )
    records_path = tmp_path / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    out_pt = tmp_path / "head.pt"
    out_json = tmp_path / "report.json"
    out_md = tmp_path / "report.md"
    module.main(
        [
            "--train-records-jsonl",
            str(records_path),
            "--out-pt",
            str(out_pt),
            "--out-json",
            str(out_json),
            "--markdown-out",
            str(out_md),
            "--epochs",
            "80",
            "--hidden-dim",
            "16",
        ]
    )

    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert out_pt.exists()
    assert out_md.exists()
    assert report["eval_count"] > 0
    assert "eval_learned" in report
