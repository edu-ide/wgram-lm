import importlib.util
import json
from pathlib import Path


def load_module():
    path = Path("scripts/146_eval_generation_verifier_rerank.py")
    spec = importlib.util.spec_from_file_location("generation_verifier_rerank_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_candidate_score_prefers_quality_and_penalizes_failures() -> None:
    module = load_module()

    good = {"quality_prob": 0.9, "repeat_prob": 0.1, "stop_prob": 0.1}
    bad = {"quality_prob": 0.7, "repeat_prob": 0.8, "stop_prob": 0.1}

    assert module.candidate_score(good) > module.candidate_score(bad)


def test_rerank_groups_selects_best_candidate_per_source_sample() -> None:
    module = load_module()
    records = [
        {
            "source_sample": 0,
            "candidate_id": 0,
            "quality_prob": 0.2,
            "repeat_prob": 0.8,
            "stop_prob": 0.1,
            "quality_target": 0.0,
            "repeat_target": 1.0,
            "stop_target": 0.0,
        },
        {
            "source_sample": 0,
            "candidate_id": 1,
            "quality_prob": 0.8,
            "repeat_prob": 0.1,
            "stop_prob": 0.1,
            "quality_target": 1.0,
            "repeat_target": 0.0,
            "stop_target": 0.0,
        },
        {
            "source_sample": 1,
            "candidate_id": 0,
            "quality_prob": 0.9,
            "repeat_prob": 0.1,
            "stop_prob": 0.1,
            "quality_target": 1.0,
            "repeat_target": 0.0,
            "stop_target": 0.0,
        },
    ]

    summary = module.summarize_rerank(records)

    assert summary["groups"] == 2
    assert summary["candidate_count"] == 3
    assert summary["baseline_quality_rate"] == 0.5
    assert summary["reranked_quality_rate"] == 1.0
    assert summary["oracle_quality_rate"] == 1.0
    assert summary["selected_changed_rate"] == 0.5


def test_rerank_script_writes_report(tmp_path: Path) -> None:
    module = load_module()
    eval_path = tmp_path / "eval.json"
    out_path = tmp_path / "rerank.json"
    eval_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "source_sample": 0,
                        "candidate_id": 0,
                        "quality_prob": 0.1,
                        "repeat_prob": 0.9,
                        "stop_prob": 0.0,
                        "quality_target": 0.0,
                        "repeat_target": 1.0,
                        "stop_target": 0.0,
                    },
                    {
                        "source_sample": 0,
                        "candidate_id": 1,
                        "quality_prob": 0.9,
                        "repeat_prob": 0.1,
                        "stop_prob": 0.0,
                        "quality_target": 1.0,
                        "repeat_target": 0.0,
                        "stop_target": 0.0,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    module.main(["--eval-json", str(eval_path), "--out", str(out_path)])

    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["reranked_quality_rate"] == 1.0
    assert report["selected"][0]["selected_candidate_id"] == 1
