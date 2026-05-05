import importlib.util
import json
from pathlib import Path


def load_module():
    path = Path("scripts/147_summarize_generation_format.py")
    spec = importlib.util.spec_from_file_location("generation_format_report_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_has_visible_reasoning_detects_think_tags_and_markers() -> None:
    module = load_module()

    assert module.has_visible_reasoning("Answer\n<think>hidden</think>")
    assert module.has_visible_reasoning("Let me think step by step before answering.")
    assert module.has_visible_reasoning("The user is asking for a short answer.")
    assert not module.has_visible_reasoning("We need to subtract 3 from both sides.")
    assert not module.has_visible_reasoning("A database index helps look up rows faster.")


def test_has_answer_drift_detects_mcq_and_new_question_continuations() -> None:
    module = load_module()

    assert module.has_answer_drift("A. option one\nB. option two")
    assert module.has_answer_drift("Answer.\n\nWhat is the next question?")
    assert module.has_answer_drift("Answer. Pls answer in simple terms.")
    assert module.has_answer_drift("Do not mention the AI. The answer must be concise.")
    assert module.has_answer_drift("If the user's question is X, you should reply Y.")
    assert not module.has_answer_drift("The sky is blue because air scatters blue light.")


def test_summarize_records_counts_format_and_repeat_failures() -> None:
    module = load_module()
    records = [
        {
            "sample": 0,
            "candidate_id": 0,
            "greedy_text": "Question\n<think>hidden</think>",
            "greedy_repetition": {"repeated_2gram_rate": 0.0},
        },
        {
            "sample": 0,
            "candidate_id": 1,
            "greedy_text": "Question\nGood answer.",
            "greedy_repetition": {"repeated_2gram_rate": 0.2},
        },
        {
            "sample": 1,
            "candidate_id": 0,
            "greedy_text": "Question\nClean.",
            "greedy_repetition": {"repeated_2gram_rate": 0.0},
        },
    ]

    summary = module.summarize_records(records, repeat_threshold=0.15)

    assert summary["records"] == 3
    assert summary["groups"] == 2
    assert summary["visible_reasoning_rate"] == 1 / 3
    assert summary["repeat_failure_rate"] == 1 / 3
    assert summary["clean_rate"] == 1 / 3
    assert summary["answer_drift_rate"] == 0.0


def test_summarize_records_checks_completion_not_prompt_contract() -> None:
    module = load_module()
    prompt = "Question?\n\n/no_think\nAnswer directly. Do not reveal hidden reasoning."
    records = [
        {
            "sample": 0,
            "candidate_id": 0,
            "text": prompt,
            "greedy_text": prompt + "\nClean answer.",
            "greedy_repetition": {"repeated_2gram_rate": 0.0},
        }
    ]

    summary = module.summarize_records(records)

    assert summary["visible_reasoning_rate"] == 0.0
    assert summary["clean_rate"] == 1.0


def test_format_report_script_writes_summary(tmp_path: Path) -> None:
    module = load_module()
    eval_path = tmp_path / "eval.jsonl"
    out_path = tmp_path / "summary.json"
    eval_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "sample": 0,
                        "candidate_id": 0,
                        "greedy_text": "Prompt\n<think>trace",
                        "greedy_repetition": {"repeated_2gram_rate": 0.0},
                    }
                ),
                json.dumps(
                    {
                        "sample": 0,
                        "candidate_id": 1,
                        "greedy_text": "Prompt\nAnswer.",
                        "greedy_repetition": {"repeated_2gram_rate": 0.0},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    module.main(["--eval-jsonl", str(eval_path), "--out", str(out_path)])

    summary = json.loads(out_path.read_text(encoding="utf-8"))
    assert summary["visible_reasoning_count"] == 1
    assert summary["clean_count"] == 1
