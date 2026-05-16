from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock


def _load_script():
    path = Path("scripts/381_eval_openai_compatible_scoped_reasoning_baseline.py")
    spec = importlib.util.spec_from_file_location("openai_compatible_scoped_reasoning_baseline", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeResponse:
    def __init__(self, payload: dict[str, object]):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class OpenAICompatibleScopedReasoningBaselineEvalTests(unittest.TestCase):
    def test_normalize_two_digit_answer(self):
        module = _load_script()

        self.assertEqual(module.normalize_two_digit_answer("7"), "07")
        self.assertEqual(module.normalize_two_digit_answer("Answer: 12."), "12")
        self.assertEqual(module.normalize_two_digit_answer("no answer"), "")

    def test_chat_completion_extracts_message_content(self):
        module = _load_script()
        with mock.patch.object(
            module.request,
            "urlopen",
            return_value=_FakeResponse(
                {"choices": [{"message": {"role": "assistant", "content": "08"}}]}
            ),
        ):
            result = module.chat_completion(
                base_url="http://127.0.0.1:18082/v1",
                model="local",
                prompt="prompt",
                max_tokens=8,
                temperature=0.0,
                timeout=1.0,
                retries=0,
            )

        self.assertEqual(result, "08")

    def test_evaluate_api_writes_report_and_predictions(self):
        module = _load_script()
        with TemporaryDirectory() as tmp:
            suite = Path(tmp) / "suite.jsonl"
            out_json = Path(tmp) / "report.json"
            out_jsonl = Path(tmp) / "predictions.jsonl"
            suite.write_text(
                json.dumps(
                    {
                        "suite_id": "suite",
                        "prompt_protocol": "operation_definitions_v1",
                        "case_id": "case-1",
                        "family": "modchain",
                        "qwen_prompt": "Prompt",
                        "answer_text": "08",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            args = module.build_arg_parser().parse_args(
                [
                    "--suite-jsonl",
                    str(suite),
                    "--out-json",
                    str(out_json),
                    "--out-jsonl",
                    str(out_jsonl),
                    "--log-every",
                    "0",
                ]
            )
            with mock.patch.object(module, "chat_completion", return_value="08"):
                report = module.evaluate_api(args)
            written_report = json.loads(out_json.read_text(encoding="utf-8"))
            written_prediction = json.loads(out_jsonl.read_text(encoding="utf-8"))

        self.assertEqual(report["score"], 1.0)
        self.assertEqual(written_report["cases"], 1)
        self.assertEqual(written_prediction["exact"], True)


if __name__ == "__main__":
    unittest.main()
