import importlib.util
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


def load_module():
    path = Path("scripts/357_build_external_language_corpus.py")
    spec = importlib.util.spec_from_file_location("external_language_corpus", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ExternalLanguageCorpusBuilderTests(unittest.TestCase):
    def test_extract_ultrachat_pairs_user_assistant_turns(self):
        module = load_module()
        row = {
            "messages": [
                {"role": "user", "content": "Why check evidence?"},
                {"role": "assistant", "content": "Evidence helps test whether a claim is supported."},
                {"role": "user", "content": "What if evidence is weak?"},
                {"role": "assistant", "content": "Say it is uncertain and avoid guessing."},
            ]
        }

        records = module.extract_ultrachat(row, max_chars=400)

        self.assertEqual(len(records), 2)
        self.assertIn("User: Why check evidence?", records[0])
        self.assertIn("Assistant: Evidence helps", records[0])
        self.assertIn("avoid guessing", records[1])

    def test_extract_alpaca_builds_answer_only_chat_record(self):
        module = load_module()
        row = {"instruction": "좋은 답변은?", "output": "좋은 답변은 근거를 제시한다."}

        records = module.extract_alpaca(row, max_chars=300)

        self.assertEqual(records, ["User: 좋은 답변은?\nAssistant: 좋은 답변은 근거를 제시한다."])

    def test_reject_text_removes_visible_think_and_url_spam(self):
        module = load_module()

        self.assertTrue(
            module.reject_text("<think>hidden</think> final", min_chars=1, max_chars=100)
        )
        self.assertTrue(
            module.reject_text(
                "a http://x http://x http://x http://x http://x",
                min_chars=1,
                max_chars=200,
            )
        )
        self.assertFalse(
            module.reject_text(
                "A clear answer gives a direct claim and a reason.",
                min_chars=10,
                max_chars=200,
            )
        )

    def test_parse_source_requires_complete_spec(self):
        module = load_module()

        parsed = module.parse_source("fineweb:HuggingFaceFW/fineweb-edu:sample-10BT:train:5")

        self.assertEqual(parsed["kind"], "fineweb")
        self.assertEqual(parsed["limit"], 5)
        with self.assertRaises(ValueError):
            module.parse_source("fineweb:missing")

    def test_parser_accepts_per_source_record_cap(self):
        module = load_module()

        args = module.build_arg_parser().parse_args(
            [
                "--max-records-per-source",
                "7",
                "--retries",
                "4",
                "--retry-sleep-base",
                "0.5",
                "--request-delay-seconds",
                "0.25",
                "--continue-on-source-error",
            ]
        )

        self.assertEqual(args.max_records_per_source, 7)
        self.assertEqual(args.retries, 4)
        self.assertEqual(args.retry_sleep_base, 0.5)
        self.assertEqual(args.request_delay_seconds, 0.25)
        self.assertTrue(args.continue_on_source_error)
        self.assertEqual(args.source, [])

    def test_build_corpus_can_continue_after_source_error(self):
        module = load_module()
        original_fetch_rows = module.fetch_rows

        def fake_fetch_rows(*, dataset, config, split, limit, page_size, **_kwargs):
            if dataset == "bad/source":
                raise RuntimeError("rate limited")
            return [{"instruction": "좋은 답변은?", "output": "좋은 답변은 근거를 제시한다."}]

        module.fetch_rows = fake_fetch_rows
        try:
            with TemporaryDirectory() as tmp:
                out = Path(tmp) / "corpus.jsonl"
                args = module.build_arg_parser().parse_args(
                    [
                        "--out",
                        str(out),
                        "--source",
                        "koalpaca:bad/source:default:train:1",
                        "--source",
                        "koalpaca:good/source:default:train:1",
                        "--continue-on-source-error",
                    ]
                )

                report = module.build_corpus(args)
        finally:
            module.fetch_rows = original_fetch_rows

        self.assertEqual(report["records"], 1)
        self.assertIn("koalpaca:bad/source:default:train", report["source_errors"])
        self.assertIn("koalpaca:good/source:default:train", report["source_counts"])


if __name__ == "__main__":
    unittest.main()
