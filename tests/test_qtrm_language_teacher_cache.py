import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def load_module():
    path = Path("scripts/355_build_qtrm_language_teacher_cache.py")
    spec = importlib.util.spec_from_file_location("qtrm_language_teacher_cache", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class QTRMLanguageTeacherCacheTests(unittest.TestCase):
    def test_collect_prompts_reads_jsonl_and_builtin(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.jsonl"
            path.write_text(json.dumps({"prompt": "Explain a clear sentence."}) + "\n", encoding="utf-8")
            args = module.build_arg_parser().parse_args(
                [
                    "--dry-run",
                    "--source-jsonl",
                    str(path),
                    "--max-records",
                    "20",
                    "--max-prompt-chars",
                    "80",
                ]
            )

            prompts = module.collect_prompts(args)

        self.assertGreaterEqual(len(prompts), 1)
        self.assertTrue(any("Explain a clear sentence" in row["prompt"] for row in prompts))

    def test_dry_run_writes_teacher_jsonl_consumable_by_bootstrap(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "teacher.jsonl"
            args = module.build_arg_parser().parse_args(
                [
                    "--dry-run",
                    "--out",
                    str(out),
                    "--max-records",
                    "2",
                    "--max-new-tokens",
                    "12",
                    "--min-answer-chars",
                    "10",
                ]
            )

            report = module.build_cache(args)
            rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
            report_exists = Path(str(out) + ".report.json").exists()

        self.assertEqual(report["written"], 2)
        self.assertEqual(len(rows), 2)
        self.assertIn("teacher_text", rows[0])
        self.assertIn("prompt", rows[0])
        self.assertIn("seed_text", rows[0])
        self.assertIn("answer", rows[0])
        self.assertNotIn("Task: continue", rows[0]["text"])
        self.assertTrue(report_exists)

    def test_repetition_filter_rejects_repeated_text(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            ["--dry-run", "--max-run-fraction", "0.2", "--min-answer-chars", "3"]
        )

        keep, reasons = module.should_keep("aaaaaaaaaaaa", args)

        self.assertFalse(keep)
        self.assertIn("answer_repetition_run_too_high", reasons)

    def test_visible_think_is_rejected_and_stripped(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(["--dry-run"])

        keep, reasons = module.should_keep("<think>hidden trace", args)

        self.assertFalse(keep)
        self.assertIn("answer_contains_visible_think", reasons)
        self.assertEqual(module.strip_think_blocks("<think>hidden trace"), "")


if __name__ == "__main__":
    unittest.main()
