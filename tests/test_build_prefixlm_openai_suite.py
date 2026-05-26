from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


def _load_script():
    path = Path("scripts/541_build_prefixlm_openai_suite.py")
    spec = importlib.util.spec_from_file_location("build_prefixlm_openai_suite", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class BuildPrefixLMOpenAISuiteTests(unittest.TestCase):
    def test_strip_instruction_wrappers_returns_condition_and_problem(self):
        module = _load_script()
        tokenizer_info = {
            "boq": "<|im_start|>",
            "eoq": "<|im_end|>",
            "condition_mapping": {
                "direct": "<|object_ref_start|>",
                "cot": "<|object_ref_end|>",
            },
        }

        condition, problem = module.strip_instruction_wrappers(
            "<|im_start|><|object_ref_start|>What is 2+2?<|im_end|>",
            tokenizer_info,
        )

        self.assertEqual(condition, "direct")
        self.assertEqual(problem, "What is 2+2?")

    def test_strip_response_removes_box_end(self):
        module = _load_script()

        self.assertEqual(module.strip_response_text("4<|box_end|>", "<|box_end|>"), "4")

    def test_write_suite_rows_emits_openai_compatible_jsonl(self):
        module = _load_script()
        rows = [
            module.SuiteRow(
                source_row=7,
                condition="direct",
                instruction="What is 2+2?",
                answer="4",
            )
        ]
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "suite.jsonl"
            report = module.write_suite_rows(
                rows,
                out_jsonl=out,
                suite_id="suite",
                prompt_protocol="hrm_text_data_io_answer_only_v1",
            )
            written = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(report["rows"], 1)
        self.assertEqual(written["suite_id"], "suite")
        self.assertEqual(written["case_id"], "epoch-row-7")
        self.assertEqual(written["condition"], "direct")
        self.assertIn("What is 2+2?", written["qwen_prompt"])
        self.assertEqual(written["answer_text"], "4")

    def test_auto_prompt_style_requests_full_solution_for_cot_rows(self):
        module = _load_script()
        rows = [
            module.SuiteRow(
                source_row=11,
                condition="cot",
                instruction="Prove 1+1=2.",
                answer="A full derivation. Therefore \\boxed{2}",
            )
        ]
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "suite.jsonl"
            module.write_suite_rows(
                rows,
                out_jsonl=out,
                suite_id="suite",
                prompt_protocol="hrm_text_data_io_auto_by_condition_v1",
                prompt_style="auto",
            )
            written = json.loads(out.read_text(encoding="utf-8"))

        self.assertIn("Return the full solution", written["qwen_prompt"])
        self.assertIn("\\boxed{}", written["qwen_prompt"])
        self.assertNotIn("Return only the final answer", written["qwen_prompt"])


if __name__ == "__main__":
    unittest.main()
