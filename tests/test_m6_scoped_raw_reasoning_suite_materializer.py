from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


def _load_script():
    path = Path("scripts/377_materialize_m6_scoped_raw_reasoning_suite.py")
    spec = importlib.util.spec_from_file_location("m6_scoped_raw_reasoning_suite", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class M6ScopedRawReasoningSuiteMaterializerTests(unittest.TestCase):
    def test_materializes_qwen_prompts_with_operation_definitions(self):
        module = _load_script()
        with TemporaryDirectory() as tmp:
            report = Path(tmp) / "qtrm.json"
            report.write_text(
                json.dumps(
                    {
                        "include_family_tag": False,
                        "train": {
                            "eval_cases": 3,
                            "eval_seed": 9337,
                            "program_len": 4,
                            "modulus": 32,
                            "task_families": "modchain,revchain,checksum",
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = module.build_arg_parser().parse_args(
                ["--qtrm-report", str(report), "--max-cases", "3"]
            )

            suite = module.build_suite(args)

        self.assertEqual(suite["case_count"], 3)
        self.assertEqual(suite["prompt_protocol"], "operation_definitions_v1")
        self.assertIn("06:", "\n".join(suite["operation_definitions"]))
        self.assertIn("Operation IDs", suite["rows"][0]["qwen_prompt"])
        self.assertRegex(suite["rows"][0]["answer_text"], r"^\d\d$")

    def test_write_suite_outputs_jsonl_and_metadata(self):
        module = _load_script()
        suite = {
            "suite_id": "suite",
            "prompt_protocol": "protocol",
            "source_qtrm_report": "report.json",
            "case_count": 1,
            "modulus": 32,
            "operation_definitions": ["00: noop"],
            "rows": [{"case_id": "case-1", "qwen_prompt": "Prompt", "answer_text": "01"}],
        }
        with TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "cases.jsonl"
            meta = Path(tmp) / "metadata.json"

            module.write_suite(suite, out_jsonl=jsonl, out_meta=meta)

            rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
            metadata = json.loads(meta.read_text(encoding="utf-8"))

        self.assertEqual(rows[0]["case_id"], "case-1")
        self.assertEqual(metadata["case_count"], 1)
        self.assertNotIn("rows", metadata)


if __name__ == "__main__":
    unittest.main()
