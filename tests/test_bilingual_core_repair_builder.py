import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def load_module():
    path = Path("scripts/358_build_bilingual_core_repair.py")
    spec = importlib.util.spec_from_file_location("bilingual_core_repair", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BilingualCoreRepairBuilderTests(unittest.TestCase):
    def test_make_records_contains_english_and_korean_families(self):
        module = load_module()

        rows = module.make_records(repeats=1)
        sources = {row["source"] for row in rows}
        text = "\n".join(row["text"] for row in rows)

        self.assertIn("bilingual_core_repair:evidence:en", sources)
        self.assertIn("bilingual_core_repair:evidence:ko", sources)
        self.assertIn("Why should evidence be checked", text)
        self.assertIn("주장을 믿기 전에", text)
        self.assertTrue(all("User:" in row["text"] and "Assistant:" in row["text"] for row in rows))

    def test_build_repair_writes_report(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "repair.jsonl"
            args = module.build_arg_parser().parse_args(["--out", str(out), "--repeats", "2"])

            report = module.build_repair(args)

            rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
            saved_report = json.loads(
                out.with_suffix(out.suffix + ".report.json").read_text(encoding="utf-8")
            )

        self.assertEqual(report["records"], len(rows))
        self.assertEqual(saved_report["records"], len(rows))
        self.assertGreater(len(rows), 40)


if __name__ == "__main__":
    unittest.main()
