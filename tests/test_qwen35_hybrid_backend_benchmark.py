import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def load_module():
    path = Path("scripts/360_benchmark_qwen35_hybrid_backend.py")
    spec = importlib.util.spec_from_file_location("qwen35_hybrid_backend_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Qwen35HybridBackendBenchmarkTests(unittest.TestCase):
    def test_cpu_attention_only_smoke_writes_report(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            args = module.build_arg_parser().parse_args(
                [
                    "--out-dir",
                    tmp,
                    "--device",
                    "cpu",
                    "--cases",
                    "attention_only",
                    "--backends",
                    "torch_gated_delta",
                    "--optimizer",
                    "adamw",
                    "--batch-size",
                    "1",
                    "--seq-len",
                    "4",
                    "--d-model",
                    "8",
                    "--n-heads",
                    "2",
                    "--n-kv-heads",
                    "1",
                    "--d-ff",
                    "16",
                    "--repeat-steps",
                    "1",
                ]
            )

            report = module.run_benchmarks(args)
            written = json.loads((Path(tmp) / "report.json").read_text(encoding="utf-8"))

        self.assertEqual(report["status"], "complete")
        self.assertEqual(written["status"], "complete")
        self.assertEqual(len(report["results"]), 1)
        row = report["results"][0]
        self.assertEqual(row["case"], "attention_only")
        self.assertEqual(row["optimizer_report"]["resolved"], "adamw")
        self.assertEqual(len(row["repeat_forward_backward_ms"]), 1)


if __name__ == "__main__":
    unittest.main()
