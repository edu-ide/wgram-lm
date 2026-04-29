import importlib.util
from pathlib import Path
import unittest


class FactEvalScriptTests(unittest.TestCase):
    def test_script_parser_defaults_to_probe_cases(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "102_eval_fact_verification_memoryos.py"
        spec = importlib.util.spec_from_file_location("eval_fact_verification_script", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        args = module.build_arg_parser().parse_args([])

        self.assertEqual(args.cases, "data/eval/fact_verification_probe.jsonl")
        self.assertEqual(args.evidence_mode, "target")
        self.assertEqual(args.jsonl_out, "runs/eval/fact_verification_probe.jsonl")


if __name__ == "__main__":
    unittest.main()
