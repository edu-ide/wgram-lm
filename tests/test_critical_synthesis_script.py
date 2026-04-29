import importlib.util
from pathlib import Path
import unittest


class CriticalSynthesisScriptTests(unittest.TestCase):
    def test_script_parser_defaults_to_probe_and_filtered_output(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "103_build_critical_synthesis_traces.py"
        spec = importlib.util.spec_from_file_location("build_critical_synthesis_script", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        args = module.build_arg_parser().parse_args([])

        self.assertEqual(args.cases, "data/eval/critical_synthesis_probe.jsonl")
        self.assertEqual(args.out, "data/filtered/critical_synthesis_traces.jsonl")
        self.assertEqual(args.max_evidence_chars, 4000)


if __name__ == "__main__":
    unittest.main()
