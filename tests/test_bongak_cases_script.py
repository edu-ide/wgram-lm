import importlib.util
from pathlib import Path
import unittest


class BongakCasesScriptTests(unittest.TestCase):
    def test_script_parser_defaults_to_bongak_docs_and_outputs(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "104_build_bongak_critical_synthesis_cases.py"
        spec = importlib.util.spec_from_file_location("build_bongak_cases_script", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        args = module.build_arg_parser().parse_args([])

        self.assertIn("본각교_요약.md", args.summary)
        self.assertIn("본각교_매뉴얼.md", args.manual)
        self.assertEqual(args.out, "data/filtered/critical_synthesis_bongak_cases.jsonl")
        self.assertEqual(args.traces_out, "data/filtered/critical_synthesis_bongak_traces.jsonl")
        self.assertEqual(args.max_cases, 30)


if __name__ == "__main__":
    unittest.main()
