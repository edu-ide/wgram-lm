import importlib.util
from pathlib import Path
import unittest


class MemoryEvalScriptTests(unittest.TestCase):
    def test_resolve_qtrm_scale_uses_override_when_present(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "95_eval_memory_retrieval.py"
        spec = importlib.util.spec_from_file_location("eval_memory_retrieval_script", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        self.assertEqual(module.resolve_qtrm_scale(0.1, None), 0.1)
        self.assertEqual(module.resolve_qtrm_scale(0.1, 0.5), 0.5)


if __name__ == "__main__":
    unittest.main()
