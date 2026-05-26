import importlib.util
from pathlib import Path
import unittest


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "547_write_prefixlm_raw_intelligence_tensorboard.py"
    spec = importlib.util.spec_from_file_location("raw_intelligence_tensorboard_writer", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class FakeWriter:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))


class RawIntelligenceTensorBoardWriterTest(unittest.TestCase):
    def test_sanitize_tag_part(self):
        module = load_module()
        self.assertEqual(module.sanitize_tag_part("reasoning arithmetic"), "reasoning_arithmetic")
        self.assertEqual(module.sanitize_tag_part("ko/한국어"), "ko")
        self.assertEqual(module.sanitize_tag_part("!!!"), "unknown")

    def test_add_scalars_skips_nonfinite(self):
        module = load_module()
        writer = FakeWriter()
        count = module.add_scalars(
            writer,
            "eval/raw_intelligence/primitive/language",
            {
                "cases": 2,
                "loss": 0.5,
                "perplexity": float("inf"),
                "token_accuracy": 0.25,
                "hits": 0,
                "accuracy": 0.0,
                "generation_accuracy": 0.0,
            },
            123,
        )
        self.assertEqual(count, 6)
        self.assertIn(("eval/raw_intelligence/primitive/language/loss", 0.5, 123), writer.scalars)
        self.assertIn(("eval/raw_intelligence/primitive/language/accuracy", 0.0, 123), writer.scalars)
        self.assertNotIn(
            ("eval/raw_intelligence/primitive/language/perplexity", float("inf"), 123),
            writer.scalars,
        )


if __name__ == "__main__":
    unittest.main()
