import unittest
from tempfile import NamedTemporaryFile


class InferMemoryTests(unittest.TestCase):
    def test_format_memory_context_includes_sources_and_budget(self):
        from wgram_lm.infer import format_memory_context

        results = [
            (0.95, {"source": "doc-a.md", "chunk_id": 2, "text": "A" * 20}),
            (0.90, {"source": "doc-b.md", "chunk_id": 4, "text": "B" * 200}),
        ]

        context = format_memory_context(results, max_chars=80)

        self.assertIn("MemoryOS evidence", context)
        self.assertIn("SOURCE=doc-a.md CHUNK=2 SCORE=0.9500", context)
        self.assertLessEqual(len(context), 80)

    def test_build_prompt_with_memory_wraps_user_prompt(self):
        from wgram_lm.infer import build_prompt_with_memory

        prompt = build_prompt_with_memory("answer this", "MemoryOS evidence\nfact")

        self.assertIn("MemoryOS evidence", prompt)
        self.assertTrue(prompt.endswith("answer this"))
        self.assertIn("Use the evidence", prompt)

    def test_load_checkpoint_state_handles_local_dataclass_checkpoint(self):
        import torch
        from wgram_lm.config import FullConfig
        from wgram_lm.infer import load_checkpoint_state

        with NamedTemporaryFile(suffix=".pt") as f:
            torch.save({"model": {"x": torch.tensor([1])}, "config": FullConfig()}, f.name)

            state = load_checkpoint_state(f.name, "cpu")

        self.assertIn("model", state)
        self.assertEqual(state["model"]["x"].tolist(), [1])


if __name__ == "__main__":
    unittest.main()
