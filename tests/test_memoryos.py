import unittest

import torch


class MemoryOSTests(unittest.TestCase):
    def test_text_memory_defaults_to_harrier_270m(self):
        from qtrm_mm.memoryos.retrieve import DEFAULT_TEXT_EMBED_MODEL as retrieve_default
        from qtrm_mm.memoryos.text_index import DEFAULT_TEXT_EMBED_MODEL as index_default

        self.assertEqual(index_default, "microsoft/harrier-oss-v1-270m")
        self.assertEqual(retrieve_default, "microsoft/harrier-oss-v1-270m")

    def test_harrier_query_encoding_uses_official_prompt_name(self):
        from qtrm_mm.memoryos.retrieve import encode_query

        class FakeModel:
            def __init__(self):
                self.calls = []

            def encode(self, texts, **kwargs):
                self.calls.append((texts, kwargs))
                return [[1.0, 0.0]]

        model = FakeModel()
        encode_query(model, "What is the archive code?", model_id="microsoft/harrier-oss-v1-270m")

        texts, kwargs = model.calls[0]
        self.assertEqual(texts, ["What is the archive code?"])
        self.assertEqual(kwargs["prompt_name"], "web_search_query")
        self.assertTrue(kwargs["normalize_embeddings"])

    def test_non_harrier_query_encoding_uses_explicit_instruction_text(self):
        from qtrm_mm.memoryos.retrieve import encode_query

        class FakeModel:
            def __init__(self):
                self.calls = []

            def encode(self, texts, **kwargs):
                self.calls.append((texts, kwargs))
                return [[1.0, 0.0]]

        model = FakeModel()
        encode_query(model, "What is the archive code?", model_id="other/embedder")

        texts, kwargs = model.calls[0]
        self.assertIn("Instruct: Retrieve relevant passages", texts[0])
        self.assertTrue(kwargs["normalize_embeddings"])

    def test_visual_embedding_output_object_is_coerced_to_tensor(self):
        from qtrm_mm.memoryos.visual_index import coerce_embedding_tensor

        class Output:
            pooler_output = torch.ones(2, 4)

        emb = coerce_embedding_tensor(Output())

        self.assertEqual(emb.shape, (2, 4))


if __name__ == "__main__":
    unittest.main()
