import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


class MemoryVectorBackendTests(unittest.TestCase):
    def test_numpy_flat_backend_round_trips_and_returns_inner_product_order(self):
        from wgram_lm.memoryos.vector_backends import build_vector_index, search_vector_index

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            vectors = np.asarray(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.9, 0.1],
                ],
                dtype="float32",
            )

            backend = build_vector_index(vectors, out, backend="numpy_flat")
            scores, ids = search_vector_index(out, np.asarray([[1.0, 0.0]], dtype="float32"), top_k=2)

            meta = json.loads((out / "vector_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(backend, "numpy_flat")
            self.assertEqual(meta["backend"], "numpy_flat")
            self.assertEqual(ids[0].tolist(), [0, 2])
            self.assertGreater(scores[0][0], scores[0][1])

    def test_faiss_hnsw_backend_declares_metadata_when_faiss_is_available(self):
        try:
            import faiss  # noqa: F401
        except Exception:
            self.skipTest("faiss is not installed")

        from wgram_lm.memoryos.vector_backends import build_vector_index, search_vector_index

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            vectors = np.eye(4, dtype="float32")

            backend = build_vector_index(vectors, out, backend="faiss_hnsw", hnsw_m=8, hnsw_ef_construction=40)
            _scores, ids = search_vector_index(out, np.asarray([[0.0, 0.0, 1.0, 0.0]], dtype="float32"), top_k=1)

            meta = json.loads((out / "vector_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(backend, "faiss_hnsw")
            self.assertEqual(meta["backend"], "faiss_hnsw")
            self.assertEqual(ids[0].tolist(), [2])

    def test_unknown_backend_fails_clearly(self):
        from wgram_lm.memoryos.vector_backends import build_vector_index

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                build_vector_index(np.eye(2, dtype="float32"), Path(tmp), backend="not-a-backend")


if __name__ == "__main__":
    unittest.main()
