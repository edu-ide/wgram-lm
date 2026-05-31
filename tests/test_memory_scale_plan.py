import unittest


class MemoryScalePlanTests(unittest.TestCase):
    def test_estimates_overlapped_chunk_count(self):
        from wgram_lm.memoryos.scale_plan import estimate_chunk_count

        self.assertEqual(estimate_chunk_count(0, chunk_tokens=512, overlap_tokens=64), 0)
        self.assertEqual(estimate_chunk_count(512, chunk_tokens=512, overlap_tokens=64), 1)
        self.assertEqual(estimate_chunk_count(513, chunk_tokens=512, overlap_tokens=64), 2)

    def test_builds_100m_memoryos_plan_with_harrier_dimensions(self):
        from wgram_lm.memoryos.scale_plan import build_memory_scale_plan

        plan = build_memory_scale_plan(
            total_tokens=100_000_000,
            chunk_tokens=512,
            overlap_tokens=64,
            embedding_dim=640,
            available_ram_gib=64,
            available_vram_gib=24,
        )

        self.assertEqual(plan.estimated_chunks, 223_215)
        self.assertAlmostEqual(plan.embedding_gib, 0.532, places=3)
        self.assertEqual(plan.build_backend, "faiss_hnsw")
        self.assertEqual(plan.serving_pattern, "retrieve-rerank-compress")
        self.assertTrue(plan.needs_latent_memory_layer)

    def test_rejects_invalid_overlap(self):
        from wgram_lm.memoryos.scale_plan import estimate_chunk_count

        with self.assertRaises(ValueError):
            estimate_chunk_count(1000, chunk_tokens=512, overlap_tokens=512)


if __name__ == "__main__":
    unittest.main()
