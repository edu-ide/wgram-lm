import unittest


class MemoryScaleBenchmarkTests(unittest.TestCase):
    def test_parse_token_targets_accepts_commas_and_suffixes(self):
        from qtrm_mm.memoryos.scale_benchmark import parse_token_targets

        self.assertEqual(parse_token_targets("1M,10M,2500000"), [1_000_000, 10_000_000, 2_500_000])

    def test_build_scale_benchmark_records_default_to_1m_and_10m(self):
        from qtrm_mm.memoryos.scale_benchmark import build_scale_benchmark_records

        records = build_scale_benchmark_records([1_000_000, 10_000_000])

        self.assertEqual([record["target_label"] for record in records], ["1M", "10M"])
        self.assertEqual(records[0]["plan"]["build_backend"], "faiss_flat")
        self.assertEqual(records[1]["plan"]["build_backend"], "faiss_flat")
        self.assertEqual(records[0]["plan"]["serving_pattern"], "retrieve-rerank-compress")
        self.assertTrue(records[1]["plan"]["needs_latent_memory_layer"])


if __name__ == "__main__":
    unittest.main()
