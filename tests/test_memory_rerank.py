import unittest


class MemoryRerankTests(unittest.TestCase):
    def test_none_reranker_preserves_retrieval_order_and_metadata(self):
        from wgram_lm.memoryos.rerank import rerank_results

        results = [
            (0.9, {"source": "a.md", "text": "alpha"}),
            (0.8, {"source": "b.md", "text": "beta"}),
        ]

        reranked = rerank_results("alpha", results, top_k=1, backend="none")

        self.assertEqual(len(reranked), 1)
        score, rec = reranked[0]
        self.assertEqual(score, 0.9)
        self.assertEqual(rec["source"], "a.md")
        self.assertEqual(rec["retrieval_score"], 0.9)
        self.assertEqual(rec["rerank_backend"], "none")

    def test_lexical_reranker_can_promote_lower_retrieval_candidate(self):
        from wgram_lm.memoryos.rerank import rerank_results

        results = [
            (0.95, {"source": "distractor.md", "text": "The west vault passphrase is jade-circuit."}),
            (0.40, {"source": "target.md", "text": "The archive access code is VX-913."}),
        ]

        reranked = rerank_results("What is the archive access code?", results, top_k=1, backend="lexical")

        score, rec = reranked[0]
        self.assertGreater(score, 0.0)
        self.assertEqual(rec["source"], "target.md")
        self.assertEqual(rec["rerank_backend"], "lexical")

    def test_cross_encoder_reranker_uses_rank_method_when_available(self):
        from wgram_lm.memoryos.rerank import rerank_results

        class FakeCrossEncoder:
            def rank(self, query, documents, top_k=None):
                self.query = query
                self.documents = documents
                return [
                    {"corpus_id": 1, "score": 7.0},
                    {"corpus_id": 0, "score": -1.0},
                ][:top_k]

        results = [
            (0.95, {"source": "distractor.md", "text": "wrong"}),
            (0.40, {"source": "target.md", "text": "right"}),
        ]

        reranked = rerank_results(
            "query",
            results,
            top_k=1,
            backend="cross_encoder",
            model=FakeCrossEncoder(),
        )

        score, rec = reranked[0]
        self.assertEqual(score, 7.0)
        self.assertEqual(rec["source"], "target.md")
        self.assertEqual(rec["rerank_backend"], "cross_encoder")


if __name__ == "__main__":
    unittest.main()
