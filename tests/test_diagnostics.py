import unittest

import torch


class DiagnosticsTests(unittest.TestCase):
    def test_next_token_diagnostics_reports_rank_entropy_and_accuracy(self):
        from qtrm_mm.diagnostics import next_token_diagnostics

        logits = torch.full((1, 4, 5), -10.0)
        input_ids = torch.tensor([[0, 2, 3, 4]])
        attention_mask = torch.tensor([[1, 1, 1, 0]])

        logits[0, 0, 2] = 10.0
        logits[0, 1, 1] = 10.0
        logits[0, 1, 3] = 9.0
        logits[0, 2, 4] = 10.0

        metrics = next_token_diagnostics(logits, input_ids, attention_mask=attention_mask)

        self.assertEqual(metrics["valid_tokens"], 2)
        self.assertAlmostEqual(metrics["target_rank_mean"], 1.5)
        self.assertAlmostEqual(metrics["target_top1_acc"], 0.5)
        self.assertGreater(metrics["entropy_mean"], 0.0)
        self.assertGreater(metrics["loss"], 0.0)

    def test_topk_token_report_decodes_tokens(self):
        from qtrm_mm.diagnostics import topk_token_report

        class FakeTokenizer:
            def decode(self, token_ids, **kwargs):
                return f"tok{token_ids[0]}"

        logits = torch.tensor([0.0, 4.0, 2.0])

        report = topk_token_report(logits, tokenizer=FakeTokenizer(), k=2)

        self.assertEqual([item["token_id"] for item in report], [1, 2])
        self.assertEqual(report[0]["token"], "tok1")
        self.assertGreater(report[0]["prob"], report[1]["prob"])

    def test_repetition_stats_focuses_on_completion(self):
        from qtrm_mm.diagnostics import repetition_stats

        stats = repetition_stats([7, 8, 8, 8, 4, 5, 4, 5], prompt_len=1)

        self.assertEqual(stats["completion_tokens"], 7)
        self.assertEqual(stats["max_token_run"], 3)
        self.assertGreater(stats["repeated_2gram_rate"], 0.0)
        self.assertEqual(stats["most_common_token_id"], 8)


if __name__ == "__main__":
    unittest.main()
