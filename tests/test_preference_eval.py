import unittest


class PreferenceEvalTests(unittest.TestCase):
    def test_summarize_preference_scores_reports_margin_accuracy(self):
        from wgram_lm.eval.preference import summarize_preference_scores

        summary = summarize_preference_scores(
            chosen_logps=[-0.5, -2.0, -1.0],
            rejected_logps=[-1.5, -1.0, -1.2],
            sample_weights=[1.0, 0.0, 2.0],
            target_margin=0.5,
        )

        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["weighted_count"], 3.0)
        self.assertAlmostEqual(summary["preference_accuracy"], 2.0 / 3.0, places=5)
        self.assertAlmostEqual(summary["weighted_preference_accuracy"], 1.0, places=5)
        self.assertAlmostEqual(summary["margin_pass_rate"], 1.0 / 3.0, places=5)
        self.assertAlmostEqual(summary["weighted_margin_pass_rate"], 1.0 / 3.0, places=5)
        self.assertAlmostEqual(summary["margin_mean"], (1.0 - 1.0 + 0.2) / 3.0, places=5)

    def test_summarize_preference_records_ignores_summary_rows(self):
        from wgram_lm.eval.preference import summarize_preference_records

        records = [
            {"chosen_logp": -0.5, "rejected_logp": -1.0, "sample_weight": 1.0},
            {"summary": {"count": 1}},
            {"chosen_logp": -2.0, "rejected_logp": -1.0, "sample_weight": 1.0},
        ]

        summary = summarize_preference_records(records, target_margin=0.0)

        self.assertEqual(summary["count"], 2)
        self.assertAlmostEqual(summary["preference_accuracy"], 0.5, places=5)
        self.assertAlmostEqual(summary["margin_mean"], -0.25, places=5)

    def test_eval_autocast_context_is_disabled_on_cpu(self):
        import contextlib
        import importlib.util
        from pathlib import Path

        script_path = Path("scripts/121_eval_preference_pairs.py")
        spec = importlib.util.spec_from_file_location("preference_eval_script", script_path)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)

        ctx = module.eval_autocast_context("cpu", use_amp=True)
        self.assertIsInstance(ctx, contextlib.nullcontext)


if __name__ == "__main__":
    unittest.main()
