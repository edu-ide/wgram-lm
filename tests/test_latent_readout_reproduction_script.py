from pathlib import Path
import importlib.util
import unittest


def _load_module():
    path = Path("scripts/331_train_latent_readout_reproduction.py")
    spec = importlib.util.spec_from_file_location("latent_readout_repro", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class LatentReadoutReproductionScriptTests(unittest.TestCase):
    def test_parser_exposes_on_policy_readout_controls(self):
        module = _load_module()

        args = module.parse_args(
            [
                "--train-steps",
                "3",
                "--scheduled-sampling-prob",
                "0.5",
                "--accept-greedy-exact",
                "0.75",
            ]
        )

        self.assertEqual(args.train_steps, 3)
        self.assertEqual(args.scheduled_sampling_prob, 0.5)
        self.assertEqual(args.accept_greedy_exact, 0.75)

    def test_case_builder_uses_digit_tokens_plus_eos(self):
        module = _load_module()

        case = module.build_case(600054, answer_len=6)

        self.assertEqual(case.answer, "600054")
        self.assertEqual(case.token_ids[:-1], [6, 0, 0, 0, 5, 4])
        self.assertEqual(case.token_ids[-1], module.EOS_TOKEN)

    def test_make_decision_requires_greedy_exact_not_teacher_forced_only(self):
        module = _load_module()

        decision = module.make_decision(
            {
                "teacher_forced_token_acc": 1.0,
                "greedy_exact": 0.25,
                "greedy_token_acc": 0.80,
            },
            accept_greedy_exact=0.75,
        )

        self.assertFalse(decision["accepted"])
        self.assertIn("greedy_exact_below_threshold", decision["reject_reasons"])

    def test_train_and_eval_smoke_reports_required_metrics(self):
        module = _load_module()

        report = module.run_experiment(
            train_steps=2,
            train_cases=16,
            eval_cases=8,
            answer_len=6,
            latent_dim=16,
            hidden_dim=32,
            scheduled_sampling_prob=0.5,
            batch_size=8,
            lr=1e-3,
            seed=0,
            device="cpu",
            accept_greedy_exact=0.10,
            log_every=0,
        )

        metrics = report["metrics"]
        self.assertIn("teacher_forced_token_acc", metrics)
        self.assertIn("greedy_exact", metrics)
        self.assertIn("greedy_token_acc", metrics)
        self.assertIn("greedy_examples", report)
        self.assertIn("accepted", report)


if __name__ == "__main__":
    unittest.main()
