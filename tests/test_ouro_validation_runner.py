import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "309_run_validation_gated_ouro_recurrent.py"
    )
    spec = importlib.util.spec_from_file_location("ouro_validation_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class OuroValidationRunnerTests(unittest.TestCase):
    def test_train_command_records_seed_save_every_and_trainable_delta(self):
        module = _load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--out-dir",
                "local_eval/example",
                "--seed",
                "17",
                "--save-every",
                "5",
                "--steps",
                "10",
            ]
        )

        command = module.build_train_command(args)

        self.assertIn("--seed", command)
        self.assertIn("17", command)
        self.assertIn("--save-every", command)
        self.assertIn("5", command)
        self.assertIn("--save-trainable-only", command)

    def test_train_command_passes_final_answer_binding_losses(self):
        module = _load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--out-dir",
                "local_eval/example",
                "--causal-prefix-supervision",
                "--target-mode",
                "final",
                "--choice-margin-weight",
                "0.4",
                "--choice-margin-mode",
                "sequence",
                "--tail-negative-margin-weight",
                "0.3",
                "--subtract-tail-counterfactual-margin-weight",
                "0.2",
                "--terminal-depth-ce-weight",
                "1.5",
                "--answer-state-loop-halt-ce-weight",
                "0.7",
            ]
        )

        command = module.build_train_command(args)

        for flag, value in {
            "--target-mode": "final",
            "--choice-margin-weight": "0.4",
            "--choice-margin-mode": "sequence",
            "--tail-negative-margin-weight": "0.3",
            "--subtract-tail-counterfactual-margin-weight": "0.2",
            "--terminal-depth-ce-weight": "1.5",
            "--answer-state-loop-halt-ce-weight": "0.7",
        }.items():
            self.assertIn(flag, command)
            self.assertEqual(command[command.index(flag) + 1], value)

    def test_summarize_eval_requires_full_to_beat_all_baselines(self):
        module = _load_module()
        rows = [
            {"mode": "donor_only_no_evidence", "hit": False},
            {"mode": "qtrm_core_off_no_evidence", "hit": False},
            {
                "mode": "qtrm_core_steps_8_answer_state_recurrent_off_no_evidence",
                "hit": False,
            },
            {"mode": "qtrm_core_steps_8_no_evidence", "hit": True},
            {"mode": "qtrm_core_steps_8_no_evidence", "hit": True},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval.jsonl"
            path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            summary = module.summarize_eval_jsonl(path)

        self.assertTrue(summary["full_beats_all_baselines"])
        self.assertEqual(summary["full_hits"], 2)
        self.assertEqual(summary["full_margin_over_best_baseline"], 2)

    def test_dry_run_preserves_manifest_and_config_snapshot(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config.yaml"
            config.write_text("model: {}\n", encoding="utf-8")
            init = tmp_path / "init.pt"
            init.write_bytes(b"checkpoint")
            out_dir = tmp_path / "out"
            args = module.build_arg_parser().parse_args(
                [
                    "--config",
                    str(config),
                    "--init-checkpoint",
                    str(init),
                    "--train-data",
                    str(tmp_path / "train.jsonl"),
                    "--eval-cases",
                    str(tmp_path / "eval.jsonl"),
                    "--out-dir",
                    str(out_dir),
                    "--dry-run",
                    "--seed",
                    "3",
                ]
            )

            report = module.run_validation_gate(args)
            loaded = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
            snapshot_exists = (out_dir / "config_snapshot.yaml").exists()

        self.assertEqual(report["decision"], "dry_run")
        self.assertEqual(loaded["seed"], 3)
        self.assertTrue(snapshot_exists)
        self.assertIn("--seed", loaded["train_command"])


if __name__ == "__main__":
    unittest.main()
