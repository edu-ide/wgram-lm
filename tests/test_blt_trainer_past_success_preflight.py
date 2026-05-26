from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "557_train_blt_d_prefixlm_dataio.py"


def load_module():
    spec = importlib.util.spec_from_file_location("blt_d_prefixlm_trainer_preflight", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BLTTrainerPastSuccessPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_parser_exposes_past_success_preflight_flags(self) -> None:
        args = self.module.build_arg_parser().parse_args(["--sampled-data", "sampled", "--out-dir", "out"])

        self.assertEqual(args.past_success_report_json, "")
        self.assertEqual(args.past_success_restoration_gate_json, "")
        self.assertEqual(args.past_success_preflight_min_steps, 1000)
        self.assertFalse(args.allow_missing_past_success_preflight)
        self.assertFalse(args.acknowledge_past_success_restoration_gap)

    def test_long_one_body_train_args_are_guarded_by_contract(self) -> None:
        args = self.module.build_arg_parser().parse_args(
            [
                "--sampled-data",
                "sampled",
                "--out-dir",
                "out",
                "--decoder-latent-mode",
                "one_body",
                "--steps",
                "1200",
            ]
        )

        with self.assertRaisesRegex(ValueError, "past-success preflight report"):
            self.module.validate_architecture_contract(args)

    def test_short_one_body_smoke_args_are_allowed_without_report(self) -> None:
        args = self.module.build_arg_parser().parse_args(
            [
                "--sampled-data",
                "sampled",
                "--out-dir",
                "out",
                "--decoder-latent-mode",
                "one_body",
                "--steps",
                "400",
            ]
        )

        self.module.validate_architecture_contract(args)


if __name__ == "__main__":
    unittest.main()
