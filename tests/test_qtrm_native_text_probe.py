import importlib.util
import argparse
import sys
import tempfile
import unittest
from pathlib import Path

import torch


def load_module():
    path = Path("scripts/336_train_qtrm_native_text_probe.py")
    spec = importlib.util.spec_from_file_location("qtrm_native_text_probe", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class QTRMNativeTextProbeTests(unittest.TestCase):
    def test_repeat_unlikelihood_penalizes_copying_current_token(self):
        module = load_module()
        logits = torch.zeros(1, 2, 4)
        logits[0, 0, 1] = 4.0
        logits[0, 1, 2] = 4.0
        x = torch.tensor([[1, 2]])
        y = torch.tensor([[3, 2]])

        loss = module.repeat_unlikelihood_loss(logits, x, y)

        self.assertGreater(float(loss), 1.0)

    def test_weighted_next_token_ce_loss_upweights_eos_targets(self):
        module = load_module()
        logits = torch.zeros(1, 2, 3)
        y = torch.tensor([[0, 1]])

        plain = module.weighted_next_token_ce_loss(
            logits,
            y,
            eos_token_id=1,
            eos_loss_weight=1.0,
        )
        weighted = module.weighted_next_token_ce_loss(
            logits,
            y,
            eos_token_id=1,
            eos_loss_weight=4.0,
        )

        self.assertTrue(torch.isfinite(weighted))
        self.assertTrue(torch.allclose(plain, weighted))

    def test_weighted_next_token_ce_loss_changes_when_eos_error_differs(self):
        module = load_module()
        logits = torch.zeros(1, 2, 3)
        logits[0, 0, 0] = 4.0
        logits[0, 1, 0] = 4.0
        y = torch.tensor([[0, 1]])

        plain = module.weighted_next_token_ce_loss(
            logits,
            y,
            eos_token_id=1,
            eos_loss_weight=1.0,
        )
        weighted = module.weighted_next_token_ce_loss(
            logits,
            y,
            eos_token_id=1,
            eos_loss_weight=4.0,
        )

        self.assertGreater(float(weighted), float(plain))

    def test_char_tokenizer_round_trips_text(self):
        module = load_module()
        tokenizer = module.CharTokenizer.from_text("abc 가나다 abc")

        encoded = tokenizer.encode("가나다 abc")

        self.assertEqual(tokenizer.decode(encoded), "가나다 abc")

    def test_windows_have_input_and_target_shifted_by_one(self):
        module = load_module()

        windows = module.make_windows([1, 2, 3, 4, 5], seq_len=3)

        self.assertEqual(windows, [([1, 2, 3], [2, 3, 4]), ([2, 3, 4], [3, 4, 5])])

    def test_load_text_concatenates_file_and_globs(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.md"
            second = root / "second.md"
            first.write_text("alpha text", encoding="utf-8")
            second.write_text("beta text", encoding="utf-8")
            args = argparse.Namespace(
                text_file=str(first),
                text_glob=[str(root / "*.md")],
                max_text_chars=0,
            )

            text = module.load_text(args)

        self.assertIn("alpha text", text)
        self.assertIn("beta text", text)
        self.assertIn("## FILE:", text)

    def test_degeneracy_metrics_detect_repeated_sample(self):
        module = load_module()

        metrics = module.sample_degeneracy("aaaaaaaaab")

        self.assertLess(metrics["unique_chars"], 3)
        self.assertGreater(metrics["max_run_fraction"], 0.5)

    def test_generate_text_stops_on_tokenizer_eos(self):
        module = load_module()

        class ToyTokenizer:
            eos_token_id = 1

            def encode(self, text):
                return [0]

            def decode(self, token_ids):
                return "".join(str(token_id) for token_id in token_ids)

        class EosModel:
            def eval(self):
                return None

            def __call__(self, x, *, think_steps):
                logits = torch.zeros((1, x.shape[1], 3), device=x.device)
                logits[:, -1, 1] = 1.0
                return logits

        text = module.generate_text(
            EosModel(),
            ToyTokenizer(),
            seed_text="x",
            seq_len=4,
            think_steps=1,
            max_new_chars=10,
            device=torch.device("cpu"),
        )

        self.assertEqual(text, "01")

    def test_char_tokenizer_uses_etx_as_eos(self):
        module = load_module()
        tokenizer = module.CharTokenizer.from_text("ab\x03")

        self.assertEqual(tokenizer.eos_token_id, tokenizer.char_to_id["\x03"])
        self.assertEqual(tokenizer.decode([tokenizer.char_to_id["a"], tokenizer.char_to_id["\x03"]]), "a")

    def test_parser_accepts_l5_language_nonregression_thresholds(self):
        module = load_module()

        args = module.build_arg_parser().parse_args(
            [
                "--target-level",
                "L5C QTRM-native language non-regression",
                "--accepted-decision",
                "accepted_l5_language_nonregression",
                "--max-full-vs-think0-loss-ratio",
                "1.05",
                "--max-full-vs-off-loss-ratio",
                "1.10",
                "--baseline-steps",
                "3",
                "--max-full-vs-baseline-loss-ratio",
                "1.20",
                "--eos-loss-weight",
                "3.5",
            ]
        )

        self.assertEqual(args.target_level, "L5C QTRM-native language non-regression")
        self.assertEqual(args.accepted_decision, "accepted_l5_language_nonregression")
        self.assertEqual(args.max_full_vs_think0_loss_ratio, 1.05)
        self.assertEqual(args.max_full_vs_off_loss_ratio, 1.10)
        self.assertEqual(args.baseline_steps, 3)
        self.assertEqual(args.max_full_vs_baseline_loss_ratio, 1.20)
        self.assertEqual(args.eos_loss_weight, 3.5)

    def test_parser_accepts_mamba3_thinking_core_placement(self):
        module = load_module()

        args = module.build_arg_parser().parse_args(
            [
                "--backbone",
                "mamba3",
                "--encode-backbone",
                "mha_etd",
                "--think-backbone",
                "mamba3",
                "--decode-backbone",
                "mha_etd",
                "--strict-backends",
            ]
        )

        self.assertEqual(args.backbone, "mamba3")
        self.assertEqual(args.encode_backbone, "mha_etd")
        self.assertEqual(args.think_backbone, "mamba3")
        self.assertEqual(args.decode_backbone, "mha_etd")
        self.assertTrue(args.strict_backends)

    def test_parser_accepts_trm_dual_z_thinking_structure(self):
        module = load_module()

        args = module.build_arg_parser().parse_args(["--think-structure", "trm_dual_z"])

        self.assertEqual(args.think_structure, "trm_dual_z")

    def test_parser_accepts_official_trm_style_thinking_core(self):
        module = load_module()

        args = module.build_arg_parser().parse_args(
            [
                "--backbone",
                "trm_official",
                "--encode-backbone",
                "mha_etd",
                "--think-backbone",
                "trm_official",
                "--decode-backbone",
                "mha_etd",
            ]
        )

        self.assertEqual(args.backbone, "trm_official")
        self.assertEqual(args.think_backbone, "trm_official")

    def test_parser_accepts_tied_embeddings(self):
        module = load_module()

        args = module.build_arg_parser().parse_args(["--tie-embeddings"])

        self.assertTrue(args.tie_embeddings)

    def test_tie_embeddings_shares_lm_head_weight(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--seq-len",
                "8",
                "--d-model",
                "16",
                "--n-heads",
                "4",
                "--n-kv-heads",
                "2",
                "--d-ff",
                "32",
                "--tie-embeddings",
            ]
        )

        model = module.build_model(args, vocab_size=32)

        self.assertEqual(
            model.lm_head.weight.data_ptr(),
            model.token_embed.weight.data_ptr(),
        )

    def test_language_reject_reasons_flag_recurrent_loss_regression(self):
        module = load_module()
        args = argparse.Namespace(
            max_random_loss_fraction=0.70,
            min_unique_chars=8.0,
            max_run_fraction=0.25,
            max_full_vs_think0_loss_ratio=1.05,
            max_full_vs_off_loss_ratio=1.10,
            max_full_vs_baseline_loss_ratio=1.20,
        )

        reasons = module.language_reject_reasons(
            args,
            full_loss=1.30,
            think0_loss=1.00,
            off_loss=1.00,
            baseline_loss=1.00,
            random_loss=3.0,
            degeneracy={"unique_chars": 16.0, "max_run_fraction": 0.05},
        )

        self.assertIn("full_loss_regressed_vs_think0", reasons)
        self.assertIn("full_loss_regressed_vs_thinking_block_off", reasons)
        self.assertIn("full_loss_regressed_vs_baseline", reasons)


if __name__ == "__main__":
    unittest.main()
