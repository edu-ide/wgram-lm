import importlib.util
import json
import sys
import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory


def load_module():
    path = Path("scripts/356_eval_qtrm_native_language_generalization.py")
    spec = importlib.util.spec_from_file_location("qtrm_native_language_generalization", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class QTRMNativeLanguageGeneralizationEvalTests(unittest.TestCase):
    def test_default_expectations_are_valid_json(self):
        module = load_module()

        expectations = json.loads(module.build_arg_parser().parse_args([]).eval_seed_expectations)

        self.assertIn("Why is checking a source important?", expectations)
        self.assertIn("How do short sentences help readers?", expectations)

    def test_checkpoint_args_merge_eval_overrides(self):
        module = load_module()
        overrides = Namespace(
            device="cpu",
            out_dir="tmp_eval",
            eval_think_steps=2,
            max_new_chars=33,
            repair_prompt_count=5,
            eval_seed_texts="User: Q\nAssistant:",
            eval_seed_expectations='{"Q": ["answer"]}',
            min_on_policy_continuation_chars=7,
            min_on_policy_keyword_hits=1,
            min_on_policy_loop_check_lines=3,
            min_on_policy_unique_line_fraction=0.5,
            max_on_policy_repeated_block_fraction=0.2,
            max_on_policy_repeated_line_fraction=0.3,
            eval_jsonl="",
        )

        args = module.merged_checkpoint_args(
            {"d_model": 128, "tokenizer_name": "Qwen/Qwen3.5-2B-Base"},
            overrides,
        )

        self.assertEqual(args.d_model, 128)
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.eval_think_steps, 2)
        self.assertEqual(args.max_new_chars, 33)
        self.assertEqual(args.repair_seed_texts, "User: Q\nAssistant:")
        self.assertEqual(args.min_on_policy_keyword_hits, 1)

    def test_eval_jsonl_overrides_seed_texts_and_expectations(self):
        module = load_module()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "suite.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "prompt": "Why does a date matter?",
                        "expected_keywords": ["date", "current"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            overrides = Namespace(
                device="cpu",
                out_dir="tmp_eval",
                eval_think_steps=2,
                max_new_chars=33,
                repair_prompt_count=5,
                eval_seed_texts="ignored",
                eval_seed_expectations='{"ignored": ["x"]}',
                eval_jsonl=str(path),
                min_on_policy_continuation_chars=7,
                min_on_policy_keyword_hits=1,
                min_on_policy_loop_check_lines=3,
                min_on_policy_unique_line_fraction=0.5,
                max_on_policy_repeated_block_fraction=0.2,
                max_on_policy_repeated_line_fraction=0.3,
            )

            args = module.merged_checkpoint_args({}, overrides)

        self.assertIn("Why does a date matter?", args.repair_seed_texts)
        expectations = json.loads(args.repair_seed_expectations)
        self.assertEqual(expectations["Why does a date matter?"], ["date", "current"])

    def test_eval_jsonl_accepts_keyword_groups(self):
        module = load_module()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "suite.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "prompt": "Why does a source date matter?",
                        "expected_keyword_groups": [
                            ["date", "time"],
                            ["source", "document"],
                            ["current", "reliability"],
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            seeds, raw_expectations = module.load_eval_suite_jsonl(path)

        self.assertIn("Why does a source date matter?", seeds)
        expectations = json.loads(raw_expectations)
        self.assertEqual(
            expectations["Why does a source date matter?"],
            ["date|time", "source|document", "current|reliability"],
        )

        metrics = module._bootstrap.semantic_relevance_metrics(
            "User: Why does a source date matter?\nAssistant:",
            "User: Why does a source date matter?\nAssistant: Time affects reliability.",
            expectations,
        )
        self.assertEqual(metrics["matched_count"], 2.0)
        self.assertEqual(metrics["matched_keywords"], ["date|time", "current|reliability"])

    def test_load_eval_suite_jsonl_rejects_missing_keywords(self):
        module = load_module()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.jsonl"
            path.write_text(
                json.dumps({"prompt": "Q", "expected_keywords": []}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                module.load_eval_suite_jsonl(path)

    def test_tokenizer_from_checkpoint_rebuilds_byte_bpe(self):
        module = load_module()
        tokenizer = module._bootstrap.ByteBPETokenizerAdapter.from_text(
            "User: q\nAssistant: a<|qtrm_eos|>",
            vocab_size=512,
            min_frequency=1,
            eos_token="<|qtrm_eos|>",
            unk_token="<|qtrm_unk|>",
        )
        payload = {
            "kind": "byte_bpe",
            "name": tokenizer.name,
            "vocab_size": tokenizer.vocab_size,
            "eos_token": tokenizer.eos_token,
            "unk_token": tokenizer.unk_token,
            "eos_id": tokenizer.eos_id,
            "tokenizer_json": tokenizer.tokenizer.to_str(),
        }

        rebuilt = module.tokenizer_from_checkpoint(
            payload,
            module._bootstrap.build_arg_parser().parse_args([]),
        )

        self.assertEqual(rebuilt.eos_token_id, tokenizer.eos_token_id)
        self.assertIn("Assistant", rebuilt.decode(rebuilt.encode("Assistant")))


if __name__ == "__main__":
    unittest.main()
