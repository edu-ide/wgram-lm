import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def load_module():
    path = Path("scripts/354_train_qtrm_native_language_bootstrap.py")
    spec = importlib.util.spec_from_file_location("qtrm_native_language_bootstrap", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class QTRMNativeLanguageBootstrapTests(unittest.TestCase):
    def test_build_stage_texts_includes_teacher_jsonl(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "teacher.jsonl"
            path.write_text(
                json.dumps({"text": "teacher continuation text"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            args = module.build_arg_parser().parse_args(
                [
                    "--tiny-repeats",
                    "1",
                    "--textbook-repeats",
                    "1",
                "--teacher-jsonl",
                str(path),
                "--repair-jsonl",
                str(path),
                "--repair-jsonl-repeats",
                "2",
                "--max-text-chars",
                "0",
            ]
            )

            stages = module.build_stage_texts(args)

        self.assertIn("teacher continuation text", stages["teacher"])
        self.assertGreaterEqual(stages["teacher"].count("teacher continuation text"), 3)
        self.assertNotIn("teacher continuation text", stages["tiny"])
        self.assertIn("A clear sentence", stages["edu"])
        self.assertIn("Assistant:", stages["edu"])
        self.assertIn("Why does ice melt?", stages["edu"])

    def test_teacher_jsonl_visible_cot_is_removed_from_language_stage(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "teacher.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "text": (
                            "clean prefix\n<think>hidden chain of thought"
                            "</think>\nclean continuation"
                        )
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            args = module.build_arg_parser().parse_args(
                [
                    "--tiny-repeats",
                    "1",
                    "--textbook-repeats",
                    "1",
                "--teacher-jsonl",
                str(path),
                "--repair-jsonl",
                str(path),
                "--repair-jsonl-repeats",
                "3",
                "--max-text-chars",
                "0",
            ]
            )

            stages = module.build_stage_texts(args)

        self.assertIn("clean prefix", stages["teacher"])
        self.assertIn("clean continuation", stages["teacher"])
        self.assertGreaterEqual(stages["teacher"].count("clean continuation"), 4)
        self.assertNotIn("<think>", stages["teacher"])
        self.assertNotIn("hidden chain of thought", stages["teacher"])

    def test_char_tokenizer_path_builds_windows(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--tiny-repeats",
                "2",
                "--textbook-repeats",
                "1",
                "--seq-len",
                "16",
            ]
        )
        stages = module.build_stage_texts(args)
        tokenizer = module.build_tokenizer(args, all_text="\n".join(stages.values()))

        windows = module.windows_for_text(tokenizer, stages["tiny"], seq_len=16)

        self.assertGreater(len(windows), 0)
        self.assertGreater(tokenizer.vocab_size, 10)

    def test_parser_accepts_language_bootstrap_args(self):
        module = load_module()

        args = module.build_arg_parser().parse_args(
            [
                "--tokenizer-name",
                "Qwen/Qwen3.5-2B-Base",
                "--compact-hf-vocab",
                "--compact-vocab-max-size",
                "1024",
                "--compact-vocab-min-count",
                "2",
                "--train-byte-bpe-tokenizer",
                "--byte-bpe-vocab-size",
                "2048",
                "--byte-bpe-min-frequency",
                "2",
                "--byte-bpe-eos-token",
                "<eos>",
                "--byte-bpe-unk-token",
                "<unk>",
                "--stage-a-steps",
                "1",
                "--stage-b-steps",
                "2",
                "--stage-c-steps",
                "3",
                "--tst-phase-steps",
                "5",
                "--tst-bag-size",
                "2",
                "--repeat-unlikelihood-weight",
                "0.2",
                "--surface-answer-repeats",
                "4",
                "--diverse-surface-answer-count",
                "8",
                "--record-separator",
                "<eos>",
                "--init-checkpoint",
                "previous.pt",
                "--pretrained-init-model",
                "Qwen/Qwen3.5-2B-Base",
                "--pretrained-init-projection",
                "slice",
                "--pretrained-init-seed",
                "123",
                "--pretrained-init-scale",
                "0.5",
                "--pretrained-init-dtype",
                "bfloat16",
                "--pretrained-init-device-map",
                "cpu",
                "--pretrained-init-allow-full-vocab",
                "--repair-jsonl",
                "repair.jsonl",
                "--repair-jsonl-repeats",
                "8",
                "--gate-anchor-repeats",
                "5",
                "--repair-prompt-count",
                "2",
                "--min-on-policy-unique-line-fraction",
                "0.75",
                "--min-on-policy-continuation-chars",
                "12",
                "--min-on-policy-keyword-hits",
                "3",
                "--min-on-policy-informative-char-fraction",
                "0.5",
                "--max-on-policy-repeated-word-fraction",
                "0.4",
                "--no-shuffle-corpus",
            ]
        )

        self.assertEqual(args.tokenizer_name, "Qwen/Qwen3.5-2B-Base")
        self.assertEqual(args.max_text_chars, 120000)
        self.assertTrue(args.compact_hf_vocab)
        self.assertEqual(args.compact_vocab_max_size, 1024)
        self.assertEqual(args.compact_vocab_min_count, 2)
        self.assertTrue(args.train_byte_bpe_tokenizer)
        self.assertEqual(args.byte_bpe_vocab_size, 2048)
        self.assertEqual(args.byte_bpe_min_frequency, 2)
        self.assertEqual(args.byte_bpe_eos_token, "<eos>")
        self.assertEqual(args.byte_bpe_unk_token, "<unk>")
        self.assertEqual(args.stage_a_steps, 1)
        self.assertEqual(args.stage_b_steps, 2)
        self.assertEqual(args.stage_c_steps, 3)
        self.assertEqual(args.tst_phase_steps, 5)
        self.assertEqual(args.tst_bag_size, 2)
        self.assertEqual(args.repeat_unlikelihood_weight, 0.2)
        self.assertEqual(args.surface_answer_repeats, 4)
        self.assertEqual(args.diverse_surface_answer_count, 8)
        self.assertEqual(args.record_separator, "<eos>")
        self.assertEqual(args.init_checkpoint, "previous.pt")
        self.assertEqual(args.pretrained_init_model, "Qwen/Qwen3.5-2B-Base")
        self.assertEqual(args.pretrained_init_projection, "slice")
        self.assertEqual(args.pretrained_init_seed, 123)
        self.assertEqual(args.pretrained_init_scale, 0.5)
        self.assertEqual(args.pretrained_init_dtype, "bfloat16")
        self.assertEqual(args.pretrained_init_device_map, "cpu")
        self.assertTrue(args.pretrained_init_allow_full_vocab)
        self.assertEqual(args.repair_jsonl, ["repair.jsonl"])
        self.assertEqual(args.repair_jsonl_repeats, 8)
        self.assertEqual(args.gate_anchor_repeats, 5)
        self.assertEqual(args.repair_prompt_count, 2)
        self.assertEqual(args.min_on_policy_unique_line_fraction, 0.75)
        self.assertEqual(args.min_on_policy_continuation_chars, 12)
        self.assertEqual(args.min_on_policy_keyword_hits, 3)
        self.assertEqual(args.min_on_policy_informative_char_fraction, 0.5)
        self.assertEqual(args.max_on_policy_repeated_word_fraction, 0.4)
        self.assertFalse(args.shuffle_corpus)

    def test_answer_surface_metrics_flags_punctuation_only_continuation(self):
        module = load_module()

        metrics = module.answer_surface_metrics(
            "User: Why?\nAssistant:",
            "User: Why?\nAssistant:,,,,,,,,,,,,,,,,",
        )

        self.assertEqual(metrics["continuation_chars"], 16.0)
        self.assertEqual(metrics["informative_char_fraction"], 0.0)
        self.assertEqual(metrics["word_count"], 0.0)

    def test_answer_surface_metrics_flags_repeated_word_loop(self):
        module = load_module()

        metrics = module.answer_surface_metrics(
            "User: Why?\nAssistant:",
            "User: Why?\nAssistant: the the the reason the the",
        )

        self.assertEqual(metrics["word_count"], 6.0)
        self.assertGreater(metrics["max_word_repeat_fraction"], 0.8)

    def test_qwen_tokenizer_path_uses_auto_eos_record_separator(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--tokenizer-name",
                "Qwen/Qwen3.5-2B-Base",
                "--tiny-repeats",
                "1",
                "--textbook-repeats",
                "1",
                "--surface-answer-repeats",
                "1",
                "--diverse-surface-answer-count",
                "1",
            ]
        )

        stages = module.build_stage_texts(args)

        self.assertIn("<|endoftext|>", stages["tiny"])
        self.assertIn("<|endoftext|>", stages["edu"])

    def test_compact_hf_tokenizer_maps_known_and_unknown_ids(self):
        module = load_module()

        class FakeTokenizer:
            eos_token_id = 2
            pad_token_id = 2
            unk_token_id = 99

            def encode(self, text, add_special_tokens=False):
                return [10, 11, 99] if "known" in text else [42]

            def decode(self, token_ids, skip_special_tokens=True):
                return "|".join(str(token_id) for token_id in token_ids)

        tokenizer = module.CompactHFTokenizerAdapter(
            name="fake",
            tokenizer=FakeTokenizer(),
            compact_to_hf=(2, 10, 11, 99),
            hf_to_compact={2: 0, 10: 1, 11: 2, 99: 3},
            unk_compact_id=3,
            eos_compact_id=0,
        )

        self.assertEqual(tokenizer.vocab_size, 4)
        self.assertEqual(tokenizer.eos_token_id, 0)
        self.assertEqual(tokenizer.encode("known text"), [1, 2, 3])
        self.assertEqual(tokenizer.encode("other text"), [3])
        self.assertEqual(tokenizer.decode([1, 2, 3]), "10|11|99")

    def test_compact_hf_tokenizer_without_unk_does_not_use_eos_as_fallback(self):
        module = load_module()

        class FakeBase:
            name = "fake"

            class tokenizer:
                eos_token_id = 2
                pad_token_id = 2
                unk_token_id = None

                @staticmethod
                def encode(text, add_special_tokens=False):
                    if "known" in text:
                        return [10, 11]
                    if "extra" in text:
                        return [11]
                    return [42]

                @staticmethod
                def decode(token_ids, skip_special_tokens=True):
                    return ",".join(str(token_id) for token_id in token_ids)

            def encode(self, text):
                return self.tokenizer.encode(text, add_special_tokens=False)

            def decode(self, token_ids):
                return self.tokenizer.decode(token_ids, skip_special_tokens=True)

        original = module.HFTokenizerAdapter.from_name
        module.HFTokenizerAdapter.from_name = staticmethod(lambda _name: FakeBase())
        try:
            tokenizer = module.CompactHFTokenizerAdapter.from_text(
                "fake",
                "known text",
                max_size=0,
                min_count=1,
                extra_texts=("extra text",),
            )
        finally:
            module.HFTokenizerAdapter.from_name = original

        self.assertEqual(tokenizer.eos_token_id, 0)
        self.assertNotEqual(tokenizer.unk_compact_id, tokenizer.eos_token_id)
        self.assertEqual(tokenizer.encode("mystery text"), [tokenizer.unk_compact_id])

    def test_compact_hf_tokenizer_from_payload_rebuilds_mapping(self):
        module = load_module()

        class FakeBase:
            name = "fake"

            class tokenizer:
                eos_token_id = 2
                pad_token_id = 2
                unk_token_id = 99

                @staticmethod
                def encode(text, add_special_tokens=False):
                    return [10, 99]

                @staticmethod
                def decode(token_ids, skip_special_tokens=True):
                    return ",".join(str(token_id) for token_id in token_ids)

        original = module.HFTokenizerAdapter.from_name
        module.HFTokenizerAdapter.from_name = staticmethod(lambda _name: FakeBase())
        try:
            tokenizer = module.CompactHFTokenizerAdapter.from_payload(
                {
                    "name": "fake",
                    "compact_to_hf": [2, 10, 99],
                    "unk_compact_id": 2,
                    "eos_compact_id": 0,
                }
            )
        finally:
            module.HFTokenizerAdapter.from_name = original

        self.assertEqual(tokenizer.encode("anything"), [1, 2])
        self.assertEqual(tokenizer.decode([0, 1]), "2,10")

    def test_compact_tokenizer_report_payload_omits_full_mapping(self):
        module = load_module()

        payload = module.tokenizer_report_payload(
            {
                "kind": "hf_compact",
                "name": "fake",
                "compact_to_hf": [2, 10, 99],
                "unk_compact_id": 2,
                "eos_compact_id": 0,
            }
        )

        self.assertEqual(payload["compact_vocab_size"], 3)
        self.assertNotIn("compact_to_hf", payload)

    def test_byte_bpe_tokenizer_roundtrips_and_exposes_eos(self):
        module = load_module()

        tokenizer = module.ByteBPETokenizerAdapter.from_text(
            "User: test\nAssistant: answer\n<|qtrm_eos|>\n",
            vocab_size=512,
            min_frequency=1,
            eos_token="<|qtrm_eos|>",
            unk_token="<|qtrm_unk|>",
            extra_texts=("User: unseen\nAssistant:",),
        )

        encoded = tokenizer.encode("User: test<|qtrm_eos|>")

        self.assertGreaterEqual(tokenizer.vocab_size, 256)
        self.assertIsNotNone(tokenizer.eos_token_id)
        self.assertIn(tokenizer.eos_token_id, encoded)
        self.assertIn("User: test", tokenizer.decode(encoded))
        self.assertNotIn("<|qtrm_eos|>", tokenizer.decode(encoded))

    def test_byte_bpe_tokenizer_payload_rebuilds(self):
        module = load_module()
        tokenizer = module.ByteBPETokenizerAdapter.from_text(
            "alpha beta gamma<|qtrm_eos|>",
            vocab_size=512,
            min_frequency=1,
            eos_token="<|qtrm_eos|>",
            unk_token="<|qtrm_unk|>",
        )
        rebuilt = module.ByteBPETokenizerAdapter.from_payload(
            {
                "kind": "byte_bpe",
                "name": tokenizer.name,
                "vocab_size": tokenizer.vocab_size,
                "eos_token": tokenizer.eos_token,
                "unk_token": tokenizer.unk_token,
                "eos_id": tokenizer.eos_id,
                "tokenizer_json": tokenizer.tokenizer.to_str(),
            }
        )

        self.assertEqual(rebuilt.eos_token_id, tokenizer.eos_token_id)
        self.assertEqual(rebuilt.decode(rebuilt.encode("alpha beta")), "alpha beta")

    def test_init_checkpoint_reuses_model_args_and_tokenizer(self):
        module = load_module()
        tokenizer = module.ByteBPETokenizerAdapter.from_text(
            "alpha beta gamma<|qtrm_eos|>",
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
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            module.torch.save(
                {
                    "args": {
                        "seq_len": 32,
                        "d_model": 48,
                        "n_heads": 4,
                        "n_kv_heads": 2,
                        "d_ff": 96,
                    },
                    "tokenizer": payload,
                    "model_state": {},
                },
                path,
            )
            args = module.build_arg_parser().parse_args(["--init-checkpoint", str(path)])
            checkpoint = module.load_init_checkpoint(args)
            rebuilt = module.tokenizer_from_payload(checkpoint["tokenizer"], args)

        self.assertEqual(args.seq_len, 32)
        self.assertEqual(args.d_model, 48)
        self.assertTrue(args.train_byte_bpe_tokenizer)
        self.assertEqual(rebuilt.decode(rebuilt.encode("alpha beta")), "alpha beta")

    def test_project_pretrained_rows_adapts_hidden_size(self):
        module = load_module()
        rows = module.torch.arange(24, dtype=module.torch.float32).view(3, 8)

        sliced = module.project_pretrained_rows(
            rows,
            target_dim=4,
            seed=1,
            mode="slice",
        )
        projected_a = module.project_pretrained_rows(
            rows,
            target_dim=4,
            seed=1,
            mode="random_projection",
        )
        projected_b = module.project_pretrained_rows(
            rows,
            target_dim=4,
            seed=1,
            mode="random_projection",
        )
        padded = module.project_pretrained_rows(
            rows[:, :2],
            target_dim=4,
            seed=1,
            mode="random_projection",
        )

        self.assertEqual(tuple(sliced.shape), (3, 4))
        self.assertEqual(sliced.tolist(), rows[:, :4].tolist())
        self.assertTrue(module.torch.allclose(projected_a, projected_b))
        self.assertEqual(tuple(padded.shape), (3, 4))
        self.assertTrue(module.torch.allclose(padded[:, :2], rows[:, :2]))
        self.assertTrue(module.torch.allclose(padded[:, 2:], module.torch.zeros(3, 2)))

    def test_compact_or_full_hf_token_ids_prefers_compact_mapping(self):
        module = load_module()

        class FakeTokenizer:
            eos_token_id = 2
            pad_token_id = 2
            unk_token_id = 99

            @staticmethod
            def encode(text, add_special_tokens=False):
                return [10, 11]

            @staticmethod
            def decode(token_ids, skip_special_tokens=True):
                return ",".join(str(token_id) for token_id in token_ids)

        tokenizer = module.CompactHFTokenizerAdapter(
            name="fake",
            tokenizer=FakeTokenizer(),
            compact_to_hf=(2, 10, 11),
            hf_to_compact={2: 0, 10: 1, 11: 2},
            unk_compact_id=0,
            eos_compact_id=0,
        )

        self.assertEqual(module.compact_or_full_hf_token_ids(tokenizer), [2, 10, 11])

    def test_byte_bpe_report_payload_omits_tokenizer_json(self):
        module = load_module()

        payload = module.tokenizer_report_payload(
            {
                "kind": "byte_bpe",
                "name": "byte_bpe_512",
                "vocab_size": 512,
                "eos_token": "<|qtrm_eos|>",
                "unk_token": "<|qtrm_unk|>",
                "eos_id": 1,
                "tokenizer_json": "{}",
            }
        )

        self.assertEqual(payload["kind"], "byte_bpe")
        self.assertEqual(payload["vocab_size"], 512)
        self.assertNotIn("tokenizer_json", payload)

    def test_byte_bpe_auto_record_separator_uses_project_eos(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--train-byte-bpe-tokenizer",
                "--byte-bpe-eos-token",
                "<eos>",
            ]
        )

        self.assertEqual(module.language_record_separator(args), "\n<eos>\n")

    def test_teacher_jsonl_records_use_auto_eos_record_separator(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "teacher.jsonl"
            path.write_text(
                json.dumps({"text": "first teacher answer"}, ensure_ascii=False)
                + "\n"
                + json.dumps({"text": "second teacher answer"}, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )
            args = module.build_arg_parser().parse_args(
                [
                    "--tokenizer-name",
                    "Qwen/Qwen3.5-2B-Base",
                    "--tiny-repeats",
                    "1",
                    "--textbook-repeats",
                    "1",
                    "--surface-answer-repeats",
                    "0",
                    "--diverse-surface-answer-count",
                    "0",
                    "--teacher-jsonl",
                    str(path),
                    "--max-text-chars",
                    "0",
                ]
            )

            stages = module.build_stage_texts(args)

        self.assertIn("first teacher answer\n<|endoftext|>\nsecond teacher answer", stages["teacher"])

    def test_gate_anchor_repeats_add_exact_repair_prompts_to_stage_c(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--train-byte-bpe-tokenizer",
                "--gate-anchor-repeats",
                "2",
                "--tiny-repeats",
                "0",
                "--textbook-repeats",
                "0",
                "--surface-answer-repeats",
                "0",
                "--diverse-surface-answer-count",
                "0",
                "--max-text-chars",
                "0",
            ]
        )

        stages = module.build_stage_texts(args)

        self.assertIn("## GATE_ANCHORS", stages["teacher"])
        self.assertEqual(stages["teacher"].count("Why should evidence be checked?"), 2)
        self.assertIn("unsupported claims", stages["teacher"])
        self.assertIn("What makes a repeated test useful?", stages["teacher"])
        self.assertIn("반복 실험은 왜 결과 판단에 도움이 되나요?", stages["teacher"])
        self.assertIn("주장을 믿기 전에 왜 근거를 확인해야 하나요?", stages["teacher"])
        self.assertIn("How can a summary stay faithful to the original text?", stages["teacher"])
        self.assertIn("How can an answer separate a claim from evidence?", stages["teacher"])
        self.assertIn("<|qtrm_eos|>", stages["teacher"])

    def test_max_text_chars_preserves_teacher_jsonl_slice_in_stage_c(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "teacher.jsonl"
            path.write_text(
                json.dumps(
                    {"text": "teacher payload must survive the cap"},
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            args = module.build_arg_parser().parse_args(
                [
                    "--tokenizer-name",
                    "Qwen/Qwen3.5-2B-Base",
                    "--tiny-repeats",
                    "64",
                    "--textbook-repeats",
                    "64",
                    "--surface-answer-repeats",
                    "64",
                    "--teacher-jsonl",
                    str(path),
                    "--max-text-chars",
                    "400",
                ]
            )

            stages = module.build_stage_texts(args)

        self.assertLessEqual(len(stages["teacher"]), 420)
        self.assertIn("teacher payload must survive", stages["teacher"])

    def test_max_text_chars_preserves_surface_answer_slice_in_stage_b(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--tokenizer-name",
                "Qwen/Qwen3.5-2B-Base",
                "--tiny-repeats",
                "64",
                "--textbook-repeats",
                "64",
                "--surface-answer-repeats",
                "64",
                "--diverse-surface-answer-count",
                "12",
                "--max-text-chars",
                "800",
            ]
        )

        stages = module.build_stage_texts(args)

        self.assertLessEqual(len(stages["edu"]), 840)
        self.assertIn("Assistant:", stages["edu"])

    def test_diverse_surface_answer_snippets_are_answer_only(self):
        module = load_module()

        snippets = module.build_diverse_surface_answer_snippets(20)

        self.assertEqual(len(snippets), 20)
        self.assertTrue(all("User:" in row and "Assistant:" in row for row in snippets))
        self.assertFalse(any("<think" in row.lower() for row in snippets))
        self.assertGreater(len(set(snippets)), 10)

    def test_surface_answer_families_are_round_robin_balanced(self):
        module = load_module()

        families = module.build_surface_answer_families()
        snippets = module.build_diverse_surface_answer_snippets(len(families) * 2)

        self.assertIn("writing", families)
        self.assertIn("uncertainty", families)
        self.assertTrue(
            any("Why can old information become wrong?" in row for row in families["evidence"])
        )
        self.assertTrue(
            any("반복 실험은 왜 결과 판단에 도움이 되나요?" in row for row in families["basics"])
        )
        self.assertTrue(
            any("주장을 믿기 전에 왜 근거를 확인해야 하나요?" in row for row in families["evidence"])
        )
        self.assertTrue(
            any("How can an answer separate a claim from evidence?" in row for row in families["evidence"])
        )
        self.assertTrue(
            any("How can a summary stay faithful to the original text?" in row for row in families["planning_summary"])
        )
        self.assertTrue(
            any("문장을 고칠 때 무엇을 먼저 확인해야 하나요?" in row for row in families["writing"])
        )
        self.assertEqual(len(snippets), len(families) * 2)
        self.assertTrue(any("How can writing become clearer?" in row for row in snippets))
        self.assertTrue(any("How can a model avoid guessing?" in row for row in snippets))

    def test_on_policy_loop_metrics_detect_replayed_dialogue(self):
        module = load_module()
        sample = (
            "User: A\nAssistant: B\n\n"
            "User: C\nAssistant: D\n\n"
            "User: A\nAssistant: B\n\n"
            "User: C\nAssistant: D\n"
        )

        metrics = module.line_loop_metrics(sample)

        self.assertLess(metrics["unique_line_fraction"], 0.7)
        self.assertGreater(metrics["max_block_repeat_fraction"], 0.24)

    def test_on_policy_loop_reject_reasons_mark_prompt_replay(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--min-on-policy-loop-check-lines",
                "4",
                "--min-on-policy-unique-line-fraction",
                "0.7",
                "--max-on-policy-repeated-block-fraction",
                "0.24",
            ]
        )
        rows = [
            {
                "sample": (
                    "User: A\nAssistant: B\n\n"
                    "User: C\nAssistant: D\n\n"
                    "User: A\nAssistant: B\n\n"
                    "User: C\nAssistant: D\n"
                )
            }
        ]

        reasons = module.on_policy_loop_reject_reasons(args, rows)

        self.assertIn("on_policy_unique_line_fraction_too_low", reasons)
        self.assertIn("on_policy_repeated_block_loop", reasons)
        self.assertIn("line_loop_metrics", rows[0])

    def test_on_policy_answer_reject_reasons_mark_short_or_cross_record_answers(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--min-on-policy-continuation-chars",
                "8",
                "--repair-seed-expectations",
                "{}",
            ]
        )
        rows = [
            {
                "seed_text": "User: Q\nAssistant:",
                "sample": "User: Q\nAssistant: ok",
            },
            {
                "seed_text": "User: Q2\nAssistant:",
                "sample": "User: Q2\nAssistant: a useful answer.\nUser: next",
            },
            {
                "seed_text": "User: Q3\nAssistant:",
                "sample": "User: Q3\nAssistant: first line\nAssistant: second line",
            },
        ]

        reasons = module.on_policy_answer_reject_reasons(args, rows)

        self.assertIn("on_policy_answer_too_short", reasons)
        self.assertIn("on_policy_cross_record_continuation", reasons)
        self.assertIn("on_policy_extra_assistant_marker", reasons)
        self.assertIn("answer_surface_metrics", rows[0])

    def test_answer_surface_metrics_mark_seed_sample_extra_assistant(self):
        module = load_module()

        metrics = module.answer_surface_metrics(
            "User: Write one sentence.\nAssistant:",
            "User: Write one sentence.\nAssistant: first line\nAssistant: second line",
        )

        self.assertTrue(metrics["contains_extra_assistant"])
        self.assertFalse(metrics["contains_next_user"])

    def test_answer_surface_metrics_marks_control_char_leak(self):
        module = load_module()

        metrics = module.answer_surface_metrics(
            "User: Q\nAssistant:",
            "User: Q\nAssistant: answer with bad \x0b control",
        )

        self.assertTrue(metrics["contains_control_char"])

    def test_on_policy_answer_reject_reasons_mark_control_char_leak(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--min-on-policy-continuation-chars",
                "8",
                "--repair-seed-expectations",
                "{}",
            ]
        )
        rows = [
            {
                "seed_text": "User: Q\nAssistant:",
                "sample": "User: Q\nAssistant: answer with bad \x0b control",
            }
        ]

        reasons = module.on_policy_answer_reject_reasons(args, rows)

        self.assertIn("on_policy_control_char_leak", reasons)

    def test_semantic_relevance_rejects_off_topic_answer(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--min-on-policy-keyword-hits",
                "2",
                "--repair-seed-expectations",
                json.dumps(
                    {
                        "How can writing become clearer?": [
                            "writing",
                            "sentences",
                            "subjects",
                        ]
                    }
                ),
            ]
        )
        rows = [
            {
                "seed_text": "User: How can writing become clearer?\nAssistant:",
                "sample": (
                    "User: How can writing become clearer?\nAssistant:"
                    " A model uses donor logits and recurrent embeddings."
                ),
            }
        ]

        reasons = module.on_policy_answer_reject_reasons(args, rows)

        self.assertIn("on_policy_semantic_relevance_too_low", reasons)
        self.assertEqual(rows[0]["semantic_relevance_metrics"]["matched_keywords"], [])

    def test_semantic_relevance_accepts_keyword_grounded_answer(self):
        module = load_module()
        metrics = module.semantic_relevance_metrics(
            "User: 좋은 답변은 무엇인가요?\nAssistant:",
            "User: 좋은 답변은 무엇인가요?\nAssistant: 좋은 답변은 질문에 직접 답하고 근거를 말한다.",
            {"좋은 답변은 무엇인가요?": ["좋은 답변", "질문", "근거"]},
        )

        self.assertEqual(metrics["matched_count"], 3.0)
        self.assertEqual(metrics["expectation_key"], "좋은 답변은 무엇인가요?")


if __name__ == "__main__":
    unittest.main()
