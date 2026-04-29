import subprocess
import sys
import unittest

import torch


class TokenizerAndDonorTests(unittest.TestCase):
    def test_hash_tokenizer_is_stable_across_processes(self):
        code = (
            "from qtrm_mm.data.jsonl_dataset import HashTokenizer;"
            "print(HashTokenizer(1024).encode('same text <image>', 8).tolist())"
        )
        first = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
        second = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
        self.assertEqual(first, second)

    def test_hf_style_tokenizer_can_be_injected(self):
        from qtrm_mm.data.jsonl_dataset import build_text_tokenizer

        class FakeTokenizer:
            pad_token_id = 7

            def __call__(self, text, **kwargs):
                self.last_text = text
                self.last_kwargs = kwargs
                return {"input_ids": torch.tensor([[11, 12, 13]])}

        fake = FakeTokenizer()
        tokenizer = build_text_tokenizer(vocab_size=99, tokenizer=fake)

        ids = tokenizer.encode("hello", max_len=6)

        self.assertEqual(ids.tolist(), [11, 12, 13, 7, 7, 7])
        self.assertEqual(fake.last_text, "hello")
        self.assertTrue(fake.last_kwargs["truncation"])
        self.assertEqual(fake.last_kwargs["max_length"], 6)

    def test_jsonl_dataset_preserves_hf_padding_attention_mask(self):
        from qtrm_mm.data.jsonl_dataset import JsonlTextVisionDataset, collate_jsonl

        class FakeTokenizer:
            pad_token_id = 7

            def __call__(self, text, **kwargs):
                return {"input_ids": torch.tensor([[11, 12]])}

        ds = JsonlTextVisionDataset(
            files=[],
            vocab_size=99,
            seq_len=5,
            visual_dim=4,
            max_visual_tokens=2,
            multimodal=False,
            tokenizer=FakeTokenizer(),
        )

        sample = ds._make_sample({"text": "hello"})
        batch = collate_jsonl([sample])

        self.assertEqual(sample["input_ids"].tolist(), [11, 12, 7, 7, 7])
        self.assertEqual(sample["attention_mask"].tolist(), [1, 1, 0, 0, 0])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 0, 0, 0]])

    def test_training_parser_accepts_donor_flag(self):
        from qtrm_mm.training.train import build_arg_parser

        args = build_arg_parser().parse_args(
            [
                "--config",
                "configs/qwen35_2b_4090.yaml",
                "--multimodal",
                "--use-donor",
                "--tokenizer-model-id",
                "Qwen/Qwen3.5-2B-Base",
            ]
        )

        self.assertTrue(args.use_donor)
        self.assertEqual(args.tokenizer_model_id, "Qwen/Qwen3.5-2B-Base")

    def test_training_parser_accepts_diagnostic_flags(self):
        from qtrm_mm.training.train import build_arg_parser

        args = build_arg_parser().parse_args(
            [
                "--config",
                "configs/qwen35_2b_4090.yaml",
                "--diag-every",
                "25",
                "--diag-max-new-tokens",
                "8",
                "--diag-prompt",
                "hello",
                "--diag-prompt",
                "안녕",
            ]
        )

        self.assertEqual(args.diag_every, 25)
        self.assertEqual(args.diag_max_new_tokens, 8)
        self.assertEqual(args.diag_prompt, ["hello", "안녕"])

    def test_prepare_donor_batch_adds_frozen_hidden_states(self):
        from qtrm_mm.training.train import prepare_donor_batch

        class DummyDonor:
            def encode_inputs(self, input_ids, attention_mask=None, return_logits=False):
                self.training = True
                out = {
                    "text_states": torch.ones(input_ids.shape[0], input_ids.shape[1], 4),
                    "attention_mask": attention_mask,
                    "visual_features": None,
                }
                if return_logits:
                    out["logits"] = torch.ones(input_ids.shape[0], input_ids.shape[1], 9)
                return out

        batch = {
            "input_ids": torch.tensor([[1, 2, 0]]),
            "attention_mask": torch.tensor([[1, 1, 0]]),
        }

        kwargs = prepare_donor_batch(DummyDonor(), batch)

        self.assertIn("text_states", kwargs)
        self.assertEqual(kwargs["text_states"].shape, (1, 3, 4))
        self.assertFalse(kwargs["text_states"].requires_grad)

    def test_prepare_donor_batch_can_add_frozen_logits(self):
        from qtrm_mm.training.train import prepare_donor_batch

        class DummyDonor:
            def encode_inputs(self, input_ids, attention_mask=None, return_logits=False):
                out = {
                    "text_states": torch.ones(input_ids.shape[0], input_ids.shape[1], 4),
                    "attention_mask": attention_mask,
                    "visual_features": None,
                }
                if return_logits:
                    out["logits"] = torch.ones(input_ids.shape[0], input_ids.shape[1], 9)
                return out

        batch = {
            "input_ids": torch.tensor([[1, 2, 0]]),
            "attention_mask": torch.tensor([[1, 1, 0]]),
        }

        kwargs = prepare_donor_batch(DummyDonor(), batch, return_logits=True)

        self.assertIn("donor_logits", kwargs)
        self.assertEqual(kwargs["donor_logits"].shape, (1, 3, 9))
        self.assertFalse(kwargs["donor_logits"].requires_grad)


if __name__ == "__main__":
    unittest.main()
