from pathlib import Path
import importlib.util
import unittest


def load_module():
    path = Path("scripts/314_train_prompt_operand_binder_probe.py")
    spec = importlib.util.spec_from_file_location("prompt_operand_binder", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PromptOperandBinderProbeTests(unittest.TestCase):
    def test_operand_target_classes_are_one_based(self):
        module = load_module()

        row = {
            "scalar_coeff": 6,
            "subtract_offset": 9,
            "scalar_initial_residual": 18,
        }

        self.assertEqual(
            module.operand_target_classes(row, scalar_vocab_size=128),
            (7, 10, 19),
        )

    def test_prompt_operand_binder_returns_field_logits(self):
        import torch

        module = load_module()
        model = module.PromptOperandBinder(
            input_dim=8,
            hidden_dim=16,
            scalar_vocab_size=32,
        )

        logits = model(
            torch.randn(2, 5, 8),
            torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]),
        )

        self.assertEqual(set(logits), set(module.FIELD_NAMES))
        self.assertEqual(logits["scalar_coeff"].shape, (2, 32))

    def test_prompt_operand_transformer_binder_returns_field_logits(self):
        import torch

        module = load_module()
        model = module.PromptOperandTransformerBinder(
            input_dim=8,
            hidden_dim=16,
            scalar_vocab_size=32,
            layers=1,
            heads=4,
            max_positions=8,
        )

        logits = model(
            torch.randn(2, 5, 8),
            torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]),
        )

        self.assertEqual(set(logits), set(module.FIELD_NAMES))
        self.assertEqual(logits["scalar_residual"].shape, (2, 32))

    def test_checkpoint_payload_preserves_token_embedding(self):
        import torch

        module = load_module()
        model = module.PromptOperandBinder(
            input_dim=8,
            hidden_dim=16,
            scalar_vocab_size=32,
        )
        token_embed = torch.nn.Embedding(11, 8)

        payload = module.checkpoint_payload(
            model=model,
            token_embed=token_embed,
            input_dim=8,
            hidden_dim=16,
            scalar_vocab_size=32,
            input_source="token_embedding",
            binder_kind="attention",
            step=3,
            eval_report={"exact_acc": 0.5},
        )

        self.assertIn("token_embed", payload)
        self.assertEqual(payload["binder_kind"], "attention")
        self.assertEqual(payload["step"], 3)


if __name__ == "__main__":
    unittest.main()
