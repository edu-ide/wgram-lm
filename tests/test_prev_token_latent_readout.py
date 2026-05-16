import types
import unittest

import torch


class PrevTokenLatentReadoutTests(unittest.TestCase):
    def _cfg(self):
        from qtrm_mm import QTRMConfig

        return QTRMConfig(
            vocab_size=64,
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            d_ff=32,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=8,
            max_visual_tokens=2,
            answer_state_loop_enabled=True,
            answer_state_loop_next_token_decoder_enabled=True,
            answer_state_loop_next_token_decoder_layers=1,
            answer_state_loop_next_token_decoder_prev_token_enabled=True,
            answer_state_loop_next_token_decoder_prev_token_gate_min=1.0,
        )

    def test_prev_token_fuse_starts_as_identity(self):
        from qtrm_mm import QTRMMultimodalModel

        torch.manual_seed(0)
        model = QTRMMultimodalModel(self._cfg())
        hidden = torch.randn(2, 3, model.cfg.d_model)
        prev_token_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])

        out = model._answer_state_loop_next_token_decoder_prev_token_hidden(
            hidden,
            prev_token_ids=prev_token_ids,
        )

        self.assertTrue(torch.allclose(out, hidden, atol=1e-6))

    def test_prev_token_fuse_can_change_hidden_by_token_id(self):
        from qtrm_mm import QTRMMultimodalModel

        torch.manual_seed(0)
        model = QTRMMultimodalModel(self._cfg())
        d_model = int(model.cfg.d_model)
        with torch.no_grad():
            model.answer_state_loop_next_token_decoder_prev_token_fuse.weight[
                :, d_model:
            ].copy_(torch.eye(d_model))

        hidden = torch.zeros(1, 2, d_model)
        prev_token_ids_a = torch.tensor([[1, 1]])
        prev_token_ids_b = torch.tensor([[2, 2]])

        out_a = model._answer_state_loop_next_token_decoder_prev_token_hidden(
            hidden,
            prev_token_ids=prev_token_ids_a,
        )
        out_b = model._answer_state_loop_next_token_decoder_prev_token_hidden(
            hidden,
            prev_token_ids=prev_token_ids_b,
        )

        self.assertFalse(torch.allclose(out_a, hidden))
        self.assertFalse(torch.allclose(out_a, out_b))

    def test_forward_passes_selected_input_tokens_as_prev_token_ids(self):
        from qtrm_mm import QTRMMultimodalModel

        model = QTRMMultimodalModel(self._cfg())
        input_ids = torch.tensor([[10, 11, 12, 13, 14]])
        selected = torch.tensor([1, 3])
        captured = {}

        def fake_compute(self, text_context_seq, **kwargs):
            prev_token_ids = kwargs["prev_token_ids"]
            captured["prev_token_ids"] = prev_token_ids.detach().clone()
            batch = int(text_context_seq.shape[0])
            seq_len = int(prev_token_ids.shape[1])
            d_model = int(self.cfg.d_model)
            vocab_size = int(self.cfg.vocab_size)
            return (
                text_context_seq.new_zeros((batch, seq_len, vocab_size)),
                text_context_seq.new_zeros((batch, seq_len, d_model)),
                text_context_seq.new_zeros((batch, 0, seq_len, d_model)),
                text_context_seq.new_empty((batch, 0)),
                text_context_seq.new_empty((batch, 0)),
                text_context_seq.new_zeros(()),
                text_context_seq.new_zeros(()),
            )

        model._compute_answer_state_loop_outputs = types.MethodType(
            fake_compute,
            model,
        )

        model(input_ids, logit_token_indices=selected)

        self.assertTrue(
            torch.equal(captured["prev_token_ids"], input_ids.index_select(1, selected))
        )


if __name__ == "__main__":
    unittest.main()
