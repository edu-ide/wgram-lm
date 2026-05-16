import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import torch


def load_module():
    path = Path("scripts/353_train_operation_order_transition_probe.py")
    spec = importlib.util.spec_from_file_location("operation_order_transition_probe", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class OperationOrderTransitionProbeTests(unittest.TestCase):
    def test_forward_and_reverse_cases_share_tokens_but_need_different_order(self):
        module = load_module()
        start = 5
        op_ids = (1, 4, 2, 7)

        forward = module.make_case(
            case_id="f",
            family="fwd",
            start=start,
            op_ids=op_ids,
            modulus=32,
        )
        reverse = module.make_case(
            case_id="r",
            family="rev",
            start=start,
            op_ids=op_ids,
            modulus=32,
        )

        self.assertEqual(forward.start, reverse.start)
        self.assertEqual(forward.op_ids, reverse.op_ids)
        self.assertNotEqual(forward.answer, reverse.answer)
        self.assertEqual(
            module.case_prompt_tokens(forward)[2:],
            module.case_prompt_tokens(reverse)[2:],
        )

    def test_trace_targets_follow_requested_operation_order(self):
        module = load_module()
        start = 5
        op_ids = (1, 4, 2, 7)
        forward = module.make_case(
            case_id="f",
            family="fwd",
            start=start,
            op_ids=op_ids,
            modulus=32,
        )
        reverse = module.make_case(
            case_id="r",
            family="rev",
            start=start,
            op_ids=op_ids,
            modulus=32,
        )

        self.assertNotEqual(
            module.case_trace_tokens(forward, modulus=32),
            module.case_trace_tokens(reverse, modulus=32),
        )
        self.assertEqual(
            module.case_trace_tokens(forward, modulus=32)[-1],
            module.value_token(forward.answer),
        )
        self.assertEqual(
            module.case_trace_tokens(reverse, modulus=32)[-1],
            module.value_token(reverse.answer),
        )

    def test_model_outputs_lm_logits_and_order_ablation_changes_answer_path(self):
        module = load_module()
        cases = module.build_cases(
            count=4,
            seed=123,
            program_len=4,
            modulus=16,
            families=("fwd", "rev"),
        )
        batch = module.cases_to_batch(cases, device=torch.device("cpu"))
        model = module.OperationOrderTransitionLM(
            vocab=module.vocab_size(16),
            max_seq_len=batch.input_ids.shape[1],
            d_model=16,
            program_len=4,
        )

        logits = model(batch.input_ids, batch.op_positions, batch.answer_positions)
        shuffled = model(
            batch.input_ids,
            batch.op_positions,
            batch.answer_positions,
            ablation="order_shuffle",
        )

        self.assertEqual(tuple(logits.shape), (4, batch.input_ids.shape[1], module.vocab_size(16)))
        self.assertFalse(torch.allclose(logits, shuffled))
        trace = model.transition_trace(batch.input_ids, batch.op_positions)
        self.assertEqual(tuple(trace.shape), (4, 4, 16))
        self.assertEqual(tuple(batch.trace_targets.shape), (4, 4))

    def test_circular_value_codec_keeps_lm_shape_and_changes_value_embeddings(self):
        module = load_module()
        cases = module.build_cases(
            count=4,
            seed=456,
            program_len=4,
            modulus=16,
            families=("fwd", "rev"),
        )
        batch = module.cases_to_batch(cases, device=torch.device("cpu"))
        learned = module.OperationOrderTransitionLM(
            vocab=module.vocab_size(16),
            max_seq_len=batch.input_ids.shape[1],
            d_model=16,
            program_len=4,
            modulus=16,
            value_codec="learned",
        )
        circular = module.OperationOrderTransitionLM(
            vocab=module.vocab_size(16),
            max_seq_len=batch.input_ids.shape[1],
            d_model=16,
            program_len=4,
            modulus=16,
            value_codec="circular",
        )

        learned_logits = learned(batch.input_ids, batch.op_positions, batch.answer_positions)
        circular_logits = circular(batch.input_ids, batch.op_positions, batch.answer_positions)

        self.assertEqual(tuple(circular_logits.shape), tuple(learned_logits.shape))
        self.assertFalse(torch.allclose(learned._embeddings(batch.input_ids), circular._embeddings(batch.input_ids)))
        self.assertEqual(circular.value_codec, "circular")

    def test_tiny_training_run_writes_report(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            code = module.main(
                [
                    "--out-dir",
                    tmp,
                    "--steps",
                    "2",
                    "--train-cases",
                    "16",
                    "--eval-cases",
                    "8",
                    "--program-len",
                    "3",
                    "--modulus",
                    "8",
                    "--d-model",
                    "16",
                    "--batch-size",
                    "4",
                    "--device",
                    "cpu",
                    "--value-codec",
                    "circular",
                    "--trace-loss-weight",
                    "0.25",
                    "--accept-min-exact",
                    "0.0",
                    "--accept-min-transition-drop",
                    "-1.0",
                    "--accept-min-order-drop",
                    "-1.0",
                ]
            )

            report = Path(tmp) / "report.json"
            self.assertEqual(code, 0)
            self.assertTrue(report.exists())


if __name__ == "__main__":
    unittest.main()
