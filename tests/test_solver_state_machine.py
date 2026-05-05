import unittest

import torch


class SolverStateMachineTests(unittest.TestCase):
    def test_char_vocab_round_trips_known_text_and_uses_unk(self):
        from qtrm_mm.agentic.solver_state_machine import CharVocab

        vocab = CharVocab.build(["abc", "123"])

        ids = vocab.encode("abc", add_eos=True)

        self.assertEqual(vocab.decode(ids), "abc")
        self.assertEqual(vocab.encode("z")[0], vocab.unk_id)

    def test_state_machine_input_text_uses_previous_state_and_operation(self):
        from qtrm_mm.agentic.solver_state_machine import state_machine_input_text

        row = {
            "prompt": "Question: Compute ((7 + 3) * 2) - 3.\nAnswer:",
            "depth": 2,
            "operation": "multiply_sum",
            "previous_state_text": "10",
        }

        text = state_machine_input_text(row)

        self.assertIn("Question: Compute", text)
        self.assertIn("Operation: multiply_sum", text)
        self.assertIn("Previous state: 10", text)
        self.assertIn("Depth: 2", text)

    def test_state_machine_input_text_prefers_question_over_full_instruction(self):
        from qtrm_mm.agentic.solver_state_machine import state_machine_input_text

        row = {
            "prompt": "Answer with only the final answer. Do not write reasoning.\nQuestion: Compute 1 + 2.\nAnswer:",
            "question": "Compute 1 + 2.",
            "depth": 1,
            "operation": "add_operands",
            "previous_state_text": "",
        }

        text = state_machine_input_text(row)

        self.assertIn("Task: Compute 1 + 2.", text)
        self.assertNotIn("Do not write reasoning", text)

    def test_operation_policy_input_excludes_target_operation(self):
        from qtrm_mm.agentic.solver_state_machine import operation_policy_input_text

        row = {
            "question": "Compute ((7 + 3) * 2) - 3.",
            "depth": 2,
            "operation": "multiply_sum",
            "previous_state_text": "10",
        }

        text = operation_policy_input_text(row)

        self.assertIn("Task: Compute", text)
        self.assertIn("Task family:", text)
        self.assertIn("Trace index:", text)
        self.assertIn("Depth: 2", text)
        self.assertIn("Previous state: 10", text)
        self.assertNotIn("multiply_sum", text)

    def test_operation_vocab_round_trips_labels(self):
        from qtrm_mm.agentic.solver_state_machine import OperationVocab

        vocab = OperationVocab.build(["add_operands", "hold_final"])

        self.assertEqual(vocab.decode(vocab.encode("hold_final")), "hold_final")

    def test_operation_policy_forward_shapes(self):
        from qtrm_mm.agentic.solver_state_machine import OperationPolicy

        model = OperationPolicy(vocab_size=19, num_operations=5, d_model=8, hidden_dim=12)
        input_ids = torch.randint(0, 19, (2, 7))
        attention_mask = torch.ones_like(input_ids)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)

        self.assertEqual(tuple(logits.shape), (2, 5))

    def test_structured_operation_policy_forward_shapes(self):
        from qtrm_mm.agentic.solver_state_machine import StructuredOperationPolicy

        model = StructuredOperationPolicy(
            num_families=3,
            num_trace_indices=4,
            num_depths=4,
            num_operations=6,
            d_model=8,
            hidden_dim=12,
        )

        logits = model(
            family_ids=torch.tensor([0, 1]),
            trace_index_ids=torch.tensor([2, 3]),
            depth_ids=torch.tensor([1, 2]),
        )

        self.assertEqual(tuple(logits.shape), (2, 6))

    def test_solver_state_machine_forward_shapes(self):
        from qtrm_mm.agentic.solver_state_machine import SolverStateMachine

        model = SolverStateMachine(vocab_size=17, d_model=8, hidden_dim=12)
        input_ids = torch.randint(0, 17, (2, 5))
        input_mask = torch.ones_like(input_ids)
        decoder_input_ids = torch.randint(0, 17, (2, 4))

        logits = model(
            input_ids=input_ids,
            attention_mask=input_mask,
            decoder_input_ids=decoder_input_ids,
        )

        self.assertEqual(tuple(logits.shape), (2, 4, 17))

    def test_target_tensors_shift_with_bos_and_eos(self):
        from qtrm_mm.agentic.solver_state_machine import CharVocab, target_tensors

        vocab = CharVocab.build(["42"])

        decoder_input, labels = target_tensors(vocab, "42", max_target_len=4)

        self.assertEqual(decoder_input.tolist(), [vocab.bos_id, vocab.token_to_id["4"], vocab.token_to_id["2"], vocab.eos_id])
        self.assertEqual(labels.tolist(), [vocab.token_to_id["4"], vocab.token_to_id["2"], vocab.eos_id, vocab.pad_id])

    def test_rollout_case_uses_predicted_previous_state(self):
        from qtrm_mm.agentic.solver_state_machine import rollout_trace_rows

        rows = [
            {
                "source_id": "case-1",
                "trace_index": 0,
                "operation": "first",
                "previous_state_text": "",
                "target_state_text": "A",
            },
            {
                "source_id": "case-1",
                "trace_index": 1,
                "operation": "second",
                "previous_state_text": "A",
                "target_state_text": "B",
            },
        ]

        calls = []

        def predict(row, previous_state):
            calls.append((row["operation"], previous_state))
            return "X" if row["operation"] == "first" else previous_state + "Y"

        records = rollout_trace_rows(rows, predict)

        self.assertEqual(calls, [("first", ""), ("second", "X")])
        self.assertEqual(records[-1]["predicted_state_text"], "XY")

    def test_execute_solver_transition_handles_arithmetic_trace(self):
        from qtrm_mm.agentic.solver_state_machine import execute_solver_transition

        row = {
            "question": "Compute ((207 + 3) * 2) - 3.",
            "operation": "add_operands",
        }
        self.assertEqual(execute_solver_transition(row, ""), "210")
        self.assertEqual(
            execute_solver_transition({**row, "operation": "multiply_sum"}, "210"),
            "420",
        )
        self.assertEqual(
            execute_solver_transition({**row, "operation": "subtract_offset"}, "420"),
            "417",
        )

    def test_execute_solver_transition_handles_list_trace(self):
        from qtrm_mm.agentic.solver_state_machine import execute_solver_transition

        row = {
            "question": "From the list [2001, 2004, 2002, 2007, 2003], keep only even numbers, double each kept number, and return comma-separated values with no spaces. If none, return EMPTY.",
            "operation": "filter_even",
        }

        self.assertEqual(execute_solver_transition(row, ""), "2004,2002")
        self.assertEqual(
            execute_solver_transition({**row, "operation": "double_filtered"}, "2004,2002"),
            "4008,4004",
        )
        self.assertEqual(
            execute_solver_transition({**row, "operation": "hold_final"}, "4008,4004"),
            "4008,4004",
        )

    def test_execute_solver_transition_handles_symbolic_trace(self):
        from qtrm_mm.agentic.solver_state_machine import execute_solver_transition

        row = {
            "question": "If A maps to green, green maps to violet, and violet maps to D, what does A map to after two mappings?",
            "operation": "first_mapping",
        }

        self.assertEqual(execute_solver_transition(row, ""), "green")
        self.assertEqual(
            execute_solver_transition({**row, "operation": "second_mapping"}, "green"),
            "violet",
        )
        self.assertEqual(
            execute_solver_transition({**row, "operation": "hold_final"}, "violet"),
            "violet",
        )

    def test_execute_solver_transition_handles_boolean_trace(self):
        from qtrm_mm.agentic.solver_state_machine import execute_solver_transition

        row = {
            "question": "Let P=TRUE, Q=FALSE, R=FALSE. Evaluate (P AND NOT Q) OR R. Answer TRUE or FALSE.",
            "operation": "not_q",
        }

        self.assertEqual(execute_solver_transition(row, ""), "TRUE")
        self.assertEqual(
            execute_solver_transition({**row, "operation": "and_with_p"}, "TRUE"),
            "TRUE",
        )
        self.assertEqual(
            execute_solver_transition({**row, "operation": "or_with_r"}, "TRUE"),
            "TRUE",
        )

    def test_rollout_solver_trace_from_operations_scores_state_and_final_answer(self):
        from qtrm_mm.agentic.solver_state_machine import rollout_solver_trace_from_operations

        row = {
            "question": "Compute ((207 + 3) * 2) - 3.",
            "chosen": "417",
            "solver_trace": [
                {"depth": 1, "operation": "add_operands", "state_text": "210"},
                {"depth": 2, "operation": "multiply_sum", "state_text": "420"},
                {"depth": 4, "operation": "subtract_offset", "state_text": "417"},
                {"depth": 8, "operation": "hold_final", "state_text": "417"},
            ],
        }

        good = rollout_solver_trace_from_operations(
            row,
            ["add_operands", "multiply_sum", "subtract_offset", "hold_final"],
        )
        bad = rollout_solver_trace_from_operations(
            row,
            ["hold_final", "multiply_sum", "subtract_offset", "hold_final"],
        )

        self.assertEqual(good["operation_exact"], "4/4")
        self.assertEqual(good["state_exact"], "4/4")
        self.assertTrue(good["final_exact_match"])
        self.assertEqual(good["predicted_final"], "417")
        self.assertEqual(bad["operation_exact"], "3/4")
        self.assertEqual(bad["state_exact"], "0/4")
        self.assertFalse(bad["final_exact_match"])

    def test_answer_from_primitive_operations_does_not_require_solver_trace(self):
        from qtrm_mm.agentic.solver_state_machine import answer_from_primitive_operations

        result = answer_from_primitive_operations(
            {
                "prompt": (
                    "Answer with only the final answer.\n"
                    "Question: Compute ((207 + 3) * 2) - 3.\n"
                    "Answer:"
                )
            },
            ["add_operands", "multiply_sum", "subtract_offset", "hold_final"],
        )

        self.assertEqual(result["answer"], "417")
        self.assertEqual(result["executed_operations"], ["add_operands", "multiply_sum", "subtract_offset"])
        self.assertEqual(result["states"], ["210", "420", "417"])

    def test_answer_from_primitive_operations_stops_after_hold_final(self):
        from qtrm_mm.agentic.solver_state_machine import answer_from_primitive_operations

        result = answer_from_primitive_operations(
            {
                "question": (
                    "From the list [2001, 2004, 2002, 2007, 2003], keep only even "
                    "numbers, double each kept number, and return comma-separated "
                    "values with no spaces. If none, return EMPTY."
                )
            },
            ["filter_even", "double_filtered", "hold_final", "add_operands"],
        )

        self.assertEqual(result["answer"], "4008,4004")
        self.assertEqual(result["executed_operations"], ["filter_even", "double_filtered"])

    def test_operation_names_from_logits_can_use_state_constraints(self):
        from qtrm_mm.agentic.solver_state_machine import operation_names_from_logits

        id_to_operation = {
            0: "filter_even",
            1: "double_filtered",
            2: "multiply_sum",
            3: "hold_final",
        }
        logits = torch.tensor(
            [
                [4.0, 0.0, 0.0, 0.0],
                [0.0, 3.0, 5.0, 1.0],
                [0.0, 0.0, 0.0, 4.0],
                [0.0, 0.0, 0.0, 4.0],
            ]
        )
        row = {
            "question": (
                "Apply the even-filter then double transform to "
                "[10002, 10005, 10003, 10008, 10004]. Print only CSV output, or EMPTY."
            )
        }

        raw = operation_names_from_logits(logits, id_to_operation)
        constrained = operation_names_from_logits(
            logits,
            id_to_operation,
            row=row,
            state_constrained=True,
        )

        self.assertEqual(raw[:2], ["filter_even", "multiply_sum"])
        self.assertEqual(
            constrained,
            ["filter_even", "double_filtered", "hold_final", "hold_final"],
        )

    def test_state_constraints_do_not_invent_first_operation(self):
        from qtrm_mm.agentic.solver_state_machine import operation_names_from_logits

        id_to_operation = {
            0: "add_operands",
            1: "filter_even",
            2: "double_filtered",
            3: "hold_final",
        }
        logits = torch.zeros((4, 4))
        row = {
            "question": (
                "Apply the even-filter then double transform to "
                "[10002, 10005, 10003, 10008, 10004]. Print only CSV output, or EMPTY."
            )
        }

        constrained = operation_names_from_logits(
            logits,
            id_to_operation,
            row=row,
            state_constrained=True,
        )

        self.assertEqual(constrained[0], "add_operands")


if __name__ == "__main__":
    unittest.main()
