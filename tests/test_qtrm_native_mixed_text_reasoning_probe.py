import importlib.util
import sys
import unittest
from pathlib import Path

import torch


def load_module():
    path = Path("scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py")
    spec = importlib.util.spec_from_file_location("qtrm_native_mixed_text_reasoning_probe", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class QTRMNativeMixedTextReasoningProbeTests(unittest.TestCase):
    def test_prompt_and_answer_are_fixed_width_text(self):
        module = load_module()
        case = module.TextReasoningCase(
            case_id="x",
            start=3,
            op_ids=(1, 7, 4, 3),
            answer=5,
        )

        prompt = module.case_prompt(case)
        answer = module.case_answer(case)

        self.assertEqual(prompt, "start 03 ops 01 07 04 03 answer ")
        self.assertEqual(answer, "05\n")

    def test_prompt_state_anchor_keeps_answer_path_visible(self):
        module = load_module()
        case = module.TextReasoningCase(
            case_id="x",
            start=3,
            op_ids=(1, 7),
            answer=5,
        )
        tokenizer = module.CharTokenizer.from_texts(
            [
                module.case_full_text(case),
                module.case_full_text(case, state_anchor=True),
            ],
            mode="number",
        )

        plain_prompt = module.case_prompt(case)
        anchor_prompt = module.case_prompt(case, state_anchor=True)
        tail_anchor_prompt = module.case_prompt(
            case,
            state_anchor=True,
            state_anchor_position="after_answer",
        )
        _x, _y, plain_prompt_len, plain_answer_len = module.cases_to_batch(
            [case],
            tokenizer=tokenizer,
            device=torch.device("cpu"),
        )
        _x, _y, anchor_prompt_len, anchor_answer_len = module.cases_to_batch(
            [case],
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            state_anchor=True,
        )

        self.assertEqual(plain_prompt, "start 03 ops 01 07 answer ")
        self.assertEqual(anchor_prompt, "start 03 ops 01 07 state answer ")
        self.assertEqual(tail_anchor_prompt, "start 03 ops 01 07 answer state ")
        self.assertGreater(anchor_prompt_len, plain_prompt_len)
        self.assertEqual(anchor_answer_len, plain_answer_len)

    def test_answer_format_valid_requires_two_digits_and_newline(self):
        module = load_module()

        self.assertTrue(module.answer_format_valid("05\n"))
        self.assertTrue(module.answer_format_valid("99\n"))
        self.assertFalse(module.answer_format_valid("5\n"))
        self.assertFalse(module.answer_format_valid("005\n"))
        self.assertFalse(module.answer_format_valid("2r\n"))
        self.assertFalse(module.answer_format_valid("\n1\n"))
        self.assertFalse(module.answer_format_valid("05"))

    def test_flexible_resume_prefix_copies_position_embeddings(self):
        module = load_module()
        source = module.NativeQTRMETDLM(
            vocab=16,
            max_seq_len=4,
            d_model=8,
            n_heads=2,
            d_ff=16,
            dropout=0.0,
            backbone="mha_etd",
        )
        target = module.NativeQTRMETDLM(
            vocab=16,
            max_seq_len=7,
            d_model=8,
            n_heads=2,
            d_ff=16,
            dropout=0.0,
            backbone="mha_etd",
        )
        with torch.no_grad():
            source.pos_embed.weight.copy_(torch.arange(32, dtype=torch.float32).view(4, 8))

        summary = module.load_model_state_flexible(target, source.state_dict())

        self.assertIn("pos_embed.weight", summary["resized_tensors"])
        self.assertEqual(summary["resized_tensors"]["pos_embed.weight"]["copied_rows"], 4)
        self.assertTrue(torch.equal(target.pos_embed.weight[:4], source.pos_embed.weight))

    def test_flexible_load_can_tail_shift_new_position_rows(self):
        module = load_module()
        source = module.NativeQTRMETDLM(
            vocab=16,
            max_seq_len=4,
            d_model=8,
            n_heads=2,
            d_ff=16,
            dropout=0.0,
            backbone="mha_etd",
        )
        target = module.NativeQTRMETDLM(
            vocab=16,
            max_seq_len=7,
            d_model=8,
            n_heads=2,
            d_ff=16,
            dropout=0.0,
            backbone="mha_etd",
        )
        with torch.no_grad():
            source.pos_embed.weight.copy_(torch.arange(32, dtype=torch.float32).view(4, 8))

        summary = module.load_model_state_flexible(
            target,
            source.state_dict(),
            pos_embed_resize_strategy="tail_shift",
        )

        self.assertEqual(summary["resized_tensors"]["pos_embed.weight"]["filled_rows"], 3)
        self.assertEqual(
            summary["resized_tensors"]["pos_embed.weight"]["resize_strategy"],
            "tail_shift",
        )
        self.assertTrue(torch.equal(target.pos_embed.weight[4:], source.pos_embed.weight[1:]))

    def test_flexible_load_remaps_vocab_rows_by_token(self):
        module = load_module()
        source = module.NativeQTRMETDLM(
            vocab=3,
            max_seq_len=4,
            d_model=8,
            n_heads=2,
            d_ff=16,
            dropout=0.0,
            backbone="mha_etd",
        )
        target = module.NativeQTRMETDLM(
            vocab=5,
            max_seq_len=4,
            d_model=8,
            n_heads=2,
            d_ff=16,
            dropout=0.0,
            backbone="mha_etd",
        )
        source_chars = tuple("aew")
        target_chars = tuple("aceuw")
        with torch.no_grad():
            source.token_embed.weight.copy_(
                torch.arange(24, dtype=torch.float32).view(3, 8)
            )
            source.lm_head.weight.copy_(
                torch.arange(24, dtype=torch.float32).view(3, 8) + 100
            )

        summary = module.load_model_state_flexible(
            target,
            source.state_dict(),
            source_chars=source_chars,
            target_chars=target_chars,
        )

        self.assertEqual(
            summary["resized_tensors"]["token_embed.weight"]["resize_strategy"],
            "token_remap",
        )
        self.assertTrue(torch.equal(target.token_embed.weight[0], source.token_embed.weight[0]))
        self.assertTrue(torch.equal(target.token_embed.weight[2], source.token_embed.weight[1]))
        self.assertTrue(torch.equal(target.token_embed.weight[4], source.token_embed.weight[2]))
        self.assertTrue(torch.equal(target.lm_head.weight[4], source.lm_head.weight[2]))
        self.assertIn("c", summary["resized_tensors"]["token_embed.weight"]["new_target_tokens"])
        self.assertIn("u", summary["resized_tensors"]["token_embed.weight"]["new_target_tokens"])

    def test_flexible_load_composes_number_and_op_tokens_from_source_chars(self):
        module = load_module()
        source = module.NativeQTRMETDLM(
            vocab=4,
            max_seq_len=4,
            d_model=4,
            n_heads=2,
            d_ff=8,
            dropout=0.0,
            backbone="mha_etd",
        )
        target = module.NativeQTRMETDLM(
            vocab=2,
            max_seq_len=4,
            d_model=4,
            n_heads=2,
            d_ff=8,
            dropout=0.0,
            backbone="mha_etd",
        )
        source_chars = ("0", "1", "o", "p")
        target_chars = ("01", "op01")
        with torch.no_grad():
            source.token_embed.weight.copy_(
                torch.tensor(
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [2.0, 2.0, 2.0, 2.0],
                        [4.0, 4.0, 4.0, 4.0],
                        [8.0, 8.0, 8.0, 8.0],
                    ]
                )
            )
            source.lm_head.weight.copy_(source.token_embed.weight + 100.0)

        summary = module.load_model_state_flexible(
            target,
            source.state_dict(),
            source_chars=source_chars,
            target_chars=target_chars,
        )

        self.assertTrue(
            torch.equal(target.token_embed.weight[0], torch.full((4,), 1.0))
        )
        self.assertTrue(
            torch.equal(target.token_embed.weight[1], torch.full((4,), 3.5))
        )
        self.assertTrue(torch.equal(target.lm_head.weight[0], torch.full((4,), 101.0)))
        self.assertIn(
            "op01",
            summary["resized_tensors"]["token_embed.weight"]["composed_target_tokens"],
        )
        self.assertNotIn(
            "op01",
            summary["resized_tensors"]["token_embed.weight"]["new_target_tokens"],
        )

    def test_answer_value_parses_only_valid_answers(self):
        module = load_module()

        self.assertEqual(module.answer_value("05\n"), 5)
        self.assertIsNone(module.answer_value("5\n"))
        self.assertIsNone(module.answer_value("2r\n"))

    def test_number_tokenizer_keeps_two_digit_values_atomic(self):
        module = load_module()
        tokenizer = module.CharTokenizer.from_texts(
            ["start 03 ops 01 07 answer 05\n"],
            mode="number",
        )

        encoded = tokenizer.encode("start 03 ops 01 07 answer 05\n")

        self.assertEqual(tokenizer.decode(encoded), "start 03 ops 01 07 answer 05\n")
        self.assertIn("05", tokenizer.char_to_id)
        self.assertIn("99", tokenizer.char_to_id)
        self.assertLess(len(encoded), len("start 03 ops 01 07 answer 05\n"))

    def test_number_tokenizer_can_limit_extra_value_tokens(self):
        module = load_module()
        tokenizer = module.CharTokenizer.from_texts(
            ["start 03 ops 01 07 answer 05\n"],
            mode="number",
            number_max_value=31,
        )

        self.assertIn("31", tokenizer.char_to_id)
        self.assertNotIn("99", tokenizer.char_to_id)

    def test_number_tokenizer_can_separate_operation_role_tokens(self):
        module = load_module()
        text = "start 03 ops 01 07 answer 05\n"
        tokenizer = module.CharTokenizer.from_texts(
            [text],
            mode="number",
            number_max_value=31,
            op_role_tokens=True,
        )

        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        value_token_ids = module.value_token_ids_for_tokenizer(tokenizer, modulus=32)

        self.assertEqual(decoded, text)
        self.assertIn("01", tokenizer.char_to_id)
        self.assertIn("op01", tokenizer.char_to_id)
        self.assertIn("op07", tokenizer.char_to_id)
        self.assertNotEqual(tokenizer.char_to_id["01"], tokenizer.char_to_id["op01"])
        self.assertEqual(tokenizer.chars[value_token_ids[1]], "01")
        self.assertNotIn(tokenizer.char_to_id["op01"], value_token_ids)

    def test_value_token_ids_for_number_tokenizer_follow_two_digit_values(self):
        module = load_module()
        tokenizer = module.CharTokenizer.from_texts(
            ["start 03 ops 01 07 answer 05\n"],
            mode="number",
            number_max_value=31,
        )

        token_ids = module.value_token_ids_for_tokenizer(tokenizer, modulus=32)

        self.assertEqual(tokenizer.chars[token_ids[0]], "00")
        self.assertEqual(tokenizer.chars[token_ids[31]], "31")
        self.assertEqual(len(token_ids), 32)

    def test_cases_to_batch_uses_token_lengths_for_number_tokenizer(self):
        module = load_module()
        case = module.TextReasoningCase(
            case_id="x",
            start=3,
            op_ids=(1, 7),
            answer=5,
        )
        tokenizer = module.CharTokenizer.from_texts(
            [module.case_full_text(case)],
            mode="number",
        )

        _x, _y, prompt_len, answer_len = module.cases_to_batch(
            [case],
            tokenizer=tokenizer,
            device=torch.device("cpu"),
        )

        self.assertEqual(prompt_len, len(tokenizer.encode(module.case_prompt(case))))
        self.assertEqual(answer_len, len(tokenizer.encode(module.case_answer(case))))
        self.assertEqual(answer_len, 2)

    def test_family_tag_prompt_is_fixed_width_for_l5_multifamily(self):
        module = load_module()
        modchain = module.TextReasoningCase(
            case_id="m",
            start=3,
            op_ids=(1, 7, 4, 3),
            answer=5,
            family="modchain",
        )
        revchain = module.TextReasoningCase(
            case_id="r",
            start=3,
            op_ids=(1, 7, 4, 3),
            answer=5,
            family="revchain",
        )

        mod_prompt = module.case_prompt(modchain, include_family_tag=True)
        rev_prompt = module.case_prompt(revchain, include_family_tag=True)

        self.assertEqual(mod_prompt, "task modchain start 03 ops 01 07 04 03 answer ")
        self.assertEqual(rev_prompt, "task revchain start 03 ops 01 07 04 03 answer ")
        self.assertEqual(len(mod_prompt), len(rev_prompt))

    def test_answer_label_prompt_preserves_fixed_width(self):
        module = load_module()
        case = module.TextReasoningCase(
            case_id="x",
            start=3,
            op_ids=(1, 7),
            answer=5,
        )

        prompt = module.case_prompt_with_answer_label(case, answer_label="answer4")
        tail_prompt = module.case_prompt_with_answer_label(
            case,
            answer_label="answer4",
            state_anchor=True,
            state_anchor_position="after_answer",
        )

        self.assertEqual(prompt, "start 03 ops 01 07 answer4")
        self.assertEqual(tail_prompt, "start 03 ops 01 07 answer4state ")
        self.assertEqual(len(prompt), len(module.case_prompt(case)))
        with self.assertRaises(ValueError):
            module.case_prompt_with_answer_label(case, answer_label="answer10")

    def test_family_answer_semantics_are_distinct(self):
        module = load_module()
        op_ids = (1, 4)

        mod_answer = module.compute_answer(
            start=3,
            op_ids=op_ids,
            family="modchain",
            modulus=8,
        )
        rev_answer = module.compute_answer(
            start=3,
            op_ids=op_ids,
            family="revchain",
            modulus=8,
        )
        checksum_answer = module.compute_answer(
            start=3,
            op_ids=op_ids,
            family="checksum",
            modulus=8,
        )

        self.assertNotEqual(mod_answer, rev_answer)
        self.assertEqual(checksum_answer, (3 + 1 + 4) % 8)

    def test_build_cases_allows_weighted_train_families(self):
        module = load_module()

        cases = module.build_cases(
            count=5,
            seed=1,
            program_len=2,
            modulus=8,
            families=("modchain", "revchain", "modchain", "revchain", "checksum"),
        )

        self.assertEqual(
            [case.family for case in cases],
            ["modchain", "revchain", "modchain", "revchain", "checksum"],
        )

    def test_order_invariant_eval_cases_keep_per_family_samples_stable(self):
        module = load_module()

        first = module.build_family_order_invariant_eval_cases(
            count=9,
            seed=17,
            program_len=3,
            modulus=8,
            families=("modchain", "revchain", "checksum"),
        )
        second = module.build_family_order_invariant_eval_cases(
            count=9,
            seed=17,
            program_len=3,
            modulus=8,
            families=("checksum", "modchain", "revchain"),
        )

        def signature_by_family(cases):
            rows = {}
            for case in cases:
                rows.setdefault(case.family, []).append(
                    (case.start, case.op_ids, case.answer)
                )
            return rows

        self.assertEqual(signature_by_family(first), signature_by_family(second))

    def test_build_cases_can_oversample_hard_ops_for_training(self):
        module = load_module()

        cases = module.build_cases(
            count=8,
            seed=1,
            program_len=3,
            modulus=8,
            hard_op_ids=(5, 7),
            hard_op_probability=1.0,
        )

        self.assertTrue(
            all(all(op_id in (5, 7) for op_id in case.op_ids) for case in cases)
        )

    def test_build_cases_can_restrict_hard_ops_to_positions(self):
        module = load_module()

        cases = module.build_cases(
            count=8,
            seed=1,
            program_len=4,
            modulus=8,
            hard_op_ids=(5, 7),
            hard_op_probability=1.0,
            hard_op_positions=(3, 4),
        )

        self.assertTrue(all(case.op_ids[2] in (5, 7) for case in cases))
        self.assertTrue(all(case.op_ids[3] in (5, 7) for case in cases))

    def test_parse_op_ids_rejects_noop_and_unknown_ops(self):
        module = load_module()

        self.assertEqual(module.parse_op_ids("05, 07 2"), (5, 7, 2))
        with self.assertRaises(ValueError):
            module.parse_op_ids("0")
        with self.assertRaises(ValueError):
            module.parse_op_ids("99")

    def test_parse_positions_rejects_non_positive_positions(self):
        module = load_module()

        self.assertEqual(module.parse_positions("5, 6 3"), (5, 6, 3))
        with self.assertRaises(ValueError):
            module.parse_positions("0")

    def test_parse_residue_moduli_requires_one_digit_moduli(self):
        module = load_module()

        self.assertEqual(module.parse_residue_moduli("2,4 8"), (2, 4, 8))
        with self.assertRaises(ValueError):
            module.parse_residue_moduli("1")
        with self.assertRaises(ValueError):
            module.parse_residue_moduli("10")

    def test_parse_preference_deltas_rejects_non_positive_values(self):
        module = load_module()

        self.assertEqual(module.parse_preference_deltas("2,4 16"), (2, 4, 16))
        with self.assertRaises(ValueError):
            module.parse_preference_deltas("0")

    def test_eval_task_families_can_differ_from_weighted_train_families(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--task-families",
                "modchain,revchain,modchain,revchain,checksum",
                "--eval-task-families",
                "modchain,revchain,checksum",
                "--train-hard-op-ids",
                "05,06,07",
                "--train-hard-op-probability",
                "0.6",
                "--train-hard-op-positions",
                "5,6",
                "--prompt-state-anchor",
                "--prompt-state-anchor-position",
                "after_answer",
                "--position-embedding-mode",
                "none",
                "--tokenizer-mode",
                "number",
                "--number-tokenizer-max-value",
                "31",
                "--number-tokenizer-op-role-tokens",
                "--value-codec",
                "circular",
                "--think-structure",
                "single_core_carrier",
                "--carrier-gate-init",
                "-4.0",
                "--carrier-state-mode",
                "state_mean",
                "--eval-family-order-invariant",
                "--core-step-op-codec-loss-weight",
                "0.2",
                "--core-step-op-codec-state-source",
                "both",
                "--core-step-op-codec-pooling",
                "mean",
                "--core-step-position-codec-loss-weight",
                "0.3",
                "--core-step-position-codec-state-source",
                "both",
                "--core-step-position-codec-pooling",
                "mean",
                "--state-trace-depth-loss-weight",
                "0.4",
                "--state-trace-depth-state-source",
                "both",
                "--state-trace-depth-min-depth",
                "2",
                "--state-trace-depth-weight-power",
                "1.5",
                "--state-trace-depth-family-dro",
                "--state-trace-depth-family-dro-temperature",
                "0.7",
            ]
        )

        self.assertEqual(
            module.train_families_for_args(args),
            ("modchain", "revchain", "modchain", "revchain", "checksum"),
        )
        self.assertEqual(
            module.eval_families_for_args(args),
            ("modchain", "revchain", "checksum"),
        )
        self.assertEqual(args.train_hard_op_ids, "05,06,07")
        self.assertEqual(args.train_hard_op_probability, 0.6)
        self.assertEqual(args.train_hard_op_positions, "5,6")
        self.assertEqual(args.position_embedding_mode, "none")
        self.assertEqual(args.tokenizer_mode, "number")
        self.assertEqual(args.number_tokenizer_max_value, 31)
        self.assertTrue(args.number_tokenizer_op_role_tokens)
        self.assertEqual(args.value_codec, "circular")
        self.assertEqual(args.think_structure, "single_core_carrier")
        self.assertEqual(args.carrier_gate_init, -4.0)
        self.assertEqual(args.carrier_state_mode, "state_mean")
        self.assertTrue(args.eval_family_order_invariant)
        self.assertEqual(args.core_step_op_codec_loss_weight, 0.2)
        self.assertEqual(args.core_step_op_codec_state_source, "both")
        self.assertEqual(args.core_step_op_codec_pooling, "mean")
        self.assertEqual(args.core_step_position_codec_loss_weight, 0.3)
        self.assertEqual(args.core_step_position_codec_state_source, "both")
        self.assertEqual(args.core_step_position_codec_pooling, "mean")
        self.assertEqual(args.state_trace_depth_loss_weight, 0.4)
        self.assertEqual(args.state_trace_depth_state_source, "both")
        self.assertEqual(args.state_trace_depth_min_depth, 2)
        self.assertEqual(args.state_trace_depth_weight_power, 1.5)
        self.assertTrue(args.state_trace_depth_family_dro)
        self.assertEqual(args.state_trace_depth_family_dro_temperature, 0.7)
        self.assertTrue(module.state_anchor_for_args(args))
        self.assertEqual(module.state_anchor_position_for_args(args), "after_answer")

    def test_active_program_len_uses_family_causal_prefix_and_recomputes_answer(self):
        module = load_module()
        mod_case = module.TextReasoningCase(
            case_id="x",
            start=3,
            op_ids=(1, 4, 2, 5),
            answer=999,
            family="modchain",
        )
        rev_case = module.TextReasoningCase(
            case_id="r",
            start=3,
            op_ids=(1, 4, 2, 5),
            answer=999,
            family="revchain",
        )

        mod_shortened = module.case_with_active_program_len(
            mod_case,
            active_len=2,
            modulus=8,
        )
        rev_shortened = module.case_with_active_program_len(
            rev_case,
            active_len=2,
            modulus=8,
        )

        self.assertEqual(
            mod_shortened.op_ids,
            (1, 4, module.NOOP_OP_ID, module.NOOP_OP_ID),
        )
        self.assertEqual(
            rev_shortened.op_ids,
            (module.NOOP_OP_ID, module.NOOP_OP_ID, 2, 5),
        )
        self.assertEqual(
            mod_shortened.answer,
            module.compute_answer(
                start=3,
                op_ids=mod_shortened.op_ids,
                family="modchain",
                modulus=8,
            ),
        )
        self.assertEqual(
            rev_shortened.answer,
            module.compute_answer(
                start=3,
                op_ids=rev_shortened.op_ids,
                family="revchain",
                modulus=8,
            ),
        )
        self.assertEqual(module.effective_program_len(rev_shortened), 2)

    def test_causal_op_id_at_depth_follows_family_order(self):
        module = load_module()
        mod_case = module.TextReasoningCase(
            case_id="m",
            start=0,
            op_ids=(1, 4, 2),
            answer=0,
            family="modchain",
        )
        rev_case = module.TextReasoningCase(
            case_id="r",
            start=0,
            op_ids=(1, 4, 2),
            answer=0,
            family="revchain",
        )

        self.assertEqual(module.causal_op_id_at_depth(mod_case, depth_index=0), 1)
        self.assertEqual(module.causal_op_id_at_depth(mod_case, depth_index=2), 2)
        self.assertEqual(module.causal_op_id_at_depth(rev_case, depth_index=0), 2)
        self.assertEqual(module.causal_op_id_at_depth(rev_case, depth_index=2), 1)
        self.assertEqual(
            module.causal_op_id_at_depth(rev_case, depth_index=3),
            module.NOOP_OP_ID,
        )

    def test_causal_op_position_at_depth_follows_family_order(self):
        module = load_module()
        mod_case = module.TextReasoningCase(
            case_id="m",
            start=0,
            op_ids=(1, 4, 2),
            answer=0,
            family="modchain",
        )
        rev_case = module.TextReasoningCase(
            case_id="r",
            start=0,
            op_ids=(1, 4, 2),
            answer=0,
            family="revchain",
        )

        self.assertEqual(module.causal_op_position_at_depth(mod_case, depth_index=0), 1)
        self.assertEqual(module.causal_op_position_at_depth(mod_case, depth_index=2), 3)
        self.assertEqual(module.causal_op_position_at_depth(rev_case, depth_index=0), 3)
        self.assertEqual(module.causal_op_position_at_depth(rev_case, depth_index=2), 1)
        self.assertEqual(module.causal_op_position_at_depth(rev_case, depth_index=3), 0)

    def test_intermediate_answer_targets_follow_family_direction(self):
        module = load_module()
        tokenizer = module.CharTokenizer.from_texts(
            ["00\n"],
            mode="number",
            number_max_value=31,
        )
        op_ids = (1, 4, 2, 5)
        modchain = module.TextReasoningCase(
            case_id="m",
            start=1,
            op_ids=op_ids,
            answer=module.compute_answer(
                start=1,
                op_ids=op_ids,
                family="modchain",
                modulus=8,
            ),
            family="modchain",
        )
        revchain = module.TextReasoningCase(
            case_id="r",
            start=1,
            op_ids=op_ids,
            answer=module.compute_answer(
                start=1,
                op_ids=op_ids,
                family="revchain",
                modulus=8,
            ),
            family="revchain",
        )

        targets = module.intermediate_answer_targets(
            [modchain, revchain],
            tokenizer=tokenizer,
            max_depth=3,
            modulus=8,
            device=torch.device("cpu"),
        )
        decoded = [
            [tokenizer.decode(item.tolist()) for item in row]
            for row in targets.cpu()
        ]

        mod_expected = [
            module.compute_answer(start=1, op_ids=op_ids[:1], family="modchain", modulus=8),
            module.compute_answer(start=1, op_ids=op_ids[:2], family="modchain", modulus=8),
            module.compute_answer(start=1, op_ids=op_ids[:3], family="modchain", modulus=8),
        ]
        rev_expected = [
            module.compute_answer(start=1, op_ids=op_ids[-1:], family="revchain", modulus=8),
            module.compute_answer(start=1, op_ids=op_ids[-2:], family="revchain", modulus=8),
            module.compute_answer(start=1, op_ids=op_ids[-3:], family="revchain", modulus=8),
        ]

        self.assertEqual(decoded[0], [f"{value:02d}\n" for value in mod_expected])
        self.assertEqual(decoded[1], [f"{value:02d}\n" for value in rev_expected])

    def test_causal_prefix_len_follows_family_order(self):
        module = load_module()
        op_ids = (1, 4, 3)
        mod_case = module.TextReasoningCase(
            case_id="m",
            start=1,
            op_ids=op_ids,
            answer=module.compute_answer(
                start=1,
                op_ids=op_ids,
                family="modchain",
                modulus=8,
            ),
            family="modchain",
        )
        rev_case = module.TextReasoningCase(
            case_id="r",
            start=1,
            op_ids=op_ids,
            answer=module.compute_answer(
                start=1,
                op_ids=op_ids,
                family="revchain",
                modulus=8,
            ),
            family="revchain",
        )

        mod_prefix = module.case_with_causal_prefix_len(
            mod_case,
            prefix_len=1,
            modulus=8,
        )
        rev_prefix = module.case_with_causal_prefix_len(
            rev_case,
            prefix_len=1,
            modulus=8,
        )

        self.assertEqual(mod_prefix.op_ids, (1, module.NOOP_OP_ID, module.NOOP_OP_ID))
        self.assertEqual(rev_prefix.op_ids, (module.NOOP_OP_ID, module.NOOP_OP_ID, 3))
        self.assertEqual(
            mod_prefix.answer,
            module.compute_answer(start=1, op_ids=(1,), family="modchain", modulus=8),
        )
        self.assertEqual(
            rev_prefix.answer,
            module.compute_answer(start=1, op_ids=(3,), family="revchain", modulus=8),
        )

    def test_make_decision_can_require_each_family_to_pass(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--accept-min-exact",
                "0.70",
                "--accept-min-depth-gain",
                "0.10",
                "--accept-min-ablation-drop",
                "0.10",
                "--accept-min-family-exact",
                "0.50",
            ]
        )
        metrics = {
            "think0": {"generation_exact": 0.10},
            "think4": {
                "generation_exact": 0.80,
                "by_family": {
                    "modchain": {"generation_exact": 0.90},
                    "revchain": {"generation_exact": 0.40},
                },
            },
            "state_reset": {"generation_exact": 0.10},
            "op_zero": {"generation_exact": 0.10},
        }

        decision = module.make_decision(metrics, args)

        self.assertFalse(decision["accepted"])
        self.assertIn("family_exact_below_threshold", decision["reject_reasons"])

    def test_accept_decision_label_can_be_overridden_for_l5(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--accepted-decision",
                "accepted_l5_multifamily",
            ]
        )
        metrics = {
            "think0": {"generation_exact": 0.10},
            "think4": {"generation_exact": 0.80, "by_family": {}},
            "state_reset": {"generation_exact": 0.10},
            "op_zero": {"generation_exact": 0.10},
            "coupling_off": {"generation_exact": 0.10},
            "z_l_zero": {"generation_exact": 0.10},
            "z_h_zero": {"generation_exact": 0.10},
        }

        decision = module.make_decision(metrics, args)

        self.assertTrue(decision["accepted"])
        self.assertEqual(decision["decision"], "accepted_l5_multifamily")

    def test_decision_ignores_non_applicable_coupling_ablation_for_single_core(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--accept-min-exact",
                "0.70",
                "--accept-min-depth-gain",
                "0.10",
                "--accept-min-ablation-drop",
                "0.10",
                "--think-structure",
                "single",
            ]
        )
        metrics = {
            "think0": {"generation_exact": 0.10},
            "think4": {"generation_exact": 0.80, "by_family": {}},
            "state_reset": {"generation_exact": 0.10},
            "op_zero": {"generation_exact": 0.10},
            "coupling_off": {"generation_exact": 0.80},
            "z_l_zero": {"generation_exact": 0.80},
            "z_h_zero": {"generation_exact": 0.80},
        }

        decision = module.make_decision(metrics, args)

        self.assertTrue(decision["accepted"])
        self.assertNotIn("coupling_off_generation_exact", decision["decisive_metrics"])
        self.assertNotIn("z_l_zero_generation_exact", decision["decisive_metrics"])

    def test_decision_reports_carrier_off_without_using_it_as_core_ablation_floor(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--accept-min-exact",
                "0.30",
                "--accept-min-depth-gain",
                "0.05",
                "--accept-min-ablation-drop",
                "0.05",
                "--think-structure",
                "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier",
            ]
        )
        metrics = {
            "think0": {"generation_exact": 0.0},
            "think4": {"generation_exact": 0.34, "by_family": {}},
            "state_reset": {"generation_exact": 0.02},
            "op_zero": {"generation_exact": 0.02},
            "z_l_zero": {"generation_exact": 0.0},
            "z_h_zero": {"generation_exact": 0.0},
            "carrier_off": {"generation_exact": 0.335},
        }

        decision = module.make_decision(metrics, args)

        self.assertTrue(decision["accepted"])
        self.assertAlmostEqual(
            decision["decisive_metrics"]["full_minus_worst_ablation"],
            0.32,
        )
        self.assertAlmostEqual(
            decision["decisive_metrics"]["full_minus_carrier_off"],
            0.0050000000000000044,
        )
        self.assertEqual(
            decision["decisive_metrics"]["carrier_off_generation_exact"],
            0.335,
        )

    def test_decision_rejects_when_coupling_or_state_ablation_keeps_score(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--accept-min-exact",
                "0.70",
                "--accept-min-depth-gain",
                "0.10",
                "--accept-min-ablation-drop",
                "0.10",
                "--think-structure",
                "trm_dual_z_coupled",
            ]
        )
        metrics = {
            "think0": {"generation_exact": 0.10},
            "think4": {"generation_exact": 0.80, "by_family": {}},
            "state_reset": {"generation_exact": 0.10},
            "op_zero": {"generation_exact": 0.10},
            "coupling_off": {"generation_exact": 0.10},
            "z_l_zero": {"generation_exact": 0.79},
            "z_h_zero": {"generation_exact": 0.10},
        }

        decision = module.make_decision(metrics, args)

        self.assertFalse(decision["accepted"])
        self.assertIn("ablation_drop_below_threshold", decision["reject_reasons"])
        self.assertEqual(decision["decisive_metrics"]["z_l_zero_generation_exact"], 0.79)

    def test_decision_does_not_require_direct_coupling_ablation_for_interactive_dual_z(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--accept-min-exact",
                "0.70",
                "--accept-min-depth-gain",
                "0.10",
                "--accept-min-ablation-drop",
                "0.10",
                "--think-structure",
                "trm_dual_z_interactive",
            ]
        )
        metrics = {
            "think0": {"generation_exact": 0.10},
            "think4": {"generation_exact": 0.80, "by_family": {}},
            "state_reset": {"generation_exact": 0.10},
            "op_zero": {"generation_exact": 0.10},
            "coupling_off": {"generation_exact": 0.80},
            "z_l_zero": {"generation_exact": 0.10},
            "z_h_zero": {"generation_exact": 0.10},
        }

        decision = module.make_decision(metrics, args)

        self.assertTrue(decision["accepted"])
        self.assertNotIn("coupling_off_generation_exact", decision["decisive_metrics"])
        self.assertIn("z_l_zero_generation_exact", decision["decisive_metrics"])

    def test_decision_can_require_adaptive_halt_nonregression_and_compute_gain(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--accept-min-exact",
                "0.20",
                "--accept-min-depth-gain",
                "0.01",
                "--accept-min-ablation-drop",
                "-1.0",
                "--accept-require-adaptive-halt",
                "--accept-max-adaptive-halt-exact-drop",
                "0.01",
                "--accept-max-mean-halt-steps",
                "2.0",
                "--accept-min-halted-fraction",
                "0.90",
            ]
        )
        metrics = {
            "think0": {"generation_exact": 0.10},
            "think4": {"generation_exact": 0.32, "by_family": {}},
            "state_reset": {"generation_exact": 0.10},
            "op_zero": {"generation_exact": 0.10},
            "adaptive_halt": {
                "generation_exact": 0.32,
                "mean_halt_steps": 1.5,
                "halted_fraction": 0.95,
            },
        }

        decision = module.make_decision(metrics, args)

        self.assertTrue(decision["accepted"])
        self.assertEqual(
            decision["decisive_metrics"]["adaptive_halt_generation_exact"],
            0.32,
        )

    def test_decision_rejects_adaptive_halt_accuracy_or_compute_regression(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--accept-min-exact",
                "0.20",
                "--accept-min-depth-gain",
                "0.01",
                "--accept-min-ablation-drop",
                "-1.0",
                "--accept-require-adaptive-halt",
                "--accept-max-adaptive-halt-exact-drop",
                "0.01",
                "--accept-max-mean-halt-steps",
                "2.0",
                "--accept-min-halted-fraction",
                "0.90",
            ]
        )
        metrics = {
            "think0": {"generation_exact": 0.10},
            "think4": {"generation_exact": 0.32, "by_family": {}},
            "state_reset": {"generation_exact": 0.10},
            "op_zero": {"generation_exact": 0.10},
            "adaptive_halt": {
                "generation_exact": 0.25,
                "mean_halt_steps": 3.5,
                "halted_fraction": 0.50,
            },
        }

        decision = module.make_decision(metrics, args)

        self.assertFalse(decision["accepted"])
        self.assertIn("adaptive_halt_exact_drop_above_threshold", decision["reject_reasons"])
        self.assertIn("adaptive_halt_mean_steps_above_threshold", decision["reject_reasons"])
        self.assertIn("adaptive_halt_fraction_below_threshold", decision["reject_reasons"])

    def test_decision_prefers_generation_telemetry_for_adaptive_halt_compute(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--accept-min-exact",
                "0.20",
                "--accept-min-depth-gain",
                "0.01",
                "--accept-min-ablation-drop",
                "-1.0",
                "--accept-require-adaptive-halt",
                "--accept-max-adaptive-halt-exact-drop",
                "0.01",
                "--accept-max-mean-halt-steps",
                "2.0",
                "--accept-min-halted-fraction",
                "0.90",
            ]
        )
        metrics = {
            "think0": {"generation_exact": 0.10},
            "think4": {"generation_exact": 0.32, "by_family": {}},
            "state_reset": {"generation_exact": 0.10},
            "op_zero": {"generation_exact": 0.10},
            "adaptive_halt": {
                "generation_exact": 0.32,
                "mean_halt_steps": 1.5,
                "halted_fraction": 0.95,
                "generation_mean_executed_think_steps": 3.2,
                "generation_mean_halted_fraction": 0.30,
            },
        }

        decision = module.make_decision(metrics, args)

        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["decisive_metrics"]["adaptive_halt_mean_steps"], 3.2)
        self.assertEqual(decision["decisive_metrics"]["adaptive_halt_halted_fraction"], 0.30)

    def test_target_level_can_be_overridden_for_runner_gates(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--target-level",
                "L5D QTRM-native Mamba3 placement scaled reasoning",
            ]
        )

        self.assertEqual(
            args.target_level,
            "L5D QTRM-native Mamba3 placement scaled reasoning",
        )

    def test_parser_accepts_trm_dual_z_thinking_structure(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(["--think-structure", "trm_dual_z"])

        self.assertEqual(args.think_structure, "trm_dual_z")

    def test_parser_accepts_diffusive_trm_thinking_structure(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--think-structure",
                "trm_dual_z_diffusive",
                "--latent-refine-loss-weight",
                "0.2",
            ]
        )

        self.assertEqual(args.think_structure, "trm_dual_z_diffusive")
        self.assertEqual(args.latent_refine_loss_weight, 0.2)

    def test_parser_accepts_diffusive_reversed_hybrid_3to1_thinking_structure(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--think-structure",
                "trm_dual_z_diffusive_reversed_hybrid_3to1",
                "--latent-refine-loss-weight",
                "0.2",
            ]
        )

        self.assertEqual(
            args.think_structure,
            "trm_dual_z_diffusive_reversed_hybrid_3to1",
        )
        self.assertEqual(args.latent_refine_loss_weight, 0.2)

    def test_parser_accepts_diffusive_reversed_hybrid_3to1_joint_readout(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--think-structure",
                "trm_dual_z_diffusive_reversed_hybrid_3to1_joint_readout",
                "--latent-refine-loss-weight",
                "0.2",
            ]
        )

        self.assertEqual(
            args.think_structure,
            "trm_dual_z_diffusive_reversed_hybrid_3to1_joint_readout",
        )
        self.assertEqual(args.latent_refine_loss_weight, 0.2)

    def test_parser_accepts_reversed_hybrid_3to1_thinking_structure(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--think-structure",
                "trm_dual_z_reversed_hybrid_3to1",
            ]
        )

        self.assertEqual(
            args.think_structure,
            "trm_dual_z_reversed_hybrid_3to1",
        )

    def test_parser_accepts_reversed_mha_etd_thinking_structure(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--think-structure",
                "trm_dual_z_reversed_mha_etd",
            ]
        )

        self.assertEqual(
            args.think_structure,
            "trm_dual_z_reversed_mha_etd",
        )

    def test_parser_accepts_nested_reversed_mha_etd_thinking_structure(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--think-structure",
                "trm_dual_z_nested_reversed_mha_etd",
            ]
        )

        self.assertEqual(
            args.think_structure,
            "trm_dual_z_nested_reversed_mha_etd",
        )

    def test_parser_accepts_nested_reversed_mha_etd_joint_readout_thinking_structure(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--think-structure",
                "trm_dual_z_nested_reversed_mha_etd_joint_readout",
            ]
        )

        self.assertEqual(
            args.think_structure,
            "trm_dual_z_nested_reversed_mha_etd_joint_readout",
        )

    def test_parser_accepts_nested_reversed_mha_etd_residual_joint_readout_thinking_structure(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--think-structure",
                "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout",
            ]
        )

        self.assertEqual(
            args.think_structure,
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout",
        )

    def test_parser_accepts_nested_residual_joint_readout_core_carrier_thinking_structure(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--think-structure",
                "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier",
            ]
        )

        self.assertEqual(
            args.think_structure,
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier",
        )

    def test_parser_accepts_nested_official_schedule_split_mixer_3to1_thinking_structure(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--think-structure",
                "trm_dual_z_nested_official_schedule_split_mixer_3to1",
            ]
        )

        self.assertEqual(
            args.think_structure,
            "trm_dual_z_nested_official_schedule_split_mixer_3to1",
        )

    def test_parser_accepts_train_only_resume_missing_params(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--train-only-resume-missing-params",
            ]
        )

        self.assertTrue(args.train_only_resume_missing_params)

    def test_parser_accepts_train_param_name_regex(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--train-param-name-regex",
                "^trm_",
            ]
        )

        self.assertEqual(args.train_param_name_regex, "^trm_")

    def test_parser_accepts_z_l_counterfactual_loss_args(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--z-l-counterfactual-loss-weight",
                "0.2",
                "--z-l-counterfactual-margin",
                "0.5",
                "--z-l-counterfactual-every",
                "3",
            ]
        )

        self.assertEqual(args.z_l_counterfactual_loss_weight, 0.2)
        self.assertEqual(args.z_l_counterfactual_margin, 0.5)
        self.assertEqual(args.z_l_counterfactual_every, 3)

    def test_parser_accepts_fast_slow_latent_loss_args(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--fast-slow-latent-loss-weight",
                "0.3",
                "--fast-slow-latent-every",
                "4",
                "--fast-slow-z-l-margin",
                "0.7",
                "--fast-slow-z-h-margin",
                "0.2",
                "--fast-slow-z-l-weight",
                "2.0",
                "--fast-slow-z-h-weight",
                "0.5",
            ]
        )

        self.assertEqual(args.fast_slow_latent_loss_weight, 0.3)
        self.assertEqual(args.fast_slow_latent_every, 4)
        self.assertEqual(args.fast_slow_z_l_margin, 0.7)
        self.assertEqual(args.fast_slow_z_h_margin, 0.2)
        self.assertEqual(args.fast_slow_z_l_weight, 2.0)
        self.assertEqual(args.fast_slow_z_h_weight, 0.5)

    def test_fast_slow_latent_counterfactual_loss_penalizes_missing_fast_margin(self):
        module = load_module()

        class DummyModel:
            def __init__(self):
                self.calls = []

            def __call__(self, input_ids, *, think_steps, z_l_zero=False, z_h_zero=False):
                self.calls.append((int(think_steps), bool(z_l_zero), bool(z_h_zero)))
                logits = torch.zeros((1, 3, 4), dtype=torch.float32)
                if z_h_zero:
                    logits[:, 1, 2] = -10.0
                return logits

        model = DummyModel()
        chosen_logits = torch.zeros((1, 3, 4), dtype=torch.float32)
        targets = torch.tensor([[0, 2, 0]], dtype=torch.long)
        loss = module.fast_slow_latent_counterfactual_loss(
            model,
            chosen_logits,
            targets,
            torch.tensor([[0, 1, 2]], dtype=torch.long),
            prompt_len=2,
            answer_len=1,
            think_steps=3,
            z_l_margin=0.6,
            z_h_margin=0.6,
            z_l_weight=2.0,
            z_h_weight=1.0,
        )

        self.assertAlmostEqual(float(loss.item()), 0.4, places=4)
        self.assertIn((3, True, False), model.calls)
        self.assertIn((3, False, True), model.calls)

    def test_parser_accepts_reversed_hybrid_semantic_carry_thinking_structure(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--think-structure",
                "trm_dual_z_reversed_hybrid_3to1_semantic_carry",
            ]
        )

        self.assertEqual(
            args.think_structure,
            "trm_dual_z_reversed_hybrid_3to1_semantic_carry",
        )

    def test_parser_accepts_reversed_hybrid_order_router_thinking_structure(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--think-structure",
                "trm_dual_z_reversed_hybrid_3to1_order_router",
            ]
        )

        self.assertEqual(
            args.think_structure,
            "trm_dual_z_reversed_hybrid_3to1_order_router",
        )

    def test_parser_accepts_reversed_hybrid_recent_order_router_thinking_structure(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--think-structure",
                "trm_dual_z_reversed_hybrid_3to1_recent_order_router",
            ]
        )

        self.assertEqual(
            args.think_structure,
            "trm_dual_z_reversed_hybrid_3to1_recent_order_router",
        )

    def test_parser_accepts_reversed_hybrid_state_gru_order_router_thinking_structure(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--think-structure",
                "trm_dual_z_reversed_hybrid_3to1_state_gru_order_router",
            ]
        )

        self.assertEqual(
            args.think_structure,
            "trm_dual_z_reversed_hybrid_3to1_state_gru_order_router",
        )

    def test_parser_accepts_reversed_hybrid_transition_state_order_router_thinking_structure(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--think-structure",
                "trm_dual_z_reversed_hybrid_3to1_transition_state_order_router",
            ]
        )

        self.assertEqual(
            args.think_structure,
            "trm_dual_z_reversed_hybrid_3to1_transition_state_order_router",
        )

    def test_order_router_probe_reports_by_family_routes(self):
        module = load_module()
        cases = [
            module.TextReasoningCase(
                case_id="m",
                start=3,
                op_ids=(1, 4),
                answer=5,
                family="modchain",
            ),
            module.TextReasoningCase(
                case_id="r",
                start=3,
                op_ids=(1, 4),
                answer=7,
                family="revchain",
            ),
        ]
        tokenizer = module.CharTokenizer.from_texts(
            [
                module.case_full_text(case, include_family_tag=True)
                for case in cases
            ],
            mode="number",
            number_max_value=31,
        )
        model = module.NativeQTRMETDLM(
            vocab=tokenizer.vocab_size,
            max_seq_len=64,
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            d_ff=32,
            dropout=0.0,
            backbone="trm_official",
            encode_backbone="mha_etd",
            think_backbone="trm_official",
            decode_backbone="mha_etd",
            think_structure="trm_dual_z_reversed_hybrid_3to1_order_router",
            trm_l_cycles=1,
        )
        with torch.no_grad():
            model.trm_order_router.weight.zero_()
            model.trm_order_router.bias.zero_()
        args = module.build_arg_parser().parse_args(
            [
                "--device",
                "cpu",
                "--task-families",
                "modchain,revchain",
                "--eval-task-families",
                "modchain,revchain",
                "--batch-size",
                "1",
            ]
        )

        metrics = module.order_router_probe_metrics(
            model,
            cases,
            args,
            tokenizer=tokenizer,
        )

        self.assertTrue(metrics["available"])
        self.assertEqual(metrics["route_names"], ["l_then_h", "h_then_l_then_h"])
        self.assertAlmostEqual(metrics["last_lh_prob"], 0.5, places=6)
        self.assertAlmostEqual(metrics["last_hlh_prob"], 0.5, places=6)
        self.assertIn("modchain", metrics["by_family"])
        self.assertIn("revchain", metrics["by_family"])
        self.assertEqual(metrics["by_family"]["modchain"]["count"], 1)
        self.assertEqual(metrics["by_active_len"]["2"]["count"], 2)

    def test_order_router_route_ablation_forces_route_and_restores_model(self):
        module = load_module()
        cases = [
            module.TextReasoningCase(
                case_id="m",
                start=3,
                op_ids=(1, 4),
                answer=5,
                family="modchain",
            )
        ]
        tokenizer = module.CharTokenizer.from_texts(
            [module.case_full_text(cases[0], include_family_tag=True)],
            mode="number",
            number_max_value=31,
        )
        model = module.NativeQTRMETDLM(
            vocab=tokenizer.vocab_size,
            max_seq_len=64,
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            d_ff=32,
            dropout=0.0,
            backbone="trm_official",
            encode_backbone="mha_etd",
            think_backbone="trm_official",
            decode_backbone="mha_etd",
            think_structure="trm_dual_z_reversed_hybrid_3to1_order_router",
            trm_l_cycles=1,
        )
        args = module.build_arg_parser().parse_args(
            [
                "--device",
                "cpu",
                "--task-families",
                "modchain",
                "--eval-task-families",
                "modchain",
                "--modulus",
                "8",
                "--program-len",
                "2",
            ]
        )

        route0 = module.evaluate(
            model,
            cases,
            args,
            tokenizer=tokenizer,
            think_steps=1,
            ablation="order_route0",
        )
        route1 = module.evaluate(
            model,
            cases,
            args,
            tokenizer=tokenizer,
            think_steps=1,
            ablation="order_route1",
        )

        self.assertEqual(route0["ablation"], "order_route0")
        self.assertEqual(route1["ablation"], "order_route1")
        self.assertFalse(hasattr(model, "trm_order_router_force_route"))

    def test_order_router_family_order_targets_mark_revchain_reverse_route(self):
        module = load_module()
        cases = [
            module.TextReasoningCase(
                case_id="c",
                start=0,
                op_ids=(1,),
                answer=0,
                family="checksum",
            ),
            module.TextReasoningCase(
                case_id="m",
                start=0,
                op_ids=(1,),
                answer=0,
                family="modchain",
            ),
            module.TextReasoningCase(
                case_id="r",
                start=0,
                op_ids=(1,),
                answer=0,
                family="revchain",
            ),
        ]

        targets = module.order_router_family_order_targets(
            cases,
            device=torch.device("cpu"),
        )

        self.assertEqual(targets.tolist(), [0, 0, 1])

    def test_order_router_family_order_loss_trains_router_without_answer_sidecar(self):
        module = load_module()
        cases = [
            module.TextReasoningCase(
                case_id="m",
                start=3,
                op_ids=(1, 4),
                answer=5,
                family="modchain",
            ),
            module.TextReasoningCase(
                case_id="r",
                start=3,
                op_ids=(1, 4),
                answer=7,
                family="revchain",
            ),
        ]
        tokenizer = module.CharTokenizer.from_texts(
            [
                module.case_full_text(case, include_family_tag=True)
                for case in cases
            ],
            mode="number",
            number_max_value=31,
        )
        model = module.NativeQTRMETDLM(
            vocab=tokenizer.vocab_size,
            max_seq_len=64,
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            d_ff=32,
            dropout=0.0,
            backbone="trm_official",
            encode_backbone="mha_etd",
            think_backbone="trm_official",
            decode_backbone="mha_etd",
            think_structure="trm_dual_z_reversed_hybrid_3to1_order_router",
            trm_l_cycles=1,
        )

        loss = module.order_router_family_order_loss(
            model,
            cases,
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            include_family_tag=True,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(loss.requires_grad)
        loss.backward()
        self.assertIsNotNone(model.trm_order_router.weight.grad)

    def test_order_router_family_order_loss_supports_nested_router(self):
        module = load_module()
        cases = [
            module.TextReasoningCase(
                case_id="m",
                start=3,
                op_ids=(1, 4),
                answer=5,
                family="modchain",
            ),
            module.TextReasoningCase(
                case_id="r",
                start=3,
                op_ids=(1, 4),
                answer=7,
                family="revchain",
            ),
        ]
        tokenizer = module.CharTokenizer.from_texts(
            [
                module.case_full_text(case, include_family_tag=True)
                for case in cases
            ],
            mode="number",
            number_max_value=31,
        )
        model = module.NativeQTRMETDLM(
            vocab=tokenizer.vocab_size,
            max_seq_len=64,
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            d_ff=32,
            dropout=0.0,
            backbone="mha_etd",
            encode_backbone="mha_etd",
            think_backbone="mha_etd",
            decode_backbone="mha_etd",
            think_structure=(
                "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_"
                "order_router"
            ),
            trm_l_cycles=1,
        )

        loss = module.order_router_family_order_loss(
            model,
            cases,
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            include_family_tag=True,
        )

        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertFalse(hasattr(model, "trm_order_router"))
        self.assertIsNotNone(model.trm_nested_order_router.weight.grad)

    def test_sequence_level_nested_order_router_uses_one_route_distribution_per_case(self):
        module = load_module()
        cases = [
            module.TextReasoningCase(
                case_id="m",
                start=3,
                op_ids=(1, 4),
                answer=5,
                family="modchain",
            )
        ]
        tokenizer = module.CharTokenizer.from_texts(
            [module.case_full_text(cases[0], include_family_tag=True)],
            mode="number",
            number_max_value=31,
        )
        model = module.NativeQTRMETDLM(
            vocab=tokenizer.vocab_size,
            max_seq_len=64,
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            d_ff=32,
            dropout=0.0,
            backbone="mha_etd",
            encode_backbone="mha_etd",
            think_backbone="mha_etd",
            decode_backbone="mha_etd",
            think_structure=(
                "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_"
                "sequence_order_router"
            ),
            trm_l_cycles=1,
        )
        prompt = module.case_prompt(cases[0], include_family_tag=True)
        input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)

        logits = module._order_router_encoded_logits(model, input_ids)

        self.assertEqual(tuple(logits.shape[-1:]), (2,))
        self.assertTrue(torch.allclose(logits[:, :1, :], logits))

    def test_forced_route_answer_loss_trains_candidate_without_router_grad(self):
        module = load_module()
        cases = [
            module.TextReasoningCase(
                case_id="m",
                start=3,
                op_ids=(1, 4),
                answer=5,
                family="modchain",
            ),
            module.TextReasoningCase(
                case_id="r",
                start=3,
                op_ids=(1, 4),
                answer=7,
                family="revchain",
            ),
        ]
        tokenizer = module.CharTokenizer.from_texts(
            [
                module.case_full_text(case, include_family_tag=True)
                for case in cases
            ],
            mode="number",
            number_max_value=31,
        )
        model = module.NativeQTRMETDLM(
            vocab=tokenizer.vocab_size,
            max_seq_len=64,
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            d_ff=32,
            dropout=0.0,
            backbone="trm_official",
            encode_backbone="mha_etd",
            think_backbone="trm_official",
            decode_backbone="mha_etd",
            think_structure="trm_dual_z_reversed_hybrid_3to1_order_router",
            trm_l_cycles=1,
        )

        loss = module.forced_route_answer_loss(
            model,
            cases,
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            include_family_tag=True,
            route=1,
            families=("revchain",),
            think_steps=1,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(loss.requires_grad)
        loss.backward()
        self.assertFalse(hasattr(model, "trm_order_router_force_route"))
        self.assertIsNone(model.trm_order_router.weight.grad)
        self.assertIsNotNone(model.lm_head.weight.grad)

    def test_forced_route_answer_loss_supports_nested_router_force_attr(self):
        module = load_module()
        cases = [
            module.TextReasoningCase(
                case_id="r",
                start=3,
                op_ids=(1, 4),
                answer=7,
                family="revchain",
            )
        ]
        tokenizer = module.CharTokenizer.from_texts(
            [module.case_full_text(cases[0], include_family_tag=True)],
            mode="number",
            number_max_value=31,
        )
        model = module.NativeQTRMETDLM(
            vocab=tokenizer.vocab_size,
            max_seq_len=64,
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            d_ff=32,
            dropout=0.0,
            backbone="mha_etd",
            encode_backbone="mha_etd",
            think_backbone="mha_etd",
            decode_backbone="mha_etd",
            think_structure=(
                "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_"
                "order_router"
            ),
            trm_l_cycles=1,
        )

        loss = module.forced_route_answer_loss(
            model,
            cases,
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            include_family_tag=True,
            route=1,
            families=("revchain",),
            think_steps=1,
        )

        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertFalse(hasattr(model, "trm_nested_order_router_force_route"))
        self.assertIsNone(model.trm_nested_order_router.weight.grad)
        self.assertIsNotNone(model.lm_head.weight.grad)

    def test_forced_route_depth_loss_trains_stepwise_candidate_without_router_grad(self):
        module = load_module()
        cases = [
            module.TextReasoningCase(
                case_id="r",
                start=3,
                op_ids=(1, 4),
                answer=7,
                family="revchain",
            )
        ]
        tokenizer = module.CharTokenizer.from_texts(
            [
                module.case_full_text(case, include_family_tag=True)
                for case in cases
            ],
            mode="number",
            number_max_value=31,
        )
        model = module.NativeQTRMETDLM(
            vocab=tokenizer.vocab_size,
            max_seq_len=64,
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            d_ff=32,
            dropout=0.0,
            backbone="trm_official",
            encode_backbone="mha_etd",
            think_backbone="trm_official",
            decode_backbone="mha_etd",
            think_structure="trm_dual_z_reversed_hybrid_3to1_order_router",
            trm_l_cycles=1,
        )

        loss = module.forced_route_intermediate_depth_loss(
            model,
            cases,
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            include_family_tag=True,
            route=1,
            families=("revchain",),
            max_depth=2,
            modulus=32,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(loss.requires_grad)
        loss.backward()
        self.assertFalse(hasattr(model, "trm_order_router_force_route"))
        self.assertIsNone(model.trm_order_router.weight.grad)
        self.assertIsNotNone(model.lm_head.weight.grad)

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

    def test_tokenizer_covers_prompt_and_answer(self):
        module = load_module()
        tokenizer = module.CharTokenizer.from_texts(["start 03 ops 01 answer 05\n"])

        encoded = tokenizer.encode("answer 05\n")

        self.assertEqual(tokenizer.decode(encoded), "answer 05\n")

    def test_parser_accepts_adaptive_halt_eval_args(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--adaptive-halt-eval",
                "--halt-threshold",
                "0.75",
                "--halt-min-steps",
                "2",
                "--adaptive-halt-loss-weight",
                "0.25",
            ]
        )

        self.assertTrue(args.adaptive_halt_eval)
        self.assertEqual(args.halt_threshold, 0.75)
        self.assertEqual(args.halt_min_steps, 2)
        self.assertEqual(args.adaptive_halt_loss_weight, 0.25)

    def test_adaptive_halt_eval_reports_runtime_telemetry(self):
        module = load_module()
        cases = module.build_cases(
            count=2,
            seed=1,
            program_len=2,
            modulus=8,
            families=("modchain",),
        )
        tokenizer = module.CharTokenizer.from_texts(
            [module.case_full_text(case) for case in cases]
        )
        model = module.NativeQTRMETDLM(
            vocab=tokenizer.vocab_size,
            max_seq_len=32,
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            d_ff=32,
            dropout=0.0,
            backbone="trm_official",
            think_structure="trm_dual_z",
        )
        with torch.no_grad():
            model.core_halt_head.weight.zero_()
            model.core_halt_head.bias.fill_(10.0)
        args = module.build_arg_parser().parse_args(
            [
                "--device",
                "cpu",
                "--modulus",
                "8",
                "--program-len",
                "2",
                "--eval-think-steps",
                "4",
                "--halt-threshold",
                "0.5",
                "--halt-min-steps",
                "1",
            ]
        )

        metrics = module.evaluate(
            model,
            cases,
            args,
            tokenizer=tokenizer,
            think_steps=4,
            ablation="adaptive_halt",
        )

        self.assertEqual(metrics["mean_halt_steps"], 1.0)
        self.assertEqual(metrics["executed_think_steps"], 1)
        self.assertEqual(metrics["halted_fraction"], 1.0)
        self.assertIn("2", metrics["by_active_len"])
        self.assertEqual(metrics["by_active_len"]["2"]["total"], 2)
        self.assertIn("generation_format_valid", metrics)
        self.assertIn("generation_format_valid", metrics["by_active_len"]["2"])
        self.assertIn("generation_format_valid", metrics["by_family"]["modchain"])
        self.assertIn("format_valid", metrics["examples"][0])
        self.assertIn("teacher_forced_token_accuracy", metrics)
        self.assertIn("teacher_forced_sequence_exact", metrics)
        self.assertIn("teacher_forced_mean_token_rank", metrics)
        self.assertIn("teacher_forced_token_top3", metrics)
        self.assertIn("teacher_forced_token_top5", metrics)
        self.assertEqual(metrics["generation_mean_halt_steps"], 1.0)
        self.assertEqual(metrics["generation_mean_executed_think_steps"], 1.0)
        self.assertEqual(metrics["halt_by_active_len"]["2"]["mean_halt_steps"], 1.0)
        self.assertEqual(metrics["halt_by_active_len"]["2"]["halted_fraction"], 1.0)

    def test_teacher_depth_halt_targets_start_at_earliest_eligible_correct_depth(self):
        module = load_module()
        correctness = torch.tensor(
            [
                [False, True, True, False],
                [False, False, False, False],
                [True, False, False, False],
                [False, False, True, True],
            ]
        )

        targets = module.teacher_depth_halt_targets_from_correctness(
            correctness,
            min_halt_step=2,
        )

        self.assertTrue(
            torch.equal(
                targets,
                torch.tensor(
                    [
                        [0.0, 1.0, 1.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 1.0],
                    ]
                ),
            )
        )

    def test_effective_program_len_uses_trailing_noop_tail(self):
        module = load_module()

        self.assertEqual(
            module.effective_program_len(
                module.TextReasoningCase(
                    case_id="partial",
                    start=0,
                    op_ids=(1, 2, module.NOOP_OP_ID, module.NOOP_OP_ID),
                    answer=0,
                )
            ),
            2,
        )
        self.assertEqual(
            module.effective_program_len(
                module.TextReasoningCase(
                    case_id="zero",
                    start=0,
                    op_ids=(module.NOOP_OP_ID, module.NOOP_OP_ID),
                    answer=0,
                )
            ),
            0,
        )

    def test_active_len_halt_targets_mark_first_required_depth(self):
        module = load_module()
        active_lengths = torch.tensor([0, 1, 3, 8])

        targets = module.active_len_halt_targets(
            active_lengths,
            max_depth=4,
            min_halt_step=2,
        )

        self.assertTrue(
            torch.equal(
                targets,
                torch.tensor(
                    [
                        [0.0, 1.0, 1.0, 1.0],
                        [0.0, 1.0, 1.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            )
        )

    def test_active_len_first_halt_targets_are_one_hot(self):
        module = load_module()
        active_lengths = torch.tensor([0, 1, 3, 8])

        targets = module.active_len_first_halt_targets(
            active_lengths,
            max_depth=4,
            min_halt_step=2,
        )

        self.assertTrue(
            torch.equal(
                targets,
                torch.tensor(
                    [
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            )
        )

    def test_intermediate_depth_loss_can_focus_on_late_depths(self):
        module = load_module()

        class RecordingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = []

            def forward(self, input_ids, *, think_steps):
                self.calls.append(int(think_steps))
                logits = torch.zeros(
                    input_ids.shape[0],
                    input_ids.shape[1],
                    4,
                    dtype=torch.float32,
                )
                logits[:, :, 0] = float(think_steps)
                return logits

        model = RecordingModel()
        input_ids = torch.zeros((2, 6), dtype=torch.long)
        targets = torch.zeros((2, 4, 2), dtype=torch.long)

        loss = module.intermediate_depth_loss(
            model,
            input_ids,
            targets,
            prompt_len=3,
            answer_len=2,
            max_depth=4,
            min_depth=3,
            depth_weight_power=1.0,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(model.calls, [3, 4])

    def test_intermediate_depth_family_dro_loss_focuses_worst_family(self):
        module = load_module()

        class FixedModel(torch.nn.Module):
            def forward(self, input_ids, *, think_steps):
                logits = torch.zeros(
                    input_ids.shape[0],
                    input_ids.shape[1],
                    2,
                    dtype=torch.float32,
                )
                logits[0, :, 0] = 5.0
                logits[0, :, 1] = -5.0
                logits[1, :, 0] = -5.0
                logits[1, :, 1] = 5.0
                return logits

        cases = [
            module.TextReasoningCase(
                case_id="easy",
                start=0,
                op_ids=(1,),
                answer=0,
                family="checksum",
            ),
            module.TextReasoningCase(
                case_id="hard",
                start=0,
                op_ids=(1,),
                answer=0,
                family="revchain",
            ),
        ]
        input_ids = torch.zeros((2, 2), dtype=torch.long)
        targets = torch.zeros((2, 1, 1), dtype=torch.long)
        loss = module.intermediate_depth_family_dro_loss(
            FixedModel(),
            input_ids,
            targets,
            cases,
            prompt_len=1,
            answer_len=1,
            max_depth=1,
            temperature=0.0,
        )
        logits = FixedModel()(input_ids, think_steps=1)
        per_case = module.answer_case_losses(
            logits,
            targets[:, 0, :],
            prompt_len=1,
            answer_len=1,
        )

        self.assertAlmostEqual(float(loss), float(per_case.max()), places=6)
        self.assertGreater(float(loss), float(per_case.mean()))

    def test_state_trace_depth_answer_loss_reads_traced_states(self):
        module = load_module()

        class TraceModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.decode = torch.nn.Identity()
                self.norm = torch.nn.Identity()
                self.lm_head = torch.nn.Identity()

            def _causal_mask(self, seq_len, device):
                return torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)

            def _run_stage(self, stage, state, *, causal_mask):
                return state

        cases = [
            module.TextReasoningCase(
                case_id="easy",
                start=0,
                op_ids=(1, 2),
                answer=0,
                family="checksum",
            ),
            module.TextReasoningCase(
                case_id="hard",
                start=0,
                op_ids=(1, 2),
                answer=0,
                family="revchain",
            ),
        ]
        trace_h = torch.zeros((2, 2, 1, 3), dtype=torch.float32)
        trace_h[0, :, 0, 0] = 6.0
        trace_h[0, :, 0, 1:] = -6.0
        trace_h[1, :, 0, 0] = -6.0
        trace_h[1, :, 0, 1] = 6.0
        targets = torch.zeros((2, 2, 1), dtype=torch.long)

        mean_loss = module.state_trace_depth_answer_loss_from_runtime(
            TraceModel(),
            {"core_state_trace_h": trace_h},
            targets,
            cases,
            prompt_len=1,
            answer_len=1,
            max_depth=2,
            state_source="h",
        )
        dro_loss = module.state_trace_depth_answer_loss_from_runtime(
            TraceModel(),
            {"core_state_trace_h": trace_h},
            targets,
            cases,
            prompt_len=1,
            answer_len=1,
            max_depth=2,
            state_source="h",
            family_dro=True,
        )

        self.assertTrue(torch.isfinite(mean_loss))
        self.assertTrue(torch.isfinite(dro_loss))
        self.assertGreater(float(dro_loss), float(mean_loss))

    def test_state_trace_depth_answer_loss_uses_shared_lm_logits_path(self):
        module = load_module()

        class BadHead(torch.nn.Module):
            def forward(self, hidden):
                logits = torch.zeros((*hidden.shape[:-1], 3), dtype=hidden.dtype)
                logits[..., 1] = 20.0
                return logits

        class TraceModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.decode = torch.nn.Identity()
                self.norm = torch.nn.Identity()
                self.lm_head = BadHead()
                self.shared_calls = 0

            def _causal_mask(self, seq_len, device):
                return torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)

            def _run_stage(self, stage, state, *, causal_mask):
                return state

            def _lm_logits(self, hidden):
                self.shared_calls += 1
                logits = torch.zeros((*hidden.shape[:-1], 3), dtype=hidden.dtype)
                logits[..., 0] = 20.0
                return logits

        trace_h = torch.zeros((1, 1, 1, 3), dtype=torch.float32, requires_grad=True)
        targets = torch.zeros((1, 1, 1), dtype=torch.long)
        loss = module.state_trace_depth_answer_loss_from_runtime(
            TraceModel(),
            {"core_state_trace_h": trace_h},
            targets,
            [
                module.TextReasoningCase(
                    case_id="x",
                    start=0,
                    op_ids=(1,),
                    answer=0,
                )
            ],
            prompt_len=1,
            answer_len=1,
            max_depth=1,
            state_source="h",
        )

        self.assertLess(float(loss.item()), 1e-4)

    def test_prefix_depth_anchor_loss_trains_noop_suffix_prefixes(self):
        module = load_module()

        class RecordingModel(torch.nn.Module):
            def __init__(self, vocab_size: int):
                super().__init__()
                self.calls = []
                self.vocab_size = int(vocab_size)

            def forward(self, input_ids, *, think_steps):
                self.calls.append(int(think_steps))
                return torch.zeros(
                    input_ids.shape[0],
                    input_ids.shape[1],
                    self.vocab_size,
                    dtype=torch.float32,
                )

        base = module.TextReasoningCase(
            case_id="x",
            start=3,
            op_ids=(1, 4, 2),
            answer=0,
        )
        prefix_cases = [
            module.case_with_active_program_len(base, active_len=depth, modulus=8)
            for depth in range(1, 4)
        ]
        tokenizer = module.CharTokenizer.from_texts(
            [module.case_full_text(case) for case in prefix_cases]
        )
        model = RecordingModel(tokenizer.vocab_size)

        loss = module.prefix_depth_anchor_loss(
            model,
            [base],
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            include_family_tag=False,
            max_depth=3,
            modulus=8,
            min_depth=2,
            depth_weight_power=1.0,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(model.calls, [2, 3])

    def test_generation_operation_breakdown_groups_failures_by_ops(self):
        module = load_module()
        case_a = module.TextReasoningCase(
            case_id="a",
            start=3,
            op_ids=(1, 4, module.NOOP_OP_ID),
            answer=5,
        )
        case_b = module.TextReasoningCase(
            case_id="b",
            start=1,
            op_ids=(2, 4, 3),
            answer=7,
        )

        breakdown = module.generation_operation_breakdown(
            [case_a, case_b],
            ["05\n", "09\n"],
            modulus=32,
        )

        self.assertEqual(breakdown["by_last_op"]["04"]["total"], 1)
        self.assertEqual(breakdown["by_last_op"]["03"]["generation_exact"], 0.0)
        self.assertEqual(breakdown["by_position_op"]["2:04"]["total"], 2)
        self.assertEqual(breakdown["by_error_delta"]["02"]["total"], 1)

    def test_generation_operation_breakdown_keeps_family_slices(self):
        module = load_module()
        mod_case = module.TextReasoningCase(
            case_id="m",
            start=3,
            op_ids=(1, 4),
            answer=5,
            family="modchain",
        )
        rev_case = module.TextReasoningCase(
            case_id="r",
            start=3,
            op_ids=(1, 4),
            answer=7,
            family="revchain",
        )

        breakdown = module.generation_operation_breakdown(
            [mod_case, rev_case],
            ["05\n", "09\n"],
            modulus=32,
        )

        self.assertEqual(
            breakdown["by_family"]["modchain"]["by_last_op"]["04"]["generation_exact"],
            1.0,
        )
        self.assertEqual(
            breakdown["by_family"]["revchain"]["by_last_op"]["04"]["generation_exact"],
            0.0,
        )
        self.assertEqual(
            breakdown["by_family"]["revchain"]["by_error_delta"]["02"]["total"],
            1,
        )

    def test_residue_auxiliary_loss_uses_fixed_width_answer_tags(self):
        module = load_module()

        class RecordingModel(torch.nn.Module):
            def __init__(self, vocab_size: int):
                super().__init__()
                self.calls = []
                self.vocab_size = int(vocab_size)

            def forward(self, input_ids, *, think_steps):
                self.calls.append((int(think_steps), tuple(input_ids.shape)))
                return torch.zeros(
                    input_ids.shape[0],
                    input_ids.shape[1],
                    self.vocab_size,
                    dtype=torch.float32,
                )

        case = module.TextReasoningCase(
            case_id="x",
            start=3,
            op_ids=(1, 4),
            answer=7,
        )
        texts = [
            module.case_full_text(case),
            module.case_prompt_with_answer_label(case, answer_label="answer2")
            + "01\n",
            module.case_prompt_with_answer_label(case, answer_label="answer4")
            + "03\n",
        ]
        tokenizer = module.CharTokenizer.from_texts(texts)
        model = RecordingModel(tokenizer.vocab_size)

        loss = module.residue_auxiliary_loss(
            model,
            [case],
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            include_family_tag=False,
            think_steps=3,
            residue_moduli=(2, 4),
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual([call[0] for call in model.calls], [3, 3])
        self.assertEqual(model.calls[0][1], model.calls[1][1])

    def test_family_dro_answer_loss_focuses_worst_family(self):
        module = load_module()
        logits = torch.tensor(
            [
                [[5.0, -5.0], [0.0, 0.0]],
                [[-5.0, 5.0], [0.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        targets = torch.tensor(
            [
                [0, 0],
                [0, 0],
            ],
            dtype=torch.long,
        )
        cases = [
            module.TextReasoningCase(
                case_id="easy",
                start=0,
                op_ids=(1,),
                answer=0,
                family="checksum",
            ),
            module.TextReasoningCase(
                case_id="hard",
                start=0,
                op_ids=(1,),
                answer=0,
                family="revchain",
            ),
        ]

        per_case = module.answer_case_losses(
            logits,
            targets,
            prompt_len=1,
            answer_len=1,
        )
        dro = module.family_dro_answer_loss(
            logits,
            targets,
            cases,
            prompt_len=1,
            answer_len=1,
            temperature=0.0,
        )

        self.assertAlmostEqual(float(dro), float(per_case.max()), places=6)
        self.assertGreater(float(dro), float(per_case.mean()))

    def test_generate_answer_beam_keeps_non_greedy_candidates(self):
        module = load_module()

        class BranchingModel(torch.nn.Module):
            def __init__(self, prompt_len: int):
                super().__init__()
                self.prompt_len = int(prompt_len)

            def forward(self, input_ids, *, think_steps):
                logits = torch.full(
                    (input_ids.shape[0], input_ids.shape[1], 4),
                    -10.0,
                    dtype=torch.float32,
                )
                generated = int(input_ids.shape[1]) - self.prompt_len
                if generated == 0:
                    logits[:, -1, 1] = 2.0
                    logits[:, -1, 2] = 1.0
                elif int(input_ids[0, -1]) == 2:
                    logits[:, -1, 3] = 3.0
                else:
                    logits[:, -1, 0] = 3.0
                return logits

        prompt_ids = torch.tensor([[0, 0]], dtype=torch.long)
        beams = module.generate_answer_beam(
            BranchingModel(prompt_len=2),
            prompt_ids,
            answer_len=2,
            think_steps=4,
            beam_width=2,
        )

        self.assertIn([2, 3], [item["token_ids"] for item in beams])

    def test_sequence_preference_loss_penalizes_rejected_answers(self):
        module = load_module()

        class PreferenceModel(torch.nn.Module):
            def forward(self, input_ids, *, think_steps):
                logits = torch.zeros(
                    input_ids.shape[0],
                    input_ids.shape[1],
                    16,
                    dtype=torch.float32,
                )
                logits[:, :, 0] = 1.0
                return logits

        case = module.TextReasoningCase(
            case_id="x",
            start=3,
            op_ids=(1, 4),
            answer=5,
        )
        rejected = module.TextReasoningCase(
            case_id="x",
            start=3,
            op_ids=(1, 4),
            answer=7,
        )
        tokenizer = module.CharTokenizer.from_texts(
            [module.case_full_text(case), module.case_full_text(rejected)]
        )
        x, y, prompt_len, answer_len = module.cases_to_batch(
            [case],
            tokenizer=tokenizer,
            device=torch.device("cpu"),
        )
        model = PreferenceModel()
        chosen_logits = model(x, think_steps=2)

        loss = module.sequence_preference_loss(
            model,
            chosen_logits,
            y,
            [case],
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            include_family_tag=False,
            prompt_len=prompt_len,
            answer_len=answer_len,
            think_steps=2,
            modulus=8,
            deltas=(2,),
            margin=0.5,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(loss), 0.0)

    def test_operation_counterfactual_loss_uses_zero_ops_prompt_with_gold_answer(self):
        module = load_module()

        class RecordingModel(torch.nn.Module):
            def __init__(self, vocab_size: int):
                super().__init__()
                self.vocab_size = int(vocab_size)
                self.calls: list[torch.Tensor] = []

            def forward(self, input_ids, *, think_steps):
                self.calls.append(input_ids.detach().cpu().clone())
                logits = torch.zeros(
                    input_ids.shape[0],
                    input_ids.shape[1],
                    self.vocab_size,
                    dtype=torch.float32,
                )
                return logits

        case = module.TextReasoningCase(
            case_id="x",
            start=3,
            op_ids=(1, 4),
            answer=5,
        )
        zero_prompt_text = module.case_prompt(
            module.zero_ops_case(case, modulus=8)
        ) + module.case_answer(case)
        tokenizer = module.CharTokenizer.from_texts(
            [module.case_full_text(case), zero_prompt_text]
        )
        x, y, prompt_len, answer_len = module.cases_to_batch(
            [case],
            tokenizer=tokenizer,
            device=torch.device("cpu"),
        )
        chosen_logits = torch.zeros(
            x.shape[0],
            x.shape[1],
            tokenizer.vocab_size,
            dtype=torch.float32,
        )
        model = RecordingModel(tokenizer.vocab_size)

        loss = module.operation_counterfactual_loss(
            model,
            chosen_logits,
            y,
            [case],
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            include_family_tag=False,
            state_anchor=False,
            state_anchor_position="before_answer",
            prompt_len=prompt_len,
            answer_len=answer_len,
            think_steps=2,
            modulus=8,
            margin=0.5,
            max_cases=1,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(len(model.calls), 1)
        counterfactual_text = tokenizer.decode(model.calls[0][0].tolist())
        self.assertIn("ops 00 00 answer 05", counterfactual_text)

    def test_operation_counterfactual_loss_can_focus_on_long_active_lengths(self):
        module = load_module()

        class RecordingModel(torch.nn.Module):
            def __init__(self, vocab_size: int):
                super().__init__()
                self.vocab_size = int(vocab_size)
                self.calls: list[torch.Tensor] = []

            def forward(self, input_ids, *, think_steps):
                self.calls.append(input_ids.detach().cpu().clone())
                return torch.zeros(
                    input_ids.shape[0],
                    input_ids.shape[1],
                    self.vocab_size,
                    dtype=torch.float32,
                )

        short_case = module.TextReasoningCase(
            case_id="short",
            start=3,
            op_ids=(1, 0, 0),
            answer=4,
        )
        long_case = module.TextReasoningCase(
            case_id="long",
            start=3,
            op_ids=(1, 4, 5),
            answer=6,
        )
        tokenizer = module.CharTokenizer.from_texts(
            [
                module.case_full_text(short_case),
                module.case_full_text(long_case),
                module.case_prompt(module.zero_ops_case(long_case, modulus=8))
                + module.case_answer(long_case),
            ]
        )
        x, y, prompt_len, answer_len = module.cases_to_batch(
            [short_case, long_case],
            tokenizer=tokenizer,
            device=torch.device("cpu"),
        )
        chosen_logits = torch.zeros(
            x.shape[0],
            x.shape[1],
            tokenizer.vocab_size,
            dtype=torch.float32,
        )
        model = RecordingModel(tokenizer.vocab_size)

        loss = module.operation_counterfactual_loss(
            model,
            chosen_logits,
            y,
            [short_case, long_case],
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            include_family_tag=False,
            state_anchor=False,
            state_anchor_position="before_answer",
            prompt_len=prompt_len,
            answer_len=answer_len,
            think_steps=2,
            modulus=8,
            margin=0.5,
            max_cases=0,
            active_len_min=3,
            active_len_max=3,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0].shape[0], 1)
        counterfactual_text = tokenizer.decode(model.calls[0][0].tolist())
        self.assertIn("ops 00 00 00 answer 06", counterfactual_text)

    def test_operation_counterfactual_schedule_can_end_after_early_phase(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--operation-counterfactual-loss-weight",
                "0.1",
                "--operation-counterfactual-warmup-steps",
                "0",
                "--operation-counterfactual-end-step",
                "400",
                "--operation-counterfactual-every",
                "2",
            ]
        )

        self.assertFalse(module.operation_counterfactual_schedule_enabled(args, 1))
        self.assertTrue(module.operation_counterfactual_schedule_enabled(args, 2))
        self.assertTrue(module.operation_counterfactual_schedule_enabled(args, 400))
        self.assertFalse(module.operation_counterfactual_schedule_enabled(args, 402))

    def test_depth_counterfactual_loss_compares_full_against_shallow_thinking(self):
        module = load_module()

        class RecordingModel(torch.nn.Module):
            def __init__(self, vocab_size: int):
                super().__init__()
                self.vocab_size = int(vocab_size)
                self.calls: list[int] = []

            def forward(self, input_ids, *, think_steps):
                self.calls.append(int(think_steps))
                return torch.zeros(
                    input_ids.shape[0],
                    input_ids.shape[1],
                    self.vocab_size,
                    dtype=torch.float32,
                )

        case = module.TextReasoningCase(
            case_id="x",
            start=3,
            op_ids=(1, 2),
            answer=6,
        )
        tokenizer = module.CharTokenizer.from_texts([module.case_full_text(case)])
        x, y, prompt_len, answer_len = module.cases_to_batch(
            [case],
            tokenizer=tokenizer,
            device=torch.device("cpu"),
        )
        chosen_logits = torch.zeros(
            x.shape[0],
            x.shape[1],
            tokenizer.vocab_size,
            dtype=torch.float32,
        )
        model = RecordingModel(tokenizer.vocab_size)

        loss = module.depth_counterfactual_loss(
            model,
            chosen_logits,
            y,
            x,
            prompt_len=prompt_len,
            answer_len=answer_len,
            counterfactual_think_steps=0,
            margin=0.5,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(float(loss.detach()), 0.5, places=5)
        self.assertEqual(model.calls, [0])

    def test_state_reset_counterfactual_loss_compares_full_against_reset_state(self):
        module = load_module()

        class RecordingModel(torch.nn.Module):
            def __init__(self, vocab_size: int):
                super().__init__()
                self.vocab_size = int(vocab_size)
                self.calls: list[tuple[int, bool]] = []

            def forward(self, input_ids, *, think_steps, state_reset_each_step=False):
                self.calls.append((int(think_steps), bool(state_reset_each_step)))
                return torch.zeros(
                    input_ids.shape[0],
                    input_ids.shape[1],
                    self.vocab_size,
                    dtype=torch.float32,
                )

        case = module.TextReasoningCase(
            case_id="x",
            start=3,
            op_ids=(1, 2),
            answer=6,
        )
        tokenizer = module.CharTokenizer.from_texts([module.case_full_text(case)])
        x, y, prompt_len, answer_len = module.cases_to_batch(
            [case],
            tokenizer=tokenizer,
            device=torch.device("cpu"),
        )
        chosen_logits = torch.zeros(
            x.shape[0],
            x.shape[1],
            tokenizer.vocab_size,
            dtype=torch.float32,
        )
        model = RecordingModel(tokenizer.vocab_size)

        loss = module.state_reset_counterfactual_loss(
            model,
            chosen_logits,
            y,
            x,
            prompt_len=prompt_len,
            answer_len=answer_len,
            think_steps=8,
            margin=0.5,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(float(loss.detach()), 0.5, places=5)
        self.assertEqual(model.calls, [(8, True)])

    def test_answer_space_ranking_loss_scores_bounded_answer_candidates(self):
        module = load_module()
        cases = [
            module.TextReasoningCase(
                case_id="x0",
                start=3,
                op_ids=(1, 4),
                answer=5,
            ),
            module.TextReasoningCase(
                case_id="x1",
                start=4,
                op_ids=(1, 4),
                answer=6,
            ),
        ]
        tokenizer = module.CharTokenizer.from_texts(
            [
                module.case_full_text(
                    module.TextReasoningCase(
                        case_id="vocab",
                        start=3,
                        op_ids=(1, 4),
                        answer=answer,
                    )
                )
                for answer in range(8)
            ]
        )

        class CountingFlatModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.batch_sizes = []

            def forward(self, input_ids, *, think_steps, **_kwargs):
                self.batch_sizes.append(int(input_ids.shape[0]))
                return torch.zeros(
                    input_ids.shape[0],
                    input_ids.shape[1],
                    tokenizer.vocab_size,
                    dtype=torch.float32,
                )

        model = CountingFlatModel()
        loss = module.answer_space_ranking_loss(
            model,
            cases,
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            include_family_tag=False,
            think_steps=2,
            modulus=8,
            max_cases=1,
            temperature=1.0,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(model.batch_sizes, [8])

    def test_answer_space_argmax_metrics_scores_all_candidates(self):
        module = load_module()
        cases = [
            module.TextReasoningCase(
                case_id="x0",
                start=3,
                op_ids=(1, 4),
                answer=0,
            ),
            module.TextReasoningCase(
                case_id="x1",
                start=4,
                op_ids=(1, 4),
                answer=1,
            ),
        ]
        tokenizer = module.CharTokenizer.from_texts(
            [
                module.case_full_text(
                    module.TextReasoningCase(
                        case_id="vocab",
                        start=3,
                        op_ids=(1, 4),
                        answer=answer,
                    )
                )
                for answer in range(8)
            ]
        )

        class FlatModel(torch.nn.Module):
            def forward(self, input_ids, *, think_steps, **_kwargs):
                return torch.zeros(
                    input_ids.shape[0],
                    input_ids.shape[1],
                    tokenizer.vocab_size,
                    dtype=torch.float32,
                )

        metrics = module.answer_space_argmax_metrics(
            FlatModel(),
            cases,
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            prompt_cases=cases,
            include_family_tag=False,
            think_steps=2,
            modulus=8,
            max_candidate_batch=4,
        )

        self.assertEqual(metrics["answer_space_argmax_exact"], 0.5)
        self.assertIn("2", metrics["answer_space_argmax_by_active_len"])

    def test_prefix_state_alignment_loss_compares_full_and_prefix_states(self):
        module = load_module()
        cases = [
            module.TextReasoningCase(
                case_id="x0",
                start=3,
                op_ids=(1, 4, module.NOOP_OP_ID),
                answer=5,
            ),
            module.TextReasoningCase(
                case_id="x1",
                start=4,
                op_ids=(1, 4, module.NOOP_OP_ID),
                answer=6,
            ),
        ]
        prefix_cases = [
            module.case_with_active_program_len(case, active_len=depth, modulus=8)
            for case in cases
            for depth in (1, 2)
        ]
        tokenizer = module.CharTokenizer.from_texts(
            [module.case_full_text(case) for case in cases + prefix_cases]
        )

        class TraceModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward_with_runtime(self, input_ids, *, think_steps, return_state_trace):
                self.calls += 1
                batch, seq = input_ids.shape
                trace = torch.zeros(
                    batch,
                    int(think_steps),
                    seq,
                    4,
                    dtype=torch.float32,
                    requires_grad=True,
                )
                return {
                    "logits": torch.zeros(batch, seq, tokenizer.vocab_size),
                    "core_state_trace_h": trace,
                    "core_state_trace_l": trace,
                }

        model = TraceModel()
        loss = module.prefix_state_alignment_loss(
            model,
            cases,
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            include_family_tag=False,
            max_depth=2,
            modulus=8,
            max_cases=1,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(model.calls, 4)

    def test_prefix_state_contrastive_loss_compares_full_and_prefix_states(self):
        module = load_module()
        cases = [
            module.TextReasoningCase(
                case_id="x0",
                start=3,
                op_ids=(1, 4, module.NOOP_OP_ID),
                answer=5,
            ),
            module.TextReasoningCase(
                case_id="x1",
                start=4,
                op_ids=(1, 4, module.NOOP_OP_ID),
                answer=6,
            ),
        ]
        prefix_cases = [
            module.case_with_active_program_len(case, active_len=depth, modulus=8)
            for case in cases
            for depth in (1, 2)
        ]
        tokenizer = module.CharTokenizer.from_texts(
            [module.case_full_text(case) for case in cases + prefix_cases]
        )

        class TraceModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward_with_runtime(self, input_ids, *, think_steps, return_state_trace):
                self.calls += 1
                batch, seq = input_ids.shape
                base = torch.arange(batch, dtype=torch.float32).view(batch, 1, 1, 1)
                trace = (
                    torch.zeros(
                        batch,
                        int(think_steps),
                        seq,
                        4,
                        dtype=torch.float32,
                    )
                    + base
                ).requires_grad_()
                return {
                    "logits": torch.zeros(batch, seq, tokenizer.vocab_size),
                    "core_state_trace_h": trace,
                    "core_state_trace_l": trace,
                }

        model = TraceModel()
        loss = module.prefix_state_contrastive_loss(
            model,
            cases,
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            include_family_tag=False,
            max_depth=2,
            modulus=8,
            max_cases=0,
            temperature=0.1,
            state_source="both",
            pooling="last",
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(loss.requires_grad)
        self.assertEqual(model.calls, 3)

    def test_reference_retention_kl_loss_compares_answer_logits(self):
        module = load_module()
        cases = [
            module.TextReasoningCase(
                case_id="x0",
                start=3,
                op_ids=(1, 4, module.NOOP_OP_ID),
                answer=5,
            ),
            module.TextReasoningCase(
                case_id="x1",
                start=4,
                op_ids=(1, 4, module.NOOP_OP_ID),
                answer=6,
            ),
        ]
        tokenizer = module.CharTokenizer.from_texts(
            [module.case_full_text(case) for case in cases]
        )

        class ConstantModel(torch.nn.Module):
            def forward(self, input_ids, *, think_steps, **_kwargs):
                return torch.zeros(
                    input_ids.shape[0],
                    input_ids.shape[1],
                    tokenizer.vocab_size,
                    dtype=torch.float32,
                )

        loss = module.reference_retention_kl_loss(
            ConstantModel(),
            ConstantModel(),
            cases,
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            include_family_tag=False,
            think_steps=2,
            modulus=8,
            active_len_min=1,
            active_len_max=2,
            max_cases=1,
            temperature=1.0,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertLess(abs(float(loss.item())), 1e-6)

    def test_active_len_replay_ce_loss_uses_gold_targets(self):
        module = load_module()
        cases = [
            module.TextReasoningCase(
                case_id="x0",
                start=3,
                op_ids=(1, 4, module.NOOP_OP_ID),
                answer=5,
            ),
            module.TextReasoningCase(
                case_id="x1",
                start=4,
                op_ids=(1, 4, module.NOOP_OP_ID),
                answer=6,
            ),
        ]
        tokenizer = module.CharTokenizer.from_texts(
            [module.case_full_text(case) for case in cases]
        )

        class FlatModel(torch.nn.Module):
            def forward(self, input_ids, *, think_steps, **_kwargs):
                return torch.zeros(
                    input_ids.shape[0],
                    input_ids.shape[1],
                    tokenizer.vocab_size,
                    dtype=torch.float32,
                    requires_grad=True,
                )

        loss = module.active_len_replay_ce_loss(
            FlatModel(),
            cases,
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            include_family_tag=False,
            think_steps=2,
            modulus=8,
            active_len_min=1,
            active_len_max=2,
            max_cases=1,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss.item()), 0.0)

    def test_core_answer_probe_features_reads_prompt_state(self):
        module = load_module()
        cases = [
            module.TextReasoningCase(
                case_id="x0",
                start=3,
                op_ids=(1, 4),
                answer=5,
            ),
            module.TextReasoningCase(
                case_id="x1",
                start=4,
                op_ids=(1, 4),
                answer=6,
            ),
        ]
        tokenizer = module.CharTokenizer.from_texts(
            [module.case_full_text(case) for case in cases]
        )

        class TraceModel(torch.nn.Module):
            def forward_with_runtime(self, input_ids, *, think_steps, return_state_trace):
                batch, seq = input_ids.shape
                trace = torch.zeros(
                    batch,
                    int(think_steps),
                    seq,
                    2,
                    dtype=torch.float32,
                )
                trace[..., 0] = input_ids[:, None, :].float()
                trace[..., 1] = 1.0
                return {"core_state_trace_h": trace}

        features, labels, active_lengths = module.core_answer_probe_features(
            TraceModel(),
            cases,
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            include_family_tag=False,
            think_steps=2,
            state_source="h",
            pooling="last",
            batch_size=1,
        )

        self.assertEqual(tuple(features.shape), (2, 2))
        self.assertEqual(labels.tolist(), [5, 6])
        self.assertEqual(active_lengths, [2, 2])

        flat_features, _, _ = module.core_answer_probe_features(
            TraceModel(),
            cases,
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            include_family_tag=False,
            think_steps=2,
            state_source="h",
            pooling="flatten",
            batch_size=2,
        )
        self.assertGreater(flat_features.shape[1], features.shape[1])

    def test_core_step_probe_features_reads_intermediate_depth_states(self):
        module = load_module()
        cases = [
            module.TextReasoningCase(
                case_id="x0",
                start=3,
                op_ids=(1, 4),
                answer=5,
            )
        ]
        tokenizer = module.CharTokenizer.from_texts(
            [module.case_full_text(case) for case in cases]
        )

        class TraceModel(torch.nn.Module):
            def forward_with_runtime(self, input_ids, *, think_steps, return_state_trace):
                batch, seq = input_ids.shape
                trace = torch.zeros(
                    batch,
                    int(think_steps),
                    seq,
                    2,
                    dtype=torch.float32,
                )
                for depth in range(int(think_steps)):
                    trace[:, depth, :, 0] = float(depth + 1)
                trace[..., 1] = input_ids[:, None, :].float()
                return {"core_state_trace_h": trace}

        features, labels, depths, active_lengths, families = module.core_step_probe_features(
            TraceModel(),
            cases,
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            include_family_tag=False,
            think_steps=2,
            state_source="h",
            pooling="last",
            batch_size=1,
            modulus=8,
        )

        self.assertEqual(tuple(features.shape), (2, 2))
        self.assertEqual(
            labels.tolist(),
            [
                module.compute_answer(
                    start=3,
                    op_ids=(1,),
                    family="modchain",
                    modulus=8,
                ),
                module.compute_answer(
                    start=3,
                    op_ids=(1, 4),
                    family="modchain",
                    modulus=8,
                ),
            ],
        )
        self.assertEqual(depths, [1, 2])
        self.assertEqual(active_lengths, [2, 2])
        self.assertEqual(families, ["modchain", "modchain"])

    def test_core_step_codec_loss_reads_prompt_only_state_trace(self):
        module = load_module()
        case = module.TextReasoningCase(
            case_id="x0",
            start=3,
            op_ids=(1, 4),
            answer=0,
        )
        trace = torch.zeros(1, 2, 5, 2, dtype=torch.float32, requires_grad=True)
        runtime = {"core_state_trace_h": trace}
        codec_head = torch.nn.Linear(2, 8)

        loss = module.core_step_codec_loss_from_runtime(
            runtime,
            [case],
            codec_head,
            prompt_len=3,
            max_depth=2,
            modulus=8,
            state_source="h",
            pooling="last",
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(loss.item()), 0.0)

    def test_core_step_codec_labels_follow_family_causal_order(self):
        module = load_module()
        op_ids = (1, 4, 3)
        cases = [
            module.TextReasoningCase(
                case_id="m",
                start=1,
                op_ids=op_ids,
                answer=module.compute_answer(
                    start=1,
                    op_ids=op_ids,
                    family="modchain",
                    modulus=8,
                ),
                family="modchain",
            ),
            module.TextReasoningCase(
                case_id="r",
                start=1,
                op_ids=op_ids,
                answer=module.compute_answer(
                    start=1,
                    op_ids=op_ids,
                    family="revchain",
                    modulus=8,
                ),
                family="revchain",
            ),
        ]
        expected_labels = [
            module.compute_answer(start=1, op_ids=op_ids[:1], family="modchain", modulus=8),
            module.compute_answer(start=1, op_ids=op_ids[-1:], family="revchain", modulus=8),
            module.compute_answer(start=1, op_ids=op_ids[:2], family="modchain", modulus=8),
            module.compute_answer(start=1, op_ids=op_ids[-2:], family="revchain", modulus=8),
            module.compute_answer(start=1, op_ids=op_ids[:3], family="modchain", modulus=8),
            module.compute_answer(start=1, op_ids=op_ids[-3:], family="revchain", modulus=8),
        ]
        trace = torch.zeros(2, 3, 5, 2, dtype=torch.float32, requires_grad=True)
        runtime = {"core_state_trace_h": trace}

        class FixedLabelHead(torch.nn.Module):
            def forward(self, x):
                logits = torch.full(
                    (x.shape[0], 8),
                    -40.0,
                    dtype=x.dtype,
                    device=x.device,
                )
                for row, label in enumerate(expected_labels):
                    logits[row, int(label)] = 40.0
                return logits

        loss = module.core_step_codec_loss_from_runtime(
            runtime,
            cases,
            FixedLabelHead(),
            prompt_len=3,
            max_depth=3,
            modulus=8,
            state_source="h",
            pooling="last",
        )

        self.assertLess(float(loss.item()), 1e-4)

    def test_core_step_op_codec_labels_follow_family_causal_order(self):
        module = load_module()
        op_ids = (1, 4, 3)
        cases = [
            module.TextReasoningCase(
                case_id="m",
                start=1,
                op_ids=op_ids,
                answer=0,
                family="modchain",
            ),
            module.TextReasoningCase(
                case_id="r",
                start=1,
                op_ids=op_ids,
                answer=0,
                family="revchain",
            ),
        ]
        expected_labels = [1, 3, 4, 4, 3, 1]
        trace = torch.zeros(2, 3, 5, 2, dtype=torch.float32, requires_grad=True)
        runtime = {"core_state_trace_l": trace}

        class FixedOpHead(torch.nn.Module):
            def forward(self, x):
                logits = torch.full(
                    (x.shape[0], len(module.OP_SPECS)),
                    -40.0,
                    dtype=x.dtype,
                    device=x.device,
                )
                for row, label in enumerate(expected_labels):
                    logits[row, int(label)] = 40.0
                return logits

        loss = module.core_step_op_codec_loss_from_runtime(
            runtime,
            cases,
            FixedOpHead(),
            prompt_len=3,
            max_depth=3,
            state_source="l",
            pooling="last",
        )

        self.assertLess(float(loss.item()), 1e-4)

    def test_core_step_position_codec_labels_follow_family_causal_order(self):
        module = load_module()
        op_ids = (1, 4, 3)
        cases = [
            module.TextReasoningCase(
                case_id="m",
                start=1,
                op_ids=op_ids,
                answer=0,
                family="modchain",
            ),
            module.TextReasoningCase(
                case_id="r",
                start=1,
                op_ids=op_ids,
                answer=0,
                family="revchain",
            ),
        ]
        expected_labels = [1, 3, 2, 2, 3, 1]
        trace = torch.zeros(2, 3, 5, 2, dtype=torch.float32, requires_grad=True)
        runtime = {"core_state_trace_l": trace}

        class FixedPositionHead(torch.nn.Module):
            def forward(self, x):
                logits = torch.full(
                    (x.shape[0], len(op_ids) + 1),
                    -40.0,
                    dtype=x.dtype,
                    device=x.device,
                )
                for row, label in enumerate(expected_labels):
                    logits[row, int(label)] = 40.0
                return logits

        loss = module.core_step_position_codec_loss_from_runtime(
            runtime,
            cases,
            FixedPositionHead(),
            prompt_len=3,
            max_depth=3,
            state_source="l",
            pooling="last",
        )

        self.assertLess(float(loss.item()), 1e-4)

    def test_online_greedy_preference_loss_uses_model_mined_wrong_answer(self):
        module = load_module()
        case = module.TextReasoningCase(
            case_id="x",
            start=3,
            op_ids=(1, 4),
            answer=5,
        )
        rejected = module.TextReasoningCase(
            case_id="x",
            start=3,
            op_ids=(1, 4),
            answer=7,
        )
        tokenizer = module.CharTokenizer.from_texts(
            [module.case_full_text(case), module.case_full_text(rejected)]
        )
        x, y, prompt_len, answer_len = module.cases_to_batch(
            [case],
            tokenizer=tokenizer,
            device=torch.device("cpu"),
        )
        wrong_ids = tokenizer.encode("07\n")

        class WrongGreedyModel(torch.nn.Module):
            def forward(self, input_ids, *, think_steps, **_kwargs):
                logits = torch.zeros(
                    input_ids.shape[0],
                    input_ids.shape[1],
                    tokenizer.vocab_size,
                    dtype=torch.float32,
                )
                generated = min(max(int(input_ids.shape[1]) - prompt_len, 0), 2)
                logits[:, -1, wrong_ids[generated]] = 5.0
                return logits

        model = WrongGreedyModel()
        chosen_logits = torch.zeros(
            x.shape[0],
            x.shape[1],
            tokenizer.vocab_size,
            dtype=torch.float32,
        )

        loss = module.online_greedy_preference_loss(
            model,
            chosen_logits,
            y,
            [case],
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            include_family_tag=False,
            prompt_len=prompt_len,
            answer_len=answer_len,
            think_steps=2,
            margin=0.5,
            max_cases=0,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(loss), 0.0)

    def test_online_greedy_preference_loss_limits_mined_cases(self):
        module = load_module()
        cases = [
            module.TextReasoningCase(
                case_id=f"x{i}",
                start=3 + i,
                op_ids=(1, 4),
                answer=5 + i,
            )
            for i in range(4)
        ]
        tokenizer = module.CharTokenizer.from_texts(
            [module.case_full_text(case) for case in cases]
            + [
                module.case_full_text(
                    module.TextReasoningCase(
                        case_id="wrong",
                        start=3,
                        op_ids=(1, 4),
                        answer=7,
                    )
                )
            ]
        )
        x, y, prompt_len, answer_len = module.cases_to_batch(
            cases,
            tokenizer=tokenizer,
            device=torch.device("cpu"),
        )
        wrong_ids = tokenizer.encode("07\n")

        class CountingWrongGreedyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.single_item_calls = 0

            def forward(self, input_ids, *, think_steps, **_kwargs):
                if int(input_ids.shape[0]) == 1:
                    self.single_item_calls += 1
                logits = torch.zeros(
                    input_ids.shape[0],
                    input_ids.shape[1],
                    tokenizer.vocab_size,
                    dtype=torch.float32,
                )
                generated = min(max(int(input_ids.shape[1]) - prompt_len, 0), 2)
                logits[:, -1, wrong_ids[generated]] = 5.0
                return logits

        model = CountingWrongGreedyModel()
        chosen_logits = torch.zeros(
            x.shape[0],
            x.shape[1],
            tokenizer.vocab_size,
            dtype=torch.float32,
        )

        loss = module.online_greedy_preference_loss(
            model,
            chosen_logits,
            y,
            cases,
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            include_family_tag=False,
            prompt_len=prompt_len,
            answer_len=answer_len,
            think_steps=2,
            margin=0.5,
            max_cases=2,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(model.single_item_calls, 6)

    def test_answer_margin_loss_penalizes_close_competitors(self):
        module = load_module()
        logits = torch.zeros((1, 4, 3), dtype=torch.float32)
        targets = torch.zeros((1, 4), dtype=torch.long)
        logits[:, 1:3, 0] = 1.0
        logits[:, 1:3, 1] = 0.8

        loss = module.answer_margin_loss(
            logits,
            targets,
            prompt_len=2,
            answer_len=2,
            margin=0.5,
        )

        self.assertGreater(float(loss), 0.0)

    def test_state_trace_anti_collapse_loss_penalizes_low_variance(self):
        module = load_module()
        runtime = {
            "core_state_trace_h": torch.zeros((2, 3, 4, 5), dtype=torch.float32),
            "logits": torch.zeros((2, 4, 6), dtype=torch.float32),
        }

        loss = module.state_trace_anti_collapse_loss(
            runtime,
            min_variance=0.5,
            min_delta_norm=0.5,
        )

        self.assertGreater(float(loss), 0.0)

    def test_latent_refinement_loss_uses_core_trace_and_shared_lm_path(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=8,
            max_seq_len=6,
            d_model=12,
            n_heads=3,
            d_ff=24,
            dropout=0.0,
            backbone="mha_etd",
        )
        input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        targets = torch.tensor([[2, 3, 4, 5, 6]], dtype=torch.long)
        runtime = model.forward_with_runtime(
            input_ids,
            think_steps=2,
            return_state_trace=True,
        )

        loss = module.latent_refinement_loss_from_runtime(
            model,
            runtime,
            targets,
            prompt_len=3,
            answer_len=2,
            min_depth=1,
            noise_std=0.0,
            depth_weight_power=0.0,
            final_kl_weight=0.0,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(loss.requires_grad)

    def test_latent_refinement_loss_calls_shared_lm_logits_path(self):
        module = load_module()

        class BadHead(torch.nn.Module):
            def forward(self, hidden):
                logits = torch.zeros((*hidden.shape[:-1], 3), dtype=hidden.dtype)
                logits[..., 1] = 20.0
                return logits

        class TraceModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.decode = torch.nn.Identity()
                self.norm = torch.nn.Identity()
                self.lm_head = BadHead()
                self.shared_calls = 0

            def _causal_mask(self, seq_len, device):
                return torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)

            def _run_stage(self, stage, state, *, causal_mask):
                return state

            def _lm_logits(self, hidden):
                self.shared_calls += 1
                logits = torch.zeros((*hidden.shape[:-1], 3), dtype=hidden.dtype)
                logits[..., 0] = 20.0
                return logits

        model = TraceModel()
        runtime = {"core_state_trace_h": torch.zeros((1, 1, 1, 3), dtype=torch.float32)}
        targets = torch.zeros((1, 1), dtype=torch.long)

        loss = module.latent_refinement_loss_from_runtime(
            model,
            runtime,
            targets,
            prompt_len=1,
            answer_len=1,
        )

        self.assertLess(float(loss.item()), 1e-4)
        self.assertEqual(model.shared_calls, 1)

    def test_latent_refinement_loss_requires_core_trace(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=8,
            max_seq_len=6,
            d_model=12,
            n_heads=3,
            d_ff=24,
            dropout=0.0,
            backbone="mha_etd",
        )
        runtime = {"logits": torch.zeros((1, 5, 8), dtype=torch.float32)}
        targets = torch.zeros((1, 5), dtype=torch.long)

        with self.assertRaisesRegex(ValueError, "core_state_trace_h"):
            module.latent_refinement_loss_from_runtime(
                model,
                runtime,
                targets,
                prompt_len=3,
                answer_len=2,
            )

    def test_halt_depth_final_answer_loss_uses_per_case_halt_depths(self):
        module = load_module()

        class RecordingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = []

            def forward(self, input_ids, *, think_steps):
                self.calls.append((int(think_steps), int(input_ids.shape[0])))
                return torch.zeros(
                    input_ids.shape[0],
                    input_ids.shape[1],
                    4,
                    dtype=torch.float32,
                )

        model = RecordingModel()
        input_ids = torch.zeros((2, 6), dtype=torch.long)
        targets = torch.zeros((2, 6), dtype=torch.long)
        cases = [
            module.TextReasoningCase(
                case_id="active2",
                start=0,
                op_ids=(1, 2, module.NOOP_OP_ID, module.NOOP_OP_ID),
                answer=0,
            ),
            module.TextReasoningCase(
                case_id="active3",
                start=0,
                op_ids=(1, 2, 3, module.NOOP_OP_ID),
                answer=0,
            ),
        ]

        loss = module.halt_depth_final_answer_loss(
            model,
            input_ids,
            targets,
            cases,
            prompt_len=3,
            answer_len=2,
            max_depth=4,
            min_halt_step=1,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(model.calls, [(2, 1), (3, 1)])

    def test_depth_sweep_summary_reports_best_depth(self):
        module = load_module()

        summary = module.depth_sweep_summary(
            {
                "think0": {
                    "generation_exact": 0.10,
                    "generation_format_valid": 0.50,
                    "by_active_len": {
                        "2": {"generation_exact": 0.20},
                    },
                },
                "think2": {
                    "generation_exact": 0.40,
                    "generation_format_valid": 1.00,
                    "by_active_len": {
                        "2": {"generation_exact": 0.60},
                    },
                },
                "think4": {
                    "generation_exact": 0.30,
                    "generation_format_valid": 1.00,
                    "by_active_len": {
                        "2": {"generation_exact": 0.30},
                    },
                },
            },
            full_depth=4,
        )

        self.assertEqual(summary["best_depth"], 2)
        self.assertEqual(summary["full_depth"], 4)
        self.assertEqual(summary["exact_by_depth"]["2"], 0.40)
        self.assertAlmostEqual(summary["best_minus_full"], 0.10)
        self.assertEqual(summary["by_active_len"]["2"]["best_depth"], 2)

    def test_periodic_eval_score_uses_rank_as_tiebreaker(self):
        module = load_module()

        weaker = {
            "generation_exact": 0.0,
            "teacher_forced_sequence_exact": 0.0,
            "teacher_forced_mean_token_rank": 7.5,
            "teacher_forced_answer_loss": 2.9,
        }
        stronger = {
            "generation_exact": 0.0,
            "teacher_forced_sequence_exact": 0.0,
            "teacher_forced_mean_token_rank": 6.1,
            "teacher_forced_answer_loss": 2.7,
        }

        self.assertGreater(
            module.periodic_eval_score(stronger),
            module.periodic_eval_score(weaker),
        )

    def test_periodic_eval_score_prefers_adaptive_balanced_checkpoint(self):
        module = load_module()

        fixed_only_better = {
            "generation_exact": 0.24,
            "adaptive_halt_generation_exact": 0.14,
            "adaptive_halt_mean_steps": 4.2,
            "teacher_forced_sequence_exact": 0.24,
            "teacher_forced_mean_token_rank": 1.6,
            "teacher_forced_answer_loss": 0.8,
        }
        strict_better = {
            "generation_exact": 0.21,
            "adaptive_halt_generation_exact": 0.20,
            "adaptive_halt_mean_steps": 4.1,
            "teacher_forced_sequence_exact": 0.21,
            "teacher_forced_mean_token_rank": 1.7,
            "teacher_forced_answer_loss": 0.82,
        }

        self.assertGreater(
            module.periodic_eval_score(strict_better),
            module.periodic_eval_score(fixed_only_better),
        )

    def test_periodic_eval_score_uses_active_len_floor(self):
        module = load_module()

        uneven = {
            "generation_exact": 0.22,
            "min_active_len_generation_exact": 0.05,
            "adaptive_halt_generation_exact": 0.22,
            "adaptive_halt_min_active_len_generation_exact": 0.04,
            "adaptive_halt_mean_steps": 4.1,
            "teacher_forced_sequence_exact": 0.22,
            "teacher_forced_mean_token_rank": 1.5,
            "teacher_forced_answer_loss": 0.8,
        }
        balanced = {
            "generation_exact": 0.22,
            "min_active_len_generation_exact": 0.12,
            "adaptive_halt_generation_exact": 0.22,
            "adaptive_halt_min_active_len_generation_exact": 0.11,
            "adaptive_halt_mean_steps": 4.1,
            "teacher_forced_sequence_exact": 0.22,
            "teacher_forced_mean_token_rank": 1.6,
            "teacher_forced_answer_loss": 0.82,
        }

        self.assertGreater(
            module.periodic_eval_score(balanced),
            module.periodic_eval_score(uneven),
        )

    def test_periodic_eval_score_active_floor_mode_prioritizes_length_floor(self):
        module = load_module()

        higher_average = {
            "generation_exact": 0.24,
            "min_active_len_generation_exact": 0.06,
            "adaptive_halt_generation_exact": 0.24,
            "adaptive_halt_min_active_len_generation_exact": 0.05,
            "adaptive_halt_mean_steps": 4.1,
            "teacher_forced_sequence_exact": 0.24,
            "teacher_forced_mean_token_rank": 1.5,
            "teacher_forced_answer_loss": 0.8,
        }
        better_floor = {
            "generation_exact": 0.20,
            "min_active_len_generation_exact": 0.13,
            "adaptive_halt_generation_exact": 0.20,
            "adaptive_halt_min_active_len_generation_exact": 0.12,
            "adaptive_halt_mean_steps": 4.1,
            "teacher_forced_sequence_exact": 0.20,
            "teacher_forced_mean_token_rank": 1.6,
            "teacher_forced_answer_loss": 0.82,
        }

        self.assertGreater(
            module.periodic_eval_score(better_floor, mode="active_floor"),
            module.periodic_eval_score(higher_average, mode="active_floor"),
        )

    def test_periodic_eval_score_family_floor_mode_prioritizes_family_floor(self):
        module = load_module()

        higher_average = {
            "generation_exact": 0.24,
            "min_active_len_generation_exact": 0.20,
            "min_family_generation_exact": 0.04,
            "teacher_forced_sequence_exact": 0.24,
            "teacher_forced_mean_token_rank": 1.5,
            "teacher_forced_answer_loss": 0.8,
        }
        better_family_floor = {
            "generation_exact": 0.18,
            "min_active_len_generation_exact": 0.12,
            "min_family_generation_exact": 0.10,
            "teacher_forced_sequence_exact": 0.18,
            "teacher_forced_mean_token_rank": 1.6,
            "teacher_forced_answer_loss": 0.82,
        }

        self.assertGreater(
            module.periodic_eval_score(better_family_floor, mode="family_floor"),
            module.periodic_eval_score(higher_average, mode="family_floor"),
        )

    def test_min_active_len_generation_exact_reads_metric_floor(self):
        module = load_module()
        metrics = {
            "generation_exact": 0.5,
            "by_active_len": {
                "3": {"generation_exact": 0.4},
                "4": {"generation_exact": 0.2},
            },
        }

        self.assertEqual(module.min_active_len_generation_exact(metrics), 0.2)

    def test_min_family_generation_exact_reads_metric_floor(self):
        module = load_module()
        metrics = {
            "generation_exact": 0.5,
            "by_family": {
                "checksum": {"generation_exact": 0.8},
                "revchain": {"generation_exact": 0.1},
            },
        }

        self.assertEqual(module.min_family_generation_exact(metrics), 0.1)

    def test_parser_accepts_active_len_halt_target_and_eval_cycle(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--adaptive-halt-target-mode",
                "active_len",
                "--adaptive-halt-active-len-target",
                "first_step",
                "--eval-active-len-cycle",
                "--eval-depth-sweep",
                "--eval-state-trace",
                "--eval-operation-breakdown",
                "--eval-core-answer-probe",
                "--eval-core-step-probe",
                "--eval-order-router-probe",
                "--eval-order-router-route-ablation",
                "--core-answer-probe-state-source",
                "both",
                "--core-answer-probe-pooling",
                "flatten",
                "--core-answer-probe-train-cases",
                "128",
                "--core-answer-probe-eval-cases",
                "64",
                "--core-answer-probe-steps",
                "20",
                "--core-answer-probe-batch-size",
                "16",
                "--core-answer-probe-lr",
                "0.02",
                "--core-answer-probe-weight-decay",
                "0.01",
                "--eval-beam-width",
                "5",
                "--eval-answer-space-argmax",
                "--eval-answer-space-argmax-batch-size",
                "64",
                "--tokenizer-mode",
                "number",
                "--number-tokenizer-max-value",
                "31",
                "--resume-from",
                "local_eval/prev/last.pt",
                "--resume-allow-missing",
                "--halt-depth-final-loss-weight",
                "0.75",
                "--depth-intermediate-min-depth",
                "3",
                "--depth-intermediate-weight-power",
                "1.5",
                "--depth-intermediate-family-dro",
                "--depth-intermediate-family-dro-temperature",
                "0.33",
                "--prefix-depth-anchor-loss-weight",
                "0.4",
                "--prefix-depth-anchor-min-depth",
                "2",
                "--prefix-depth-anchor-weight-power",
                "1.25",
                "--residue-aux-loss-weight",
                "0.3",
                "--residue-aux-moduli",
                "2,4,8",
                "--answer-margin-loss-weight",
                "0.25",
                "--answer-margin",
                "0.75",
                "--family-dro-loss-weight",
                "0.23",
                "--family-dro-temperature",
                "0.17",
                "--sequence-preference-loss-weight",
                "0.15",
                "--sequence-preference-deltas",
                "2,4,8",
                "--sequence-preference-margin",
                "0.5",
                "--operation-counterfactual-loss-weight",
                "0.21",
                "--operation-counterfactual-margin",
                "0.35",
                "--operation-counterfactual-max-cases",
                "7",
                "--operation-counterfactual-every",
                "11",
                "--operation-counterfactual-warmup-steps",
                "200",
                "--operation-counterfactual-end-step",
                "600",
                "--operation-counterfactual-active-len-min",
                "6",
                "--operation-counterfactual-active-len-max",
                "8",
                "--depth-counterfactual-loss-weight",
                "0.17",
                "--depth-counterfactual-margin",
                "0.45",
                "--depth-counterfactual-think-steps",
                "0",
                "--depth-counterfactual-every",
                "9",
                "--state-reset-counterfactual-loss-weight",
                "0.19",
                "--state-reset-counterfactual-margin",
                "0.55",
                "--state-reset-counterfactual-every",
                "10",
                "--answer-space-ranking-loss-weight",
                "0.07",
                "--answer-space-ranking-max-cases",
                "3",
                "--answer-space-ranking-every",
                "5",
                "--answer-space-ranking-temperature",
                "0.75",
                "--order-router-aux-loss-weight",
                "0.04",
                "--order-router-aux-target-mode",
                "family_order",
                "--order-router-lr-multiplier",
                "100.0",
                "--forced-route-answer-loss-weight",
                "0.29",
                "--forced-route-answer-route",
                "1",
                "--forced-route-answer-families",
                "revchain",
                "--forced-route-answer-max-cases",
                "9",
                "--forced-route-answer-every",
                "4",
                "--forced-route-depth-loss-weight",
                "0.31",
                "--forced-route-depth-route",
                "1",
                "--forced-route-depth-families",
                "revchain",
                "--forced-route-depth-max-cases",
                "5",
                "--forced-route-depth-every",
                "3",
                "--forced-route-depth-min-depth",
                "2",
                "--forced-route-depth-weight-power",
                "1.3",
                "--prefix-state-alignment-loss-weight",
                "0.09",
                "--prefix-state-alignment-max-cases",
                "2",
                "--prefix-state-alignment-every",
                "4",
                "--prefix-state-contrastive-loss-weight",
                "0.13",
                "--prefix-state-contrastive-max-cases",
                "5",
                "--prefix-state-contrastive-every",
                "6",
                "--prefix-state-contrastive-temperature",
                "0.2",
                "--prefix-state-contrastive-state-source",
                "both",
                "--prefix-state-contrastive-pooling",
                "mean",
                "--retention-reference-checkpoint",
                "resume",
                "--retention-kl-loss-weight",
                "0.11",
                "--retention-active-len-min",
                "3",
                "--retention-active-len-max",
                "5",
                "--retention-max-cases",
                "6",
                "--retention-every",
                "7",
                "--retention-temperature",
                "1.25",
                "--active-len-replay-loss-weight",
                "0.33",
                "--active-len-replay-min",
                "3",
                "--active-len-replay-max",
                "5",
                "--active-len-replay-max-cases",
                "8",
                "--active-len-replay-every",
                "9",
                "--online-greedy-preference-loss-weight",
                "0.05",
                "--online-greedy-preference-margin",
                "0.25",
                "--online-greedy-preference-max-cases",
                "4",
                "--online-greedy-preference-every",
                "3",
                "--state-trace-anti-collapse-loss-weight",
                "0.2",
                "--state-trace-min-variance",
                "0.6",
                "--state-trace-min-delta-norm",
                "3.5",
                "--latent-refine-loss-weight",
                "0.17",
                "--latent-refine-min-depth",
                "2",
                "--latent-refine-noise-std",
                "0.03",
                "--latent-refine-depth-weight-power",
                "1.25",
                "--latent-refine-final-kl-weight",
                "0.4",
                "--core-step-codec-loss-weight",
                "0.12",
                "--core-step-codec-state-source",
                "both",
                "--core-step-codec-pooling",
                "mean",
                "--eval-during-training-every",
                "100",
                "--eval-during-training-cases",
                "0",
                "--periodic-eval-score-mode",
                "family_floor",
                "--eval-initial-checkpoint",
                "--restore-best-eval-checkpoint",
            ]
        )

        self.assertEqual(args.adaptive_halt_target_mode, "active_len")
        self.assertEqual(args.adaptive_halt_active_len_target, "first_step")
        self.assertTrue(args.eval_active_len_cycle)
        self.assertTrue(args.eval_depth_sweep)
        self.assertTrue(args.eval_state_trace)
        self.assertTrue(args.eval_operation_breakdown)
        self.assertTrue(args.eval_core_answer_probe)
        self.assertTrue(args.eval_core_step_probe)
        self.assertTrue(args.eval_order_router_probe)
        self.assertTrue(args.eval_order_router_route_ablation)
        self.assertEqual(args.core_answer_probe_state_source, "both")
        self.assertEqual(args.core_answer_probe_pooling, "flatten")
        self.assertEqual(args.core_answer_probe_train_cases, 128)
        self.assertEqual(args.core_answer_probe_eval_cases, 64)
        self.assertEqual(args.core_answer_probe_steps, 20)
        self.assertEqual(args.core_answer_probe_batch_size, 16)
        self.assertEqual(args.core_answer_probe_lr, 0.02)
        self.assertEqual(args.core_answer_probe_weight_decay, 0.01)
        self.assertEqual(args.eval_beam_width, 5)
        self.assertTrue(args.eval_answer_space_argmax)
        self.assertEqual(args.eval_answer_space_argmax_batch_size, 64)
        self.assertEqual(args.tokenizer_mode, "number")
        self.assertEqual(args.number_tokenizer_max_value, 31)
        self.assertEqual(args.resume_from, "local_eval/prev/last.pt")
        self.assertTrue(args.resume_allow_missing)
        self.assertEqual(args.halt_depth_final_loss_weight, 0.75)
        self.assertEqual(args.depth_intermediate_min_depth, 3)
        self.assertEqual(args.depth_intermediate_weight_power, 1.5)
        self.assertTrue(args.depth_intermediate_family_dro)
        self.assertEqual(args.depth_intermediate_family_dro_temperature, 0.33)
        self.assertEqual(args.prefix_depth_anchor_loss_weight, 0.4)
        self.assertEqual(args.prefix_depth_anchor_min_depth, 2)
        self.assertEqual(args.prefix_depth_anchor_weight_power, 1.25)
        self.assertEqual(args.residue_aux_loss_weight, 0.3)
        self.assertEqual(args.residue_aux_moduli, "2,4,8")
        self.assertEqual(args.answer_margin_loss_weight, 0.25)
        self.assertEqual(args.answer_margin, 0.75)
        self.assertEqual(args.family_dro_loss_weight, 0.23)
        self.assertEqual(args.family_dro_temperature, 0.17)
        self.assertEqual(args.sequence_preference_loss_weight, 0.15)
        self.assertEqual(args.sequence_preference_deltas, "2,4,8")
        self.assertEqual(args.sequence_preference_margin, 0.5)
        self.assertEqual(args.operation_counterfactual_loss_weight, 0.21)
        self.assertEqual(args.operation_counterfactual_margin, 0.35)
        self.assertEqual(args.operation_counterfactual_max_cases, 7)
        self.assertEqual(args.operation_counterfactual_every, 11)
        self.assertEqual(args.operation_counterfactual_warmup_steps, 200)
        self.assertEqual(args.operation_counterfactual_end_step, 600)
        self.assertEqual(args.operation_counterfactual_active_len_min, 6)
        self.assertEqual(args.operation_counterfactual_active_len_max, 8)
        self.assertEqual(args.depth_counterfactual_loss_weight, 0.17)
        self.assertEqual(args.depth_counterfactual_margin, 0.45)
        self.assertEqual(args.depth_counterfactual_think_steps, 0)
        self.assertEqual(args.depth_counterfactual_every, 9)
        self.assertEqual(args.state_reset_counterfactual_loss_weight, 0.19)
        self.assertEqual(args.state_reset_counterfactual_margin, 0.55)
        self.assertEqual(args.state_reset_counterfactual_every, 10)
        self.assertEqual(args.answer_space_ranking_loss_weight, 0.07)
        self.assertEqual(args.answer_space_ranking_max_cases, 3)
        self.assertEqual(args.answer_space_ranking_every, 5)
        self.assertEqual(args.answer_space_ranking_temperature, 0.75)
        self.assertEqual(args.order_router_aux_loss_weight, 0.04)
        self.assertEqual(args.order_router_aux_target_mode, "family_order")
        self.assertEqual(args.order_router_lr_multiplier, 100.0)
        self.assertEqual(args.forced_route_answer_loss_weight, 0.29)
        self.assertEqual(args.forced_route_answer_route, 1)
        self.assertEqual(args.forced_route_answer_families, "revchain")
        self.assertEqual(args.forced_route_answer_max_cases, 9)
        self.assertEqual(args.forced_route_answer_every, 4)
        self.assertEqual(args.forced_route_depth_loss_weight, 0.31)
        self.assertEqual(args.forced_route_depth_route, 1)
        self.assertEqual(args.forced_route_depth_families, "revchain")
        self.assertEqual(args.forced_route_depth_max_cases, 5)
        self.assertEqual(args.forced_route_depth_every, 3)
        self.assertEqual(args.forced_route_depth_min_depth, 2)
        self.assertEqual(args.forced_route_depth_weight_power, 1.3)
        self.assertEqual(args.prefix_state_alignment_loss_weight, 0.09)
        self.assertEqual(args.prefix_state_alignment_max_cases, 2)
        self.assertEqual(args.prefix_state_alignment_every, 4)
        self.assertEqual(args.prefix_state_contrastive_loss_weight, 0.13)
        self.assertEqual(args.prefix_state_contrastive_max_cases, 5)
        self.assertEqual(args.prefix_state_contrastive_every, 6)
        self.assertEqual(args.prefix_state_contrastive_temperature, 0.2)
        self.assertEqual(args.prefix_state_contrastive_state_source, "both")
        self.assertEqual(args.prefix_state_contrastive_pooling, "mean")
        self.assertEqual(args.retention_reference_checkpoint, "resume")
        self.assertEqual(args.retention_kl_loss_weight, 0.11)
        self.assertEqual(args.retention_active_len_min, 3)
        self.assertEqual(args.retention_active_len_max, 5)
        self.assertEqual(args.retention_max_cases, 6)
        self.assertEqual(args.retention_every, 7)
        self.assertEqual(args.retention_temperature, 1.25)
        self.assertEqual(args.active_len_replay_loss_weight, 0.33)
        self.assertEqual(args.active_len_replay_min, 3)
        self.assertEqual(args.active_len_replay_max, 5)
        self.assertEqual(args.active_len_replay_max_cases, 8)
        self.assertEqual(args.active_len_replay_every, 9)
        self.assertEqual(args.online_greedy_preference_loss_weight, 0.05)
        self.assertEqual(args.online_greedy_preference_margin, 0.25)
        self.assertEqual(args.online_greedy_preference_max_cases, 4)
        self.assertEqual(args.online_greedy_preference_every, 3)
        self.assertEqual(args.state_trace_anti_collapse_loss_weight, 0.2)
        self.assertEqual(args.state_trace_min_variance, 0.6)
        self.assertEqual(args.state_trace_min_delta_norm, 3.5)
        self.assertEqual(args.latent_refine_loss_weight, 0.17)
        self.assertEqual(args.latent_refine_min_depth, 2)
        self.assertEqual(args.latent_refine_noise_std, 0.03)
        self.assertEqual(args.latent_refine_depth_weight_power, 1.25)
        self.assertEqual(args.latent_refine_final_kl_weight, 0.4)
        self.assertEqual(args.core_step_codec_loss_weight, 0.12)
        self.assertEqual(args.core_step_codec_state_source, "both")
        self.assertEqual(args.core_step_codec_pooling, "mean")
        self.assertEqual(args.eval_during_training_every, 100)
        self.assertEqual(args.eval_during_training_cases, 0)
        self.assertEqual(args.periodic_eval_score_mode, "family_floor")
        self.assertTrue(args.eval_initial_checkpoint)
        self.assertTrue(args.restore_best_eval_checkpoint)

    def test_eval_active_len_cycle_rewrites_eval_cases_deterministically(self):
        module = load_module()
        cases = module.build_cases(
            count=5,
            seed=1,
            program_len=4,
            modulus=8,
            families=("modchain",),
        )

        cycled = module.apply_eval_active_len_cycle(cases, modulus=8)

        self.assertEqual(
            [module.effective_program_len(case) for case in cycled],
            [0, 1, 2, 3, 4],
        )

    def test_eval_active_len_cycle_can_focus_on_hard_lengths(self):
        module = load_module()
        cases = module.build_cases(
            count=5,
            seed=1,
            program_len=4,
            modulus=8,
            families=("modchain",),
        )

        cycled = module.apply_eval_active_len_cycle(
            cases,
            modulus=8,
            min_active_len=2,
            max_active_len=4,
        )

        self.assertEqual(
            [module.effective_program_len(case) for case in cycled],
            [2, 3, 4, 2, 3],
        )

    def test_eval_active_len_cycle_balances_each_family_independently(self):
        module = load_module()
        cases = module.build_cases(
            count=6,
            seed=1,
            program_len=4,
            modulus=8,
            families=("modchain", "revchain"),
        )

        cycled = module.apply_eval_active_len_cycle(
            cases,
            modulus=8,
            min_active_len=2,
            max_active_len=4,
        )

        self.assertEqual(
            [(case.family, module.effective_program_len(case)) for case in cycled],
            [
                ("modchain", 2),
                ("revchain", 2),
                ("modchain", 3),
                ("revchain", 3),
                ("modchain", 4),
                ("revchain", 4),
            ],
        )
        self.assertEqual(
            cycled[1].op_ids[:2],
            (module.NOOP_OP_ID, module.NOOP_OP_ID),
        )
        self.assertEqual(
            cycled[3].op_ids[:1],
            (module.NOOP_OP_ID,),
        )

    def test_active_len_batch_cycle_mixes_difficulties_each_step(self):
        module = load_module()
        cases = module.build_cases(
            count=6,
            seed=1,
            program_len=4,
            modulus=8,
            families=("modchain",),
        )

        mixed = module.apply_active_len_batch_cycle(cases, step=1, modulus=8)

        self.assertEqual(
            [module.effective_program_len(case) for case in mixed],
            [0, 1, 2, 3, 4, 0],
        )

        hard_mixed = module.apply_active_len_batch_cycle(
            cases,
            step=2,
            modulus=8,
            min_active_len=2,
            max_active_len=4,
        )

        self.assertEqual(
            [module.effective_program_len(case) for case in hard_mixed],
            [3, 4, 2, 3, 4, 2],
        )

    def test_active_len_batch_cycle_uses_revchain_causal_suffix(self):
        module = load_module()
        cases = [
            module.TextReasoningCase(
                case_id="r0",
                start=1,
                op_ids=(1, 2, 3, 4),
                answer=0,
                family="revchain",
            )
        ]

        mixed = module.apply_active_len_batch_cycle(
            cases,
            step=3,
            modulus=8,
            min_active_len=0,
            max_active_len=4,
        )

        self.assertEqual(
            mixed[0].op_ids,
            (module.NOOP_OP_ID, module.NOOP_OP_ID, 3, 4),
        )
        self.assertEqual(module.effective_program_len(mixed[0]), 2)

    def test_parser_accepts_active_len_batch_cycle(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--active-len-batch-cycle",
                "--active-len-cycle-min",
                "2",
                "--active-len-cycle-max",
                "4",
                "--train-active-len-cycle-min",
                "5",
                "--train-active-len-cycle-max",
                "6",
            ]
        )

        self.assertTrue(args.active_len_batch_cycle)
        self.assertEqual(args.active_len_cycle_min, 2)
        self.assertEqual(args.active_len_cycle_max, 4)
        self.assertEqual(args.train_active_len_cycle_min, 5)
        self.assertEqual(args.train_active_len_cycle_max, 6)

    def test_halt_loss_input_can_use_prompt_only_context(self):
        module = load_module()
        input_ids = torch.arange(20).view(2, 10)

        prompt_only = module.halt_loss_input_for_context(
            input_ids,
            prompt_len=6,
            context="prompt",
        )
        full = module.halt_loss_input_for_context(
            input_ids,
            prompt_len=6,
            context="full",
        )

        self.assertTrue(torch.equal(prompt_only, input_ids[:, :6]))
        self.assertTrue(torch.equal(full, input_ids))

    def test_halt_loss_inputs_can_use_all_generation_prefixes(self):
        module = load_module()
        input_ids = torch.arange(20).view(2, 10)

        prefixes = module.halt_loss_inputs_for_context(
            input_ids,
            prompt_len=8,
            context="prefixes",
        )

        self.assertEqual([item.shape[1] for item in prefixes], [8, 9, 10])
        self.assertTrue(torch.equal(prefixes[0], input_ids[:, :8]))
        self.assertTrue(torch.equal(prefixes[-1], input_ids))

    def test_parser_accepts_prompt_only_adaptive_halt_loss_context(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            ["--adaptive-halt-loss-context", "prefixes"]
        )

        self.assertEqual(args.adaptive_halt_loss_context, "prefixes")

    def test_parser_accepts_mean_halt_pooling(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(["--halt-pooling", "dedicated"])

        self.assertEqual(args.halt_pooling, "dedicated")


if __name__ == "__main__":
    unittest.main()
