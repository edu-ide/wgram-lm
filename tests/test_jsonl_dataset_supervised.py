import unittest


class JsonlDatasetSupervisedTests(unittest.TestCase):
    def test_prompt_answer_rows_mask_prompt_tokens_in_labels(self):
        from qtrm_mm.data.jsonl_dataset import JsonlTextVisionDataset

        ds = JsonlTextVisionDataset(
            files=[],
            vocab_size=512,
            seq_len=32,
            visual_dim=8,
            max_visual_tokens=1,
            multimodal=False,
        )

        sample = ds._make_sample({"prompt": "Question: alpha beta?", "answer": "Answer: OMEGA"})
        labels = sample["labels"]
        unmasked = (labels != -100).nonzero().flatten()

        self.assertGreater(int(unmasked.numel()), 0)
        first_answer_index = int(unmasked[0])
        self.assertGreater(first_answer_index, 0)
        self.assertTrue((labels[:first_answer_index] == -100).all())
        self.assertTrue((labels[first_answer_index:][labels[first_answer_index:] != 0] != -100).any())

    def test_workspace_evidence_injection_splits_memoryos_prompt(self):
        from qtrm_mm.data.jsonl_dataset import (
            JsonlTextVisionDataset,
            split_memory_prompt_for_workspace,
        )

        prompt = (
            "MemoryOS evidence\n"
            "SOURCE=archive.md CHUNK=0 SCORE=1.0000\n"
            "The access code is VX-913.\n\n"
            "Use the evidence above when it is relevant. If it is not relevant, answer from the prompt.\n\n"
            "User prompt:\n"
            "Answer using only the evidence.\n"
            "Question: What is the access code?"
        )

        visible_prompt, evidence_text = split_memory_prompt_for_workspace(prompt)

        self.assertNotIn("VX-913", visible_prompt)
        self.assertIn("Question: What is the access code?", visible_prompt)
        self.assertIn("MemoryOS evidence", evidence_text)
        self.assertIn("VX-913", evidence_text)

        ds = JsonlTextVisionDataset(
            files=[],
            vocab_size=512,
            seq_len=48,
            visual_dim=8,
            max_visual_tokens=1,
            multimodal=False,
            workspace_evidence_injection=True,
        )
        sample = ds._make_sample({"prompt": prompt, "answer": "Answer: VX-913"})

        self.assertIn("workspace_input_ids", sample)
        self.assertIn("workspace_attention_mask", sample)
        self.assertGreater(int(sample["workspace_attention_mask"].sum().item()), 0)
        self.assertEqual(sample["workspace_input_ids"].shape, sample["input_ids"].shape)

        plain = ds._make_sample({"prompt": "Question: plain?", "answer": "Answer: UNKNOWN"})
        self.assertIn("workspace_input_ids", plain)
        self.assertEqual(int(plain["workspace_attention_mask"].sum().item()), 0)

    def test_dual_workspace_evidence_injection_keeps_evidence_visible_and_in_workspace(self):
        import torch

        from qtrm_mm.data.jsonl_dataset import JsonlTextVisionDataset

        class SimpleTokenizer:
            pad_token_id = 0
            bos_token_id = 1
            eos_token_id = 2

            def __init__(self):
                self.token_to_id = {"<pad>": 0, "<bos>": 1, "<eos>": 2}
                self.id_to_token = {0: "<pad>", 1: "<bos>", 2: "<eos>"}

            def __call__(self, text, **kwargs):
                max_length = kwargs.get("max_length", 512)
                tokens = str(text or "").replace("\n", " ").split()
                ids = [self.bos_token_id]
                for token in tokens[: max_length - 2]:
                    if token not in self.token_to_id:
                        idx = len(self.token_to_id)
                        self.token_to_id[token] = idx
                        self.id_to_token[idx] = token
                    ids.append(self.token_to_id[token])
                ids.append(self.eos_token_id)
                return {"input_ids": torch.tensor([ids], dtype=torch.long)}

            def decode(self, ids):
                return " ".join(
                    self.id_to_token.get(int(item), "?")
                    for item in ids
                    if int(item) != self.pad_token_id
                )

        prompt = (
            "MemoryOS evidence\n"
            "SOURCE=archive.md CHUNK=0 SCORE=1.0000\n"
            "The access code is VX-913.\n\n"
            "Use the evidence above when it is relevant. If it is not relevant, answer from the prompt.\n\n"
            "User prompt:\n"
            "Answer using only the evidence.\n"
            "Question: What is the access code?"
        )
        ds = JsonlTextVisionDataset(
            files=[],
            vocab_size=512,
            seq_len=64,
            visual_dim=8,
            max_visual_tokens=1,
            multimodal=False,
            tokenizer=SimpleTokenizer(),
            workspace_evidence_injection=True,
            workspace_evidence_injection_mode="dual",
        )

        sample = ds._make_sample({"prompt": prompt, "answer": "Answer: VX-913"})
        decoded_input = ds.tok.tokenizer.decode(sample["input_ids"].tolist())
        decoded_workspace = ds.tok.tokenizer.decode(sample["workspace_input_ids"].tolist())

        self.assertIn("MemoryOS evidence", decoded_input)
        self.assertIn("VX-913", decoded_input)
        self.assertIn("MemoryOS evidence", decoded_workspace)
        self.assertIn("VX-913", decoded_workspace)
        self.assertGreater(int(sample["workspace_attention_mask"].sum().item()), 0)

    def test_preference_rows_emit_rejected_sequence_and_confidence_weight(self):
        from qtrm_mm.data.jsonl_dataset import JsonlTextVisionDataset, collate_jsonl

        ds = JsonlTextVisionDataset(
            files=[],
            vocab_size=512,
            seq_len=40,
            visual_dim=8,
            max_visual_tokens=1,
            multimodal=False,
        )

        sample = ds._make_sample(
            {
                "prompt": "Question: choose the verified answer.",
                "chosen": "Answer: VERIFIED",
                "rejected": "Answer: guessed",
                "preference_weight": 0.7,
            }
        )

        self.assertIn("labels", sample)
        self.assertIn("preference_rejected_input_ids", sample)
        self.assertIn("preference_rejected_attention_mask", sample)
        self.assertIn("preference_rejected_labels", sample)
        self.assertIn("preference_sample_weight", sample)
        self.assertEqual(sample["preference_rejected_input_ids"].shape, sample["input_ids"].shape)
        self.assertGreater(int((sample["preference_rejected_labels"] != -100).sum().item()), 0)
        self.assertAlmostEqual(float(sample["preference_sample_weight"].item()), 0.7, places=5)

        batch = collate_jsonl([sample])

        self.assertIn("preference_rejected_input_ids", batch)
        self.assertIn("preference_rejected_labels", batch)
        self.assertIn("preference_sample_weight", batch)
        self.assertEqual(batch["preference_sample_weight"].shape, (1,))

    def test_rows_emit_answer_decision_targets(self):
        from qtrm_mm.data.jsonl_dataset import JsonlTextVisionDataset, collate_jsonl

        ds = JsonlTextVisionDataset(
            files=[],
            vocab_size=512,
            seq_len=40,
            visual_dim=8,
            max_visual_tokens=1,
            multimodal=False,
        )

        sample = ds._make_sample(
            {
                "prompt": "Question: should this be blocked?",
                "answer": "Answer: guessed",
                "answer_decision_target": 1.0,
                "answer_decision_sample_weight": 0.75,
            }
        )

        self.assertIn("answer_decision_target", sample)
        self.assertIn("answer_decision_sample_weight", sample)
        self.assertAlmostEqual(float(sample["answer_decision_target"].item()), 1.0, places=5)
        self.assertAlmostEqual(float(sample["answer_decision_sample_weight"].item()), 0.75, places=5)

        batch = collate_jsonl([sample])

        self.assertIn("answer_decision_target", batch)
        self.assertIn("answer_decision_sample_weight", batch)
        self.assertEqual(batch["answer_decision_target"].shape, (1,))

    def test_rows_emit_answer_decision_telemetry_features(self):
        from qtrm_mm.data.jsonl_dataset import JsonlTextVisionDataset, collate_jsonl

        ds = JsonlTextVisionDataset(
            files=[],
            vocab_size=512,
            seq_len=40,
            visual_dim=8,
            max_visual_tokens=1,
            multimodal=False,
        )

        sample = ds._make_sample(
            {
                "prompt": "Question: should this be blocked?",
                "answer": "Answer: guessed",
                "answer_decision_target": 1.0,
                "answer_decision_features": [0.7, 0.2, 1.0],
            }
        )

        self.assertIn("answer_decision_features", sample)
        self.assertEqual(sample["answer_decision_features"].shape, (3,))
        self.assertAlmostEqual(float(sample["answer_decision_features"][0].item()), 0.7, places=5)

        batch = collate_jsonl([sample])

        self.assertIn("answer_decision_features", batch)
        self.assertEqual(batch["answer_decision_features"].shape, (1, 3))

    def test_trace_replay_rows_emit_action_policy_targets(self):
        from qtrm_mm.agentic.cognitive_loop import Action
        from qtrm_mm.data.jsonl_dataset import JsonlTextVisionDataset, collate_jsonl

        ds = JsonlTextVisionDataset(
            files=[],
            vocab_size=512,
            seq_len=48,
            visual_dim=8,
            max_visual_tokens=1,
            multimodal=False,
        )

        sample = ds._make_sample(
            {
                "type": "trace_replay",
                "chat_prompt": "<user>\nWhat is the archive code?\n</user>",
                "workspace_context": "The archive code is VX-913.",
                "action_target": "VERIFY_EVIDENCE",
                "reward": 0.75,
                "controller_signal": [1.0, 0.0],
            }
        )

        self.assertIn("action_target", sample)
        self.assertIn("action_sample_weight", sample)
        self.assertIn("controller_signal", sample)
        self.assertEqual(int(sample["action_target"].item()), Action.VERIFY_EVIDENCE.id)
        self.assertAlmostEqual(float(sample["action_sample_weight"].item()), 0.75, places=5)
        self.assertEqual(sample["controller_signal"].shape, (2,))

        batch = collate_jsonl([sample])

        self.assertEqual(batch["action_target"].shape, (1,))
        self.assertEqual(batch["controller_signal"].shape, (1, 2))
        self.assertEqual(int(batch["action_target"][0].item()), Action.VERIFY_EVIDENCE.id)
        self.assertGreater(int(batch["workspace_attention_mask"].sum().item()), 0)

    def test_trace_replay_rows_encode_step_and_state_in_action_input(self):
        import torch

        from qtrm_mm.data.jsonl_dataset import JsonlTextVisionDataset

        class SimpleTokenizer:
            pad_token_id = 0
            bos_token_id = 1
            eos_token_id = 2

            def __init__(self):
                self.token_to_id = {"<pad>": 0, "<bos>": 1, "<eos>": 2}
                self.id_to_token = {0: "<pad>", 1: "<bos>", 2: "<eos>"}

            def __call__(self, text, **kwargs):
                max_length = kwargs.get("max_length", 512)
                tokens = str(text or "").replace("\n", " ").split()
                ids = [self.bos_token_id]
                for token in tokens[: max_length - 2]:
                    if token not in self.token_to_id:
                        idx = len(self.token_to_id)
                        self.token_to_id[token] = idx
                        self.id_to_token[idx] = token
                    ids.append(self.token_to_id[token])
                ids.append(self.eos_token_id)
                return {"input_ids": torch.tensor([ids], dtype=torch.long)}

            def decode(self, ids):
                return " ".join(
                    self.id_to_token.get(int(item), "?")
                    for item in ids
                    if int(item) != self.pad_token_id
                )

        ds = JsonlTextVisionDataset(
            files=[],
            vocab_size=512,
            seq_len=64,
            visual_dim=8,
            max_visual_tokens=1,
            multimodal=False,
            tokenizer=SimpleTokenizer(),
        )
        common = {
            "type": "trace_replay",
            "chat_prompt": "Question: What is the archive code?",
            "workspace_context": "The archive code is VX-913.",
            "action_target": "VERIFY_EVIDENCE",
        }

        retrieve = ds._make_sample({**common, "step": 0, "state_summary": "need retrieval"})
        verify = ds._make_sample(
            {
                **common,
                "step": 1,
                "state_summary": "evidence retrieved",
                "previous_observation": "retrieved evidence",
            }
        )

        self.assertFalse(torch.equal(retrieve["input_ids"], verify["input_ids"]))
        decoded = ds.tok.tokenizer.decode(verify["input_ids"].tolist())
        self.assertIn("trace_step=1", decoded)
        self.assertIn("evidence retrieved", decoded)
        self.assertIn("previous_observation", decoded)
        self.assertIn("next_action_query", decoded)

    def test_trace_replay_can_hide_step_text_for_signal_conditioning(self):
        import torch

        from qtrm_mm.data.jsonl_dataset import JsonlTextVisionDataset

        class SimpleTokenizer:
            pad_token_id = 0
            bos_token_id = 1
            eos_token_id = 2

            def __init__(self):
                self.token_to_id = {"<pad>": 0, "<bos>": 1, "<eos>": 2}
                self.id_to_token = {0: "<pad>", 1: "<bos>", 2: "<eos>"}

            def __call__(self, text, **kwargs):
                max_length = kwargs.get("max_length", 512)
                tokens = str(text or "").replace("\n", " ").split()
                ids = [self.bos_token_id]
                for token in tokens[: max_length - 2]:
                    if token not in self.token_to_id:
                        idx = len(self.token_to_id)
                        self.token_to_id[token] = idx
                        self.id_to_token[idx] = token
                    ids.append(self.token_to_id[token])
                ids.append(self.eos_token_id)
                return {"input_ids": torch.tensor([ids], dtype=torch.long)}

            def decode(self, ids):
                return " ".join(
                    self.id_to_token.get(int(item), "?")
                    for item in ids
                    if int(item) != self.pad_token_id
                )

        ds = JsonlTextVisionDataset(
            files=[],
            vocab_size=512,
            seq_len=64,
            visual_dim=8,
            max_visual_tokens=1,
            multimodal=False,
            tokenizer=SimpleTokenizer(),
        )
        first = ds._make_sample(
            {
                "type": "trace_replay",
                "step": 0,
                "state_summary": "Decide from signal.",
                "chat_prompt": "Question: x",
                "workspace_context": "Evidence: x",
                "action_target": "RETRIEVE_MEMORY",
                "controller_signal": [0.0, 0.0],
                "hide_trace_step_from_input": True,
            }
        )
        second = ds._make_sample(
            {
                "type": "trace_replay",
                "step": 2,
                "state_summary": "Decide from signal.",
                "chat_prompt": "Question: x",
                "workspace_context": "Evidence: x",
                "action_target": "ANSWER",
                "controller_signal": [1.0, 1.0],
                "hide_trace_step_from_input": True,
            }
        )

        self.assertTrue(torch.equal(first["input_ids"], second["input_ids"]))
        decoded = ds.tok.tokenizer.decode(first["input_ids"].tolist())
        self.assertNotIn("trace_step=", decoded)
        self.assertIn("controller_signal", first)

    def test_workspace_counterfactual_rows_emit_separate_workspace_sequence(self):
        from qtrm_mm.data.jsonl_dataset import JsonlTextVisionDataset, collate_jsonl

        prompt = (
            "MemoryOS evidence\n"
            "SOURCE=true.md CHUNK=0 SCORE=1.0000\n"
            "The code is TRUE-7.\n\n"
            "Use the evidence above when it is relevant. If it is not relevant, answer from the prompt.\n\n"
            "User prompt:\n"
            "Answer using only the evidence.\n"
            "Question: What is the code?"
        )
        ds = JsonlTextVisionDataset(
            files=[],
            vocab_size=512,
            seq_len=48,
            visual_dim=8,
            max_visual_tokens=1,
            multimodal=False,
            workspace_evidence_injection=True,
        )

        sample = ds._make_sample(
            {
                "prompt": prompt,
                "chosen": "Answer: TRUE-7",
                "rejected": "Answer: FALSE-3",
                "counterfactual_workspace_text": (
                    "MemoryOS evidence\n"
                    "SOURCE=false.md CHUNK=0 SCORE=1.0000\n"
                    "The code is FALSE-3."
                ),
            }
        )

        self.assertIn("workspace_input_ids", sample)
        self.assertIn("workspace_counterfactual_input_ids", sample)
        self.assertIn("logical_support_target", sample)
        self.assertIn("causal_evidence_target", sample)
        self.assertGreater(int(sample["workspace_attention_mask"].sum().item()), 0)
        self.assertGreater(
            int(sample["workspace_counterfactual_attention_mask"].sum().item()),
            0,
        )
        self.assertEqual(float(sample["logical_support_target"].item()), 1.0)
        self.assertEqual(float(sample["causal_evidence_target"].item()), 1.0)

        batch = collate_jsonl([sample])

        self.assertIn("workspace_counterfactual_input_ids", batch)
        self.assertIn("logical_support_target", batch)
        self.assertIn("causal_evidence_target", batch)
        self.assertEqual(
            batch["workspace_counterfactual_input_ids"].shape,
            batch["input_ids"].shape,
        )

    def test_collate_jsonl_handles_mixed_optional_training_targets(self):
        from qtrm_mm.data.jsonl_dataset import JsonlTextVisionDataset, collate_jsonl

        evidence_prompt = (
            "MemoryOS evidence\n"
            "SOURCE=doc.md CHUNK=1 SCORE=1.0000\n"
            "The code is VX-913.\n\n"
            "Use the evidence above when it is relevant. If it is not relevant, answer from the prompt.\n\n"
            "User prompt:\n"
            "What is the code?"
        )
        ds = JsonlTextVisionDataset(
            files=[],
            vocab_size=512,
            seq_len=48,
            visual_dim=8,
            max_visual_tokens=1,
            multimodal=False,
            workspace_evidence_injection=True,
        )

        preference_sample = ds._make_sample(
            {
                "prompt": "Choose the verified answer.",
                "chosen": "Answer: VERIFIED",
                "rejected": "Answer: guessed",
            }
        )
        evidence_sample = ds._make_sample(
            {
                "prompt": evidence_prompt,
                "answer": "Answer: VX-913",
            }
        )

        batch = collate_jsonl([preference_sample, evidence_sample])

        self.assertIn("preference_rejected_input_ids", batch)
        self.assertIn("logical_support_target", batch)
        self.assertEqual(batch["preference_rejected_input_ids"].shape, batch["input_ids"].shape)
        self.assertEqual(batch["logical_support_target"].shape, (2,))
        self.assertEqual(float(batch["preference_sample_weight"][0].item()), 1.0)
        self.assertEqual(float(batch["preference_sample_weight"][1].item()), 0.0)
        self.assertEqual(float(batch["logical_support_target"][0].item()), 0.0)
        self.assertEqual(float(batch["logical_support_target"][1].item()), 1.0)

    def test_generation_verifier_rows_emit_targets(self):
        from qtrm_mm.data.jsonl_dataset import JsonlTextVisionDataset, collate_jsonl

        ds = JsonlTextVisionDataset(
            files=[],
            vocab_size=512,
            seq_len=48,
            visual_dim=8,
            max_visual_tokens=1,
            multimodal=False,
        )

        sample = ds._make_sample(
            {
                "text": "Question: 5 + 7?\n\nAnswer: 12. Question: 5 + 7?",
                "generation_verifier_repeat_target": 1.0,
                "generation_verifier_stop_target": 1.0,
                "generation_verifier_quality_target": 0.0,
                "generation_verifier_sample_weight": 0.8,
            }
        )

        self.assertIn("generation_verifier_repeat_target", sample)
        self.assertIn("generation_verifier_stop_target", sample)
        self.assertIn("generation_verifier_quality_target", sample)
        self.assertIn("generation_verifier_sample_weight", sample)

        batch = collate_jsonl([sample])

        self.assertEqual(batch["generation_verifier_repeat_target"].shape, (1,))
        self.assertEqual(float(batch["generation_verifier_repeat_target"][0].item()), 1.0)
        self.assertEqual(float(batch["generation_verifier_stop_target"][0].item()), 1.0)
        self.assertEqual(float(batch["generation_verifier_quality_target"][0].item()), 0.0)
        self.assertAlmostEqual(
            float(batch["generation_verifier_sample_weight"][0].item()),
            0.8,
            places=5,
        )


if __name__ == "__main__":
    unittest.main()
