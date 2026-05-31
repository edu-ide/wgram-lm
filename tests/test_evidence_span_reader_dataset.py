import json
import tempfile
import unittest
from pathlib import Path


def load_span_reader_module():
    import importlib.util

    script = Path("scripts/build_evidence_span_reader_dataset.py")
    spec = importlib.util.spec_from_file_location("evidence_span_reader_dataset", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class EvidenceSpanReaderDatasetTests(unittest.TestCase):
    def test_builds_prompt_conditioned_span_row_from_hidden_workspace_evidence(self):
        module = load_span_reader_module()
        row = {
            "case_id": "garnet-ko",
            "category": "multi_hop",
            "prompt": (
                "MemoryOS evidence\n"
                "SOURCE=garnet_note.md CHUNK=0 SCORE=1.0000\n"
                "Garnet-실험-112 실험의 책임자는 윤서아이다.\n\n"
                "Use the evidence above when it is relevant. If it is not relevant, answer from the prompt.\n\n"
                "User prompt:\n"
                "Question: Garnet 현장 노트의 책임자는 누구인가요?"
            ),
            "answer": "Answer: 윤서아",
        }

        built = module.build_span_reader_rows([row])

        self.assertEqual(len(built), 1)
        out = built[0]
        self.assertEqual(out["type"], "evidence_span_reader")
        self.assertEqual(out["answer_text"], "윤서아")
        self.assertFalse(out["no_answer"])
        self.assertIn("Question: Garnet", out["visible_prompt"])
        self.assertIn("책임자는 윤서아", out["workspace_evidence"])
        span = out["answer_span"]
        self.assertEqual(span["text"], "윤서아")
        self.assertEqual(
            out["workspace_evidence"][span["start_char"] : span["end_char"]],
            "윤서아",
        )

    def test_marks_unknown_as_no_answer_even_when_evidence_contains_distractors(self):
        module = load_span_reader_module()
        row = {
            "case_id": "unknown-ko",
            "expected_unknown": True,
            "prompt": (
                "MemoryOS evidence\n"
                "SOURCE=anonymous.md CHUNK=0 SCORE=1.0000\n"
                "익명 메모: 북쪽 통신실의 암구호는 구름-39이다.\n\n"
                "Use the evidence above when it is relevant. If it is not relevant, answer from the prompt.\n\n"
                "User prompt:\n"
                "Question: 북쪽 통신실의 현재 비상 암구호는 무엇인가요?"
            ),
            "answer": "Answer: UNKNOWN",
        }

        built = module.build_span_reader_rows([row])

        self.assertEqual(len(built), 1)
        self.assertTrue(built[0]["no_answer"])
        self.assertEqual(built[0]["answer_text"], "UNKNOWN")
        self.assertIsNone(built[0]["answer_span"])

    def test_script_writes_limited_span_rows(self):
        module = load_span_reader_module()

        with tempfile.TemporaryDirectory() as tmp:
            in_path = Path(tmp) / "train.jsonl"
            out_path = Path(tmp) / "span.jsonl"
            rows = [
                {
                    "case_id": "case-a",
                    "prompt": (
                        "MemoryOS evidence\nA is Alpha.\n\n"
                        "Use the evidence above when it is relevant. If it is not relevant, answer from the prompt.\n\n"
                        "User prompt:\nQuestion: What is A?"
                    ),
                    "answer": "Answer: Alpha",
                },
                {
                    "case_id": "case-b",
                    "prompt": (
                        "MemoryOS evidence\nB is Beta.\n\n"
                        "Use the evidence above when it is relevant. If it is not relevant, answer from the prompt.\n\n"
                        "User prompt:\nQuestion: What is B?"
                    ),
                    "answer": "Answer: Beta",
                },
            ]
            in_path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                encoding="utf-8",
            )

            count = module.write_span_reader_rows(in_path, out_path, max_rows=1)

            written = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(count, 1)
            self.assertEqual(written[0]["case_id"], "case-a")

    def test_jsonl_dataset_emits_span_reader_targets(self):
        from wgram_lm.data.jsonl_dataset import JsonlTextVisionDataset

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "span.jsonl"
            row = {
                "type": "evidence_span_reader",
                "case_id": "case-span",
                "visible_prompt": "Question: What is A?",
                "workspace_evidence": "A is Alpha.",
                "workspace_text": "A is Alpha.",
                "answer": "Answer: Alpha",
                "answer_text": "Alpha",
                "no_answer": False,
                "answer_span": {"text": "Alpha"},
            }
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            ds = JsonlTextVisionDataset(
                files=[str(path)],
                vocab_size=256,
                seq_len=32,
                visual_dim=8,
                max_visual_tokens=4,
                multimodal=False,
                shuffle_buffer=1,
                workspace_evidence_injection=True,
            )

            sample = next(iter(ds))

            self.assertIn("workspace_input_ids", sample)
            self.assertIn("evidence_span_start_target", sample)
            self.assertGreaterEqual(int(sample["evidence_span_start_target"]), 0)
            self.assertGreaterEqual(int(sample["evidence_span_end_target"]), 0)
            self.assertEqual(float(sample["evidence_span_no_answer_target"]), 0.0)
            self.assertEqual(float(sample["evidence_span_sample_weight"]), 1.0)

    def test_jsonl_dataset_keeps_text_span_targets_beyond_visual_token_limit(self):
        from wgram_lm.data.jsonl_dataset import JsonlTextVisionDataset

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "span.jsonl"
            row = {
                "type": "evidence_span_reader",
                "case_id": "late-span",
                "visible_prompt": "Question: What is the answer?",
                "workspace_evidence": "one two three four five six seven Alpha.",
                "workspace_text": "one two three four five six seven Alpha.",
                "answer": "Answer: Alpha",
                "answer_text": "Alpha",
                "no_answer": False,
                "answer_span": {"text": "Alpha"},
            }
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            ds = JsonlTextVisionDataset(
                files=[str(path)],
                vocab_size=256,
                seq_len=32,
                visual_dim=8,
                max_visual_tokens=4,
                multimodal=False,
                shuffle_buffer=1,
                workspace_evidence_injection=True,
            )

            sample = next(iter(ds))

            self.assertGreaterEqual(int(sample["evidence_span_start_target"]), 4)
            self.assertGreaterEqual(int(sample["evidence_span_end_target"]), 4)
            self.assertEqual(float(sample["evidence_span_sample_weight"]), 1.0)

    def test_jsonl_dataset_ssot_span_reader_targets_canonical_prompt_tokens(self):
        from wgram_lm.data.jsonl_dataset import JsonlTextVisionDataset

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "span.jsonl"
            row = {
                "type": "evidence_span_reader",
                "case_id": "ssot-span",
                "visible_prompt": "Question: What is the answer?",
                "workspace_evidence": (
                    "MemoryOS evidence\n"
                    "SOURCE=a.md CHUNK=0 SCORE=1.0000\n"
                    "The answer is Alpha."
                ),
                "workspace_text": (
                    "MemoryOS evidence\n"
                    "SOURCE=a.md CHUNK=0 SCORE=1.0000\n"
                    "The answer is Alpha."
                ),
                "answer": "Answer: Alpha",
                "answer_text": "Alpha",
                "no_answer": False,
                "answer_span": {"text": "Alpha"},
            }
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            ds = JsonlTextVisionDataset(
                files=[str(path)],
                vocab_size=256,
                seq_len=96,
                visual_dim=8,
                max_visual_tokens=4,
                multimodal=False,
                shuffle_buffer=1,
                workspace_evidence_injection=True,
                workspace_evidence_injection_mode="ssot",
            )

            sample = next(iter(ds))

            self.assertGreaterEqual(int(sample["evidence_span_start_target"]), 0)
            self.assertGreaterEqual(int(sample["evidence_span_end_target"]), 0)
            self.assertEqual(int(sample["workspace_attention_mask"].sum().item()), 0)
            self.assertEqual(float(sample["evidence_span_sample_weight"]), 1.0)

    def test_training_mix_builder_adds_counterfactual_hard_negatives(self):
        import importlib.util

        script = Path("scripts/build_evidence_span_reader_training_mix.py")
        spec = importlib.util.spec_from_file_location("span_reader_mix_builder", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.jsonl"
            cases = root / "cases.jsonl"
            out = root / "out.jsonl"
            base.write_text(
                json.dumps(
                    {
                        "type": "evidence_span_reader",
                        "case_id": "base",
                        "prompt": "What is the code?",
                        "visible_prompt": "What is the code?",
                        "workspace_text": (
                            "MemoryOS evidence\n"
                            "SOURCE=base CHUNK=0 SCORE=1.0000\n"
                            "The code is VX-1."
                        ),
                        "workspace_evidence": (
                            "MemoryOS evidence\n"
                            "SOURCE=base CHUNK=0 SCORE=1.0000\n"
                            "The code is VX-1."
                        ),
                        "answer": "Answer: VX-1",
                        "answer_text": "VX-1",
                        "no_answer": False,
                        "answer_span": {
                            "start_char": 58,
                            "end_char": 62,
                            "text": "VX-1",
                        },
                        "span_status": "found",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            cases.write_text(
                json.dumps(
                    {
                        "id": "swap-a",
                        "instruction": "Use hidden evidence only.",
                        "question": "What is the current code?",
                        "expected_unknown": True,
                        "answer_aliases": ["UNKNOWN"],
                        "evidence": [],
                        "distractors": [
                            {
                                "source": "other.md",
                                "chunk_id": 0,
                                "text": "Another record says the owner is 윤서아.",
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            count = module.write_training_mix(
                base,
                out,
                hard_negative_cases=cases,
                hard_negative_top_k=3,
                hard_negative_repeat=2,
            )
            rows = [
                json.loads(line)
                for line in out.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(count, 3)
        hard = rows[1]
        self.assertEqual(hard["type"], "evidence_span_reader")
        self.assertTrue(hard["no_answer"])
        self.assertEqual(hard["answer"], "Answer: UNKNOWN")
        self.assertEqual(hard["span_status"], "hard_no_answer")
        self.assertTrue(hard["workspace_text"].startswith("MemoryOS evidence"))
        self.assertIn("윤서아", hard["workspace_text"])
        self.assertEqual(rows[2]["case_id"], "swap-a")


if __name__ == "__main__":
    unittest.main()
