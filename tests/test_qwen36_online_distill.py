from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


def _load_script():
    path = Path("scripts/198_train_qwen36_online_distill.py")
    spec = importlib.util.spec_from_file_location("qwen36_online_distill", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Qwen36OnlineDistillTests(unittest.TestCase):
    def test_clean_teacher_answer_accepts_json_answer(self) -> None:
        from wgram_lm.distill.online_qwen36 import clean_teacher_answer

        answer = clean_teacher_answer('{"answer": "17", "trace_summary": "computed"}')

        self.assertEqual(answer, "17")

    def test_clean_teacher_answer_strips_labels_and_extra_lines(self) -> None:
        from wgram_lm.distill.online_qwen36 import clean_teacher_answer

        answer = clean_teacher_answer("Answer: violet\nExplanation: mapping chain")

        self.assertEqual(answer, "violet")

    def test_build_online_teacher_prompt_keeps_prompt_as_single_source(self) -> None:
        from wgram_lm.distill.online_qwen36 import build_online_teacher_prompt

        prompt = build_online_teacher_prompt("Question: 1+1?\nAnswer:")

        self.assertIn("Question: 1+1?", prompt)
        self.assertIn("Return only the final answer", prompt)

    def test_teacher_answer_record_round_trips_schema(self) -> None:
        from wgram_lm.distill.online_qwen36 import teacher_answer_record

        record = teacher_answer_record(
            prompt="Question: 1+1?\nAnswer:",
            answer="2",
            teacher_model="/models/Qwen3.6-27B",
        )

        self.assertTrue(record.prompt.startswith("Question"))
        self.assertEqual(record.answer, "2")
        self.assertEqual(record.teacher_model, "/models/Qwen3.6-27B")

    def test_online_distill_script_parser_exposes_teacher_model_path(self) -> None:
        module = _load_script()

        args = module.build_arg_parser().parse_args(
            [
                "--config",
                "config.yaml",
                "--cases-jsonl",
                "cases.jsonl",
                "--init-checkpoint",
                "last.pt",
                "--qwen36-model-path",
                "/models/Qwen3.6-27B",
                "--steps",
                "1",
                "--out-dir",
                "runs/out",
            ]
        )

        self.assertEqual(args.qwen36_model_path, "/models/Qwen3.6-27B")
        self.assertGreater(args.teacher_max_new_tokens, 0)

    def test_teacher_loader_falls_back_to_image_text_model(self) -> None:
        module = _load_script()
        calls = []

        class FailingCausal:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                calls.append("causal")
                raise ValueError("unsupported config")

        class ImageText:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                calls.append("image_text")
                return "loaded-image-text"

        model = module.load_teacher_model(
            "/models/Qwen3.6-27B",
            teacher_dtype="bf16",
            device_map="auto",
            auto_model_for_causal_lm=FailingCausal,
            auto_model_for_image_text_to_text=ImageText,
        )

        self.assertEqual(model, "loaded-image-text")
        self.assertEqual(calls, ["causal", "image_text"])


if __name__ == "__main__":
    unittest.main()
