import unittest


class CleanFilterTests(unittest.TestCase):
    def test_quality_filter_rejects_boilerplate_course_unit(self):
        from qtrm_mm.data.clean_filter import quality_reason

        text = (
            "Course Unit: Themed Settings and Storytelling. "
            "This course unit will delve into the world of the world of examples. "
            "Students will learn through this course unit."
        )

        self.assertEqual(quality_reason({"type": "text", "text": text}), "boilerplate")

    def test_quality_filter_rejects_repeated_scienceqa_letter_prompt(self):
        from qtrm_mm.data.clean_filter import quality_reason

        text = (
            "Lecture: Experiments can be designed to answer specific questions. "
            "Question: Which choice is correct? Choices: A. one B. two Answer with the letter. "
            "Lecture: Experiments can be designed to answer specific questions. "
            "Question: Which choice is correct? Choices: A. one B. two Answer with the letter."
        )

        self.assertEqual(quality_reason({"type": "multimodal_sft", "text": text}), "boilerplate")

    def test_quality_filter_keeps_clean_math_solution(self):
        from qtrm_mm.data.clean_filter import quality_reason

        text = (
            "Problem: If x + 3 = 7, solve for x. "
            "Solution: Subtract 3 from both sides to get x = 4. "
            "The answer is 4."
        )

        self.assertIsNone(quality_reason({"type": "math", "text": text}, min_words=10, max_words=80))

    def test_normalize_row_drops_images_for_text_pilot(self):
        from qtrm_mm.data.clean_filter import normalize_row

        row = {
            "type": "multimodal_sft",
            "source": "example",
            "images": ["a.jpg"],
            "text": "<image>\nQuestion: What is shown?\n\nAnswer: A diagram.",
        }

        out = normalize_row(row, drop_images=True)

        self.assertNotIn("images", out)
        self.assertNotIn("<image>", out["text"])
        self.assertEqual(out["source"], "example")


if __name__ == "__main__":
    unittest.main()
