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


if __name__ == "__main__":
    unittest.main()
