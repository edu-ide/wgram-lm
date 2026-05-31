from __future__ import annotations

import unittest
from unittest.mock import patch


class QwenDonorQuantizationTests(unittest.TestCase):
    def test_4bit_quantization_is_disabled_when_bitsandbytes_is_missing(self) -> None:
        from wgram_lm.qwen_donor import _build_4bit_quantization_config

        with patch("wgram_lm.qwen_donor._bitsandbytes_available", return_value=False):
            self.assertIsNone(_build_4bit_quantization_config(load_in_4bit=True))

    def test_4bit_quantization_is_disabled_when_not_requested(self) -> None:
        from wgram_lm.qwen_donor import _build_4bit_quantization_config

        self.assertIsNone(_build_4bit_quantization_config(load_in_4bit=False))


if __name__ == "__main__":
    unittest.main()
