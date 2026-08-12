"""Dependency-free unit tests for Hugging Face inference text handling."""

import unittest

from huggingface_lm_inference import HuggingFaceLanguageModelInference


class HuggingFaceInferenceTextTest(unittest.TestCase):
    def test_truncate_at_first_semicolon(self):
        self.assertEqual(
            HuggingFaceLanguageModelInference._truncate_at_semicolon(
                "  e : C a b e 00 ; ignored"
            ),
            "e : C a b e 00 ;",
        )

    def test_preserve_unfinished_candidate(self):
        self.assertEqual(
            HuggingFaceLanguageModelInference._truncate_at_semicolon("  e : C a b e 00  "),
            "e : C a b e 00",
        )


if __name__ == "__main__":
    unittest.main()

