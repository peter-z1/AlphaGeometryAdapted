"""Unit tests for backend-neutral PyTorch inference entry points."""

import unittest
from unittest.mock import Mock

from pytorch_lm_inference import PytorchLanguageModelInference


class PytorchInferenceDispatchTest(unittest.TestCase):
    def setUp(self):
        self.model = PytorchLanguageModelInference.__new__(
            PytorchLanguageModelInference
        )
        self.model._decode = Mock(  # pylint: disable=protected-access
            return_value={"seqs_str": [], "scores": []}
        )

    def test_completion_decode_does_not_stop_at_first_semicolon(self):
        self.model.completion_decode("prompt")
        self.model._decode.assert_called_once_with(  # pylint: disable=protected-access
            "prompt", stop_at_semicolon=False
        )

    def test_proof_search_decode_stops_at_first_semicolon(self):
        self.model.beam_decode("prompt", [";"])
        self.model._decode.assert_called_once_with(  # pylint: disable=protected-access
            "prompt", stop_at_semicolon=True
        )

    def test_proof_search_rejects_unknown_stop_sequence(self):
        with self.assertRaisesRegex(ValueError, "only ';' is supported"):
            self.model.beam_decode("prompt", ["."])


if __name__ == "__main__":
    unittest.main()
