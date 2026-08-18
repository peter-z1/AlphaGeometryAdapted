"""Offline unit tests for the Qwen3.5 experiment scaffolding."""

from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

import prepare_data
import train


class FakeTokenizer:
    eos_token_id = 99
    pad_token_id = 0

    @staticmethod
    def encode(text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(character) for character in text]


class PrepareDataTest(unittest.TestCase):
    def test_systematic_indices_are_exact_and_reproducible(self):
        first = prepare_data.systematic_indices(100, 13, 7)
        second = prepare_data.systematic_indices(100, 13, 7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 13)
        self.assertEqual(len(set(first)), 13)
        self.assertEqual(first, sorted(first))

    def test_sample_text_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            source.write_text("".join(f"row-{i}\n" for i in range(20)), encoding="utf-8")
            output = root / "sample.txt"
            report = prepare_data.sample_text_file(source, output, 20, 5, 2)
            self.assertEqual(report["output_rows"], 5)
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 5)

    def test_normalize_auxiliary_adds_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train.jsonl.gz"
            with gzip.open(source, "wt", encoding="utf-8") as dst:
                dst.write(json.dumps({"prompt": "p", "target": "t;"}) + "\n")
            output = root / "train.jsonl"
            report = prepare_data.normalize_auxiliary(source, output)
            row = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["output_rows"], 1)
            self.assertEqual(row["completion"], "t;")


class EncodingTest(unittest.TestCase):
    def test_auxiliary_loss_masks_prompt(self):
        ids, labels = train.encode_auxiliary(FakeTokenizer(), "abc", "x;", 32, "error")
        self.assertEqual(ids, [97, 98, 99, 32, 120, 59, 99])
        self.assertEqual(labels, [-100, -100, -100, -100, 120, 59, 99])

    def test_syntax_loss_covers_all_tokens(self):
        ids, labels = train.encode_syntax(FakeTokenizer(), "ab", 8, "error")
        self.assertEqual(ids, [97, 98, 99])
        self.assertEqual(labels, ids)

    def test_auxiliary_left_truncation_preserves_target(self):
        ids, labels = train.encode_auxiliary(FakeTokenizer(), "abcdef", "x;", 5, "truncate_left")
        self.assertEqual(ids[-3:], [120, 59, 99])
        self.assertEqual(labels[-3:], [120, 59, 99])


class ConfigurationTest(unittest.TestCase):
    def test_checked_in_configs_validate(self):
        config_dir = Path(__file__).with_name("configs")
        for path in config_dir.glob("*.json"):
            with self.subTest(path=path.name):
                train.validate_config(json.loads(path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
