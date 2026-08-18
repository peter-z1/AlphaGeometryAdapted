"""Audit Qwen token lengths and AlphaGeometry punctuation before training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "outputs" / "qwen3_5_9b" / "data"
DEFAULT_OUT = REPO_ROOT / "outputs" / "qwen3_5_9b" / "tokenizer_audit.json"
DEFAULT_REVISION = "68c46c4b3498877f3ef123c856ecfde50c39f404"


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def summarize(values: list[int], limit: int) -> dict[str, object]:
    return {
        "rows": len(values),
        "min": min(values, default=0),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values, default=0),
        "over_limit": sum(value > limit for value in values),
        "limit": limit,
    }


def text_rows(path: Path, max_rows: int) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as src:
        for index, line in enumerate(src):
            if max_rows and index >= max_rows:
                break
            if line.strip():
                yield line.rstrip("\n")


def jsonl_rows(path: Path, max_rows: int) -> Iterable[dict[str, object]]:
    with path.open("r", encoding="utf-8") as src:
        for index, line in enumerate(src):
            if max_rows and index >= max_rows:
                break
            if line.strip():
                yield json.loads(line)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B-Base")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max_rows", type=int, default=10_000)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--cache_dir", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "install experiments/qwen3_5_9b/requirements.txt before the tokenizer audit"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, cache_dir=args.cache_dir
    )
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)

    syntax_lengths = [
        len(encode(text))
        for text in text_rows(args.data_dir / "syntax" / "train.txt", args.max_rows)
    ]
    prompt_lengths: list[int] = []
    target_lengths: list[int] = []
    combined_lengths: list[int] = []
    unknown_tokens = 0
    for row in jsonl_rows(args.data_dir / "auxiliary" / "train.jsonl", args.max_rows):
        prompt_ids = encode(str(row["prompt"]).rstrip() + " ")
        target_ids = encode(str(row["target"]).strip())
        prompt_lengths.append(len(prompt_ids))
        target_lengths.append(len(target_ids))
        combined_lengths.append(len(prompt_ids) + len(target_ids) + 1)
        if tokenizer.unk_token_id is not None:
            unknown_tokens += (prompt_ids + target_ids).count(tokenizer.unk_token_id)

    fragments = ["{S}", "{F1}", "x00", ":", ";", " ? ", "D", "T", "C", "00", "99"]
    fragment_audit = {
        fragment: {
            "ids": encode(fragment),
            "tokens": tokenizer.convert_ids_to_tokens(encode(fragment)),
        }
        for fragment in fragments
    }
    report = {
        "model": args.model,
        "revision": args.revision,
        "vocab_size": len(tokenizer),
        "max_length": args.max_length,
        "syntax": summarize(syntax_lengths, args.max_length),
        "auxiliary_prompt": summarize(prompt_lengths, args.max_length),
        "auxiliary_target": summarize(target_lengths, args.max_length),
        "auxiliary_combined": summarize(combined_lengths, args.max_length),
        "unknown_tokens": unknown_tokens,
        "fragments": fragment_audit,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["auxiliary_combined"]["over_limit"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
