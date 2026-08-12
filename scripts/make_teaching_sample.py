#!/usr/bin/env python3
"""Copy a deterministic line sample from a large language-model corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="newline-delimited source corpus")
    parser.add_argument("output", type=Path, help="sample text file to create")
    parser.add_argument("--lines", type=int, default=2000)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="optional JSON manifest (defaults to <output>.manifest.json)",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    if args.lines < 1:
        raise ValueError("--lines must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    copied = 0
    with args.input.open("rb") as source, args.output.open("wb") as target:
        for line in source:
            if copied >= args.lines:
                break
            target.write(line)
            copied += 1

    manifest_path = args.manifest or args.output.with_suffix(
        args.output.suffix + ".manifest.json"
    )
    manifest = {
        "method": "first_n_lines",
        "requested_lines": args.lines,
        "copied_lines": copied,
        "source_name": args.input.name,
        "output_name": args.output.name,
        "output_bytes": args.output.stat().st_size,
        "output_sha256": sha256(args.output),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
