#!/usr/bin/env python3
"""Verify every size and SHA-256 listed in an educational release manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release_dir", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("release/manifest-v4.json"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    failed = False
    artifacts = manifest.get("artifacts", manifest.get("files"))
    if not isinstance(artifacts, list):
        raise ValueError("manifest must contain an 'artifacts' or 'files' list")
    for artifact in artifacts:
        path = args.release_dir / artifact["path"]
        if not path.is_file():
            print(f"MISSING  {artifact['path']}")
            failed = True
            continue
        size_ok = path.stat().st_size == artifact["bytes"]
        digest_ok = sha256(path) == artifact["sha256"]
        state = "OK" if size_ok and digest_ok else "FAILED"
        print(f"{state:7}  {artifact['path']}")
        failed |= not (size_ok and digest_ok)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
