"""Prepare deterministic Qwen3.5 syntax and auxiliary experiment data.

The full pretraining corpus is large, so this script takes an exactly sized,
systematic sample without holding its lines in memory.  The already stable
auxiliary train/validation/test split is copied to plain JSONL and gains a
``completion`` alias understood by generic prompt-completion tooling.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
from pathlib import Path
from typing import Iterable, TextIO


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
RELEASE_NAME = "alphageometry_educational_release_v4"


def first_existing(paths: Iterable[Path]) -> Path:
    candidates = list(paths)
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


DEFAULT_SYNTAX_DIR = first_existing(
    [
        REPO_ROOT / "artifacts" / RELEASE_NAME / "datasets" / "pretraining_rich_v2",
        WORKSPACE_ROOT
        / "release_artifacts"
        / RELEASE_NAME
        / "datasets"
        / "pretraining_rich_v2",
    ]
)
DEFAULT_AUXILIARY_DIR = REPO_ROOT / "data" / "generated" / "auxiliary_v4"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "qwen3_5_9b" / "data"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as src:
        while chunk := src.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def systematic_indices(total_rows: int, target_rows: int, seed: int) -> list[int]:
    """Return exactly ``target_rows`` sorted indices spread across the source."""
    if total_rows < 0 or target_rows < 0:
        raise ValueError("row counts must be non-negative")
    if target_rows > total_rows:
        raise ValueError(
            f"cannot select {target_rows} rows from a {total_rows}-row source"
        )
    if target_rows == 0:
        return []
    if target_rows == total_rows:
        return list(range(total_rows))

    rng = random.Random(seed)
    phase = rng.random()
    indices = [int((index + phase) * total_rows / target_rows) for index in range(target_rows)]
    if len(set(indices)) != target_rows:
        raise AssertionError("systematic sampler unexpectedly produced duplicate indices")
    return indices


def sample_text_file(
    source: Path,
    destination: Path,
    total_rows: int,
    target_rows: int,
    seed: int,
) -> dict[str, object]:
    """Write an exactly sized deterministic sample and audit its source."""
    selected = iter(systematic_indices(total_rows, target_rows, seed))
    next_index = next(selected, None)
    source_digest = hashlib.sha256()
    output_digest = hashlib.sha256()
    rows_read = 0
    rows_written = 0

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with source.open("rb") as src, temporary.open("wb") as dst:
        for row_index, line in enumerate(src):
            source_digest.update(line)
            rows_read += 1
            if row_index != next_index:
                continue
            dst.write(line)
            output_digest.update(line)
            rows_written += 1
            next_index = next(selected, None)

    if rows_read != total_rows:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"{source} has {rows_read} rows, but its release summary says {total_rows}"
        )
    if rows_written != target_rows or next_index is not None:
        temporary.unlink(missing_ok=True)
        raise AssertionError(
            f"selected {rows_written} rows from {source}; expected {target_rows}"
        )
    temporary.replace(destination)
    return {
        "source": str(source),
        "source_rows": rows_read,
        "source_sha256": source_digest.hexdigest(),
        "output": str(destination),
        "output_rows": rows_written,
        "output_sha256": output_digest.hexdigest(),
        "seed": seed,
    }


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def normalize_auxiliary(source: Path, destination: Path) -> dict[str, object]:
    """Copy one auxiliary split while preserving all audit metadata."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    rows = 0
    output_digest = hashlib.sha256()
    with open_text(source) as src, temporary.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_no}: invalid JSONL: {exc}") from exc
            if not row.get("prompt") or not row.get("target"):
                raise ValueError(f"{source}:{line_no}: missing prompt or target")
            row = dict(row)
            row["completion"] = row["target"]
            encoded = (json.dumps(row, sort_keys=True) + "\n").encode("utf-8")
            dst.write(encoded.decode("utf-8"))
            output_digest.update(encoded)
            rows += 1
    temporary.replace(destination)
    return {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "output": str(destination),
        "output_rows": rows,
        "output_sha256": output_digest.hexdigest(),
    }


def release_counts(syntax_dir: Path) -> dict[str, int]:
    summary_path = syntax_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"missing {summary_path}; place the v4 release bundle under artifacts/ "
            "or pass --syntax_source_dir"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {name: int(value) for name, value in summary["splits"].items()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--syntax_source_dir", type=Path, default=DEFAULT_SYNTAX_DIR)
    parser.add_argument("--auxiliary_source_dir", type=Path, default=DEFAULT_AUXILIARY_DIR)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--syntax_rows", type=int, default=250_000)
    parser.add_argument("--syntax_val_rows", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=35)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing prepared-data directory",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.force:
        raise FileExistsError(
            f"{args.out_dir} is not empty; pass --force to replace prepared files"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "experiment": "qwen3_5_9b",
        "seed": args.seed,
        "syntax": {},
        "auxiliary": {},
    }
    if args.syntax_rows or args.syntax_val_rows:
        counts = release_counts(args.syntax_source_dir)
        selections = {
            "train": (args.syntax_rows, args.seed),
            "val": (args.syntax_val_rows, args.seed + 1),
        }
        for split, (target_rows, seed) in selections.items():
            manifest["syntax"][split] = sample_text_file(
                args.syntax_source_dir / f"{split}.txt",
                args.out_dir / "syntax" / f"{split}.txt",
                total_rows=counts[split],
                target_rows=target_rows,
                seed=seed,
            )

    for split in ("train", "val", "test"):
        source = args.auxiliary_source_dir / f"{split}.jsonl.gz"
        if not source.exists():
            source = args.auxiliary_source_dir / f"{split}.jsonl"
        if not source.exists():
            raise FileNotFoundError(f"missing auxiliary split: {source}")
        manifest["auxiliary"][split] = normalize_auxiliary(
            source, args.out_dir / "auxiliary" / f"{split}.jsonl"
        )

    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"manifest": str(manifest_path), **manifest}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
