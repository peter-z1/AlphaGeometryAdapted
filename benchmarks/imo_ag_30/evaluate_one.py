"""Evaluate one canonical IMO-AG-30 problem with released AlphaGeometry-LM."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import logging
import os
from pathlib import Path
import signal
import sys
import time
from typing import Iterator


SUITE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SUITE_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/alphageometry-matplotlib")

import numpy as np  # pylint: disable=wrong-import-position

import alphageometry  # pylint: disable=wrong-import-position
import graph as gh  # pylint: disable=wrong-import-position
import lm_inference  # pylint: disable=wrong-import-position
import numericals as nm  # pylint: disable=wrong-import-position
import problem as pr  # pylint: disable=wrong-import-position


class SearchTimeoutError(TimeoutError):
    pass


@contextmanager
def deadline(seconds: int) -> Iterator[None]:
    def handle_timeout(_signum, _frame):
        raise SearchTimeoutError(f"exceeded {seconds} seconds")

    previous = signal.signal(signal.SIGALRM, handle_timeout)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def save(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem_index", type=int, required=True)
    parser.add_argument(
        "--problems_file", type=Path, default=REPO_ROOT / "examples/imo_ag_30.txt"
    )
    parser.add_argument("--defs_file", type=Path, default=REPO_ROOT / "data/defs.txt")
    parser.add_argument("--rules_file", type=Path, default=REPO_ROOT / "data/rules.txt")
    parser.add_argument(
        "--model_file",
        type=Path,
        default=REPO_ROOT / "src/chatllm/quantized/alphageometry-lm-f32.bin",
    )
    parser.add_argument("--out_root", type=Path, default=SUITE_DIR / "outputs")
    parser.add_argument("--model_batch_size", type=int, default=32)
    parser.add_argument("--beam_size", type=int, default=512)
    parser.add_argument("--search_depth", type=int, default=16)
    parser.add_argument("--timeout_seconds", type=int, default=240)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--enforce_point_distance_checks",
        action="store_true",
        help="Enforce diagram-only too-close/too-far heuristics (off for the canonical suite)",
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.WARNING)
    problems = pr.Problem.from_txt_file(str(args.problems_file), translate=True)
    if not 0 <= args.problem_index < len(problems):
        raise ValueError(
            f"problem_index must be between 0 and {len(problems) - 1}"
        )
    problem = problems[args.problem_index]
    out_dir = args.out_root / problem.url
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    proof_path = out_dir / "proof.txt"

    summary = {
        "problem_index": args.problem_index,
        "problem_name": problem.url,
        "formalization": problem.txt(),
        "model": "released AlphaGeometry-LM 152M",
        "model_batch_size": args.model_batch_size,
        "beam_size": args.beam_size,
        "search_depth": args.search_depth,
        "timeout_seconds": args.timeout_seconds,
        "status": "loading",
        "solved": None,
        "numeric_goal_true": None,
        "complete": False,
    }
    save(summary_path, summary)

    definitions = pr.Definition.from_txt_file(str(args.defs_file), to_dict=True)
    rules = pr.Theorem.from_txt_file(str(args.rules_file), to_dict=True)
    alphageometry.DEFINITIONS = definitions
    alphageometry.RULES = rules

    if not args.enforce_point_distance_checks:
        # These tests only control diagram aesthetics. Some official benchmark
        # inputs use fixed coordinates that violate the fork's thresholds even
        # though their incidence relations remain nondegenerate.
        nm.check_too_close = lambda *_args, **_kwargs: False
        nm.check_too_far = lambda *_args, **_kwargs: False

    np.random.seed(args.seed)
    # Several canonical inputs name unordered circle intersections. The
    # original builder uses the theorem goal to resample the intended branch,
    # so validation must retain the goal as well.
    try:
        graph, _ = alphageometry.build_problem_with_retries(problem, definitions)
        goal_args = [graph.get(name, lambda: int(name)) for name in problem.goal.args]
        summary["numeric_goal_true"] = bool(nm.check(problem.goal.name, goal_args))
    except Exception as exc:
        summary.update(
            status="error",
            error=f"precheck {type(exc).__name__}: {exc}",
            complete=True,
        )
        save(summary_path, summary)
        return 1
    if not summary["numeric_goal_true"]:
        summary.update(status="error", error="numerical goal check failed", complete=True)
        save(summary_path, summary)
        return 1

    model = lm_inference.LanguageModelInference(
        model_file=str(args.model_file),
        mode="beam_search",
        batch_size=args.model_batch_size,
    )
    decode_calls = [0]
    original_beam_decode = model.beam_decode

    def tracked_beam_decode(*decode_args, **decode_kwargs):
        decode_calls[0] += 1
        return original_beam_decode(*decode_args, **decode_kwargs)

    model.beam_decode = tracked_beam_decode

    summary["status"] = "running"
    save(summary_path, summary)
    started = time.monotonic()
    try:
        with deadline(args.timeout_seconds):
            np.random.seed(args.seed)
            summary["solved"] = bool(
                alphageometry.run_alphageometry(
                    model=model,
                    p=problem,
                    search_depth=args.search_depth,
                    beam_size=args.beam_size,
                    out_file=str(proof_path),
                )
            )
            summary["status"] = "solved" if summary["solved"] else "unsolved"
    except SearchTimeoutError as exc:
        summary["status"] = "timeout"
        summary["error"] = str(exc)
    except Exception as exc:  # retain a result instead of losing the array task
        logging.exception("Evaluation failed")
        summary["status"] = "error"
        summary["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        summary["elapsed_seconds"] = round(time.monotonic() - started, 3)
        summary["lm_decode_calls"] = decode_calls[0]
        if summary["solved"]:
            summary["solver_stage"] = (
                "ddar" if decode_calls[0] == 0 else "lm_auxiliary_then_ddar"
            )
        elif summary["status"] == "timeout":
            summary["solver_stage"] = (
                "ddar" if decode_calls[0] == 0 else "lm_auxiliary_search"
            )
        summary["proof"] = str(proof_path) if summary["solved"] else None
        summary["complete"] = True
        save(summary_path, summary)

    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    raise SystemExit(main())
