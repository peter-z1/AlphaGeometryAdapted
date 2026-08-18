"""Evaluate one language-model checkpoint on held-out data and proof search.

The model is loaded once.  The evaluator first measures top-1/top-k completion
quality on the locked strict auxiliary test split, then runs the same model on
IMO-AG-30 and the final 33 JGEX problems (indices 198--230).  Results are
written incrementally. Hugging Face adapters and native PyTorch checkpoints use
the same metrics and proof-search limits.
"""

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
from typing import Any, Iterator


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "outputs/qwen3_5_9b/matplotlib"))

import numpy as np  # pylint: disable=wrong-import-position

import alphageometry  # pylint: disable=wrong-import-position
from huggingface_lm_inference import (  # pylint: disable=wrong-import-position
    HuggingFaceLanguageModelInference,
)
import numericals as nm  # pylint: disable=wrong-import-position
import problem as pr  # pylint: disable=wrong-import-position


DEFAULT_MODEL = "Qwen/Qwen3.5-9B-Base"
DEFAULT_REVISION = "68c46c4b3498877f3ef123c856ecfde50c39f404"
DEFAULT_CACHE = REPO_ROOT / "outputs/qwen3_5_9b/hf_cache"
DEFAULT_TEST = REPO_ROOT / "outputs/qwen3_5_9b/data/auxiliary/test.jsonl"
DEFAULT_OUT = REPO_ROOT / "outputs/qwen3_5_9b/evaluations"


class SearchTimeoutError(TimeoutError):
    """Raised when one end-to-end theorem exceeds its independent budget."""


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


def normalize_completion(text: str) -> str:
    """Compare formal completions independently of repeated whitespace."""
    return " ".join(text.strip().split())


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def append_jsonl(handle, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, sort_keys=True) + "\n")
    handle.flush()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as src:
        return [json.loads(line) for line in src if line.strip()]


def aggregate_status(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "total": len(rows),
        "solved": sum(row.get("solved") is True for row in rows),
        "direct_ddar": sum(row.get("solver_stage") == "ddar" for row in rows),
        "lm_auxiliary_then_ddar": sum(
            row.get("solver_stage") == "lm_auxiliary_then_ddar" for row in rows
        ),
        "unsolved": sum(row.get("status") == "unsolved" for row in rows),
        "timeouts": sum(row.get("status") == "timeout" for row in rows),
        "errors": sum(row.get("status") == "error" for row in rows),
    }


def evaluate_heldout(
    model: Any,
    rows: list[dict[str, Any]],
    output_path: Path,
    definitions: dict[str, pr.Definition],
) -> dict[str, Any]:
    counts = {
        "rows": len(rows),
        "stored_target_valid": 0,
        "top1_exact": 0,
        "topk_exact": 0,
        "top1_exact_and_valid": 0,
        "topk_exact_and_valid": 0,
        "top1_valid": 0,
        "topk_any_valid": 0,
        "generation_errors": 0,
        "graph_errors": 0,
    }
    started = time.monotonic()
    with output_path.open("w", encoding="utf-8", buffering=1) as dst:
        for index, row in enumerate(rows):
            target = normalize_completion(str(row["target"]))
            record: dict[str, Any] = {
                "index": index,
                "id": row.get("id"),
                "target": target,
            }
            try:
                # Held-out labels may contain several dependent point clauses,
                # so completion evaluation runs to the model EOS.  End-to-end
                # proof search still decodes one construction clause at a time.
                outputs = model.completion_decode(str(row["prompt"]))
                predictions = [normalize_completion(text) for text in outputs["seqs_str"]]
                record["predictions"] = predictions
                record["scores"] = outputs["scores"]
                exact = [prediction == target for prediction in predictions]
                record["top1_exact"] = bool(exact and exact[0])
                record["topk_exact"] = any(exact)
                counts["top1_exact"] += int(record["top1_exact"])
                counts["topk_exact"] += int(record["topk_exact"])

                valid = [False] * len(predictions)
                validation_errors = ["not checked"] * len(predictions)
                try:
                    # Hidden auxiliaries deliberately leave gaps in point names.
                    # Translating here would compact those names and can make the
                    # stored target appear to redefine an existing point.
                    problem = pr.Problem.from_txt(
                        str(row["visible_problem"]), translate=False
                    )
                    graph, _ = alphageometry.build_problem_with_retries(
                        problem, definitions, max_attempts=20
                    )
                    target_translation = (
                        alphageometry.try_translate_constrained_sequence_to_construct(
                            target, graph
                        )
                    )
                    record["target_translation"] = target_translation
                    record["stored_target_valid"] = not target_translation.startswith(
                        "ERROR:"
                    )
                    counts["stored_target_valid"] += int(
                        record["stored_target_valid"]
                    )

                    valid = []
                    validation_errors = []
                    for prediction in predictions:
                        translation = (
                            alphageometry.try_translate_constrained_sequence_to_construct(
                                prediction, graph
                            )
                        )
                        is_valid = not translation.startswith("ERROR:")
                        valid.append(is_valid)
                        validation_errors.append("" if is_valid else translation)
                except Exception as exc:  # numerical audit must not lose predictions
                    counts["graph_errors"] += 1
                    record["graph_error"] = f"{type(exc).__name__}: {exc}"
                record["valid"] = valid
                record["validation_errors"] = validation_errors
                record["top1_valid"] = bool(valid and valid[0])
                record["topk_any_valid"] = any(valid)
                record["top1_exact_and_valid"] = bool(
                    exact and valid and exact[0] and valid[0]
                )
                record["topk_exact_and_valid"] = any(
                    is_exact and is_valid
                    for is_exact, is_valid in zip(exact, valid)
                )
                counts["top1_valid"] += int(record["top1_valid"])
                counts["topk_any_valid"] += int(record["topk_any_valid"])
                counts["top1_exact_and_valid"] += int(
                    record["top1_exact_and_valid"]
                )
                counts["topk_exact_and_valid"] += int(
                    record["topk_exact_and_valid"]
                )
            except Exception as exc:
                counts["generation_errors"] += 1
                record["error"] = f"{type(exc).__name__}: {exc}"
            append_jsonl(dst, record)
    counts["elapsed_seconds"] = round(time.monotonic() - started, 3)
    counts["top1_exact_rate"] = counts["top1_exact"] / max(1, counts["rows"])
    counts["topk_exact_rate"] = counts["topk_exact"] / max(1, counts["rows"])
    counts["stored_target_valid_rate"] = (
        counts["stored_target_valid"] / max(1, counts["rows"])
    )
    counts["top1_exact_and_valid_rate"] = (
        counts["top1_exact_and_valid"] / max(1, counts["rows"])
    )
    counts["topk_exact_and_valid_rate"] = (
        counts["topk_exact_and_valid"] / max(1, counts["rows"])
    )
    counts["top1_valid_rate"] = counts["top1_valid"] / max(1, counts["rows"])
    counts["topk_any_valid_rate"] = counts["topk_any_valid"] / max(1, counts["rows"])
    return counts


def evaluate_problem(
    model: Any,
    problem: pr.Problem,
    problem_index: int,
    suite_name: str,
    out_dir: Path,
    definitions: dict[str, pr.Definition],
    timeout_seconds: int,
    search_depth: int,
    beam_size: int,
    seed: int,
) -> dict[str, Any]:
    proof_path = out_dir / suite_name / problem.url / "proof.txt"
    proof_path.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "problem_index": problem_index,
        "problem_name": problem.url,
        "formalization": problem.txt(),
        "status": "precheck",
        "solved": None,
        "solver_stage": None,
        "lm_decode_calls": 0,
    }
    np.random.seed(seed)
    try:
        graph, _ = alphageometry.build_problem_with_retries(problem, definitions)
        goal_args = [graph.get(name, lambda: int(name)) for name in problem.goal.args]
        result["numeric_goal_true"] = bool(nm.check(problem.goal.name, goal_args))
        if not result["numeric_goal_true"]:
            raise ValueError("numerical goal check failed")
    except Exception as exc:
        result.update(
            status="error",
            error=f"precheck {type(exc).__name__}: {exc}",
        )
        return result

    decode_calls = [0]
    search_stats: dict[str, int] = {}
    original_decode = model.beam_decode

    def tracked_decode(*args, **kwargs):
        decode_calls[0] += 1
        return original_decode(*args, **kwargs)

    model.beam_decode = tracked_decode  # type: ignore[method-assign]
    started = time.monotonic()
    try:
        result["status"] = "running"
        with deadline(timeout_seconds):
            np.random.seed(seed)
            result["solved"] = bool(
                alphageometry.run_alphageometry(
                    model=model,
                    p=problem,
                    search_depth=search_depth,
                    beam_size=beam_size,
                    out_file=str(proof_path),
                    search_stats=search_stats,
                )
            )
            result["status"] = "solved" if result["solved"] else "unsolved"
    except SearchTimeoutError as exc:
        result.update(status="timeout", error=str(exc))
    except Exception as exc:
        logging.exception("Evaluation failed for %s", problem.url)
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
    finally:
        model.beam_decode = original_decode  # type: ignore[method-assign]
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        result["lm_decode_calls"] = decode_calls[0]
        result["search_stats"] = search_stats
        if result.get("solved"):
            result["solver_stage"] = (
                "ddar" if decode_calls[0] == 0 else "lm_auxiliary_then_ddar"
            )
            result["proof"] = str(proof_path)
        elif decode_calls[0] > 0:
            result["solver_stage"] = "lm_auxiliary_search"
    return result


def suite_specs() -> list[tuple[str, Path, int, int]]:
    return [
        ("imo_ag_30", REPO_ROOT / "examples/imo_ag_30.txt", 0, 30),
        ("jgex_33", REPO_ROOT / "examples/jgex_ag_231.txt", 198, 33),
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--backend", choices=("huggingface", "pytorch"), default="huggingface"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--test_file", type=Path, default=DEFAULT_TEST)
    parser.add_argument(
        "--eq_test_file",
        type=Path,
        help="optional held-out eqangle/eqratio test split scored after --test_file",
    )
    parser.add_argument("--out_root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--model_batch_size", type=int, default=8)
    parser.add_argument("--max_decode_len", type=int, default=128)
    parser.add_argument("--beam_size", type=int, default=8)
    parser.add_argument("--search_depth", type=int, default=2)
    parser.add_argument("--timeout_seconds", type=int, default=180)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_test_examples", type=int, default=0)
    parser.add_argument("--skip_heldout", action="store_true")
    parser.add_argument("--skip_proof_search", action="store_true")
    parser.add_argument("--validate_only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    heldout_rows = load_jsonl(args.test_file)
    eq_heldout_rows = load_jsonl(args.eq_test_file) if args.eq_test_file else []
    if args.max_test_examples:
        heldout_rows = heldout_rows[: args.max_test_examples]
        eq_heldout_rows = eq_heldout_rows[: args.max_test_examples]
    suites: list[tuple[str, list[pr.Problem], int]] = []
    for suite_name, path, start, count in suite_specs():
        all_problems = pr.Problem.from_txt_file(str(path), translate=True)
        selected = all_problems[start : start + count]
        if len(selected) != count:
            raise ValueError(f"{path} provided {len(selected)} of {count} requested problems")
        suites.append((suite_name, selected, start))

    validation = {
        "label": args.label,
        "backend": args.backend,
        "test_file": str(args.test_file),
        "eq_test_file": str(args.eq_test_file) if args.eq_test_file else None,
        "adapter": str(args.adapter) if args.adapter else None,
        "adapter_exists": args.adapter.is_dir() if args.adapter else True,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "checkpoint_exists": args.checkpoint.is_file() if args.checkpoint else False,
        "tokenizer": str(args.tokenizer) if args.tokenizer else None,
        "tokenizer_exists": args.tokenizer.is_file() if args.tokenizer else False,
        "heldout_examples": len(heldout_rows),
        "eq_heldout_examples": len(eq_heldout_rows),
        "suites": {name: len(problems) for name, problems, _ in suites},
    }
    required_artifacts_exist = validation["adapter_exists"]
    if args.backend == "pytorch":
        required_artifacts_exist = bool(
            validation["checkpoint_exists"] and validation["tokenizer_exists"]
        )
    if args.validate_only:
        print(json.dumps(validation, indent=2, sort_keys=True))
        return 0 if required_artifacts_exist else 2
    if args.backend == "huggingface" and not validation["adapter_exists"]:
        raise FileNotFoundError(f"adapter does not exist: {args.adapter}")
    if args.backend == "pytorch" and not args.checkpoint:
        raise ValueError("--checkpoint is required for the PyTorch backend")
    if args.backend == "pytorch" and not validation["checkpoint_exists"]:
        raise FileNotFoundError(f"checkpoint does not exist: {args.checkpoint}")
    if args.backend == "pytorch" and not args.tokenizer:
        raise ValueError("--tokenizer is required for the PyTorch backend")
    if args.backend == "pytorch" and not validation["tokenizer_exists"]:
        raise FileNotFoundError(f"tokenizer does not exist: {args.tokenizer}")

    logging.getLogger().setLevel(logging.WARNING)
    definitions = pr.Definition.from_txt_file(
        str(REPO_ROOT / "data/defs.txt"), to_dict=True
    )
    rules = pr.Theorem.from_txt_file(str(REPO_ROOT / "data/rules.txt"), to_dict=True)
    alphageometry.DEFINITIONS = definitions
    alphageometry.RULES = rules
    nm.check_too_close = lambda *_args, **_kwargs: False
    nm.check_too_far = lambda *_args, **_kwargs: False

    if args.backend == "huggingface":
        model = HuggingFaceLanguageModelInference(
            model_name=args.model,
            revision=args.revision,
            adapter=args.adapter or "",
            batch_size=args.model_batch_size,
            max_decode_len=args.max_decode_len,
            device=args.device,
            dtype="bfloat16",
            load_in_4bit=args.device == "cuda",
            cache_dir=args.cache_dir,
        )
    else:
        from pytorch_lm_inference import (  # pylint: disable=import-outside-toplevel
            PytorchLanguageModelInference,
        )

        model = PytorchLanguageModelInference(
            checkpoint=args.checkpoint,
            tokenizer=args.tokenizer,
            batch_size=args.model_batch_size,
            max_decode_len=args.max_decode_len,
            device=args.device,
        )

    output_dir = args.out_root / args.label
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary: dict[str, Any] = {
        **validation,
        "model": args.model if args.backend == "huggingface" else None,
        "revision": args.revision if args.backend == "huggingface" else None,
        "settings": {
            "model_batch_size": args.model_batch_size,
            "max_decode_len": args.max_decode_len,
            "beam_size": args.beam_size,
            "search_depth": args.search_depth,
            "timeout_seconds": args.timeout_seconds,
            "seed": args.seed,
        },
        "status": "heldout",
        "complete": False,
        "heldout": None,
        "eq_heldout": None,
        "proof_search": {},
    }
    atomic_json(summary_path, summary)
    if not args.skip_heldout:
        summary["heldout"] = evaluate_heldout(
            model,
            heldout_rows,
            output_dir / "heldout_predictions.jsonl",
            definitions,
        )
        atomic_json(summary_path, summary)
        if eq_heldout_rows:
            summary["status"] = "eq_heldout"
            summary["eq_heldout"] = evaluate_heldout(
                model,
                eq_heldout_rows,
                output_dir / "eq_heldout_predictions.jsonl",
                definitions,
            )
            atomic_json(summary_path, summary)

    for suite_name, problems, original_start in ([] if args.skip_proof_search else suites):
        suite_rows: list[dict[str, Any]] = []
        summary["status"] = f"proof_search:{suite_name}"
        for offset, problem in enumerate(problems):
            result = evaluate_problem(
                model=model,
                problem=problem,
                problem_index=original_start + offset,
                suite_name=suite_name,
                out_dir=output_dir,
                definitions=definitions,
                timeout_seconds=args.timeout_seconds,
                search_depth=args.search_depth,
                beam_size=args.beam_size,
                seed=args.seed,
            )
            suite_rows.append(result)
            summary["proof_search"][suite_name] = {
                "counts": aggregate_status(suite_rows),
                "results": suite_rows,
            }
            atomic_json(summary_path, summary)

    summary.update(status="complete", complete=True)
    atomic_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
