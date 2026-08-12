"""Evaluate the trained PyTorch LM on the orthocenter auxiliary example."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import alphageometry
import graph as gh
import problem as pr
from pytorch_lm_inference import PytorchLanguageModelInference


PROBLEM_TXT = (
    "a b c = triangle a b c; "
    "d = on_tline d b a c, on_tline d c a b ? perp a d b c"
)

EXPECTED_AUX = "e = on_line e a c, on_line e b d"

EXPECTED_AUX_PROBLEM_TXT = (
    "a b c = triangle a b c; "
    "d = on_tline d b a c, on_tline d c a b; "
    f"{EXPECTED_AUX} ? perp a d b c"
)


def load_definitions(path: Path) -> dict[str, pr.Definition]:
    return pr.Definition.from_txt_file(str(path), to_dict=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--defs_file", type=Path, default=Path("data/defs.txt"))
    parser.add_argument("--rules_file", type=Path, default=Path("data/rules.txt"))
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/eval/orthocenter_model"))
    parser.add_argument("--beam_size", type=int, default=8)
    parser.add_argument("--search_depth", type=int, default=2)
    parser.add_argument("--max_decode_len", type=int, default=96)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def run_problem(
    problem_txt: str,
    definitions: dict[str, pr.Definition],
    out_file: Path,
) -> tuple[pr.Problem, gh.Graph, bool]:
    problem = pr.Problem.from_txt(problem_txt, translate=True)
    graph, _ = gh.Graph.build_problem(problem, definitions)
    solved = alphageometry.run_ddar(graph, problem, str(out_file))
    return problem, graph, solved


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    definitions = load_definitions(args.defs_file)
    rules = pr.Theorem.from_txt_file(str(args.rules_file), to_dict=True)
    alphageometry.DEFINITIONS = definitions
    alphageometry.RULES = rules

    visible_problem, visible_graph, ddar_solved = run_problem(
        PROBLEM_TXT,
        definitions,
        args.out_dir / "ddar_visible_solution.txt",
    )
    aux_problem, _aux_graph, expected_aux_solved = run_problem(
        EXPECTED_AUX_PROBLEM_TXT,
        definitions,
        args.out_dir / "ddar_expected_aux_solution.txt",
    )

    setup_prompt = visible_problem.setup_str_from_problem(definitions)
    lm_prompt = setup_prompt + " {F1} x00"
    model = PytorchLanguageModelInference(
        checkpoint=args.checkpoint,
        tokenizer=args.tokenizer,
        batch_size=args.beam_size,
        max_decode_len=args.max_decode_len,
        device=args.device,
    )
    model_outputs = model.beam_decode(lm_prompt, eos_tokens=[";"])
    translations = [
        alphageometry.try_translate_constrained_to_construct(output, visible_graph)
        for output in model_outputs["seqs_str"]
    ]

    alphageometry_solution = args.out_dir / "alphageometry_solution.txt"
    alphageometry_solved = alphageometry.run_alphageometry(
        model=model,
        p=visible_problem,
        search_depth=args.search_depth,
        beam_size=args.beam_size,
        out_file=str(alphageometry_solution),
    )

    expected_aux_prompt = aux_problem.setup_str_from_problem(definitions)
    summary = {
        "problem": visible_problem.txt(),
        "expected_auxiliary": EXPECTED_AUX,
        "expected_auxiliary_problem": aux_problem.txt(),
        "setup_prompt": setup_prompt,
        "lm_prompt": lm_prompt,
        "expected_auxiliary_prompt": expected_aux_prompt,
        "expected_auxiliary_target": "e : C a c e 02 C b d e 03 ;",
        "ddar_visible_solved": ddar_solved,
        "ddar_expected_aux_solved": expected_aux_solved,
        "alphageometry_solved": alphageometry_solved,
        "model_outputs": [
            {"text": text, "score": score, "translation": translation}
            for text, score, translation in zip(
                model_outputs["seqs_str"], model_outputs["scores"], translations
            )
        ],
        "visible_ddar_solution": str(args.out_dir / "ddar_visible_solution.txt"),
        "expected_aux_solution": str(args.out_dir / "ddar_expected_aux_solution.txt"),
        "alphageometry_solution": str(alphageometry_solution),
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    raise SystemExit(main())
