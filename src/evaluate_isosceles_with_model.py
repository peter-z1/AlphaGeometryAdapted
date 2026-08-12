"""Evaluate the trained PyTorch LM on the basic isosceles triangle theorem."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import alphageometry
import graph as gh
import problem as pr
from pytorch_lm_inference import PytorchLanguageModelInference


GIVEN_ISO_DEFINITION = """given_iso a b c
c : a b c
 =
a : ; b : ; c : cong a b a c
isos
"""

PROBLEM_TXT = (
    "a b c = given_iso a b c ? eqangle a b b c b c a c"
)


def load_definitions(path: Path) -> dict[str, pr.Definition]:
    definitions = pr.Definition.from_txt_file(str(path), to_dict=True)
    given_iso = pr.Definition.from_txt(GIVEN_ISO_DEFINITION)
    definitions[given_iso.construction.name] = given_iso
    return definitions


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--defs_file", type=Path, default=Path("data/defs.txt"))
    parser.add_argument("--rules_file", type=Path, default=Path("data/rules.txt"))
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/eval/isosceles_model"))
    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument("--search_depth", type=int, default=1)
    parser.add_argument("--max_decode_len", type=int, default=24)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    definitions = load_definitions(args.defs_file)
    rules = pr.Theorem.from_txt_file(str(args.rules_file), to_dict=True)
    alphageometry.DEFINITIONS = definitions
    alphageometry.RULES = rules

    problem = pr.Problem.from_txt(PROBLEM_TXT, translate=True)
    g, _ = gh.Graph.build_problem(problem, definitions)
    ddar_solution = args.out_dir / "ddar_solution.txt"
    alphageometry_solution = args.out_dir / "alphageometry_solution.txt"

    ddar_solved = alphageometry.run_ddar(g, problem, str(ddar_solution))

    setup_prompt = problem.setup_str_from_problem(definitions)
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
        alphageometry.try_translate_constrained_to_construct(output, g)
        for output in model_outputs["seqs_str"]
    ]

    alphageometry_solved = alphageometry.run_alphageometry(
        model=model,
        p=problem,
        search_depth=args.search_depth,
        beam_size=args.beam_size,
        out_file=str(alphageometry_solution),
    )

    summary = {
        "problem": PROBLEM_TXT,
        "setup_prompt": setup_prompt,
        "lm_prompt": lm_prompt,
        "ddar_solved": ddar_solved,
        "alphageometry_solved": alphageometry_solved,
        "note": (
            "This theorem is solved by DD+AR before auxiliary LM search is needed; "
            "the trained PyTorch model was still loaded and decoded once on the theorem prompt."
        ),
        "model_outputs": [
            {"text": text, "score": score, "translation": translation}
            for text, score, translation in zip(
                model_outputs["seqs_str"], model_outputs["scores"], translations
            )
        ],
        "ddar_solution": str(ddar_solution),
        "alphageometry_solution": str(alphageometry_solution),
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if alphageometry_solved else 1


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    raise SystemExit(main())
