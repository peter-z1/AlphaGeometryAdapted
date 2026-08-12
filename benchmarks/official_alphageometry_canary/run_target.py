"""Instrumented official AlphaGeometry search with parallel DD+AR checks.

The language model and search semantics are the released DeepMind versions.
Independent DD+AR checks from a search depth are distributed across CPU
workers and overlap the next GPU decodes, restoring optimizations omitted from
the public reference runner.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Dict, List, Tuple

from parallel_ddar_worker import evaluate_candidate, init_worker


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / "external" / "google_deepmind_alphageometry"
MELIAD = OFFICIAL / "meliad_lib" / "meliad"


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--problem_name", required=True)
  parser.add_argument("--problems_file", type=Path,
                      default=OFFICIAL / "imo_ag_30.txt")
  parser.add_argument("--defs_file", type=Path,
                      default=OFFICIAL / "defs.txt")
  parser.add_argument("--rules_file", type=Path,
                      default=OFFICIAL / "rules.txt")
  parser.add_argument("--checkpoint_dir", type=Path,
                      default=OFFICIAL / "ag_ckpt_vocab")
  parser.add_argument("--vocab_path", type=Path,
                      default=OFFICIAL / "ag_ckpt_vocab" / "geometry.757.model")
  parser.add_argument("--output_root", type=Path,
                      default=ROOT / "benchmarks" /
                      "official_alphageometry_canary" / "outputs" /
                      "hard_targets")
  parser.add_argument("--model_batch_size", type=int, default=32)
  parser.add_argument("--beam_size", type=int, default=512)
  parser.add_argument("--search_depth", type=int, default=16)
  parser.add_argument("--workers", type=int, default=31)
  parser.add_argument("--candidate_timeout_seconds", type=int, default=900)
  return parser.parse_args()


class EventWriter:
  """Append-only progress log that remains useful during interrupted jobs."""

  def __init__(self, path: Path):
    self._handle = path.open("w", encoding="utf-8", buffering=1)

  def write(self, event: str, **fields: Any) -> None:
    record = {"time_unix": time.time(), "event": event}
    record.update(fields)
    self._handle.write(json.dumps(record, sort_keys=True) + "\n")
    self._handle.flush()

  def close(self) -> None:
    self._handle.close()


def write_json(path: Path, value: Dict[str, Any]) -> None:
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  temporary.replace(path)


def main() -> int:
  args = parse_args()
  if args.workers < 1:
    raise ValueError("--workers must be positive")

  problem_dir = args.output_root / args.problem_name
  problem_dir.mkdir(parents=True, exist_ok=True)
  event_path = problem_dir / "events.jsonl"
  summary_path = problem_dir / "summary.json"
  proof_path = problem_dir / "proof.txt"
  events = EventWriter(event_path)
  started = time.time()
  summary: Dict[str, Any] = {
      "problem_name": args.problem_name,
      "status": "starting",
      "started_unix": started,
      "completed_unix": None,
      "elapsed_seconds": None,
      "configuration": {
          "model_batch_size": args.model_batch_size,
          "beam_size": args.beam_size,
          "search_depth": args.search_depth,
          "workers": args.workers,
          "candidate_timeout_seconds": args.candidate_timeout_seconds,
          "checkpoint_dir": str(args.checkpoint_dir),
          "problems_file": str(args.problems_file),
      },
      "counters": {
          "lm_decode_calls": 0,
          "lm_outputs": 0,
          "valid_constructions": 0,
          "invalid_constructions": 0,
          "ddar_solved": 0,
          "ddar_unsolved": 0,
          "ddar_timeout": 0,
          "ddar_error": 0,
      },
      "depth_completed": -1,
      "proof_path": None,
      "solution": None,
      "error": None,
  }
  write_json(summary_path, summary)
  events.write("runner_start", configuration=summary["configuration"])

  # Spawn CPU-only workers before importing JAX/TensorFlow in this process.
  context = mp.get_context("spawn")
  pool = context.Pool(
      processes=args.workers,
      initializer=init_worker,
      initargs=(str(args.defs_file), str(args.rules_file)),
  )

  try:
    sys.path.insert(0, str(MELIAD))
    sys.path.insert(0, str(OFFICIAL))
    import alphageometry as ag  # pylint: disable=import-outside-toplevel
    import ddar  # pylint: disable=import-outside-toplevel
    import graph as gh  # pylint: disable=import-outside-toplevel
    import lm_inference  # pylint: disable=import-outside-toplevel
    import problem as pr  # pylint: disable=import-outside-toplevel

    definitions = pr.Definition.from_txt_file(str(args.defs_file), to_dict=True)
    rules = pr.Theorem.from_txt_file(str(args.rules_file), to_dict=True)
    ag.DEFINITIONS = definitions
    ag.RULES = rules

    problems = pr.Problem.from_txt_file(
        str(args.problems_file), to_dict=True, translate=True
    )
    if args.problem_name not in problems:
      raise ValueError(
          "problem {!r} not found in {}".format(
              args.problem_name, args.problems_file
          )
      )
    original_problem = problems[args.problem_name]

    gin_files = [
        "base_htrans.gin",
        "size/medium_150M.gin",
        "options/positions_t5.gin",
        "options/lr_cosine_decay.gin",
        "options/seq_1024_nocache.gin",
        "geometry_150M_generate.gin",
    ]
    gin_params = [
        "DecoderOnlyLanguageModelGenerate.output_token_losses=True",
        "TransformerTaskConfig.batch_size={}".format(args.model_batch_size),
        "TransformerTaskConfig.sequence_length=128",
        "Trainer.restore_state_variables=False",
    ]
    events.write("model_loading")
    lm_inference.parse_gin_configuration(
        gin_files, gin_params, gin_paths=[str(MELIAD / "transformer" / "configs"),
                                         str(OFFICIAL)]
    )
    model = lm_inference.LanguageModelInference(
        str(args.vocab_path), str(args.checkpoint_dir), mode="beam_search"
    )
    events.write("model_loaded", batch_size=int(model.batch_size))

    def build(problem: Any) -> Any:
      graph, _ = gh.Graph.build_problem(problem, definitions, verbose=False)
      return graph

    def solved(graph: Any, problem: Any) -> bool:
      goal_args = graph.names2nodes(problem.goal.args)
      return bool(graph.check(problem.goal.name, goal_args))

    def solve_for_proof(problem_string: str) -> bool:
      problem = pr.Problem.from_txt(problem_string)
      graph = build(problem)
      ddar.solve(graph, rules, problem, max_level=1000)
      if not solved(graph, problem):
        return False
      ag.write_solution(graph, problem, str(proof_path))
      return True

    root_problem_string = original_problem.txt()
    root_graph = build(original_problem)
    events.write("root_ddar_start")
    ddar.solve(root_graph, rules, original_problem, max_level=1000)
    if solved(root_graph, original_problem):
      ag.write_solution(root_graph, original_problem, str(proof_path))
      summary["status"] = "solved"
      summary["proof_path"] = str(proof_path)
      summary["solution"] = {"stage": "direct_ddar", "depth": 0,
                             "auxiliary_constructions": []}
      events.write("solution", solution=summary["solution"])
    else:
      events.write("root_ddar_unsolved")
      prompt = original_problem.setup_str_from_problem(definitions) + " {F1} x00"
      beam = ag.BeamQueue(max_size=args.beam_size)
      beam.add((root_graph, prompt, root_problem_string, []), 0.0)
      summary["status"] = "running"
      write_json(summary_path, summary)

      found = False
      candidate_serial = 0
      for depth in range(args.search_depth):
        events.write("depth_start", depth=depth, beam_nodes=len(beam))
        new_beam = ag.BeamQueue(max_size=args.beam_size)
        pending_batches: List[Dict[str, Any]] = []

        for node_index, (previous_score, node) in enumerate(beam):
          graph, node_prompt, problem_string, aux_history = node
          events.write("decode_start", depth=depth, node_index=node_index,
                       previous_score=float(previous_score))
          outputs = model.beam_decode(node_prompt, eos_tokens=[";"])
          summary["counters"]["lm_decode_calls"] += 1

          candidates: List[Tuple[str, str, float]] = []
          for lm_output, score in zip(outputs["seqs_str"], outputs["scores"]):
            translation = ag.try_translate_constrained_to_construct(
                lm_output, graph
            )
            candidates.append((lm_output, translation, float(score)))
          candidates.reverse()
          summary["counters"]["lm_outputs"] += len(candidates)

          jobs: List[Dict[str, Any]] = []
          metadata: Dict[int, Dict[str, Any]] = {}
          for rank, (lm_output, translation, score) in enumerate(candidates):
            candidate_serial += 1
            candidate_id = candidate_serial
            event_fields = {
                "depth": depth,
                "node_index": node_index,
                "rank": rank,
                "candidate_id": candidate_id,
                "score": score,
                "lm_output": lm_output,
                "translation": translation,
            }
            if translation.startswith("ERROR:"):
              summary["counters"]["invalid_constructions"] += 1
              events.write("invalid_construction", **event_fields)
              continue

            summary["counters"]["valid_constructions"] += 1
            candidate_problem_string = ag.insert_aux_to_premise(
                problem_string, translation
            )
            metadata[candidate_id] = {
                **event_fields,
                "problem_string": candidate_problem_string,
                "next_prompt": node_prompt + " " + lm_output + " x00",
                "aux_history": aux_history + [translation],
            }
            jobs.append({
                "candidate_id": candidate_id,
                "problem_string": candidate_problem_string,
                "timeout_seconds": args.candidate_timeout_seconds,
            })
            events.write("valid_construction", **event_fields)

          # Submit immediately, then continue decoding the next beam node. This
          # keeps all CPU workers packed and overlaps DD+AR with GPU inference.
          result_handle = (
              pool.map_async(evaluate_candidate, jobs, chunksize=1)
              if jobs else None
          )
          pending_batches.append({
              "previous_score": float(previous_score),
              "metadata": metadata,
              "result_handle": result_handle,
          })

        for batch in pending_batches:
          results = (
              batch["result_handle"].get()
              if batch["result_handle"] is not None else []
          )
          for result in results:
            events.write("ddar_result", **result)
            counter = "ddar_" + result["status"]
            summary["counters"][counter] += 1
          write_json(summary_path, summary)

          for result in results:
            metadata = batch["metadata"]
            item = metadata[result["candidate_id"]]
            if result["solved"]:
              # Re-run only the winning candidate in the parent so the official
              # proof writer has the saturated graph and full traceback.
              if not solve_for_proof(item["problem_string"]):
                raise RuntimeError(
                    "worker solved candidate but proof replay did not"
                )
              summary["status"] = "solved"
              summary["proof_path"] = str(proof_path)
              summary["solution"] = {
                  "stage": "lm_auxiliary_then_ddar",
                  "depth": depth + 1,
                  "cumulative_score": batch["previous_score"] + item["score"],
                  "lm_output": item["lm_output"],
                  "auxiliary_constructions": item["aux_history"],
              }
              events.write("solution", solution=summary["solution"])
              found = True
              break

            if result["status"] == "unsolved":
              next_problem = pr.Problem.from_txt(item["problem_string"])
              next_graph = build(next_problem)
              new_beam.add(
                  (next_graph, item["next_prompt"], item["problem_string"],
                   item["aux_history"]),
                  batch["previous_score"] + item["score"],
              )

          if found:
            break

        if found:
          break
        beam = new_beam
        summary["depth_completed"] = depth
        events.write("depth_complete", depth=depth, beam_nodes=len(beam))
        write_json(summary_path, summary)
        if len(beam) == 0:
          summary["status"] = "exhausted"
          events.write("search_exhausted", depth=depth)
          break
      else:
        summary["status"] = "depth_limit"

    summary["completed_unix"] = time.time()
    summary["elapsed_seconds"] = round(summary["completed_unix"] - started, 3)
    write_json(summary_path, summary)
    events.write("runner_complete", status=summary["status"],
                 elapsed_seconds=summary["elapsed_seconds"])
    if summary["status"] == "solved":
      # A whole depth may have been queued to overlap inference and DD+AR.
      # Do not wait for lower-priority candidates after a proof is found.
      pool.terminate()
    else:
      pool.close()
    pool.join()
    events.close()
    return 0 if summary["status"] == "solved" else 2
  except Exception:  # pylint: disable=broad-except
    summary["status"] = "error"
    summary["error"] = traceback.format_exc(limit=30)
    summary["completed_unix"] = time.time()
    summary["elapsed_seconds"] = round(summary["completed_unix"] - started, 3)
    write_json(summary_path, summary)
    events.write("runner_error", error=summary["error"])
    pool.terminate()
    pool.join()
    events.close()
    raise


if __name__ == "__main__":
  sys.exit(main())
