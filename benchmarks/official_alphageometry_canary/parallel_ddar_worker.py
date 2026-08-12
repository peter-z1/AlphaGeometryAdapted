"""CPU worker for checking one AlphaGeometry auxiliary construction."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import sys
import time
import traceback
from typing import Any, Dict

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / "external" / "google_deepmind_alphageometry"
sys.path.insert(0, str(OFFICIAL))

import ddar  # pylint: disable=wrong-import-position
import graph as gh  # pylint: disable=wrong-import-position
import problem as pr  # pylint: disable=wrong-import-position


DEFINITIONS = None
RULES = None


class CandidateTimeout(Exception):
  """Raised when one candidate exceeds its CPU allowance."""


def _raise_timeout(_signum: int, _frame: Any) -> None:
  raise CandidateTimeout()


def init_worker(defs_file: str, rules_file: str) -> None:
  """Load immutable DD+AR inputs once in each spawned worker."""
  global DEFINITIONS
  global RULES
  DEFINITIONS = pr.Definition.from_txt_file(defs_file, to_dict=True)
  RULES = pr.Theorem.from_txt_file(rules_file, to_dict=True)
  np.random.seed((os.getpid() * 1103515245 + int(time.time())) % (2**32))
  signal.signal(signal.SIGALRM, _raise_timeout)


def evaluate_candidate(payload: Dict[str, Any]) -> Dict[str, Any]:
  """Build and saturate one candidate theorem, returning JSON-safe metadata."""
  started = time.monotonic()
  result = {
      "candidate_id": payload["candidate_id"],
      "status": "error",
      "solved": False,
      "elapsed_seconds": 0.0,
      "error": None,
  }
  timeout_seconds = int(payload.get("timeout_seconds", 300))

  if timeout_seconds > 0:
    signal.alarm(timeout_seconds)
  try:
    if DEFINITIONS is None or RULES is None:
      raise RuntimeError("DD+AR worker was not initialized")
    problem = pr.Problem.from_txt(payload["problem_string"])
    graph, _ = gh.Graph.build_problem(problem, DEFINITIONS, verbose=False)
    ddar.solve(graph, RULES, problem, max_level=1000)
    goal_args = graph.names2nodes(problem.goal.args)
    result["solved"] = bool(graph.check(problem.goal.name, goal_args))
    result["status"] = "solved" if result["solved"] else "unsolved"
  except CandidateTimeout:
    result["status"] = "timeout"
    result["error"] = "candidate exceeded {} seconds".format(timeout_seconds)
  except Exception:  # pylint: disable=broad-except
    result["status"] = "error"
    result["error"] = traceback.format_exc(limit=20)
  finally:
    signal.alarm(0)
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)

  return result
