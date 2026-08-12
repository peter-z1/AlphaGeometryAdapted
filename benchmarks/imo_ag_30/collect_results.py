"""Collect the 30 per-problem AlphaGeometry benchmark summaries."""

from __future__ import annotations

import json
from pathlib import Path
import sys


SUITE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SUITE_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import problem as pr  # pylint: disable=wrong-import-position


def main() -> int:
    problems = pr.Problem.from_txt_file(
        str(REPO_ROOT / "examples/imo_ag_30.txt"), translate=True
    )
    rows = []
    for index, problem in enumerate(problems):
        path = SUITE_DIR / "outputs" / problem.url / "summary.json"
        if path.is_file():
            result = json.loads(path.read_text(encoding="utf-8"))
        else:
            result = {
                "problem_index": index,
                "problem_name": problem.url,
                "status": "pending",
                "solved": None,
                "complete": False,
            }
        rows.append(result)

    counts = {
        "total": len(rows),
        "solved": sum(row.get("solved") is True for row in rows),
        "unsolved": sum(row.get("status") == "unsolved" for row in rows),
        "timeouts": sum(row.get("status") == "timeout" for row in rows),
        "errors": sum(row.get("status") == "error" for row in rows),
        "pending": sum(not row.get("complete") for row in rows),
    }
    payload = {"complete": counts["pending"] == 0, "counts": counts, "results": rows}
    output_json = SUITE_DIR / "results.json"
    output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Released AlphaGeometry-LM on `imo_ag_30.txt`",
        "",
        "Settings: 32 LM candidates, beam 512, depth 16, four-minute search / five-minute job cap.",
        "",
        "| # | Problem | Result | Seconds |",
        "| ---: | --- | --- | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['problem_index'] + 1} | `{row['problem_name']}` | "
            f"{row.get('status', 'pending')} | {row.get('elapsed_seconds', '')} |"
        )
    output_md = SUITE_DIR / "results.md"
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(counts, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
