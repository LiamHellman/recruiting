"""Ranking eval: the scorer's order against the hand-made ranking in evals/golden/.

Offline and free: it runs the deterministic scorer on stored evidence, with no LLM calls.

    uv run --project backend python evals/run_ranking_eval.py            # print results
    uv run --project backend python evals/run_ranking_eval.py --record   # also log to RESULTS.md
"""

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.evals.ranking import evaluate_posting, load_golden  # noqa: E402
from app.evals.results import append_result_row  # noqa: E402
from app.scoring.policy import POLICY_V1  # noqa: E402
from app.taxonomy.loader import load_taxonomy  # noqa: E402

GOLDEN = ROOT / "evals" / "golden"
RESULTS = ROOT / "evals" / "RESULTS.md"
HEADING = "## Ranking (`evals/run_ranking_eval.py`)"


def _tau(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true", help="append a row to RESULTS.md")
    parser.add_argument("--note", default="", help="note for the RESULTS.md row")
    args = parser.parse_args()

    taxonomy = load_taxonomy()
    golden = load_golden(GOLDEN, taxonomy)
    results = [evaluate_posting(golden, posting, taxonomy) for posting in golden.postings]

    for result in results:
        print(f"\n{result.posting.title}  tau={_tau(result.tau)}")
        print(f"  {'candidate':20} {'score':>6} {'tier':>4}  flags")
        for row in result.rows:
            print(f"  {row.slug:20} {row.total:6.2f} {row.tier:4}  {'; '.join(row.result.labels)}")
        for high, low in result.disagreements:
            print(f"  scorer prefers {high} over {low}; the ranking disagrees")

    if args.record:
        cells = ", ".join(f"{r.posting.slug} {_tau(r.tau)}" for r in results)
        row = (
            f"| {date.today().isoformat()} | scoring {POLICY_V1.version} "
            f"| {len(golden.candidates)} | {cells} | {args.note} |"
        )
        append_result_row(RESULTS, HEADING, row)
        print(f"\nRecorded in {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
