"""CV-extractor eval against the hand-labelled fictional CVs in evals/cvs/.

Makes live Claude calls (costs money; needs ANTHROPIC_API_KEY in backend/.env):

    uv run --project backend python evals/run_cv_eval.py            # print results
    uv run --project backend python evals/run_cv_eval.py --record   # also log to RESULTS.md
"""

import argparse
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.api.deps import anthropic_backend  # noqa: E402
from app.evals.cv import evaluate_cvs, load_cv_cases, summarize_cvs  # noqa: E402
from app.evals.posting import pct  # noqa: E402
from app.evals.results import append_result_row  # noqa: E402
from app.extraction.cv import CV_PROMPT  # noqa: E402
from app.llm.gateway import (  # noqa: E402
    Budgets,
    Gateway,
    InMemoryResultCache,
    InMemoryUsageLedger,
)
from app.llm.prompts import load_prompt  # noqa: E402
from app.taxonomy.loader import load_taxonomy  # noqa: E402

CVS = ROOT / "evals" / "cvs"
RESULTS = ROOT / "evals" / "RESULTS.md"
HEADING = "## CV extractor (`evals/run_cv_eval.py`)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true", help="append a row to RESULTS.md")
    parser.add_argument("--version", type=int, default=CV_PROMPT[1], help="prompt version")
    parser.add_argument("--note", default="", help="note for the RESULTS.md row")
    args = parser.parse_args()

    taxonomy = load_taxonomy()
    prompt = load_prompt(CV_PROMPT[0], args.version)
    cases = load_cv_cases(CVS, taxonomy)
    ledger = InMemoryUsageLedger()
    gateway = Gateway(
        anthropic_backend(),
        InMemoryResultCache(),  # fresh cache: every run measures the model
        ledger,
        Budgets(per_job=10**7, per_tenant=10**7),
    )
    results = evaluate_cvs(gateway, cases, taxonomy, prompt)
    summary = summarize_cvs(results)
    cost = sum((r.cost_usd or Decimal(0) for r in ledger.records), Decimal(0))

    print(f"{prompt.ref} on {prompt.model}, {summary.cases} CVs, ${cost:.4f}\n")
    print(f"{'cv':24} {'P/R':>11} dates  rejected  leaks")
    for r in results:
        print(
            f"{r.slug:24} {pct(r.claimed.precision):>5}/{pct(r.claimed.recall):<5} "
            f"{r.dates_right}/{r.dates_total}    {r.rejected:>3}/{r.checked:<4} {len(r.leaks):>4}"
        )
        if r.false_positives:
            print(f"{'':24} extra:  {', '.join(r.false_positives)}")
        if r.misses:
            print(f"{'':24} missed: {', '.join(r.misses)}")
        if r.leaks:
            print(f"{'':24} LEAKED: {', '.join(r.leaks)}")
    print(
        f"\nclaimed P/R {pct(summary.claimed.precision)}/{pct(summary.claimed.recall)}  "
        f"dates {pct(summary.dates)}  quote rejections {pct(summary.quote_rejection_rate)}  "
        f"leaks {summary.leaks}  unmapped {summary.unmapped}"
    )

    if args.record:
        row = (
            f"| {date.today().isoformat()} | cv | {prompt.ref} | {prompt.model} "
            f"| {summary.cases} "
            f"| {pct(summary.claimed.precision)} / {pct(summary.claimed.recall)} "
            f"| {pct(summary.dates)} | {pct(summary.quote_rejection_rate)} "
            f"| {summary.leaks} | ${cost:.4f} | {args.note} |"
        )
        append_result_row(RESULTS, HEADING, row)
        print(f"\nRecorded in {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
