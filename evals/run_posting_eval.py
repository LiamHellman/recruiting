"""Posting-parser eval against the hand-labelled postings in evals/postings/.

Makes live Claude calls (costs money; needs ANTHROPIC_API_KEY in backend/.env):

    uv run --project backend python evals/run_posting_eval.py            # print results
    uv run --project backend python evals/run_posting_eval.py --record   # also log to RESULTS.md
"""

import argparse
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.api.deps import anthropic_backend  # noqa: E402
from app.config import DEFAULT_TENANT_ID  # noqa: E402
from app.evals.posting import evaluate, load_cases, pct, summarize  # noqa: E402
from app.evals.results import append_result_row  # noqa: E402
from app.extraction.posting import POSTING_PROMPT  # noqa: E402
from app.llm.gateway import (  # noqa: E402
    Budgets,
    CallContext,
    Gateway,
    InMemoryResultCache,
    InMemoryUsageLedger,
)
from app.llm.prompts import load_prompt  # noqa: E402
from app.taxonomy.loader import load_taxonomy  # noqa: E402

POSTINGS = ROOT / "evals" / "postings"
RESULTS = ROOT / "evals" / "RESULTS.md"
HEADING = "## Posting parser (`evals/run_posting_eval.py`)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true", help="append a row to RESULTS.md")
    parser.add_argument("--version", type=int, default=POSTING_PROMPT[1], help="prompt version")
    parser.add_argument("--note", default="", help="note for the RESULTS.md row")
    args = parser.parse_args()

    taxonomy = load_taxonomy()
    prompt = load_prompt(POSTING_PROMPT[0], args.version)
    cases = load_cases(POSTINGS, taxonomy)
    ledger = InMemoryUsageLedger()
    gateway = Gateway(
        anthropic_backend(),
        InMemoryResultCache(),  # fresh cache: every run measures the model
        ledger,
        Budgets(per_job=10**7, per_tenant=10**7),
    )
    results = evaluate(gateway, cases, taxonomy, CallContext(tenant_id=DEFAULT_TENANT_ID), prompt)
    summary = summarize(results)
    cost = sum((r.cost_usd or Decimal(0) for r in ledger.records), Decimal(0))

    print(f"{prompt.ref} on {prompt.model}, {summary.cases} postings, ${cost:.4f}\n")
    print(f"{'posting':34} {'must P/R':>11} {'nice P/R':>11} {'any P/R':>11} rejected  unmapped")
    for r in results:
        print(
            f"{r.slug:34} {pct(r.must.precision):>5}/{pct(r.must.recall):<5} "
            f"{pct(r.nice.precision):>5}/{pct(r.nice.recall):<5} "
            f"{pct(r.either.precision):>5}/{pct(r.either.recall):<5} "
            f"{r.rejected_quotes:>8}  {', '.join(r.unmapped) or '-'}"
        )
    print(
        f"\nmust P/R {pct(summary.must.precision)}/{pct(summary.must.recall)}  "
        f"nice P/R {pct(summary.nice.precision)}/{pct(summary.nice.recall)}  "
        f"any-list P/R {pct(summary.either.precision)}/{pct(summary.either.recall)}  "
        f"placement {pct(summary.placement)}  seniority {pct(summary.seniority)}  "
        f"quote rejections {pct(summary.quote_rejection_rate)}"
    )

    if args.record:
        row = (
            f"| {date.today().isoformat()} | posting | {prompt.ref} | {prompt.model} "
            f"| {summary.cases} "
            f"| {pct(summary.must.precision)} / {pct(summary.must.recall)} "
            f"| {pct(summary.nice.precision)} / {pct(summary.nice.recall)} "
            f"| {pct(summary.either.precision)} / {pct(summary.either.recall)} "
            f"| {pct(summary.placement)} | {pct(summary.seniority)} "
            f"| {pct(summary.quote_rejection_rate)} | ${cost:.4f} | {args.note} |"
        )
        append_result_row(RESULTS, HEADING, row)
        print(f"\nRecorded in {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
