"""README-extractor eval against the hand-labelled fictional READMEs in evals/readmes/.

Makes live Claude calls (costs money; needs ANTHROPIC_API_KEY in backend/.env):

    uv run --project backend python evals/run_readme_eval.py            # print results
    uv run --project backend python evals/run_readme_eval.py --record   # also log to RESULTS.md
"""

import argparse
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.api.deps import anthropic_backend  # noqa: E402
from app.evals.posting import pct  # noqa: E402
from app.evals.readme import evaluate_readmes, load_readme_cases, summarize_readmes  # noqa: E402
from app.evals.results import append_result_row  # noqa: E402
from app.extraction.readme import README_PROMPT  # noqa: E402
from app.llm.gateway import (  # noqa: E402
    Budgets,
    Gateway,
    InMemoryResultCache,
    InMemoryUsageLedger,
)
from app.llm.prompts import load_prompt  # noqa: E402
from app.taxonomy.loader import load_taxonomy  # noqa: E402

READMES = ROOT / "evals" / "readmes"
RESULTS = ROOT / "evals" / "RESULTS.md"
HEADING = "## README extractor (`evals/run_readme_eval.py`)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true", help="append a row to RESULTS.md")
    parser.add_argument("--version", type=int, default=README_PROMPT[1], help="prompt version")
    parser.add_argument("--note", default="", help="note for the RESULTS.md row")
    args = parser.parse_args()

    taxonomy = load_taxonomy()
    prompt = load_prompt(README_PROMPT[0], args.version)
    cases = load_readme_cases(READMES, taxonomy)
    ledger = InMemoryUsageLedger()
    gateway = Gateway(
        anthropic_backend(),
        InMemoryResultCache(),  # fresh cache: every run measures the model
        ledger,
        Budgets(per_job=10**7, per_tenant=10**7),
    )
    results = evaluate_readmes(gateway, cases, taxonomy, prompt)
    summary = summarize_readmes(results)
    cost = sum((r.cost_usd or Decimal(0) for r in ledger.records), Decimal(0))

    print(f"{prompt.ref} on {prompt.model}, {summary.cases} READMEs, ${cost:.4f}\n")
    print(f"{'readme':24} {'P/R':>11}  rejected  leaks")
    for r in results:
        print(
            f"{r.slug:24} {pct(r.counts.precision):>5}/{pct(r.counts.recall):<5} "
            f"{r.rejected:>4}/{r.checked:<4} {len(r.leaks):>4}"
        )
        if r.false_positives:
            print(f"{'':24} extra:  {', '.join(r.false_positives)}")
        if r.misses:
            print(f"{'':24} missed: {', '.join(r.misses)}")
        if r.leaks:
            print(f"{'':24} LEAKED: {', '.join(r.leaks)}")
    print(
        f"\nP/R {pct(summary.counts.precision)}/{pct(summary.counts.recall)}  "
        f"quote rejections {pct(summary.quote_rejection_rate)}  leaks {summary.leaks}"
    )

    if args.record:
        row = (
            f"| {date.today().isoformat()} | readme | {prompt.ref} | {prompt.model} "
            f"| {summary.cases} "
            f"| {pct(summary.counts.precision)} / {pct(summary.counts.recall)} "
            f"| {pct(summary.quote_rejection_rate)} | {summary.leaks} | ${cost:.4f} | {args.note} |"
        )
        append_result_row(RESULTS, HEADING, row)
        print(f"\nRecorded in {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
