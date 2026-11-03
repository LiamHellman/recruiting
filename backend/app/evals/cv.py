"""CV-extractor eval: compare claimed-skill evidence with hand labels in evals/cvs/.

Scoring here is pure apart from the extractor call; the CLI in evals/run_cv_eval.py makes
the live calls. Besides precision/recall it measures dates (the latest `date_last_seen` per
labelled skill), the quote rejection rate, and redaction leaks: fragments of the candidate's
name or email found in the text sent to the model (must be zero).
"""

import re
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

from app.config import DEFAULT_TENANT_ID
from app.evals.posting import Counts, compare, label_key
from app.extraction.redact import find_emails
from app.llm.gateway import Gateway
from app.llm.prompts import Prompt
from app.sources.base import CandidateRef, IngestResult
from app.sources.cv import CvPayload, CvSource
from app.taxonomy.loader import Taxonomy, term_key
from app.taxonomy.normalize import normalize_term

_EVAL_NS = uuid.UUID("6f1d3c52-6a0e-4c55-9d2c-4a8e0b7f6c11")


@dataclass(frozen=True)
class CvCase:
    slug: str
    text: str
    name: str
    emails: tuple[str, ...]
    received_on: date
    claimed: frozenset[str]
    acceptable: frozenset[str]
    last_seen: dict[str, date]


def load_cv_cases(directory: Path, taxonomy: Taxonomy) -> list[CvCase]:
    cases = []
    for path in sorted(directory.glob("*.txt")):
        labels_path = path.with_suffix(".labels.yaml")
        labels = yaml.safe_load(labels_path.read_text(encoding="utf-8")) or {}
        keys: dict[str, frozenset[str]] = {}
        for list_name in ("claimed", "acceptable"):
            items = [str(item) for item in labels.get(list_name) or []]
            for item in items:
                if item.startswith("skill:") and item not in taxonomy:
                    raise ValueError(f"{labels_path.name}: unknown ids [{item!r}]")
                if not item.startswith("skill:") and (mapped := normalize_term(item, taxonomy)):
                    raise ValueError(f"{labels_path.name}: label {item!r} as {mapped}")
            keys[list_name] = frozenset(label_key(item) for item in items)
            if len(keys[list_name]) != len(items):
                raise ValueError(f"{labels_path.name}: labelled twice within {list_name}")
        if overlap := keys["claimed"] & keys["acceptable"]:
            raise ValueError(f"{labels_path.name}: labelled twice: {sorted(overlap)}")
        last_seen = {str(k): v for k, v in (labels.get("last_seen") or {}).items()}
        if stray := set(last_seen) - keys["claimed"]:
            raise ValueError(f"{labels_path.name}: last_seen for unclaimed {sorted(stray)}")
        if not labels.get("name") or not isinstance(labels.get("received_on"), date):
            raise ValueError(f"{labels_path.name}: needs name and received_on (YYYY-MM-DD)")
        cases.append(
            CvCase(
                slug=path.stem,
                text=path.read_text(encoding="utf-8"),
                name=str(labels["name"]),
                emails=tuple(str(e) for e in labels.get("emails") or []),
                received_on=labels["received_on"],
                claimed=keys["claimed"],
                acceptable=keys["acceptable"],
                last_seen=last_seen,
            )
        )
    return cases


def redaction_leaks(
    sent_text: str, name: str, emails: Iterable[str], taxonomy: Taxonomy
) -> list[str]:
    """Name parts (3+ letters, not skill names) and emails that survived redaction."""
    fragments = [p for p in name.split() if len(p) >= 3 and taxonomy.lookup(p) is None]
    fragments += list(emails)
    return [
        f for f in fragments if re.search(rf"(?<!\w){re.escape(f)}(?!\w)", sent_text, re.IGNORECASE)
    ]


@dataclass(frozen=True)
class CvCaseResult:
    slug: str
    claimed: Counts
    false_positives: tuple[str, ...]
    misses: tuple[str, ...]
    dates_right: int
    dates_total: int
    checked: int
    rejected: int
    leaks: tuple[str, ...]
    unmapped: tuple[str, ...]


def score_cv_case(case: CvCase, result: IngestResult, leaks: list[str]) -> CvCaseResult:
    predicted = {item.skill_id for item in result.evidence} | {
        f"term:{term_key(t)}" for t in result.report.unmapped_terms
    }
    latest: dict[str, date] = {}
    for item in result.evidence:
        if item.date_last_seen and item.date_last_seen > latest.get(item.skill_id, date.min):
            latest[item.skill_id] = item.date_last_seen
    return CvCaseResult(
        slug=case.slug,
        claimed=compare(predicted, case.claimed, case.acceptable),
        false_positives=tuple(sorted(predicted - case.claimed - case.acceptable)),
        misses=tuple(sorted(case.claimed - predicted)),
        dates_right=sum(latest.get(skill) == day for skill, day in case.last_seen.items()),
        dates_total=len(case.last_seen),
        checked=result.report.checked,
        rejected=len(result.report.rejected),
        leaks=tuple(leaks),
        unmapped=tuple(result.report.unmapped_terms),
    )


def evaluate_cvs(
    gateway: Gateway,
    cases: list[CvCase],
    taxonomy: Taxonomy,
    prompt: Prompt | None = None,
) -> list[CvCaseResult]:
    source = CvSource(gateway, taxonomy, prompt)
    results = []
    for case in cases:
        candidate = CandidateRef(
            id=uuid.uuid5(_EVAL_NS, case.slug),
            tenant_id=DEFAULT_TENANT_ID,
            job_id=None,
            names=(case.name,),
            emails=case.emails,
        )
        payload = CvPayload(
            document_id=uuid.uuid5(_EVAL_NS, f"doc:{case.slug}"),
            text=case.text,
            received_on=case.received_on,
            emails=tuple(find_emails(case.text)),
        )
        sent = source.redact(candidate, payload).text
        leaks = redaction_leaks(sent, case.name, case.emails, taxonomy)
        results.append(score_cv_case(case, source.ingest(candidate, payload), leaks))
    return results


@dataclass(frozen=True)
class CvSummary:
    cases: int
    claimed: Counts
    dates: float | None
    quote_rejection_rate: float | None
    leaks: int
    unmapped: int


def summarize_cvs(results: list[CvCaseResult]) -> CvSummary:
    dates_total = sum(r.dates_total for r in results)
    checked = sum(r.checked for r in results)
    return CvSummary(
        cases=len(results),
        claimed=sum((r.claimed for r in results), Counts()),
        dates=sum(r.dates_right for r in results) / dates_total if dates_total else None,
        quote_rejection_rate=sum(r.rejected for r in results) / checked if checked else None,
        leaks=sum(len(r.leaks) for r in results),
        unmapped=sum(len(r.unmapped) for r in results),
    )
