"""Posting-parser eval: compare parsed skills with hand labels in evals/postings/.

Scoring here is pure; the CLI in evals/run_posting_eval.py makes the live calls.
Skills compare by canonical id when mapped, otherwise by the normalized term, so the eval
also measures whether unknown terms (e.g. "Airflow") are surfaced for the recruiter.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml

from app.extraction.posting import parse_posting
from app.llm.gateway import CallContext, Gateway
from app.llm.prompts import Prompt
from app.schemas.job import JobSkill, JobSpec
from app.taxonomy.loader import Taxonomy, term_key
from app.taxonomy.normalize import normalize_term


def label_key(label: str) -> str:
    return label if label.startswith("skill:") else f"term:{term_key(label)}"


def skill_key(skill: JobSkill) -> str:
    return skill.skill_id or f"term:{term_key(skill.raw_text)}"


@dataclass(frozen=True)
class PostingCase:
    slug: str
    posting_text: str
    must_have: frozenset[str]
    nice_to_have: frozenset[str]
    acceptable: frozenset[str]  # defensible either way: neither a hit nor a miss
    seniority: str | None


def load_cases(directory: Path, taxonomy: Taxonomy) -> list[PostingCase]:
    cases = []
    for posting in sorted(directory.glob("*.md")):
        labels_path = posting.with_suffix(".labels.yaml")
        labels = yaml.safe_load(labels_path.read_text(encoding="utf-8")) or {}
        keys: dict[str, frozenset[str]] = {}
        for name in ("must_have", "nice_to_have", "acceptable"):
            items = [str(item) for item in labels.get(name) or []]
            for item in items:
                if item.startswith("skill:") and item not in taxonomy:
                    raise ValueError(f"{labels_path.name}: unknown ids [{item!r}]")
                if not item.startswith("skill:") and (mapped := normalize_term(item, taxonomy)):
                    raise ValueError(f"{labels_path.name}: label {item!r} as {mapped}")
            keys[name] = frozenset(label_key(item) for item in items)
            if len(keys[name]) != len(items):
                raise ValueError(f"{labels_path.name}: labelled twice within {name}")
        overlap = (keys["must_have"] & keys["nice_to_have"]) | (
            keys["acceptable"] & (keys["must_have"] | keys["nice_to_have"])
        )
        if overlap:
            raise ValueError(f"{labels_path.name}: labelled twice: {sorted(overlap)}")
        cases.append(
            PostingCase(
                slug=posting.stem,
                posting_text=posting.read_text(encoding="utf-8"),
                must_have=keys["must_have"],
                nice_to_have=keys["nice_to_have"],
                acceptable=keys["acceptable"],
                seniority=labels.get("seniority"),
            )
        )
    return cases


@dataclass(frozen=True)
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __add__(self, other: "Counts") -> "Counts":
        return Counts(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)

    @property
    def precision(self) -> float | None:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else None

    @property
    def recall(self) -> float | None:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else None


def compare(predicted: Iterable[str], gold: Iterable[str], acceptable: Iterable[str]) -> Counts:
    pred, gold_set, ok = set(predicted), set(gold), set(acceptable)
    return Counts(tp=len(pred & gold_set), fp=len(pred - gold_set - ok), fn=len(gold_set - pred))


@dataclass(frozen=True)
class CaseResult:
    slug: str
    must: Counts
    nice: Counts
    either: Counts  # ignoring which list
    placed_right: int  # of skills found in both prediction and labels
    placed_total: int
    seniority_ok: bool | None
    accepted_quotes: int
    rejected_quotes: int
    unmapped: tuple[str, ...]


def score_case(case: PostingCase, spec: JobSpec, rejected_quotes: int) -> CaseResult:
    pred_must = {skill_key(s) for s in spec.must_have}
    pred_nice = {skill_key(s) for s in spec.nice_to_have}
    pred_any = pred_must | pred_nice
    gold_any = case.must_have | case.nice_to_have
    found = pred_any & gold_any
    return CaseResult(
        slug=case.slug,
        must=compare(pred_must, case.must_have, case.acceptable),
        nice=compare(pred_nice, case.nice_to_have, case.acceptable),
        either=compare(pred_any, gold_any, case.acceptable),
        placed_right=sum((k in pred_must) == (k in case.must_have) for k in found),
        placed_total=len(found),
        seniority_ok=None if case.seniority is None else spec.seniority == case.seniority,
        accepted_quotes=len(pred_any),
        rejected_quotes=rejected_quotes,
        unmapped=tuple(spec.unmapped()),
    )


def evaluate(
    gateway: Gateway,
    cases: list[PostingCase],
    taxonomy: Taxonomy,
    ctx: CallContext,
    prompt: Prompt | None = None,
) -> list[CaseResult]:
    results = []
    for case in cases:
        parsed = parse_posting(gateway, case.posting_text, ctx, taxonomy, prompt)
        results.append(score_case(case, parsed.spec, len(parsed.report.rejected)))
    return results


@dataclass(frozen=True)
class Summary:
    cases: int
    must: Counts
    nice: Counts
    either: Counts
    placement: float | None
    seniority: float | None
    quote_rejection_rate: float | None


def summarize(results: list[CaseResult]) -> Summary:
    placed = sum(r.placed_total for r in results)
    graded = [r.seniority_ok for r in results if r.seniority_ok is not None]
    quotes = sum(r.accepted_quotes + r.rejected_quotes for r in results)
    return Summary(
        cases=len(results),
        must=sum((r.must for r in results), Counts()),
        nice=sum((r.nice for r in results), Counts()),
        either=sum((r.either for r in results), Counts()),
        placement=sum(r.placed_right for r in results) / placed if placed else None,
        seniority=sum(graded) / len(graded) if graded else None,
        quote_rejection_rate=sum(r.rejected_quotes for r in results) / quotes if quotes else None,
    )


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"
