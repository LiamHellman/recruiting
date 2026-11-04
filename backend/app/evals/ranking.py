"""Ranking eval: the scorer's order against a hand-made ranking (Kendall's tau). Pure, apart
from reading the golden set; no LLM, so it is free and can run on every scoring change."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from itertools import combinations
from pathlib import Path
from typing import Any

import yaml

from app.schemas.evidence import EvidenceItem
from app.schemas.github import DepthSignals, RepoActivity
from app.schemas.job import JobSkill, JobSpec
from app.schemas.score import ScoreResult
from app.scoring.depth import depth_score
from app.scoring.scorer import score_candidate
from app.taxonomy.loader import Taxonomy


def kendall_tau_b(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Kendall's tau-b, which allows ties in either list. 1 is the same order, -1 the reverse,
    0 unrelated. None if either list has no differences at all."""
    if len(a) != len(b):
        raise ValueError("the lists must have the same length")
    concordant = discordant = tied_a = tied_b = 0
    for i, j in combinations(range(len(a)), 2):
        da, db = a[i] - a[j], b[i] - b[j]
        if da == 0:
            tied_a += 1
        if db == 0:
            tied_b += 1
        if da * db > 0:
            concordant += 1
        elif da * db < 0:
            discordant += 1
    pairs = len(a) * (len(a) - 1) // 2
    denominator = math.sqrt((pairs - tied_a) * (pairs - tied_b))
    return (concordant - discordant) / denominator if denominator else None


@dataclass(frozen=True)
class GoldenCandidate:
    slug: str
    evidence: list[EvidenceItem]
    depth: float


@dataclass(frozen=True)
class GoldenPosting:
    slug: str
    title: str
    spec: JobSpec
    tiers: list[list[str]]


@dataclass(frozen=True)
class GoldenSet:
    as_of: date
    candidates: dict[str, GoldenCandidate]
    postings: list[GoldenPosting]


def _signals(raw: Mapping[str, Any] | None) -> DepthSignals | None:
    if raw is None:
        return None
    return DepthSignals(
        credited_repos=1,
        longest_active_days=raw["days"],
        longest_active_repo=None,
        activity=[RepoActivity(days=raw["days"], commits=raw["commits"])],
        repos_with_tests=["repo"] if raw["tests"] else [],
        repos_with_ci=["repo"] if raw["ci"] else [],
        repos_with_docker=["repo"] if raw["docker"] else [],
        merged_external_prs=raw["external_prs"],
        external_repos=[],
        max_stars=0,
    )


def _item(slug: str, raw: Mapping[str, Any]) -> EvidenceItem:
    source = raw.get("source", "cv")
    return EvidenceItem(
        skill_id=f"skill:{raw['skill']}",
        level=raw["level"],
        source=source,
        locator=f"golden:{slug}:{source}:{raw['skill']}",
        quote="golden fixture",
        date_last_seen=raw["seen"],
        ownership=raw.get("ownership", 1.0),
        confidence=1.0,
    )


def _skills(slugs: Sequence[str]) -> list[JobSkill]:
    return [JobSkill(raw_text=s, skill_id=f"skill:{s}") for s in slugs]


def load_golden(directory: Path, taxonomy: Taxonomy) -> GoldenSet:
    """Read the golden set and refuse anything inconsistent, so a typo can't skew tau."""
    postings_doc = yaml.safe_load((directory / "postings.yaml").read_text(encoding="utf-8"))
    candidates_doc = yaml.safe_load((directory / "candidates.yaml").read_text(encoding="utf-8"))
    candidates = {
        slug: GoldenCandidate(
            slug,
            [_item(slug, item) for item in raw["evidence"]],
            depth_score(_signals(raw.get("depth"))),
        )
        for slug, raw in candidates_doc["candidates"].items()
    }
    postings = [
        GoldenPosting(
            slug,
            raw["title"],
            JobSpec(must_have=_skills(raw["must_have"]), nice_to_have=_skills(raw["nice_to_have"])),
            raw["ranking"],
        )
        for slug, raw in postings_doc["postings"].items()
    ]
    problems: list[str] = []
    for candidate in candidates.values():
        problems += [
            f"{candidate.slug}: unknown {item.skill_id}"
            for item in candidate.evidence
            if item.skill_id not in taxonomy
        ]
    for posting in postings:
        problems += [
            f"{posting.slug}: unknown {skill.skill_id}"
            for skill in [*posting.spec.must_have, *posting.spec.nice_to_have]
            if skill.skill_id not in taxonomy
        ]
        ranked = [slug for tier in posting.tiers for slug in tier]
        if len(ranked) != len(set(ranked)) or set(ranked) != set(candidates):
            problems.append(f"{posting.slug}: the ranking must list every candidate once")
    if problems:
        raise ValueError("invalid golden set:\n  " + "\n  ".join(problems))
    return GoldenSet(postings_doc["as_of"], candidates, postings)


@dataclass(frozen=True)
class RankedRow:
    slug: str
    total: float
    tier: int  # 1 is best
    result: ScoreResult


@dataclass(frozen=True)
class RankingResult:
    posting: GoldenPosting
    rows: list[RankedRow]  # by score, best first
    tau: float | None
    disagreements: list[tuple[str, str]]  # (scorer's higher, whom the ranking prefers)


def evaluate_posting(
    golden: GoldenSet, posting: GoldenPosting, taxonomy: Taxonomy
) -> RankingResult:
    tier_of = {slug: n for n, tier in enumerate(posting.tiers, start=1) for slug in tier}
    rows = [
        RankedRow(
            slug,
            result.total,
            tier_of[slug],
            result,
        )
        for slug, candidate in golden.candidates.items()
        for result in [
            score_candidate(
                posting.spec, candidate.evidence, candidate.depth, taxonomy, golden.as_of
            )
        ]
    ]
    rows.sort(key=lambda r: (-r.total, r.slug))
    # Negated tiers, so a better tier is a larger number, like a higher score.
    tau = kendall_tau_b([r.total for r in rows], [-r.tier for r in rows])
    disagreements = [
        (high.slug, low.slug)
        for high, low in combinations(rows, 2)
        if high.total > low.total and high.tier > low.tier
    ]
    return RankingResult(posting, rows, tau, disagreements)
