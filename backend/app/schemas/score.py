"""What the scorer returns. Stored with the candidate and shown to the recruiter, so every
number can be traced to an evidence item, an adjacency edge or a recruiter's override."""

from datetime import date
from typing import Literal

from pydantic import BaseModel

from app.schemas.evidence import EvidenceLevel, EvidenceSource
from app.schemas.job import SkillList

# A recruiter's level for a skill. "none" means they found no such skill.
OverrideLevel = Literal["claimed", "demonstrated", "deep", "none"]
CoverageStatus = Literal["demonstrated", "claimed_only", "missing"]
# Where a skill's strength came from.
CreditKind = Literal["evidence", "child", "adjacent", "override", "none"]


class ItemCredit(BaseModel):
    """The evidence item behind a strength, and how it was valued."""

    source: EvidenceSource
    locator: str
    level: EvidenceLevel
    date_last_seen: date | None
    recency: float
    ownership: float
    value: float  # level x recency x ownership


class SkillScore(BaseModel):
    skill_id: str
    raw_text: str
    bucket: SkillList
    weight: float
    direct: float  # best own evidence (or a group's children), before adjacency
    strength: float  # max(direct, adjacent direct x similarity); what enters the total
    credit_kind: CreditKind
    credit_skill: str | None  # the child or adjacent skill the credit came through
    similarity: float | None  # for adjacent credit
    best_item: ItemCredit | None  # None for an override, or when there is no evidence
    coverage: CoverageStatus
    override_level: OverrideLevel | None


class Coverage(BaseModel):
    """Must-haves by evidence status; shown beside every score."""

    demonstrated: list[str]  # directly, through a child, or through adjacency
    claimed_only: list[str]
    missing: list[str]


class ScoreResult(BaseModel):
    version: str
    as_of: date
    total: float  # rounded to 2 decimals
    max_total: float  # below 100 only when the posting has no nice-to-haves
    must_have_avg: float
    nice_to_have_avg: float
    depth: float
    skills: list[SkillScore]
    coverage: Coverage
    needs_human_review: bool  # gate: a must-have has no credit at all. Never a rejection.
    needs_more_evidence: bool  # a must-have is claimed only or missing
    labels: list[str]
