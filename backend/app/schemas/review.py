"""The recruiter's review API: overrides, decisions and the ranked list."""

import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.schemas.job import SkillId
from app.schemas.score import Coverage, OverrideLevel

# Overrides and decisions need a reason. It may name a person, so it never enters the audit log.
Reason = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
Decision = Literal["advance", "hold", "reject"]
RankingStatus = Literal["scored", "pending", "failed"]


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: Reason


class LevelOverrideRequest(_Request):
    """Set a skill's level from what the recruiter knows, e.g. after a technical screen.
    `level: "none"` means they found no such skill; `level: null` removes the override."""

    kind: Literal["level"]
    skill_id: SkillId
    level: OverrideLevel | None


class EvidenceOverrideRequest(_Request):
    """Mark an evidence item wrong, so it stops counting; `wrong: false` undoes it."""

    kind: Literal["evidence"]
    evidence_id: uuid.UUID
    wrong: bool = True


class NoteRequest(_Request):
    """A note for the record; the text is the reason. Does not change the score."""

    kind: Literal["note"]


class IdentityOverrideRequest(_Request):
    """Whether the GitHub account is the candidate's. Rejecting removes its evidence and
    depth signals from the score."""

    kind: Literal["identity"]
    decision: Literal["confirmed", "rejected"]


OverrideRequest = Annotated[
    LevelOverrideRequest | EvidenceOverrideRequest | NoteRequest | IdentityOverrideRequest,
    Field(discriminator="kind"),
]


class DecisionRequest(_Request):
    decision: Decision


class OverrideOut(BaseModel):
    id: int
    kind: str
    target: dict[str, Any]
    reason: str
    actor: str
    created_at: datetime


class DecisionOut(BaseModel):
    id: int
    decision: Decision
    reason: str
    actor: str
    score_total: float | None
    created_at: datetime


class PipelineStepOut(BaseModel):
    step: str
    status: str
    attempts: int
    error: str | None
    updated_at: datetime


class DeadLetterOut(PipelineStepOut):
    candidate_id: str
    job_id: str


class RankedCandidate(BaseModel):
    candidate_id: str
    full_name: str
    rank: int | None  # equal totals share a rank; None until scored
    status: RankingStatus
    stale: bool  # a newer score is on its way; these numbers are the previous ones
    total: float | None
    max_total: float | None
    must_have_avg: float | None
    nice_to_have_avg: float | None
    depth: float | None
    coverage: Coverage | None
    needs_human_review: bool
    needs_more_evidence: bool
    # Depth is 0 because no public code was shown, which is not the same as shallow work.
    low_visibility: bool
    labels: list[str]
    as_of: date | None
    decision: DecisionOut | None
    llm_cost_usd: Decimal


class RankingsOut(BaseModel):
    job_id: str
    candidates: list[RankedCandidate]
