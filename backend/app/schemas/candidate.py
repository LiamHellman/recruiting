from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel

from app.schemas.evidence import CvProfile, EvidenceItem, EvidenceSource, RejectedItem
from app.schemas.github import GitHubProfile
from app.schemas.review import DecisionOut, OverrideOut, PipelineStepOut
from app.schemas.score import ScoreResult


class ConsentOut(BaseModel):
    source: EvidenceSource
    notice_version: str
    granted_at: datetime
    revoked_at: datetime | None


class DocumentOut(BaseModel):
    id: str
    kind: str
    content_type: str
    size_bytes: int
    sha256: str
    page_count: int
    links: list[str]
    created_at: datetime


class SourceRunOut(BaseModel):
    id: str
    source: EvidenceSource
    status: Literal["succeeded", "failed"]
    prompt_ref: str | None
    model: str | None
    model_served: str | None
    cached: bool | None
    checked: int
    accepted: int
    rejected: list[RejectedItem]
    unmapped_terms: list[str]
    redactions: dict[str, int]
    fetch: dict[str, int]
    error: str | None
    created_at: datetime


class CandidateOut(BaseModel):
    id: str
    job_id: str
    full_name: str
    email: str | None
    github_handle: str | None
    github_handle_from: str | None
    site_url: str | None
    created_at: datetime
    consents: list[ConsentOut]
    documents: list[DocumentOut]
    runs: list[SourceRunOut]  # newest first
    pipeline: list[PipelineStepOut]
    llm_cost_usd: Decimal


class EvidenceOut(EvidenceItem):
    id: str
    run_id: str
    marked_wrong: bool  # by a recruiter; kept for the record, not counted


class CandidateEvidenceOut(BaseModel):
    candidate_id: str
    evidence: list[EvidenceOut]
    cv_profile: CvProfile | None
    github_profile: GitHubProfile | None
    score: ScoreResult | None
    overrides: list[OverrideOut]  # oldest first
    decisions: list[DecisionOut]  # oldest first
