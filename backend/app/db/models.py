import uuid
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import CheckConstraint, ForeignKey, Index, Numeric, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, BigIntPK, JSONType
from app.schemas.evidence import EVIDENCE_SOURCES

_SOURCES_SQL = ", ".join(f"'{s}'" for s in EVIDENCE_SOURCES)


def utcnow() -> datetime:
    return datetime.now(UTC)


class Tenant(Base):
    __tablename__ = "tenants"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class AuditLog(Base):
    """Append-only. Postgres rejects UPDATE, DELETE and TRUNCATE on this table (migration 0001).

    Payloads hold IDs and hashes, never raw personal data, so deleting a candidate never
    requires editing the log.
    """

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id"), index=True)
    occurred_at: Mapped[datetime] = mapped_column(default=utcnow)
    actor: Mapped[str] = mapped_column(String(200))
    action: Mapped[str] = mapped_column(String(100), index=True)
    entity_type: Mapped[str] = mapped_column(String(50))
    entity_id: Mapped[str] = mapped_column(String(100), index=True)
    payload: Mapped[dict[str, Any]]


class LLMCacheEntry(Base):
    """Validated LLM outputs keyed by (prompt version, model, input hash, output schema)."""

    __tablename__ = "llm_cache"

    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id"), primary_key=True)
    cache_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    prompt_ref: Mapped[str] = mapped_column(String(100))
    model: Mapped[str] = mapped_column(String(100))
    input_hash: Mapped[str] = mapped_column(String(64))
    output: Mapped[dict[str, Any]]
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class LLMUsage(Base):
    """One row per gateway call (cache hits included, with zero tokens)."""

    __tablename__ = "llm_usage"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id"), index=True)
    job_id: Mapped[uuid.UUID | None] = mapped_column(index=True)
    prompt_ref: Mapped[str] = mapped_column(String(100))
    model_requested: Mapped[str] = mapped_column(String(100))
    model_served: Mapped[str | None] = mapped_column(String(100))
    cached: Mapped[bool]
    input_tokens: Mapped[int]
    output_tokens: Mapped[int]
    cache_read_input_tokens: Mapped[int]
    cache_creation_input_tokens: Mapped[int]
    cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(12, 6))
    latency_ms: Mapped[int]
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (CheckConstraint("status IN ('draft', 'confirmed')", name="ck_jobs_status"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id"), index=True)
    title: Mapped[str | None] = mapped_column(String(300))
    posting_text: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20))  # "draft" | "confirmed"
    spec: Mapped[dict[str, Any]]  # JobSpec.model_dump(mode="json")
    parse_report: Mapped[dict[str, Any]]  # ParseReport.model_dump(mode="json")
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow)
    confirmed_at: Mapped[datetime | None]


class Candidate(Base):
    """An applicant to one job. Name and contact details are personal data: shown to the
    recruiter and used to redact CV text before LLM calls; never scored, never audit-logged.
    Child rows are deleted with the candidate."""

    __tablename__ = "candidates"
    __table_args__ = (
        CheckConstraint(
            "github_handle_from IN ('form', 'cv')", name="ck_candidates_github_handle_from"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id"), index=True)
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("jobs.id"), index=True)
    full_name: Mapped[str] = mapped_column(String(200))
    email: Mapped[str | None] = mapped_column(String(320))
    github_handle: Mapped[str | None] = mapped_column(String(39))
    # "form" (typed by the candidate) or "cv" (the one GitHub profile link in their CV).
    github_handle_from: Mapped[str | None] = mapped_column(String(10))
    site_url: Mapped[str | None] = mapped_column(String(2000))
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class Consent(Base):
    """One grant per (candidate, source). Revoking sets `revoked_at`; a new grant is a new
    row. At most one active grant per source."""

    __tablename__ = "consents"
    __table_args__ = (
        CheckConstraint(f"source IN ({_SOURCES_SQL})", name="ck_consents_source"),
        Index(
            "uq_consents_active",
            "candidate_id",
            "source",
            unique=True,
            postgresql_where=text("revoked_at IS NULL"),
            sqlite_where=text("revoked_at IS NULL"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id"), index=True)
    candidate_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("candidates.id", ondelete="CASCADE"), index=True
    )
    source: Mapped[str] = mapped_column(String(30))
    notice_version: Mapped[str] = mapped_column(String(50))
    granted_at: Mapped[datetime] = mapped_column(default=utcnow)
    revoked_at: Mapped[datetime | None]


class Document(Base):
    """An uploaded file. The bytes live in the file store; `text` is the extracted text that
    every locator's offsets refer to, so it is never re-extracted in place."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id"), index=True)
    candidate_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("candidates.id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[str] = mapped_column(String(30))  # "cv"
    content_type: Mapped[str] = mapped_column(String(100))
    sha256: Mapped[str] = mapped_column(String(64))
    size_bytes: Mapped[int]
    storage_key: Mapped[str] = mapped_column(String(500))
    text: Mapped[str] = mapped_column(Text)
    pages: Mapped[list[list[int]]] = mapped_column(JSONType)  # [start, end) per page
    links: Mapped[list[str]] = mapped_column(JSONType)  # deterministic, from file + text
    emails: Mapped[list[str]] = mapped_column(JSONType)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class SourceRun(Base):
    """One run of one source for one candidate: what was produced, what was rejected and
    why. The report holds rejected quotes (source text), so it lives here, not in the
    audit log."""

    __tablename__ = "source_runs"
    __table_args__ = (
        CheckConstraint("status IN ('succeeded', 'failed')", name="ck_source_runs_status"),
        CheckConstraint(f"source IN ({_SOURCES_SQL})", name="ck_source_runs_source"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id"), index=True)
    candidate_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("candidates.id", ondelete="CASCADE"), index=True
    )
    source: Mapped[str] = mapped_column(String(30))
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE")
    )
    status: Mapped[str] = mapped_column(String(20))
    report: Mapped[dict[str, Any] | None] = mapped_column(JSONType)  # IngestReport
    profile: Mapped[dict[str, Any] | None] = mapped_column(JSONType)  # e.g. CvProfile
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class EvidenceRecord(Base):
    """A stored `EvidenceItem`. A new successful run of a source replaces that source's
    evidence for the candidate."""

    __tablename__ = "evidence_items"
    __table_args__ = (
        CheckConstraint(
            "level IN ('claimed', 'demonstrated', 'deep')", name="ck_evidence_items_level"
        ),
        CheckConstraint(f"source IN ({_SOURCES_SQL})", name="ck_evidence_items_source"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id"), index=True)
    candidate_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("candidates.id", ondelete="CASCADE"), index=True
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("source_runs.id", ondelete="CASCADE"), index=True
    )
    skill_id: Mapped[str] = mapped_column(String(100), index=True)
    level: Mapped[str] = mapped_column(String(20))
    source: Mapped[str] = mapped_column(String(30))
    locator: Mapped[str] = mapped_column(String(500))
    quote: Mapped[str] = mapped_column(Text)
    date_last_seen: Mapped[date | None]
    ownership: Mapped[float]
    confidence: Mapped[float]
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class GitHubCacheEntry(Base):
    """GitHub API answers collected for one candidate, already reduced to the fields the
    collector uses (see app/github/client.py). Scoped to the candidate rather than shared,
    so there is no store of GitHub users: rows go when the candidate is deleted or revokes
    GitHub consent."""

    __tablename__ = "github_cache"

    candidate_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("candidates.id", ondelete="CASCADE"), primary_key=True
    )
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id"), index=True)
    data: Mapped[Any] = mapped_column(JSONType, nullable=True)
    etag: Mapped[str | None] = mapped_column(String(200))
    links: Mapped[dict[str, str]] = mapped_column(JSONType)
    immutable: Mapped[bool]
    fetched_at: Mapped[datetime]
