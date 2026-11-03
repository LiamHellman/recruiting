"""candidates, consents, documents, source runs, evidence items

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-30
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SOURCES = "'cv', 'github', 'site', 'linkedin_pdf', 'linkedin_export', 'work_sample'"


def _now(name: str) -> sa.Column:  # type: ignore[type-arg]
    return sa.Column(name, sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())


def _tenant() -> sa.Column:  # type: ignore[type-arg]
    return sa.Column("tenant_id", sa.Uuid(), sa.ForeignKey("tenants.id"), nullable=False)


def _candidate() -> sa.Column:  # type: ignore[type-arg]
    return sa.Column(
        "candidate_id",
        sa.Uuid(),
        sa.ForeignKey("candidates.id", ondelete="CASCADE"),
        nullable=False,
    )


def upgrade() -> None:
    op.create_table(
        "candidates",
        sa.Column("id", sa.Uuid(), primary_key=True),
        _tenant(),
        sa.Column("job_id", sa.Uuid(), sa.ForeignKey("jobs.id"), nullable=False),
        sa.Column("full_name", sa.String(200), nullable=False),
        sa.Column("email", sa.String(320), nullable=True),
        sa.Column("github_handle", sa.String(39), nullable=True),
        sa.Column("site_url", sa.String(2000), nullable=True),
        _now("created_at"),
    )
    op.create_index("ix_candidates_tenant_id", "candidates", ["tenant_id"])
    op.create_index("ix_candidates_job_id", "candidates", ["job_id"])

    op.create_table(
        "consents",
        sa.Column("id", sa.Uuid(), primary_key=True),
        _tenant(),
        _candidate(),
        sa.Column("source", sa.String(30), nullable=False),
        sa.Column("notice_version", sa.String(50), nullable=False),
        _now("granted_at"),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(f"source IN ({SOURCES})", name="ck_consents_source"),
    )
    op.create_index("ix_consents_tenant_id", "consents", ["tenant_id"])
    op.create_index("ix_consents_candidate_id", "consents", ["candidate_id"])
    op.create_index(
        "uq_consents_active",
        "consents",
        ["candidate_id", "source"],
        unique=True,
        postgresql_where=sa.text("revoked_at IS NULL"),
    )

    op.create_table(
        "documents",
        sa.Column("id", sa.Uuid(), primary_key=True),
        _tenant(),
        _candidate(),
        sa.Column("kind", sa.String(30), nullable=False),
        sa.Column("content_type", sa.String(100), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("storage_key", sa.String(500), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("pages", postgresql.JSONB(), nullable=False),
        sa.Column("links", postgresql.JSONB(), nullable=False),
        sa.Column("emails", postgresql.JSONB(), nullable=False),
        _now("created_at"),
    )
    op.create_index("ix_documents_tenant_id", "documents", ["tenant_id"])
    op.create_index("ix_documents_candidate_id", "documents", ["candidate_id"])

    op.create_table(
        "source_runs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        _tenant(),
        _candidate(),
        sa.Column("source", sa.String(30), nullable=False),
        sa.Column(
            "document_id",
            sa.Uuid(),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("report", postgresql.JSONB(), nullable=True),
        sa.Column("profile", postgresql.JSONB(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        _now("created_at"),
        sa.CheckConstraint("status IN ('succeeded', 'failed')", name="ck_source_runs_status"),
        sa.CheckConstraint(f"source IN ({SOURCES})", name="ck_source_runs_source"),
    )
    op.create_index("ix_source_runs_tenant_id", "source_runs", ["tenant_id"])
    op.create_index("ix_source_runs_candidate_id", "source_runs", ["candidate_id"])

    op.create_table(
        "evidence_items",
        sa.Column("id", sa.Uuid(), primary_key=True),
        _tenant(),
        _candidate(),
        sa.Column(
            "run_id",
            sa.Uuid(),
            sa.ForeignKey("source_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("skill_id", sa.String(100), nullable=False),
        sa.Column("level", sa.String(20), nullable=False),
        sa.Column("source", sa.String(30), nullable=False),
        sa.Column("locator", sa.String(500), nullable=False),
        sa.Column("quote", sa.Text(), nullable=False),
        sa.Column("date_last_seen", sa.Date(), nullable=True),
        sa.Column("ownership", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        _now("created_at"),
        sa.CheckConstraint(
            "level IN ('claimed', 'demonstrated', 'deep')", name="ck_evidence_items_level"
        ),
        sa.CheckConstraint(f"source IN ({SOURCES})", name="ck_evidence_items_source"),
    )
    op.create_index("ix_evidence_items_tenant_id", "evidence_items", ["tenant_id"])
    op.create_index("ix_evidence_items_candidate_id", "evidence_items", ["candidate_id"])
    op.create_index("ix_evidence_items_run_id", "evidence_items", ["run_id"])
    op.create_index("ix_evidence_items_skill_id", "evidence_items", ["skill_id"])


def downgrade() -> None:
    op.drop_table("evidence_items")
    op.drop_table("source_runs")
    op.drop_table("documents")
    op.drop_table("consents")
    op.drop_table("candidates")
