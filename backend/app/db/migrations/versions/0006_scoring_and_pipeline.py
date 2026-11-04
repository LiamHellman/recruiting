"""pipeline steps, scores, overrides, decisions; cost per candidate

Revision ID: 0006
Revises: 0005
Create Date: 2026-11-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _candidate_fk(**kwargs: bool) -> sa.Column[sa.Uuid]:
    return sa.Column(
        "candidate_id",
        sa.Uuid(),
        sa.ForeignKey("candidates.id", ondelete="CASCADE"),
        **kwargs,
    )


def _tenant_fk() -> sa.Column[sa.Uuid]:
    return sa.Column("tenant_id", sa.Uuid(), sa.ForeignKey("tenants.id"), nullable=False)


def upgrade() -> None:
    op.add_column(
        "source_runs",
        sa.Column("retryable", sa.Boolean(), server_default=sa.false(), nullable=False),
    )
    op.add_column(
        "llm_usage",
        sa.Column(
            "candidate_id",
            sa.Uuid(),
            sa.ForeignKey("candidates.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.create_index("ix_llm_usage_candidate_id", "llm_usage", ["candidate_id"])

    op.create_table(
        "pipeline_steps",
        _candidate_fk(primary_key=True),
        sa.Column("step", sa.String(30), primary_key=True),
        _tenant_fk(),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("input_hash", sa.String(64), nullable=True),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'retrying', 'succeeded', 'failed', 'skipped', "
            "'dead_letter')",
            name="ck_pipeline_steps_status",
        ),
    )
    op.create_index("ix_pipeline_steps_status", "pipeline_steps", ["tenant_id", "status"])

    op.create_table(
        "candidate_scores",
        _candidate_fk(primary_key=True),
        _tenant_fk(),
        sa.Column("job_id", sa.Uuid(), sa.ForeignKey("jobs.id"), nullable=False),
        sa.Column("version", sa.String(20), nullable=False),
        sa.Column("spec_hash", sa.String(64), nullable=False),
        sa.Column("input_hash", sa.String(64), nullable=False),
        sa.Column("total", sa.Float(), nullable=False),
        sa.Column("has_depth_signals", sa.Boolean(), nullable=False),
        sa.Column("result", postgresql.JSONB(), nullable=False),
        sa.Column("scored_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_candidate_scores_tenant_id", "candidate_scores", ["tenant_id"])
    op.create_index("ix_candidate_scores_job_id", "candidate_scores", ["job_id"])

    op.create_table(
        "candidate_overrides",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        _tenant_fk(),
        _candidate_fk(nullable=False),
        sa.Column("kind", sa.String(20), nullable=False),
        sa.Column("target", postgresql.JSONB(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("actor", sa.String(200), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "kind IN ('level', 'evidence', 'note', 'identity')",
            name="ck_candidate_overrides_kind",
        ),
    )
    op.create_index("ix_candidate_overrides_tenant_id", "candidate_overrides", ["tenant_id"])
    op.create_index("ix_candidate_overrides_candidate_id", "candidate_overrides", ["candidate_id"])

    op.create_table(
        "candidate_decisions",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        _tenant_fk(),
        _candidate_fk(nullable=False),
        sa.Column("decision", sa.String(20), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("actor", sa.String(200), nullable=False),
        sa.Column("score_total", sa.Float(), nullable=True),
        sa.Column("score_input_hash", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "decision IN ('advance', 'hold', 'reject')",
            name="ck_candidate_decisions_decision",
        ),
    )
    op.create_index("ix_candidate_decisions_tenant_id", "candidate_decisions", ["tenant_id"])
    op.create_index("ix_candidate_decisions_candidate_id", "candidate_decisions", ["candidate_id"])


def downgrade() -> None:
    op.drop_table("candidate_decisions")
    op.drop_table("candidate_overrides")
    op.drop_table("candidate_scores")
    op.drop_table("pipeline_steps")
    op.drop_index("ix_llm_usage_candidate_id", table_name="llm_usage")
    op.drop_column("llm_usage", "candidate_id")
    op.drop_column("source_runs", "retryable")
