"""github cache, candidates.github_handle_from

Revision ID: 0005
Revises: 0004
Create Date: 2026-11-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("candidates", sa.Column("github_handle_from", sa.String(10), nullable=True))
    op.create_check_constraint(
        "ck_candidates_github_handle_from",
        "candidates",
        "github_handle_from IN ('form', 'cv')",
    )
    op.create_table(
        "github_cache",
        sa.Column(
            "candidate_id",
            sa.Uuid(),
            sa.ForeignKey("candidates.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("key", sa.String(64), primary_key=True),
        sa.Column("tenant_id", sa.Uuid(), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column("data", postgresql.JSONB(), nullable=True),
        sa.Column("etag", sa.String(200), nullable=True),
        sa.Column("links", postgresql.JSONB(), nullable=False),
        sa.Column("immutable", sa.Boolean(), nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_github_cache_tenant_id", "github_cache", ["tenant_id"])


def downgrade() -> None:
    op.drop_index("ix_github_cache_tenant_id", table_name="github_cache")
    op.drop_table("github_cache")
    op.drop_constraint("ck_candidates_github_handle_from", "candidates", type_="check")
    op.drop_column("candidates", "github_handle_from")
