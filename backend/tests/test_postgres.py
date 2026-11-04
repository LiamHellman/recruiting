"""Runs against a real, disposable PostgreSQL (TEST_DATABASE_URL); skipped otherwise.

Covers what SQLite cannot: the Alembic migrations and the append-only audit_log trigger.
The database is migrated to head and back to base, so point it at a throwaway database.
"""

import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import Connection, Engine, create_engine, text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.orm import sessionmaker

from app.audit.log import record
from app.config import DEFAULT_TENANT_ID
from app.db.base import Base
from app.llm.fake import FakeBackend
from app.llm.gateway import Budgets, CallContext, Gateway
from app.llm.prompts import load_prompt
from app.llm.stores import SqlResultCache, SqlUsageLedger
from app.schemas.job import ParsedPosting
from tests.conftest import posting_reply

URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL not set"),
]


@pytest.fixture(scope="module")
def engine() -> Iterator[Engine]:
    assert URL
    config = Config(str(Path(__file__).parents[1] / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", URL)
    command.upgrade(config, "head")
    engine = create_engine(URL)
    yield engine
    engine.dispose()
    command.downgrade(config, "base")


def test_default_tenant_is_seeded(engine: Engine) -> None:
    with engine.connect() as conn:
        name = conn.execute(
            text("SELECT name FROM tenants WHERE id = :id"), {"id": DEFAULT_TENANT_ID}
        ).scalar_one()
    assert name == "default"


def test_audit_log_accepts_inserts(engine: Engine) -> None:
    with sessionmaker(bind=engine).begin() as session:
        record(
            session,
            tenant_id=DEFAULT_TENANT_ID,
            actor="test",
            action="test.inserted",
            entity_type="test",
            entity_id="1",
            payload={"hash": "abc"},
        )
    with engine.connect() as conn:
        count = conn.execute(
            text("SELECT count(*) FROM audit_log WHERE action = 'test.inserted'")
        ).scalar_one()
    assert count == 1


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE audit_log SET actor = 'tampered'",
        "DELETE FROM audit_log",
        "TRUNCATE audit_log",
    ],
)
def test_audit_log_is_append_only(engine: Engine, statement: str) -> None:
    with engine.connect() as conn:
        conn.execute(
            text(
                "INSERT INTO audit_log (tenant_id, actor, action, entity_type, entity_id, payload)"
                " VALUES (:t, 'test', 'test.row', 'test', '2', '{}')"
            ),
            {"t": DEFAULT_TENANT_ID},
        )
        conn.commit()
        with pytest.raises(DBAPIError, match="append-only"):
            conn.execute(text(statement))
        conn.rollback()


def test_sql_cache_and_ledger_serve_repeat_calls_from_the_database(engine: Engine) -> None:
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    backend = FakeBackend(replies={"posting_parser": posting_reply()})

    def gateway() -> Gateway:
        return Gateway(
            backend,
            SqlResultCache(sessions),
            SqlUsageLedger(sessions),
            Budgets(per_job=10**9, per_tenant=10**9),
        )

    prompt = load_prompt("posting_parser", 1)
    ctx = CallContext(tenant_id=DEFAULT_TENANT_ID, job_id=uuid.uuid4())
    first = gateway().parse(prompt, "posting", ParsedPosting, ctx)
    second = gateway().parse(prompt, "posting", ParsedPosting, ctx)  # fresh gateway, same DB

    assert len(backend.calls) == 1
    assert second.cached
    assert second.output == first.output
    ledger = SqlUsageLedger(sessions)
    assert ledger.tokens_used(DEFAULT_TENANT_ID, ctx.job_id) == first.usage.total > 0


def _candidate_with_children(conn: Connection) -> uuid.UUID:
    job_id, candidate_id, document_id, run_id = (uuid.uuid4() for _ in range(4))
    t = DEFAULT_TENANT_ID
    conn.execute(
        text(
            "INSERT INTO jobs (id, tenant_id, posting_text, status, spec, parse_report)"
            " VALUES (:j, :t, 'posting', 'draft', '{}', '{}')"
        ),
        {"j": job_id, "t": t},
    )
    conn.execute(
        text("INSERT INTO candidates (id, tenant_id, job_id, full_name) VALUES (:c, :t, :j, 'X')"),
        {"c": candidate_id, "t": t, "j": job_id},
    )
    conn.execute(
        text(
            "INSERT INTO consents (id, tenant_id, candidate_id, source, notice_version)"
            " VALUES (:id, :t, :c, 'cv', 'v1')"
        ),
        {"id": uuid.uuid4(), "t": t, "c": candidate_id},
    )
    conn.execute(
        text(
            "INSERT INTO documents (id, tenant_id, candidate_id, kind, content_type, sha256,"
            " size_bytes, storage_key, text, pages, links, emails)"
            " VALUES (:d, :t, :c, 'cv', 'application/pdf', 'h', 1, 'k', 'Python', '[]', '[]',"
            " '[]')"
        ),
        {"d": document_id, "t": t, "c": candidate_id},
    )
    conn.execute(
        text(
            "INSERT INTO source_runs (id, tenant_id, candidate_id, source, document_id, status)"
            " VALUES (:r, :t, :c, 'cv', :d, 'succeeded')"
        ),
        {"r": run_id, "t": t, "c": candidate_id, "d": document_id},
    )
    conn.execute(
        text(
            "INSERT INTO evidence_items (id, tenant_id, candidate_id, run_id, skill_id, level,"
            " source, locator, quote, ownership, confidence)"
            " VALUES (:id, :t, :c, :r, 'skill:python', 'claimed', 'cv', 'cv:d#0-6', 'Python',"
            " 1, 1)"
        ),
        {"id": uuid.uuid4(), "t": t, "c": candidate_id, "r": run_id},
    )
    conn.execute(
        text(
            "INSERT INTO github_cache (candidate_id, key, tenant_id, data, links, immutable,"
            " fetched_at) VALUES (:c, 'k', :t, '{}', '{}', true, now())"
        ),
        {"t": t, "c": candidate_id},
    )
    return candidate_id


def test_only_one_active_consent_per_source(engine: Engine) -> None:
    with engine.connect() as conn:
        candidate_id = _candidate_with_children(conn)
        conn.commit()
        insert = text(
            "INSERT INTO consents (id, tenant_id, candidate_id, source, notice_version)"
            " VALUES (:id, :t, :c, 'cv', 'v2')"
        )
        params = {"t": DEFAULT_TENANT_ID, "c": candidate_id}
        with pytest.raises(IntegrityError, match="uq_consents_active"):
            conn.execute(insert, {"id": uuid.uuid4(), **params})
        conn.rollback()
        conn.execute(text("UPDATE consents SET revoked_at = now() WHERE candidate_id = :c"), params)
        conn.execute(insert, {"id": uuid.uuid4(), **params})  # re-granting after revoking
        conn.commit()


def test_unknown_sources_are_refused(engine: Engine) -> None:
    with engine.connect() as conn:
        candidate_id = _candidate_with_children(conn)
        with pytest.raises(IntegrityError, match="ck_consents_source"):
            conn.execute(
                text(
                    "INSERT INTO consents (id, tenant_id, candidate_id, source, notice_version)"
                    " VALUES (:id, :t, :c, 'linkedin_scrape', 'v1')"
                ),
                {"id": uuid.uuid4(), "t": DEFAULT_TENANT_ID, "c": candidate_id},
            )
        conn.rollback()


def test_deleting_a_candidate_deletes_their_data_but_not_the_audit_log(engine: Engine) -> None:
    with engine.connect() as conn:
        candidate_id = _candidate_with_children(conn)
        record_count = conn.execute(text("SELECT count(*) FROM audit_log")).scalar_one()
        conn.execute(text("DELETE FROM candidates WHERE id = :c"), {"c": candidate_id})
        for table in (
            "consents",
            "documents",
            "source_runs",
            "evidence_items",
            "github_cache",
            "pipeline_steps",
            "candidate_scores",
            "candidate_overrides",
            "candidate_decisions",
        ):
            remaining = conn.execute(
                text(f"SELECT count(*) FROM {table} WHERE candidate_id = :c"),
                {"c": candidate_id},
            ).scalar_one()
            assert remaining == 0, table
        assert conn.execute(text("SELECT count(*) FROM audit_log")).scalar_one() == record_count
        conn.commit()


def test_migrations_match_the_models(engine: Engine) -> None:
    with engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    assert diff == []
