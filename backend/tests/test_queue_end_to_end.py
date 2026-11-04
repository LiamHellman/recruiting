"""Phase 4's "Done when": a job with 10 fictional candidates runs end to end through the queue
into a ranked list with coverage and flags, and a re-run makes no new LLM calls.

This uses a real Celery worker and real task messages, twice. "in_process" runs on Celery's
in-memory broker and a file database (the worker has its own thread), so it needs nothing
installed. "real" runs on PostgreSQL and Redis, when TEST_DATABASE_URL and TEST_REDIS_URL are
set; it migrates that database up and down and flushes that Redis database, so point both at
throwaway ones. Either way, FakeGateway and the fake GitHub stand in for the network.
"""

import os
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
import redis
from alembic import command
from alembic.config import Config
from celery.contrib.testing.worker import start_worker
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from app.api.deps import (
    build_registry,
    get_file_store,
    get_gateway,
    get_github_clients,
    get_pipeline_queue,
)
from app.config import DEFAULT_TENANT_ID
from app.db.base import Base
from app.db.models import PipelineStep, Tenant
from app.db.session import get_session
from app.github.cache import SqlHttpCache
from app.llm.fake import FakeGateway
from app.main import create_app
from app.pipeline import tasks
from app.pipeline.graph import PipelineContext
from app.sources.base import CandidateRef
from app.sources.github import GitHubClientFactory
from app.storage import InMemoryFileStore
from app.taxonomy.loader import load_taxonomy
from tests.conftest import posting_reply
from tests.github_world import AMIRA_TODAY, fake_github, readme_reply
from tests.review_world import (
    apply_amira,
    apply_ben,
    apply_with_cv,
    confirm_job,
    create_job,
    rankings,
    scanning_cv_reply,
)

OTHERS = [
    ("03-carmen-rojas", "Carmen Rojas"),
    ("04-devi-raman", "Devi Raman"),
    ("05-elif-kaya", "Elif Kaya"),
    ("06-farid-haidari", "Farid Haidari"),
    ("07-grace-lindqvist", "Grace Lindqvist"),
    ("08-hiroshi-tanaka", "Hiroshi Tanaka"),
    ("09-ines-moreau", "Inès Moreau"),
    ("10-ruby-castellanos", "Ruby Castellanos"),
]
SETTLED = {"succeeded", "failed", "skipped", "dead_letter"}
DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
REDIS_URL = os.environ.get("TEST_REDIS_URL")


@pytest.fixture
def in_process_stack(tmp_path: Path) -> Iterator[tuple[sessionmaker[Session], str]]:
    engine = create_engine(
        f"sqlite:///{tmp_path / 'queue.db'}",
        connect_args={"check_same_thread": False, "timeout": 30},
        poolclass=NullPool,
    )

    def on_connect(connection, _record):
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")

    event.listen(engine, "connect", on_connect)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory.begin() as session:
        session.add(Tenant(id=DEFAULT_TENANT_ID, name="default"))
    yield factory, "memory://"
    engine.dispose()


@pytest.fixture
def real_stack() -> Iterator[tuple[sessionmaker[Session], str]]:
    assert DATABASE_URL
    assert REDIS_URL
    config = Config(str(Path(__file__).parents[1] / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", DATABASE_URL)
    command.upgrade(config, "head")
    broker = redis.Redis.from_url(REDIS_URL)
    broker.flushdb()
    engine = create_engine(DATABASE_URL)
    yield sessionmaker(bind=engine, expire_on_commit=False), REDIS_URL
    engine.dispose()
    command.downgrade(config, "base")
    broker.flushdb()
    broker.close()


@pytest.fixture(
    params=[
        "in_process",
        pytest.param(
            "real",
            marks=[
                pytest.mark.postgres,
                pytest.mark.skipif(
                    not (DATABASE_URL and REDIS_URL),
                    reason="TEST_DATABASE_URL and TEST_REDIS_URL not set",
                ),
            ],
        ),
    ]
)
def stack(request: pytest.FixtureRequest) -> tuple[sessionmaker[Session], str]:
    result: tuple[sessionmaker[Session], str] = request.getfixturevalue(f"{request.param}_stack")
    return result


@pytest.fixture
def stack_sessions(stack: tuple[sessionmaker[Session], str]) -> sessionmaker[Session]:
    return stack[0]


@pytest.fixture
def worker_gateway() -> FakeGateway:
    return FakeGateway(
        {
            "posting_parser": posting_reply(),
            "cv_extractor": scanning_cv_reply,
            "readme_extractor": readme_reply,
        }
    )


@pytest.fixture
def queue_client(
    stack: tuple[sessionmaker[Session], str],
    worker_gateway: FakeGateway,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TestClient]:
    sessions, broker_url = stack

    def cache_for(candidate: CandidateRef) -> SqlHttpCache:
        return SqlHttpCache(sessions, candidate.tenant_id, candidate.id)

    clients = GitHubClientFactory("test-token", cache_for, transport=fake_github().transport())
    taxonomy = load_taxonomy()
    ctx = PipelineContext(
        sessions,
        build_registry(worker_gateway, taxonomy, clients),
        taxonomy,
        today=lambda: AMIRA_TODAY,
    )
    monkeypatch.setattr(tasks, "context_provider", lambda: ctx)
    monkeypatch.setitem(tasks.celery_app.conf, "broker_url", broker_url)
    # Celery keeps its publisher pool; drop it so that this test's broker is the one used.
    tasks.celery_app.close()
    tasks.celery_app.amqp._producer_pool = None

    app = create_app()

    def session_override() -> Iterator[Session]:
        with sessions() as session:
            yield session

    store = InMemoryFileStore()
    app.dependency_overrides[get_session] = session_override
    app.dependency_overrides[get_gateway] = lambda: worker_gateway
    app.dependency_overrides[get_file_store] = lambda: store
    app.dependency_overrides[get_github_clients] = lambda: clients
    app.dependency_overrides[get_pipeline_queue] = tasks.CeleryQueue  # the real queue

    with (
        start_worker(tasks.celery_app, pool="solo", perform_ping_check=False, loglevel="WARNING"),
        TestClient(app) as test_client,
    ):
        yield test_client


def wait_for_pipelines(
    sessions: sessionmaker[Session], candidate_ids: list[uuid.UUID], timeout: float = 60
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with sessions() as session:
            rows = session.scalars(
                select(PipelineStep).where(PipelineStep.candidate_id.in_(candidate_ids))
            ).all()
        if len(rows) == 3 * len(candidate_ids) and all(r.status in SETTLED for r in rows):
            return
        time.sleep(0.05)
    states = sorted((str(r.candidate_id)[:8], r.step, r.status) for r in rows)
    raise AssertionError(f"pipelines did not settle: {states}")


def test_ten_candidates_through_the_queue_to_a_ranked_list(
    queue_client: TestClient,
    stack_sessions: sessionmaker[Session],
    worker_gateway: FakeGateway,
):
    job_id = create_job(queue_client)
    confirm_job(queue_client, job_id)

    applied = [apply_amira(queue_client, job_id), apply_ben(queue_client, job_id)]
    applied += [apply_with_cv(queue_client, job_id, slug, name) for slug, name in OTHERS]
    ids = [uuid.UUID(c["id"]) for c in applied]
    wait_for_pipelines(stack_sessions, ids)

    rows = rankings(queue_client, job_id)
    assert len(rows) == 10
    assert {r["status"] for r in rows} == {"scored"}
    assert all(r["rank"] is not None for r in rows)
    assert [r["total"] for r in rows] == sorted((r["total"] for r in rows), reverse=True)
    assert rows[0]["full_name"] == "Amira Haddad"
    assert all(r["coverage"] is not None for r in rows)
    # Candidates who match poorly are labelled, not removed.
    assert any("missing evidence: needs human review" in r["labels"] for r in rows)
    assert any(r["low_visibility"] for r in rows)

    # Re-queue every pipeline: no LLM call, no change to any score.
    calls = len(worker_gateway.calls)
    totals = [r["total"] for r in rows]
    for candidate_id in ids:
        response = queue_client.post(f"/candidates/{candidate_id}/pipeline")
        assert response.status_code == 202
    wait_for_pipelines(stack_sessions, ids)
    assert len(worker_gateway.calls) == calls
    assert [r["total"] for r in rankings(queue_client, job_id)] == totals


def test_an_override_reaches_the_worker_and_changes_the_ranking(
    queue_client: TestClient, stack_sessions: sessionmaker[Session]
):
    job_id = create_job(queue_client)
    confirm_job(queue_client, job_id)
    ben = apply_ben(queue_client, job_id)
    identifier = uuid.UUID(ben["id"])
    wait_for_pipelines(stack_sessions, [identifier])
    before = rankings(queue_client, job_id)[0]["total"]

    response = queue_client.post(
        f"/candidates/{ben['id']}/overrides",
        json={
            "kind": "level",
            "skill_id": "skill:postgresql",
            "level": "deep",
            "reason": "technical screen",
        },
    )
    assert response.status_code == 201
    wait_for_pipelines(stack_sessions, [identifier])
    assert rankings(queue_client, job_id)[0]["total"] > before
