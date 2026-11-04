import dataclasses
import random
import uuid
from datetime import UTC, date, datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient
from kombu.exceptions import OperationalError
from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

from app.api.deps import build_registry, get_pipeline_queue
from app.db.models import (
    AuditLog,
    Candidate,
    CandidateScore,
    Consent,
    EvidenceRecord,
    GitHubCacheEntry,
    LLMUsage,
    PipelineStep,
    SourceRun,
)
from app.llm.fake import FakeBackend, FakeGateway
from app.llm.gateway import Budgets, Gateway, InMemoryResultCache, RetryableLLMError
from app.llm.stores import SqlUsageLedger
from app.pipeline import tasks
from app.pipeline.graph import (
    InlineQueue,
    PipelineContext,
    QueueUnavailable,
    StepName,
    execute_step,
    next_step,
    run_inline,
    source_payload,
)
from app.pipeline.ingest import run_source
from app.schemas.evidence import EvidenceItem, EvidenceSource, IngestReport
from app.sources.base import CandidateRef, IngestResult, SourceRegistry
from tests.conftest import posting_reply
from tests.review_world import (
    apply_amira,
    apply_ben,
    apply_with_cv,
    confirm_job,
    create_job,
    rankings,
    row_for,
    scanning_cv_reply,
)


@pytest.fixture(autouse=True)
def any_cv(gateway: FakeGateway) -> None:
    gateway.backend.replies["cv_extractor"] = scanning_cv_reply


@pytest.fixture
def job_id(client: TestClient) -> str:
    return create_job(client)


def use_queue(client: TestClient, ctx: PipelineContext) -> None:
    client.app.dependency_overrides[get_pipeline_queue] = lambda: InlineQueue(ctx)  # type: ignore[attr-defined]


def steps(sessions: sessionmaker[Session], candidate_id: Any) -> dict[str, PipelineStep]:
    with sessions() as session:
        rows = session.scalars(
            select(PipelineStep).where(PipelineStep.candidate_id == uuid.UUID(str(candidate_id)))
        )
        return {row.step: row for row in rows}


def actions(sessions: sessionmaker[Session]) -> list[str]:
    with sessions() as session:
        return list(session.scalars(select(AuditLog.action).order_by(AuditLog.id)))


def retryable_errors(count: int) -> list[Exception]:
    return [RetryableLLMError("overloaded") for _ in range(count)]


# Gateway retries each model call 4 times, so 4 errors fail one attempt of a step.
ONE_FAILED_ATTEMPT = 4


# --- The graph -----------------------------------------------------------------------------


def test_graph_order():
    assert next_step("cv") == "github"
    assert next_step("github") == "score"
    assert next_step("score") is None


def test_a_candidate_goes_through_every_step_and_is_scored(
    client: TestClient, job_id: str, sessions: sessionmaker[Session]
):
    confirm_job(client, job_id)
    candidate = apply_amira(client, job_id)
    assert {s["step"]: s["status"] for s in candidate["pipeline"]} == {
        "cv": "succeeded",
        "github": "succeeded",
        "score": "succeeded",
    }
    assert rankings(client, job_id)[0]["status"] == "scored"
    assert actions(sessions)[-3:] == ["source.ingested", "source.ingested", "candidate.scored"]


def test_skips_a_source_the_candidate_did_not_share(client: TestClient, job_id: str):
    confirm_job(client, job_id)
    candidate = apply_with_cv(client, job_id, "02-ben-okafor", "Ben Okafor")
    assert {s["step"]: s["status"] for s in candidate["pipeline"]} == {
        "cv": "succeeded",
        "github": "skipped",
        "score": "succeeded",
    }


def test_a_draft_job_collects_evidence_and_scores_when_confirmed(
    client: TestClient, job_id: str, sessions: sessionmaker[Session]
):
    candidate = apply_amira(client, job_id)
    assert steps(sessions, candidate["id"])["score"].status == "skipped"
    assert client.get(f"/jobs/{job_id}/rankings").status_code == 409
    confirm_job(client, job_id)
    assert steps(sessions, candidate["id"])["score"].status == "succeeded"
    assert rankings(client, job_id)[0]["total"] is not None


# --- Idempotence ---------------------------------------------------------------------------


def test_a_repeated_step_does_nothing(
    client: TestClient,
    job_id: str,
    sessions: sessionmaker[Session],
    gateway: FakeGateway,
    github: Any,
    pipeline_context: PipelineContext,
):
    confirm_job(client, job_id)
    candidate = apply_amira(client, job_id)
    calls, requests, log = len(gateway.calls), len(github.requests), actions(sessions)
    with sessions() as session:
        runs = len(session.scalars(select(SourceRun)).all())

    for step in ("cv", "github", "score"):
        outcome = execute_step(pipeline_context, uuid.UUID(candidate["id"]), step)
        assert (outcome.status, outcome.ran) == ("succeeded", False)

    assert len(gateway.calls) == calls
    assert len(github.requests) == requests
    assert actions(sessions) == log
    with sessions() as session:
        assert len(session.scalars(select(SourceRun)).all()) == runs


def test_requeueing_a_finished_candidate_makes_no_llm_calls(
    client: TestClient, job_id: str, gateway: FakeGateway, sessions: sessionmaker[Session]
):
    confirm_job(client, job_id)
    candidate = apply_amira(client, job_id)
    calls, log = len(gateway.calls), actions(sessions)
    response = client.post(f"/candidates/{candidate['id']}/pipeline")
    assert response.status_code == 202
    assert len(gateway.calls) == calls
    assert actions(sessions) == log


def test_an_explicit_rerun_is_served_from_the_llm_cache_and_rescoring_keeps_the_score(
    client: TestClient, job_id: str, gateway: FakeGateway, sessions: sessionmaker[Session]
):
    confirm_job(client, job_id)
    candidate = apply_amira(client, job_id)
    before = rankings(client, job_id)[0]["total"]
    calls = len(gateway.calls)
    for source in ("cv", "github"):
        response = client.post(f"/candidates/{candidate['id']}/sources/{source}/runs")
        assert response.status_code == 201
    assert len(gateway.calls) == calls
    assert rankings(client, job_id)[0]["total"] == before
    assert actions(sessions).count("candidate.scored") == 1  # same inputs: nothing rewritten


def test_a_new_input_runs_the_step_again(
    client: TestClient,
    job_id: str,
    sessions: sessionmaker[Session],
    gateway: FakeGateway,
    pipeline_context: PipelineContext,
):
    candidate = apply_amira(client, job_id)
    with sessions.begin() as session:
        row = session.get(PipelineStep, (uuid.UUID(candidate["id"]), "cv"))
        assert row is not None
        row.input_hash = "something-else"  # as after a new prompt version
    outcome = execute_step(pipeline_context, uuid.UUID(candidate["id"]), "cv")
    assert (outcome.status, outcome.ran) == ("succeeded", True)


def test_score_input_hash_changes_with_the_day(
    client: TestClient,
    job_id: str,
    sessions: sessionmaker[Session],
    pipeline_context: PipelineContext,
):
    confirm_job(client, job_id)
    candidate = apply_amira(client, job_id)
    later = dataclasses.replace(pipeline_context, today=lambda: date(2031, 1, 1))
    outcome = execute_step(later, uuid.UUID(candidate["id"]), "score")
    assert outcome.ran
    with sessions() as session:
        score = session.get(CandidateScore, uuid.UUID(candidate["id"]))
        assert score is not None
        assert score.result["as_of"] == "2031-01-01"


# --- Retries and dead letters --------------------------------------------------------------


def test_an_outage_is_retried_with_backoff_and_then_succeeds(
    client: TestClient,
    job_id: str,
    sessions: sessionmaker[Session],
    gateway: FakeGateway,
    pipeline_context: PipelineContext,
):
    ctx = dataclasses.replace(pipeline_context, max_attempts=3)
    candidate = apply_with_cv(client, job_id, "02-ben-okafor", "Ben Okafor")
    identifier = uuid.UUID(candidate["id"])
    with sessions.begin() as session:  # the first run already succeeded; go again
        row = session.get(PipelineStep, (identifier, "cv"))
        assert row is not None
        row.status = "pending"
    gateway.cache.entries.clear()
    gateway.backend.errors.extend(retryable_errors(ONE_FAILED_ATTEMPT))

    first = execute_step(ctx, identifier, "cv")
    assert first.status == "retrying"
    assert first.retry_in is not None and 15 <= first.retry_in <= 30
    assert steps(sessions, identifier)["cv"].attempts == 1

    second = execute_step(ctx, identifier, "cv")
    assert second.status == "succeeded"
    assert steps(sessions, identifier)["cv"].attempts == 2
    assert steps(sessions, identifier)["cv"].error is None


def test_retries_end_in_a_dead_letter_and_the_candidate_is_still_scored(
    client: TestClient,
    job_id: str,
    sessions: sessionmaker[Session],
    gateway: FakeGateway,
    pipeline_context: PipelineContext,
):
    confirm_job(client, job_id)
    use_queue(client, dataclasses.replace(pipeline_context, max_attempts=2))
    gateway.backend.errors.extend(retryable_errors(2 * ONE_FAILED_ATTEMPT))
    candidate = apply_with_cv(client, job_id, "02-ben-okafor", "Ben Okafor")

    state = steps(sessions, candidate["id"])
    assert state["cv"].status == "dead_letter"
    assert state["cv"].attempts == 2
    assert state["score"].status == "succeeded"
    assert "pipeline.dead_letter" in actions(sessions)

    # The candidate stays in the list, with a score of what is known and a clear label.
    row = rankings(client, job_id)[0]
    assert row["status"] == "scored"
    assert row["total"] == 0
    assert row["needs_human_review"]
    assert "cv could not be processed: score may be incomplete" in row["labels"]

    letters = client.get("/pipeline/dead-letters").json()
    assert [(d["candidate_id"], d["step"]) for d in letters] == [(candidate["id"], "cv")]

    # After the outage: re-queue, and the candidate is whole.
    assert client.post(f"/candidates/{candidate['id']}/pipeline").status_code == 202
    assert steps(sessions, candidate["id"])["cv"].status == "succeeded"
    assert client.get("/pipeline/dead-letters").json() == []
    row = rankings(client, job_id)[0]
    assert row["total"] > 0
    assert not any("could not be processed" in label for label in row["labels"])


def test_a_failure_that_cannot_pass_is_not_retried(
    client: TestClient,
    job_id: str,
    sessions: sessionmaker[Session],
    github: Any,
    pipeline_context: PipelineContext,
):
    use_queue(client, dataclasses.replace(pipeline_context, max_attempts=3))
    candidate = apply_with_cv(
        client,
        job_id,
        "02-ben-okafor",
        "Ben Okafor",
        github_handle="no-such-account",
        consent_github="true",
    )
    state = steps(sessions, candidate["id"])["github"]
    assert (state.status, state.attempts) == ("failed", 1)
    assert "not found" in (state.error or "")


class _BrokenSource:
    name: EvidenceSource = "cv"

    def input_key(self, candidate: CandidateRef, payload: object) -> str:
        return "key"

    def ingest(self, candidate: CandidateRef, payload: object) -> IngestResult:
        raise RuntimeError("SECRET candidate text in an error message")


def test_a_bug_is_retried_then_dead_lettered_keeping_only_the_error_type(
    client: TestClient,
    job_id: str,
    sessions: sessionmaker[Session],
    pipeline_context: PipelineContext,
):
    candidate = apply_amira(client, job_id)
    identifier = uuid.UUID(candidate["id"])
    with sessions.begin() as session:
        row = session.get(PipelineStep, (identifier, "cv"))
        assert row is not None
        row.status = "pending"
    ctx = dataclasses.replace(
        pipeline_context, registry=SourceRegistry([_BrokenSource()]), max_attempts=2
    )
    assert execute_step(ctx, identifier, "cv").status == "retrying"
    assert execute_step(ctx, identifier, "cv").status == "dead_letter"
    state = steps(sessions, identifier)["cv"]
    assert state.error == "RuntimeError"
    with sessions() as session:
        entry = session.scalars(
            select(AuditLog).where(AuditLog.action == "pipeline.dead_letter")
        ).one()
    assert "SECRET" not in str(entry.payload)


def test_a_step_that_keeps_killing_its_worker_is_dead_lettered(
    client: TestClient,
    job_id: str,
    sessions: sessionmaker[Session],
    gateway: FakeGateway,
    pipeline_context: PipelineContext,
):
    candidate = apply_amira(client, job_id)
    identifier = uuid.UUID(candidate["id"])
    with sessions.begin() as session:
        row = session.get(PipelineStep, (identifier, "cv"))
        assert row is not None
        row.status, row.attempts = "running", 4  # four deliveries that never settled
    ctx = dataclasses.replace(pipeline_context, max_attempts=4)
    calls = len(gateway.calls)
    assert execute_step(ctx, identifier, "cv").status == "dead_letter"
    assert len(gateway.calls) == calls


def test_backoff_doubles_up_to_the_cap_with_jitter(pipeline_context: PipelineContext):
    ctx = dataclasses.replace(pipeline_context, rng=random.Random(1))
    delays = [ctx.retry_delay(attempt) for attempt in range(1, 8)]
    for attempt, delay in enumerate(delays, start=1):
        full = min(30.0 * 2 ** (attempt - 1), 900.0)
        assert full / 2 <= delay <= full


# --- Consent while queued ------------------------------------------------------------------


def test_consent_revoked_before_a_step_runs_means_it_is_skipped(
    client: TestClient,
    job_id: str,
    sessions: sessionmaker[Session],
    github: Any,
    pipeline_context: PipelineContext,
):
    candidate = apply_amira(client, job_id)
    identifier = uuid.UUID(candidate["id"])
    assert client.delete(f"/candidates/{identifier}/consents/github").status_code == 200
    requests = len(github.requests)
    assert execute_step(pipeline_context, identifier, "github").status == "skipped"
    assert len(github.requests) == requests


class _RevokedMidRun:
    """A source during whose run the candidate withdraws consent."""

    name: EvidenceSource = "github"

    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self._sessions = sessions

    def input_key(self, candidate: CandidateRef, payload: object) -> str:
        return "key"

    def ingest(self, candidate: CandidateRef, payload: object) -> IngestResult:
        with self._sessions.begin() as session:
            # What collection stores as it goes, which must not outlive the revocation.
            session.add(
                GitHubCacheEntry(
                    candidate_id=candidate.id,
                    key="fetched-during-the-run",
                    tenant_id=candidate.tenant_id,
                    data={"login": "ben-okafor"},
                    links={},
                    immutable=False,
                    fetched_at=datetime.now(UTC),
                )
            )
            consent = session.scalars(
                select(Consent).where(
                    Consent.candidate_id == candidate.id, Consent.source == "github"
                )
            ).one()
            consent.revoked_at = datetime.now(UTC)
        item = EvidenceItem(
            skill_id="skill:python",
            level="deep",
            source="github",
            locator="pr:a/b#1",
            quote="Python",
            date_last_seen=None,
            ownership=1.0,
            confidence=1.0,
        )
        report = IngestReport(
            source="github", checked=1, rejected=[], accepted=1, unmapped_terms=[]
        )
        return IngestResult(evidence=[item], report=report)


def test_consent_revoked_during_a_run_discards_what_the_run_produced(
    client: TestClient,
    job_id: str,
    sessions: sessionmaker[Session],
    pipeline_context: PipelineContext,
):
    candidate = apply_with_cv(
        client,
        job_id,
        "02-ben-okafor",
        "Ben Okafor",
        github_handle="ben-okafor",
        consent_github="true",
    )
    identifier = uuid.UUID(candidate["id"])
    with sessions.begin() as session:
        session.execute(delete(EvidenceRecord).where(EvidenceRecord.source == "github"))
        row = session.get(PipelineStep, (identifier, "github"))
        assert row is not None
        row.status = "pending"
    ctx = dataclasses.replace(pipeline_context, registry=SourceRegistry([_RevokedMidRun(sessions)]))
    outcome = execute_step(ctx, identifier, "github")
    assert outcome.status == "skipped"
    with sessions() as session:
        assert not session.scalars(
            select(EvidenceRecord).where(EvidenceRecord.source == "github")
        ).all()
        assert not session.scalars(select(GitHubCacheEntry)).all()
    assert actions(sessions)[-1] == "source.refused"


def test_candidate_deleted_while_queued_is_a_no_op(
    pipeline_context: PipelineContext, sessions: sessionmaker[Session]
):
    outcome = execute_step(pipeline_context, uuid.uuid4(), "cv")
    assert (outcome.status, outcome.ran) == ("skipped", False)


def test_unknown_step_is_a_programming_error(pipeline_context: PipelineContext):
    with pytest.raises(ValueError, match="unknown pipeline step"):
        execute_step(pipeline_context, uuid.uuid4(), "laser")


# --- Queue outages -------------------------------------------------------------------------


class _DownQueue:
    def enqueue(self, candidate_id: uuid.UUID, step: StepName) -> None:
        raise QueueUnavailable("no broker")


def test_a_queue_outage_leaves_the_application_stored_and_visibly_pending(
    client: TestClient, job_id: str, pipeline_context: PipelineContext
):
    confirm_job(client, job_id)
    client.app.dependency_overrides[get_pipeline_queue] = lambda: _DownQueue()  # type: ignore[attr-defined]
    candidate = apply_amira(client, job_id)
    assert {s["status"] for s in candidate["pipeline"]} == {"pending"}
    row = rankings(client, job_id)[0]
    assert (row["status"], row["rank"], row["total"]) == ("pending", None, None)
    url = f"/candidates/{candidate['id']}/pipeline"
    assert client.post(url).status_code == 503

    use_queue(client, pipeline_context)
    assert client.post(url).status_code == 202
    assert rankings(client, job_id)[0]["status"] == "scored"


def test_celery_queue_reports_a_broker_outage(monkeypatch: pytest.MonkeyPatch):
    def fail(*args: Any) -> None:
        raise OperationalError("connection refused")

    monkeypatch.setattr(tasks.run_step, "delay", fail)
    with pytest.raises(QueueUnavailable):
        tasks.CeleryQueue().enqueue(uuid.uuid4(), "cv")


# --- Cost per candidate --------------------------------------------------------------------


def test_llm_cost_is_recorded_per_candidate_and_a_rerun_adds_none(
    client: TestClient,
    job_id: str,
    sessions: sessionmaker[Session],
    pipeline_context: PipelineContext,
    github_clients: Any,
):
    ledgered = Gateway(
        FakeBackend(replies={"posting_parser": posting_reply(), "cv_extractor": scanning_cv_reply}),
        InMemoryResultCache(),
        SqlUsageLedger(sessions),
        Budgets(per_job=10**9, per_tenant=10**9),
        sleep=lambda _s: None,
    )
    ctx = dataclasses.replace(
        pipeline_context,
        registry=build_registry(ledgered, pipeline_context.taxonomy, github_clients),
    )
    use_queue(client, ctx)
    confirm_job(client, job_id)
    first = apply_with_cv(client, job_id, "01-amira-haddad", "Amira Haddad")
    second = apply_with_cv(client, job_id, "02-ben-okafor", "Ben Okafor")
    assert float(first["llm_cost_usd"]) > 0
    assert float(second["llm_cost_usd"]) > 0

    with sessions() as session:
        usage = session.scalars(select(LLMUsage)).all()
    assert {u.candidate_id for u in usage} == {uuid.UUID(first["id"]), uuid.UUID(second["id"])}

    # The same CV text again is a cache hit, recorded at zero cost.
    client.post(f"/candidates/{first['id']}/sources/cv/runs")
    rows = rankings(client, job_id)
    assert row_for(rows, first)["llm_cost_usd"] == first["llm_cost_usd"]


def test_worked_example_candidates_through_the_whole_pipeline(client: TestClient, job_id: str):
    """Amira and Ben from CLAUDE.md, from CV upload and GitHub account to a ranked list. The
    totals differ from the worked example's 67.25 / 46.5 because the fictional accounts yield
    their own depth (0.815 / 0.2575) and Amira's CV also claims Redis."""
    confirm_job(client, job_id)
    amira, ben = apply_amira(client, job_id), apply_ben(client, job_id)
    rows = rankings(client, job_id)
    assert [r["candidate_id"] for r in rows] == [amira["id"], ben["id"]]
    assert [r["rank"] for r in rows] == [1, 2]

    top, second = rows
    assert top["depth"] == pytest.approx(0.815, abs=0.001)
    assert top["must_have_avg"] == pytest.approx((1.0 + 0.7 + 0.75) / 3)
    assert top["total"] == pytest.approx(70.81, abs=0.01)
    assert top["coverage"]["demonstrated"] == [
        "skill:python",
        "skill:fastapi",
        "skill:postgresql",
    ]
    assert top["needs_more_evidence"] is False

    assert second["depth"] == pytest.approx(0.2575, abs=0.001)
    assert second["total"] == pytest.approx(45.86, abs=0.01)
    assert second["coverage"]["claimed_only"] == ["skill:postgresql"]
    assert second["needs_more_evidence"] is True
    assert second["needs_human_review"] is False


def test_inline_runner_retries_without_waiting(
    client: TestClient,
    job_id: str,
    sessions: sessionmaker[Session],
    gateway: FakeGateway,
    pipeline_context: PipelineContext,
):
    candidate = apply_with_cv(client, job_id, "02-ben-okafor", "Ben Okafor")
    identifier = uuid.UUID(candidate["id"])
    with sessions.begin() as session:
        row = session.get(PipelineStep, (identifier, "cv"))
        assert row is not None
        row.status = "pending"
    gateway.cache.entries.clear()
    gateway.backend.errors.extend(retryable_errors(ONE_FAILED_ATTEMPT))
    run_inline(dataclasses.replace(pipeline_context, max_attempts=2), identifier, "cv")
    assert steps(sessions, identifier)["cv"].status == "succeeded"


def test_run_source_marks_only_outages_retryable(
    client: TestClient,
    job_id: str,
    sessions: sessionmaker[Session],
    gateway: FakeGateway,
    pipeline_context: PipelineContext,
):
    candidate = apply_with_cv(client, job_id, "02-ben-okafor", "Ben Okafor")
    gateway.cache.entries.clear()
    gateway.backend.errors.extend(retryable_errors(ONE_FAILED_ATTEMPT))
    with sessions() as session:
        row = session.get(Candidate, uuid.UUID(candidate["id"]))
        assert row is not None
        prepared = source_payload(session, row, "cv")
        assert prepared is not None
        run = run_source(session, pipeline_context.registry, row, "cv", prepared[0], actor="test")
    assert (run.status, run.retryable) == ("failed", True)
