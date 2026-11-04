"""The fixed per-candidate task graph: cv -> github -> score.

Each source step is parse/collect, extract, validate and (GitHub) resolve identity, because
that is one `ProfileEvidenceSource.ingest` call; its sub-steps are already re-runnable through
the LLM and GitHub caches. This module knows nothing about Celery, so all of the logic can be
tested without a broker; `tasks.py` is a thin wrapper.

- Idempotent: a source step that succeeded on the same input hash is not run again, and a
  score whose inputs are unchanged is not rewritten.
- A step that fails for a reason that could pass (an outage) is retried with backoff, then
  dead-lettered. Attempts are counted in the database, so a step that keeps killing its worker
  is dead-lettered too.
- A failed source never blocks the score. The candidate stays in the list with what is known,
  labelled with the source that could not be processed.
"""

import logging
import random
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any, Literal, Protocol, cast

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.audit.log import record
from app.db.models import Candidate, Job, PipelineStep, SourceRun
from app.pipeline.ingest import (
    active_consents,
    candidate_ref,
    cv_payload,
    github_payload,
    latest_cv,
    run_source,
)
from app.pipeline.score import PIPELINE_ACTOR, score_one
from app.sources.base import ConsentMissing, SourceRegistry
from app.taxonomy.loader import Taxonomy

logger = logging.getLogger(__name__)

StepName = Literal["cv", "github", "score"]
GRAPH: tuple[StepName, ...] = ("cv", "github", "score")
SOURCE_STEPS: tuple[StepName, ...] = ("cv", "github")
# Statuses a new request starts over from. Attempts carry over between the others.
_SETTLED = frozenset({"pending", "succeeded", "failed", "skipped", "dead_letter"})


def next_step(step: str) -> StepName | None:
    index = GRAPH.index(cast(StepName, step)) + 1
    return GRAPH[index] if index < len(GRAPH) else None


@dataclass(frozen=True)
class PipelineContext:
    sessions: sessionmaker[Session]
    registry: SourceRegistry
    taxonomy: Taxonomy
    today: Callable[[], date] = field(default_factory=lambda: lambda: datetime.now(UTC).date())
    max_attempts: int = 4
    retry_base_seconds: float = 30.0
    retry_cap_seconds: float = 900.0  # below the broker's visibility timeout
    rng: random.Random = field(default_factory=random.Random)

    def retry_delay(self, attempt: int) -> float:
        backoff = min(self.retry_base_seconds * 2.0 ** (attempt - 1), self.retry_cap_seconds)
        return backoff * self.rng.uniform(0.5, 1.0)


@dataclass(frozen=True)
class StepOutcome:
    status: str
    ran: bool = True  # False: nothing to do (already done on this input)
    retry_in: float | None = None  # set: run this step again after this many seconds


class QueueUnavailable(Exception):
    """The broker could not be reached. The work stays visibly pending and can be re-queued
    through POST /candidates/{id}/pipeline."""


class PipelineQueue(Protocol):
    def enqueue(self, candidate_id: uuid.UUID, step: StepName) -> None: ...


# --- Steps --------------------------------------------------------------------------------


def _step_row(session: Session, candidate: Candidate, step: str) -> PipelineStep:
    row = session.get(PipelineStep, (candidate.id, step))
    if row is None:
        row = PipelineStep(
            candidate_id=candidate.id,
            step=step,
            tenant_id=candidate.tenant_id,
            status="pending",
            attempts=0,
        )
        session.add(row)
    return row


def _finish(
    session: Session, row: PipelineStep, status: str, error: str | None = None, *, ran: bool = True
) -> StepOutcome:
    row.status = status
    row.error = error
    session.commit()
    return StepOutcome(status, ran=ran)


def _start(session: Session, row: PipelineStep, input_hash: str | None = None) -> None:
    row.attempts = 1 if row.status in _SETTLED else row.attempts + 1
    row.status = "running"
    row.error = None
    row.input_hash = input_hash
    session.commit()


def _dead_letter(
    session: Session, candidate: Candidate, row: PipelineStep, error: str, error_type: str
) -> StepOutcome:
    record(
        session,
        tenant_id=candidate.tenant_id,
        actor=PIPELINE_ACTOR,
        action="pipeline.dead_letter",
        entity_type="candidate",
        entity_id=candidate.id,
        payload={"step": row.step, "attempts": row.attempts, "error": error_type},
    )
    logger.error("candidate %s step %s dead-lettered: %s", candidate.id, row.step, error_type)
    return _finish(session, row, "dead_letter", error)


def _retry_or_dead_letter(
    ctx: PipelineContext,
    session: Session,
    candidate: Candidate,
    row: PipelineStep,
    error: str,
    error_type: str,
) -> StepOutcome:
    if row.attempts >= ctx.max_attempts:
        return _dead_letter(session, candidate, row, error, error_type)
    row.status = "retrying"
    row.error = error
    session.commit()
    return StepOutcome("retrying", retry_in=ctx.retry_delay(row.attempts))


def source_payload(
    session: Session, candidate: Candidate, step: str
) -> tuple[Any, uuid.UUID | None] | None:
    """What a source step would run on, or None if the candidate gave nothing for it."""
    if step == "cv":
        document = latest_cv(session, candidate.id)
        return None if document is None else (cv_payload(document, candidate), document.id)
    if not candidate.github_handle:
        return None
    return github_payload(session, candidate), None


def _source_step(
    ctx: PipelineContext, session: Session, candidate: Candidate, row: PipelineStep
) -> StepOutcome:
    step = row.step
    if step not in active_consents(session, candidate.id):
        return _finish(session, row, "skipped", "no active consent")
    prepared = source_payload(session, candidate, step)
    if prepared is None:
        return _finish(session, row, "skipped", "nothing submitted for this source")
    payload, document_id = prepared
    key = ctx.registry.input_key(candidate_ref(candidate), step, payload)
    if row.status == "succeeded" and row.input_hash == key:
        return StepOutcome("succeeded", ran=False)

    _start(session, row, key)
    try:
        run = run_source(
            session,
            ctx.registry,
            candidate,
            step,
            payload,
            actor=PIPELINE_ACTOR,
            document_id=document_id,
        )
    except ConsentMissing:  # revoked while the step ran; nothing from it was kept
        return _finish(session, _step_row(session, candidate, step), "skipped", "no active consent")
    if run.status == "succeeded":
        return _finish(session, row, "succeeded")
    error = run.error or "failed"
    if run.retryable:
        return _retry_or_dead_letter(ctx, session, candidate, row, error, error.split(":")[0])
    return _finish(session, row, "failed", error)


def _score_step(
    ctx: PipelineContext, session: Session, candidate: Candidate, row: PipelineStep
) -> StepOutcome:
    job = session.get(Job, candidate.job_id)
    if job is None or job.status != "confirmed":  # scored when the recruiter confirms the job
        return _finish(session, row, "skipped", "job is not confirmed")
    _start(session, row)
    changed = score_one(session, candidate, job, ctx.taxonomy, ctx.today())
    return _finish(session, row, "succeeded", ran=changed)


def execute_step(ctx: PipelineContext, candidate_id: uuid.UUID, step: str) -> StepOutcome:
    """Run one step once and record the outcome. Never raises for a failure of the step."""
    if step not in GRAPH:
        raise ValueError(f"unknown pipeline step {step!r}")
    with ctx.sessions() as session:
        candidate = session.get(Candidate, candidate_id)
        if candidate is None:  # deleted while queued
            return StepOutcome("skipped", ran=False)
        row = _step_row(session, candidate, step)
        if row.status not in _SETTLED and row.attempts >= ctx.max_attempts:
            # Every attempt so far ended without settling: the step keeps killing its worker.
            return _dead_letter(session, candidate, row, row.error or "worker lost", "WorkerLost")
        try:
            if step == "score":
                return _score_step(ctx, session, candidate, row)
            return _source_step(ctx, session, candidate, row)
        except Exception as exc:
            # A bug or a database error. Only the type is stored: messages of unexpected
            # errors can carry SQL parameters, which can be personal data.
            logger.exception("candidate %s step %s raised", candidate_id, step)
            session.rollback()
            row = _step_row(session, candidate, step)
            name = type(exc).__name__
            return _retry_or_dead_letter(ctx, session, candidate, row, name, name)


def run_inline(ctx: PipelineContext, candidate_id: uuid.UUID, step: StepName) -> None:
    """Run from `step` to the end of the graph in this process, retrying without waiting."""
    current: StepName | None = step
    while current is not None:
        if execute_step(ctx, candidate_id, current).retry_in is None:
            current = next_step(current)


class InlineQueue:
    """Runs work immediately in the caller's process. For tests and offline evals."""

    def __init__(self, ctx: PipelineContext) -> None:
        self.ctx = ctx

    def enqueue(self, candidate_id: uuid.UUID, step: StepName) -> None:
        run_inline(self.ctx, candidate_id, step)


# --- Requesting work ----------------------------------------------------------------------


def mark_pending(session: Session, candidate: Candidate, steps: Iterable[str]) -> None:
    for step in steps:
        row = _step_row(session, candidate, step)
        row.status = "pending"
        row.attempts = 0
        row.error = None


def settle_source_step(
    session: Session, candidate: Candidate, step: str, run: SourceRun, input_hash: str
) -> None:
    """Record a run started outside the queue (POST .../sources/{source}/runs) on its step."""
    row = _step_row(session, candidate, step)
    row.status = run.status
    row.error = run.error
    row.input_hash = input_hash
    row.attempts = 1
    session.commit()


def _enqueue(queue: PipelineQueue, candidate_id: uuid.UUID, step: StepName) -> bool:
    try:
        queue.enqueue(candidate_id, step)
    except QueueUnavailable:
        logger.error("queue unavailable: candidate %s step %s stays pending", candidate_id, step)
        return False
    return True


def start_pipeline(session: Session, queue: PipelineQueue, candidate: Candidate) -> bool:
    """Queue the whole graph for a stored candidate. False if the queue was unavailable."""
    mark_pending(session, candidate, GRAPH)
    session.commit()
    return _enqueue(queue, candidate.id, GRAPH[0])


def resume_pipeline(session: Session, queue: PipelineQueue, candidate: Candidate) -> bool:
    """Queue again from the first step that did not succeed, after a dead letter or an outage.
    Later source steps that already succeeded are skipped by their input hash."""
    rows = {
        row.step: row
        for row in session.scalars(
            select(PipelineStep).where(PipelineStep.candidate_id == candidate.id)
        )
    }
    todo = [s for s in GRAPH if s not in rows or rows[s].status != "succeeded"] or ["score"]
    mark_pending(session, candidate, todo)
    session.commit()
    return _enqueue(queue, candidate.id, todo[0])


def request_rescore(
    session: Session, queue: PipelineQueue, candidates: Iterable[Candidate]
) -> None:
    """Re-score after anything that changes a score's inputs. Commits the session."""
    chosen = list(candidates)
    for candidate in chosen:
        mark_pending(session, candidate, ["score"])
    session.commit()
    for candidate in chosen:
        _enqueue(queue, candidate.id, "score")
