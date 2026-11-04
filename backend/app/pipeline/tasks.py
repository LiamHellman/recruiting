"""Celery wiring for the task graph in `graph.py`. Start a worker with:

    uv run celery -A app.pipeline.tasks worker --loglevel=info --pool=solo

(`--pool=solo` is for Windows; leave it out elsewhere.) Tasks carry only IDs, so nothing
personal passes through the broker.
"""

import uuid
from functools import lru_cache
from typing import Any

from celery import Celery
from kombu.exceptions import OperationalError

from app.config import get_settings
from app.pipeline.graph import (
    PipelineContext,
    QueueUnavailable,
    StepName,
    execute_step,
    next_step,
)

celery_app = Celery("recruiting")
celery_app.conf.update(
    broker_url=get_settings().redis_url,
    task_ignore_result=True,
    task_acks_late=True,  # a worker that dies mid-step leaves the task to be redelivered
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,  # steps are slow: don't hoard them
    task_default_queue="pipeline",
    broker_connection_retry_on_startup=True,
    broker_transport_options={"visibility_timeout": 3600},
    task_publish_retry_policy={"max_retries": 2, "interval_start": 0.1, "interval_step": 0.2},
)


@lru_cache
def default_context() -> PipelineContext:
    # Imported here: the API's wiring imports this module for the queue.
    from app.api.deps import build_registry, get_gateway, get_github_clients, get_taxonomy
    from app.db.session import get_sessionmaker

    taxonomy = get_taxonomy()
    registry = build_registry(get_gateway(), taxonomy, get_github_clients())
    return PipelineContext(get_sessionmaker(), registry, taxonomy)


# Replaced in tests.
context_provider = default_context


@celery_app.task(name="pipeline.step", bind=True, max_retries=None)  # type: ignore[untyped-decorator]
def run_step(self: Any, candidate_id: str, step: str) -> None:
    """One step, then the next. Retries are capped by the attempts counted in the database."""
    outcome = execute_step(context_provider(), uuid.UUID(candidate_id), step)
    if outcome.retry_in is not None:
        raise self.retry(countdown=outcome.retry_in)
    following = next_step(step)
    if following is not None:
        run_step.delay(candidate_id, following)


class CeleryQueue:
    def enqueue(self, candidate_id: uuid.UUID, step: StepName) -> None:
        try:
            run_step.delay(str(candidate_id), step)
        except OperationalError as exc:
            raise QueueUnavailable(str(exc)) from exc
