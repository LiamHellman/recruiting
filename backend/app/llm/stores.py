"""Database-backed cache and usage ledger for the gateway.

Each write runs in its own short transaction so usage is recorded even when the caller's
request later fails.
"""

import uuid
from contextlib import suppress
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import LLMCacheEntry, LLMUsage
from app.llm.gateway import UsageRecord


class SqlResultCache:
    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self._sessions = sessions

    def get(self, tenant_id: uuid.UUID, key: str) -> dict[str, Any] | None:
        with self._sessions() as session:
            entry = session.get(LLMCacheEntry, (tenant_id, key))
            return entry.output if entry is not None else None

    def put(
        self,
        tenant_id: uuid.UUID,
        key: str,
        *,
        prompt_ref: str,
        model: str,
        input_hash: str,
        output: dict[str, Any],
    ) -> None:
        # A concurrent call may cache the same key first; either entry is valid.
        with suppress(IntegrityError), self._sessions.begin() as session:
            session.merge(
                LLMCacheEntry(
                    tenant_id=tenant_id,
                    cache_key=key,
                    prompt_ref=prompt_ref,
                    model=model,
                    input_hash=input_hash,
                    output=output,
                )
            )


class SqlUsageLedger:
    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self._sessions = sessions

    def record(self, record: UsageRecord) -> None:
        with self._sessions.begin() as session:
            session.add(
                LLMUsage(
                    tenant_id=record.tenant_id,
                    job_id=record.job_id,
                    prompt_ref=record.prompt_ref,
                    model_requested=record.model_requested,
                    model_served=record.model_served,
                    cached=record.cached,
                    input_tokens=record.usage.input_tokens,
                    output_tokens=record.usage.output_tokens,
                    cache_read_input_tokens=record.usage.cache_read_input_tokens,
                    cache_creation_input_tokens=record.usage.cache_creation_input_tokens,
                    cost_usd=record.cost_usd,
                    latency_ms=record.latency_ms,
                )
            )

    def tokens_used(self, tenant_id: uuid.UUID, job_id: uuid.UUID | None = None) -> int:
        total = (
            LLMUsage.input_tokens
            + LLMUsage.output_tokens
            + LLMUsage.cache_read_input_tokens
            + LLMUsage.cache_creation_input_tokens
        )
        query = select(func.coalesce(func.sum(total), 0)).where(LLMUsage.tenant_id == tenant_id)
        if job_id is not None:
            query = query.where(LLMUsage.job_id == job_id)
        with self._sessions() as session:
            return int(session.execute(query).scalar_one())
