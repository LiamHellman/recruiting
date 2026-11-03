"""Database-backed `HttpCache`, scoped to one candidate. Writes run in their own short
transactions, like the LLM cache, so a later failure in the run keeps what was fetched."""

import uuid

from sqlalchemy import delete
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import GitHubCacheEntry
from app.github.client import CachedResponse


class SqlHttpCache:
    def __init__(
        self, sessions: sessionmaker[Session], tenant_id: uuid.UUID, candidate_id: uuid.UUID
    ) -> None:
        self._sessions = sessions
        self._tenant_id = tenant_id
        self._candidate_id = candidate_id

    def get(self, key: str) -> CachedResponse | None:
        with self._sessions() as session:
            row = session.get(GitHubCacheEntry, (self._candidate_id, key))
            if row is None:
                return None
            return CachedResponse(
                data=row.data,
                etag=row.etag,
                links=dict(row.links),
                immutable=row.immutable,
                fetched_at=row.fetched_at,
            )

    def put(self, key: str, entry: CachedResponse) -> None:
        with self._sessions.begin() as session:
            session.merge(
                GitHubCacheEntry(
                    candidate_id=self._candidate_id,
                    key=key,
                    tenant_id=self._tenant_id,
                    data=entry.data,
                    etag=entry.etag,
                    links=entry.links,
                    immutable=entry.immutable,
                    fetched_at=entry.fetched_at,
                )
            )


def clear_github_cache(session: Session, candidate_id: uuid.UUID) -> None:
    """In the caller's transaction (e.g. when GitHub consent is revoked)."""
    session.execute(delete(GitHubCacheEntry).where(GitHubCacheEntry.candidate_id == candidate_id))
