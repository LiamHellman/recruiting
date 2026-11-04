"""The evidence-source interface and the registry that gates it on consent.

Every source (CV, GitHub, personal site, LinkedIn uploads, ...) implements
`ProfileEvidenceSource`. Sources are only ever called through `SourceRegistry.ingest`, which
refuses any source the candidate has not actively consented to.
"""

import uuid
from collections.abc import Iterable, Set
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.schemas.evidence import EvidenceItem, EvidenceSource, IngestReport


@dataclass(frozen=True)
class CandidateRef:
    id: uuid.UUID
    tenant_id: uuid.UUID
    job_id: uuid.UUID | None
    # Used only to redact the candidate's identity before LLM calls; never a scoring input.
    names: tuple[str, ...] = ()
    emails: tuple[str, ...] = ()


@dataclass(frozen=True)
class IngestResult:
    evidence: list[EvidenceItem]
    report: IngestReport
    # Source-specific structured output for the recruiter (e.g. a CvProfile), JSON-ready.
    profile: dict[str, Any] | None = field(default=None)


class ProfileEvidenceSource[P](Protocol):
    @property
    def name(self) -> EvidenceSource: ...

    def ingest(self, candidate: CandidateRef, payload: P) -> IngestResult: ...

    def input_key(self, candidate: CandidateRef, payload: P) -> str:
        """Hash of everything that determines the result, so the pipeline can skip a step
        that already succeeded on the same input."""
        ...


class ConsentMissing(Exception):
    def __init__(self, source: str) -> None:
        super().__init__(f"no active consent for source {source!r}")
        self.source = source


class UnknownSource(Exception):
    pass


class SourceError(Exception):
    """A source could not collect its data (e.g. the GitHub account does not exist, or a
    rate limit would take too long). Recorded as a failed run; the message is shown to the
    recruiter, so it must not contain secrets. `retryable` says whether trying again later
    could help."""

    retryable: bool = False


class SourceRegistry:
    def __init__(self, sources: Iterable[ProfileEvidenceSource[Any]]) -> None:
        self._sources: dict[str, ProfileEvidenceSource[Any]] = {}
        for source in sources:
            if source.name in self._sources:
                raise ValueError(f"source {source.name!r} registered twice")
            self._sources[source.name] = source

    def __contains__(self, name: object) -> bool:
        return name in self._sources

    def input_key(self, candidate: CandidateRef, source: str, payload: Any) -> str:
        impl = self._sources.get(source)
        if impl is None:
            raise UnknownSource(f"no source named {source!r}")
        return impl.input_key(candidate, payload)

    def ingest(
        self,
        candidate: CandidateRef,
        source: str,
        payload: Any,
        *,
        consented: Set[str],
    ) -> IngestResult:
        """Run one source for one candidate. `consented` is the set of sources with active
        consent, read just before the call; the source is refused before it sees any data
        unless it is in that set."""
        if source not in consented:
            raise ConsentMissing(source)
        impl = self._sources.get(source)
        if impl is None:
            raise UnknownSource(f"no source named {source!r}")
        result = impl.ingest(candidate, payload)
        stray = {item.source for item in result.evidence} - {source}
        if stray:
            raise ValueError(f"source {source!r} returned evidence labelled {sorted(stray)}")
        return result
