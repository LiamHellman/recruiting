"""Run one evidence source for one candidate and persist the outcome.

This is the synchronous core that the task graph (`graph.py`) wraps. Consent is read from
the database immediately before the source runs, and `SourceRegistry.ingest` refuses any
source without an active grant. Consent is read again before anything is stored, because a
queued run can take minutes. Refusals and failures are audit-logged.
"""

import uuid
from datetime import UTC, datetime
from typing import Any, cast

from sqlalchemy import CursorResult, delete, select, update
from sqlalchemy.orm import Session

from app.audit.log import record, stable_hash
from app.db.models import (
    Candidate,
    CandidateOverride,
    Consent,
    Document,
    EvidenceRecord,
    PipelineStep,
    SourceRun,
)
from app.github.cache import clear_github_cache
from app.llm.gateway import LLMError, RetryableLLMError
from app.sources.base import CandidateRef, ConsentMissing, SourceError, SourceRegistry
from app.sources.cv import CvPayload
from app.sources.github import GitHubPayload


def active_consents(session: Session, candidate_id: uuid.UUID) -> frozenset[str]:
    rows = session.scalars(
        select(Consent.source).where(
            Consent.candidate_id == candidate_id, Consent.revoked_at.is_(None)
        )
    )
    return frozenset(rows)


def candidate_ref(candidate: Candidate) -> CandidateRef:
    return CandidateRef(
        id=candidate.id,
        tenant_id=candidate.tenant_id,
        job_id=candidate.job_id,
        names=(candidate.full_name,),
        emails=(candidate.email,) if candidate.email else (),
    )


def latest_cv(session: Session, candidate_id: uuid.UUID) -> Document | None:
    return session.scalars(
        select(Document)
        .where(Document.candidate_id == candidate_id, Document.kind == "cv")
        .order_by(Document.created_at.desc())
        .limit(1)
    ).first()


def latest_profile(session: Session, candidate_id: uuid.UUID, source: str) -> dict[str, Any] | None:
    run = session.scalars(
        select(SourceRun)
        .where(
            SourceRun.candidate_id == candidate_id,
            SourceRun.source == source,
            SourceRun.status == "succeeded",
        )
        .order_by(SourceRun.created_at.desc())
        .limit(1)
    ).first()
    return run.profile if run is not None else None


def cv_payload(document: Document, candidate: Candidate) -> CvPayload:
    return CvPayload(
        document_id=document.id,
        text=document.text,
        received_on=candidate.created_at.date(),
        emails=tuple(document.emails),
    )


def github_payload(session: Session, candidate: Candidate) -> GitHubPayload:
    if not candidate.github_handle:
        raise ValueError("candidate gave no GitHub handle")
    cv = latest_cv(session, candidate.id)
    return GitHubPayload(
        handle=candidate.github_handle,
        handle_from="cv" if candidate.github_handle_from == "cv" else "form",
        collected_on=datetime.now(UTC).date(),
        site_url=candidate.site_url,
        cv_links=tuple(cv.links) if cv else (),
        emails=tuple(cv.emails) if cv else (),
    )


def lock_candidate(session: Session, candidate_id: uuid.UUID) -> None:
    """Serialize consent changes with whatever stores or scores a candidate's data."""
    session.execute(select(Candidate.id).where(Candidate.id == candidate_id).with_for_update())


def purge_source_data(session: Session, candidate_id: uuid.UUID, source: str) -> int:
    """Delete what was derived from a source, within the caller's transaction: evidence,
    profile and report (which quote it), fetched data, recruiter overrides that point at it
    (locators name repositories), and its pipeline step. Returns the evidence removed."""
    removed = cast(
        CursorResult[Any],
        session.execute(
            delete(EvidenceRecord).where(
                EvidenceRecord.candidate_id == candidate_id, EvidenceRecord.source == source
            )
        ),
    )
    if source == "github":
        clear_github_cache(session, candidate_id)
    session.execute(
        update(SourceRun)
        .where(SourceRun.candidate_id == candidate_id, SourceRun.source == source)
        .values(profile=None, report=None)
    )
    for override in session.scalars(
        select(CandidateOverride).where(
            CandidateOverride.candidate_id == candidate_id,
            CandidateOverride.kind.in_(("evidence", "identity")),
        )
    ):
        about_source = (
            override.target.get("source") == source
            if override.kind == "evidence"
            else source == "github"
        )
        if about_source:
            session.delete(override)
    session.execute(
        delete(PipelineStep).where(
            PipelineStep.candidate_id == candidate_id, PipelineStep.step == source
        )
    )
    return removed.rowcount


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, SourceError):
        return exc.retryable
    return isinstance(exc, RetryableLLMError)


def run_source(
    session: Session,
    registry: SourceRegistry,
    candidate: Candidate,
    source: str,
    payload: Any,
    *,
    actor: str,
    document_id: uuid.UUID | None = None,
) -> SourceRun:
    """Run `source` and commit the run, its evidence and audit entries.

    Raises `ConsentMissing` (after committing an audit entry) when the candidate has no
    active consent for the source, either before the run or by the time it finished. A model
    or collection failure (e.g. a GitHub account that does not exist) is recorded as a failed
    run, not raised. Evidence from a successful run replaces earlier evidence from the same
    source.
    """
    try:
        result = registry.ingest(
            candidate_ref(candidate),
            source,
            payload,
            consented=active_consents(session, candidate.id),
        )
    except ConsentMissing:
        record(
            session,
            tenant_id=candidate.tenant_id,
            actor=actor,
            action="source.refused",
            entity_type="candidate",
            entity_id=candidate.id,
            payload={"source": source, "reason": "no active consent"},
        )
        session.commit()
        raise
    except (LLMError, SourceError) as exc:
        run = SourceRun(
            id=uuid.uuid4(),
            tenant_id=candidate.tenant_id,
            candidate_id=candidate.id,
            source=source,
            document_id=document_id,
            status="failed",
            report=None,
            profile=None,
            error=f"{type(exc).__name__}: {exc}"[:2000],
            retryable=_is_retryable(exc),
            created_at=datetime.now(UTC),
        )
        session.add(run)
        record(
            session,
            tenant_id=candidate.tenant_id,
            actor=actor,
            action="source.failed",
            entity_type="candidate",
            entity_id=candidate.id,
            payload={"source": source, "run_id": str(run.id), "error": type(exc).__name__},
        )
        session.commit()
        return run

    # The run may have outlasted the candidate's consent. Under this lock, a revocation has
    # either committed (and is seen here) or waits until the evidence is stored and then
    # deletes it.
    lock_candidate(session, candidate.id)
    if source not in active_consents(session, candidate.id):
        purge_source_data(session, candidate.id, source)  # e.g. what collection just cached
        record(
            session,
            tenant_id=candidate.tenant_id,
            actor=actor,
            action="source.refused",
            entity_type="candidate",
            entity_id=candidate.id,
            payload={"source": source, "reason": "consent revoked during the run"},
        )
        session.commit()
        raise ConsentMissing(source)

    report = result.report
    run = SourceRun(
        id=uuid.uuid4(),
        tenant_id=candidate.tenant_id,
        candidate_id=candidate.id,
        source=source,
        document_id=document_id,
        status="succeeded",
        report=report.model_dump(mode="json"),
        profile=result.profile,
        error=None,
        created_at=datetime.now(UTC),
    )
    session.add(run)
    session.execute(
        delete(EvidenceRecord).where(
            EvidenceRecord.candidate_id == candidate.id, EvidenceRecord.source == source
        )
    )
    session.add_all(
        EvidenceRecord(
            tenant_id=candidate.tenant_id,
            candidate_id=candidate.id,
            run_id=run.id,
            **item.model_dump(),
        )
        for item in result.evidence
    )
    record(
        session,
        tenant_id=candidate.tenant_id,
        actor=actor,
        action="source.ingested",
        entity_type="candidate",
        entity_id=candidate.id,
        payload={
            "source": source,
            "run_id": str(run.id),
            "document_id": str(document_id) if document_id else None,
            "prompt_ref": report.prompt_ref,
            "model": report.model,
            "model_served": report.model_served,
            "llm_cached": report.cached,
            "input_sha256": report.input_sha256,
            # A candidate's skills are personal data: hash them, as with all audit payloads.
            "evidence_sha256": stable_hash(
                [item.model_dump(mode="json") for item in result.evidence]
            ),
            "evidence_count": len(result.evidence),
            "checked": report.checked,
            "rejected": len(report.rejected),
            "rejection_reasons": sorted({r.reason for r in report.rejected}),
            "unmapped_count": len(report.unmapped_terms),
            "redactions": report.redactions,
            "fetch": report.fetch,
        },
    )
    session.commit()
    return run
