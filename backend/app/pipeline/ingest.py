"""Run one evidence source for one candidate and persist the outcome.

This is the synchronous core that the Phase 4 task graph will wrap. Consent is read from
the database immediately before the source runs, and `SourceRegistry.ingest` refuses any
source without an active grant. Refusals and failures are audit-logged.
"""

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.audit.log import record, stable_hash
from app.db.models import Candidate, Consent, EvidenceRecord, SourceRun
from app.llm.gateway import LLMError
from app.sources.base import CandidateRef, ConsentMissing, SourceError, SourceRegistry


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
    active consent for the source. A model or collection failure (e.g. a GitHub account that
    does not exist) is recorded as a failed run, not raised.
    Evidence from a successful run replaces earlier evidence from the same source.
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
