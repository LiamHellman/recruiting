import uuid
from datetime import UTC, datetime
from typing import Annotated, Any, cast
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import CursorResult, delete, select, update
from sqlalchemy.orm import Session

from app.api.deps import get_actor, get_file_store, get_registry, get_tenant_id
from app.api.jobs import load_job
from app.audit.log import record, sha256_hex
from app.db.models import Candidate, Consent, Document, EvidenceRecord, SourceRun
from app.db.session import get_session
from app.extraction.redact import find_emails, find_links, normalize_url
from app.extraction.text import MAX_BYTES, DocumentError, extract_document
from app.github.cache import clear_github_cache
from app.identity.github import handle_from_cv
from app.pipeline.ingest import run_source
from app.schemas.candidate import (
    CandidateEvidenceOut,
    CandidateOut,
    ConsentOut,
    DocumentOut,
    EvidenceOut,
    SourceRunOut,
)
from app.schemas.evidence import CvProfile, EvidenceSource, IngestReport
from app.schemas.github import GitHubProfile, HandleSource
from app.sources.base import ConsentMissing, SourceRegistry
from app.sources.cv import CvPayload
from app.sources.github import GitHubPayload
from app.storage import FileStore

router = APIRouter(tags=["candidates"])

SessionDep = Annotated[Session, Depends(get_session)]
TenantDep = Annotated[uuid.UUID, Depends(get_tenant_id)]
ActorDep = Annotated[str, Depends(get_actor)]
RegistryDep = Annotated[SourceRegistry, Depends(get_registry)]
StoreDep = Annotated[FileStore, Depends(get_file_store)]

# GitHub's rules: alphanumerics and single hyphens, not at either end; 39 characters max.
GITHUB_HANDLE = r"^[A-Za-z0-9](?:-?[A-Za-z0-9])*$"
EMAIL = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
NOTICE_VERSION = r"^[A-Za-z0-9._-]{1,50}$"


def _unprocessable(detail: str) -> HTTPException:
    return HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail)


def _check_site_url(url: str) -> str:
    """Shape only. Nothing is fetched here; Phase 6 adds the fetcher and its SSRF guards."""
    parts = urlsplit(url.strip())
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.username:
        raise _unprocessable("site_url must be an http(s) URL")
    return url.strip()


def load_candidate(session: Session, tenant_id: uuid.UUID, candidate_id: uuid.UUID) -> Candidate:
    candidate = session.get(Candidate, candidate_id)
    if candidate is None or candidate.tenant_id != tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "candidate not found")
    return candidate


def _run_out(run: SourceRun) -> SourceRunOut:
    report = IngestReport.model_validate(run.report) if run.report else None
    return SourceRunOut(
        id=str(run.id),
        source=run.source,
        status=run.status,
        prompt_ref=report.prompt_ref if report else None,
        model=report.model if report else None,
        model_served=report.model_served if report else None,
        cached=report.cached if report else None,
        checked=report.checked if report else 0,
        accepted=report.accepted if report else 0,
        rejected=report.rejected if report else [],
        unmapped_terms=report.unmapped_terms if report else [],
        redactions=report.redactions if report else {},
        fetch=report.fetch if report else {},
        error=run.error,
        created_at=run.created_at,
    )


def _candidate_out(session: Session, candidate: Candidate) -> CandidateOut:
    consents = session.scalars(
        select(Consent).where(Consent.candidate_id == candidate.id).order_by(Consent.granted_at)
    )
    documents = session.scalars(
        select(Document).where(Document.candidate_id == candidate.id).order_by(Document.created_at)
    )
    runs = session.scalars(
        select(SourceRun)
        .where(SourceRun.candidate_id == candidate.id)
        .order_by(SourceRun.created_at.desc())
    )
    return CandidateOut(
        id=str(candidate.id),
        job_id=str(candidate.job_id),
        full_name=candidate.full_name,
        email=candidate.email,
        github_handle=candidate.github_handle,
        github_handle_from=candidate.github_handle_from,
        site_url=candidate.site_url,
        created_at=candidate.created_at,
        consents=[
            ConsentOut(
                source=c.source,
                notice_version=c.notice_version,
                granted_at=c.granted_at,
                revoked_at=c.revoked_at,
            )
            for c in consents
        ],
        documents=[
            DocumentOut(
                id=str(d.id),
                kind=d.kind,
                content_type=d.content_type,
                size_bytes=d.size_bytes,
                sha256=d.sha256,
                page_count=len(d.pages),
                links=d.links,
                created_at=d.created_at,
            )
            for d in documents
        ],
        runs=[_run_out(run) for run in runs],
    )


@router.post("/jobs/{job_id}/candidates", status_code=status.HTTP_201_CREATED)
def create_candidate(
    job_id: uuid.UUID,
    session: SessionDep,
    tenant_id: TenantDep,
    actor: ActorDep,
    registry: RegistryDep,
    store: StoreDep,
    cv: Annotated[UploadFile, File(description="The candidate's CV, PDF or DOCX")],
    full_name: Annotated[str, Form(min_length=1, max_length=200)],
    notice_version: Annotated[str, Form(pattern=NOTICE_VERSION)],
    consent_cv: Annotated[bool, Form()] = False,
    email: Annotated[str | None, Form(max_length=320, pattern=EMAIL)] = None,
    github_handle: Annotated[str | None, Form(max_length=39, pattern=GITHUB_HANDLE)] = None,
    consent_github: Annotated[bool, Form()] = False,
    site_url: Annotated[str | None, Form(max_length=2000)] = None,
    consent_site: Annotated[bool, Form()] = False,
) -> CandidateOut:
    """Apply to a job with a CV and optional sources, with consent per source.

    A source the candidate did not consent to is rejected, not stored. With consent_github
    but no github_handle, the handle is the one GitHub profile the CV links to. The CV, then
    GitHub, are processed immediately; a failure leaves the candidate stored with a failed
    run that can be retried.
    """
    job = load_job(session, tenant_id, job_id)
    if not consent_cv:
        raise _unprocessable("consent_cv is required to process the CV")
    if github_handle and not consent_github:
        raise _unprocessable("give a GitHub handle together with consent_github, or neither")
    if bool(site_url) != consent_site:
        raise _unprocessable("give a site_url together with consent_site, or neither")
    site = _check_site_url(site_url) if site_url else None

    data = cv.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "CV is larger than 10 MB")
    try:
        extracted = extract_document(data)
    except DocumentError as exc:
        raise _unprocessable(str(exc)) from exc
    links = list(dict.fromkeys([*map(normalize_url, extracted.links), *find_links(extracted.text)]))
    handle_from: HandleSource | None = "form" if github_handle else None
    if consent_github and not github_handle:
        github_handle = handle_from_cv(links)
        if github_handle is None:
            raise _unprocessable(
                "consent_github needs a GitHub handle: give github_handle, or link exactly "
                "one GitHub profile in the CV"
            )
        handle_from = "cv"

    now = datetime.now(UTC)
    candidate = Candidate(
        id=uuid.uuid4(),
        tenant_id=tenant_id,
        job_id=job.id,
        full_name=" ".join(full_name.split()),
        email=email.strip().lower() if email else None,
        github_handle=github_handle,
        github_handle_from=handle_from,
        site_url=site,
        created_at=now,
    )
    session.add(candidate)
    sources: list[EvidenceSource] = ["cv"]
    if consent_github:
        sources.append("github")
    if consent_site:
        sources.append("site")
    for source in sources:
        session.add(
            Consent(
                tenant_id=tenant_id,
                candidate_id=candidate.id,
                source=source,
                notice_version=notice_version,
                granted_at=now,
            )
        )
    document = Document(
        id=uuid.uuid4(),
        tenant_id=tenant_id,
        candidate_id=candidate.id,
        kind="cv",
        content_type=extracted.content_type,
        sha256=sha256_hex(data),
        size_bytes=len(data),
        storage_key=f"{tenant_id}/{candidate.id}/{uuid.uuid4()}",
        text=extracted.text,
        pages=[list(span) for span in extracted.pages],
        links=links,
        emails=find_emails(extracted.text),
        created_at=now,
    )
    session.add(document)
    session.flush()  # candidate before its children, whatever the unit-of-work order
    record(
        session,
        tenant_id=tenant_id,
        actor=actor,
        action="candidate.applied",
        entity_type="candidate",
        entity_id=candidate.id,
        payload={
            "job_id": str(job.id),
            "consents": [{"source": s, "notice_version": notice_version} for s in sources],
            "github_handle_from": handle_from,
            "document_id": str(document.id),
            "document_sha256": document.sha256,
            "content_type": document.content_type,
            "size_bytes": document.size_bytes,
        },
    )
    store.put(document.storage_key, data)
    try:
        session.commit()
    except Exception:
        store.delete(document.storage_key)
        raise

    run_source(
        session,
        registry,
        candidate,
        "cv",
        _cv_payload(document, candidate),
        actor=actor,
        document_id=document.id,
    )
    if consent_github:
        run_source(
            session, registry, candidate, "github", _github_payload(session, candidate), actor=actor
        )
    return _candidate_out(session, candidate)


def _cv_payload(document: Document, candidate: Candidate) -> CvPayload:
    return CvPayload(
        document_id=document.id,
        text=document.text,
        received_on=candidate.created_at.date(),
        emails=tuple(document.emails),
    )


def _latest_cv(session: Session, candidate: Candidate) -> Document | None:
    return session.scalars(
        select(Document)
        .where(Document.candidate_id == candidate.id, Document.kind == "cv")
        .order_by(Document.created_at.desc())
        .limit(1)
    ).first()


def _github_payload(session: Session, candidate: Candidate) -> GitHubPayload:
    if not candidate.github_handle:
        raise _unprocessable("candidate gave no GitHub handle")
    cv = _latest_cv(session, candidate)
    return GitHubPayload(
        handle=candidate.github_handle,
        handle_from="cv" if candidate.github_handle_from == "cv" else "form",
        collected_on=datetime.now(UTC).date(),
        site_url=candidate.site_url,
        cv_links=tuple(cv.links) if cv else (),
        emails=tuple(cv.emails) if cv else (),
    )


def _latest_profile(session: Session, candidate: Candidate, source: str) -> dict[str, Any] | None:
    run = session.scalars(
        select(SourceRun)
        .where(
            SourceRun.candidate_id == candidate.id,
            SourceRun.source == source,
            SourceRun.status == "succeeded",
        )
        .order_by(SourceRun.created_at.desc())
        .limit(1)
    ).first()
    return run.profile if run is not None else None


@router.get("/candidates/{candidate_id}")
def get_candidate(
    candidate_id: uuid.UUID, session: SessionDep, tenant_id: TenantDep
) -> CandidateOut:
    return _candidate_out(session, load_candidate(session, tenant_id, candidate_id))


@router.get("/candidates/{candidate_id}/evidence")
def get_candidate_evidence(
    candidate_id: uuid.UUID, session: SessionDep, tenant_id: TenantDep
) -> CandidateEvidenceOut:
    candidate = load_candidate(session, tenant_id, candidate_id)
    rows = session.scalars(
        select(EvidenceRecord)
        .where(EvidenceRecord.candidate_id == candidate.id)
        .order_by(EvidenceRecord.skill_id, EvidenceRecord.locator)
    )
    cv_profile = _latest_profile(session, candidate, "cv")
    github_profile = _latest_profile(session, candidate, "github")
    return CandidateEvidenceOut(
        candidate_id=str(candidate.id),
        evidence=[
            EvidenceOut(
                id=str(row.id),
                run_id=str(row.run_id),
                skill_id=row.skill_id,
                level=row.level,
                source=row.source,
                locator=row.locator,
                quote=row.quote,
                date_last_seen=row.date_last_seen,
                ownership=row.ownership,
                confidence=row.confidence,
            )
            for row in rows
        ],
        cv_profile=CvProfile.model_validate(cv_profile) if cv_profile else None,
        github_profile=GitHubProfile.model_validate(github_profile) if github_profile else None,
    )


@router.post("/candidates/{candidate_id}/sources/{source}/runs", status_code=201)
def rerun_source(
    candidate_id: uuid.UUID,
    source: EvidenceSource,
    session: SessionDep,
    tenant_id: TenantDep,
    actor: ActorDep,
    registry: RegistryDep,
) -> SourceRunOut:
    """Run a source again, e.g. after a failed run, a new prompt version, or (GitHub) to pick
    up new activity. A GitHub re-run revalidates cached answers instead of refetching."""
    candidate = load_candidate(session, tenant_id, candidate_id)
    if source not in ("cv", "github") or source not in registry:
        raise _unprocessable(f"source {source!r} is not available yet")
    try:
        if source == "cv":
            document = _latest_cv(session, candidate)
            if document is None:
                raise _unprocessable("candidate has no CV")
            run = run_source(
                session,
                registry,
                candidate,
                source,
                _cv_payload(document, candidate),
                actor=actor,
                document_id=document.id,
            )
        else:
            payload = _github_payload(session, candidate)
            run = run_source(session, registry, candidate, source, payload, actor=actor)
    except ConsentMissing as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    return _run_out(run)


@router.delete("/candidates/{candidate_id}/consents/{source}")
def revoke_consent(
    candidate_id: uuid.UUID,
    source: EvidenceSource,
    session: SessionDep,
    tenant_id: TenantDep,
    actor: ActorDep,
) -> CandidateOut:
    """Record that the candidate withdrew consent for a source. Nothing more is collected
    from it, and what was derived from it (evidence, profile, run reports, fetched GitHub
    data) is deleted so it cannot be scored or shown. The uploaded file stays until the
    candidate's data is deleted (Phase 8.3/8.5)."""
    candidate = load_candidate(session, tenant_id, candidate_id)
    consent = session.scalars(
        select(Consent).where(
            Consent.candidate_id == candidate.id,
            Consent.source == source,
            Consent.revoked_at.is_(None),
        )
    ).first()
    if consent is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no active consent for {source!r}")
    consent.revoked_at = datetime.now(UTC)
    removed = cast(
        CursorResult[Any],
        session.execute(
            delete(EvidenceRecord).where(
                EvidenceRecord.candidate_id == candidate.id, EvidenceRecord.source == source
            )
        ),
    )
    # Other derived data goes too: the profile and the report (which quotes the source),
    # and for GitHub everything fetched (the cache).
    if source == "github":
        clear_github_cache(session, candidate.id)
    session.execute(
        update(SourceRun)
        .where(SourceRun.candidate_id == candidate.id, SourceRun.source == source)
        .values(profile=None, report=None)
    )
    record(
        session,
        tenant_id=tenant_id,
        actor=actor,
        action="consent.revoked",
        entity_type="candidate",
        entity_id=candidate.id,
        payload={
            "source": source,
            "notice_version": consent.notice_version,
            "evidence_removed": removed.rowcount,
        },
    )
    session.commit()
    return _candidate_out(session, candidate)
