import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import (
    get_actor,
    get_file_store,
    get_pipeline_queue,
    get_registry,
    get_tenant_id,
)
from app.api.jobs import load_job
from app.audit.log import record, sha256_hex
from app.db.models import (
    Candidate,
    CandidateDecision,
    CandidateOverride,
    CandidateScore,
    Consent,
    Document,
    EvidenceRecord,
    LLMUsage,
    PipelineStep,
    SourceRun,
)
from app.db.session import get_session
from app.extraction.redact import find_emails, find_links, normalize_url
from app.extraction.text import MAX_BYTES, DocumentError, extract_document
from app.identity.github import handle_from_cv
from app.pipeline.graph import (
    PipelineQueue,
    request_rescore,
    settle_source_step,
    source_payload,
    start_pipeline,
)
from app.pipeline.ingest import (
    candidate_ref,
    latest_profile,
    lock_candidate,
    purge_source_data,
    run_source,
)
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
from app.schemas.review import DecisionOut, OverrideOut, PipelineStepOut
from app.schemas.score import ScoreResult
from app.scoring.inputs import fold_overrides
from app.sources.base import ConsentMissing, SourceRegistry
from app.storage import FileStore

router = APIRouter(tags=["candidates"])

SessionDep = Annotated[Session, Depends(get_session)]
TenantDep = Annotated[uuid.UUID, Depends(get_tenant_id)]
ActorDep = Annotated[str, Depends(get_actor)]
RegistryDep = Annotated[SourceRegistry, Depends(get_registry)]
StoreDep = Annotated[FileStore, Depends(get_file_store)]
QueueDep = Annotated[PipelineQueue, Depends(get_pipeline_queue)]

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


def step_out(step: PipelineStep) -> PipelineStepOut:
    return PipelineStepOut(
        step=step.step,
        status=step.status,
        attempts=step.attempts,
        error=step.error,
        updated_at=step.updated_at,
    )


def override_out(override: CandidateOverride) -> OverrideOut:
    return OverrideOut(
        id=override.id,
        kind=override.kind,
        target=override.target,
        reason=override.reason,
        actor=override.actor,
        created_at=override.created_at,
    )


def decision_out(decision: CandidateDecision) -> DecisionOut:
    return DecisionOut(
        id=decision.id,
        decision=decision.decision,  # a check constraint guards it
        reason=decision.reason,
        actor=decision.actor,
        score_total=decision.score_total,
        created_at=decision.created_at,
    )


def _candidate_out(session: Session, candidate: Candidate) -> CandidateOut:
    # A queue running in this process (tests, offline evals) has written since these loaded.
    session.expire_all()
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
    pipeline = session.scalars(
        select(PipelineStep).where(PipelineStep.candidate_id == candidate.id)
    )
    cost = session.execute(
        select(func.coalesce(func.sum(LLMUsage.cost_usd), 0)).where(
            LLMUsage.candidate_id == candidate.id
        )
    ).scalar_one()
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
        pipeline=[step_out(step) for step in pipeline],
        llm_cost_usd=Decimal(cost or 0),
    )


@router.post("/jobs/{job_id}/candidates", status_code=status.HTTP_201_CREATED)
def create_candidate(
    job_id: uuid.UUID,
    session: SessionDep,
    tenant_id: TenantDep,
    actor: ActorDep,
    store: StoreDep,
    queue: QueueDep,
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
    but no github_handle, the handle is the one GitHub profile the CV links to. Processing
    (CV, then GitHub, then the score) is queued; `pipeline` in the response and in
    GET /candidates/{id} shows how far it got. A source that fails leaves the candidate stored
    with a failed run that can be retried.
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

    start_pipeline(session, queue, candidate)  # a queue outage leaves the steps pending
    return _candidate_out(session, candidate)


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
    cv_profile = latest_profile(session, candidate.id, "cv")
    github_profile = latest_profile(session, candidate.id, "github")
    overrides = list(
        session.scalars(
            select(CandidateOverride)
            .where(CandidateOverride.candidate_id == candidate.id)
            .order_by(CandidateOverride.id)
        )
    )
    wrong = fold_overrides((o.kind, o.target) for o in overrides).wrong
    decisions = session.scalars(
        select(CandidateDecision)
        .where(CandidateDecision.candidate_id == candidate.id)
        .order_by(CandidateDecision.id)
    )
    score = session.get(CandidateScore, candidate.id)
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
                marked_wrong=(row.skill_id, row.source, row.locator) in wrong,
            )
            for row in rows
        ],
        cv_profile=CvProfile.model_validate(cv_profile) if cv_profile else None,
        github_profile=GitHubProfile.model_validate(github_profile) if github_profile else None,
        score=ScoreResult.model_validate(score.result) if score else None,
        overrides=[override_out(o) for o in overrides],
        decisions=[decision_out(d) for d in decisions],
    )


@router.post("/candidates/{candidate_id}/sources/{source}/runs", status_code=201)
def rerun_source(
    candidate_id: uuid.UUID,
    source: EvidenceSource,
    session: SessionDep,
    tenant_id: TenantDep,
    actor: ActorDep,
    registry: RegistryDep,
    queue: QueueDep,
) -> SourceRunOut:
    """Run a source again now, e.g. after a failed run, a new prompt version, or (GitHub) to
    pick up new activity, then queue a new score. A GitHub re-run revalidates cached answers
    instead of refetching."""
    candidate = load_candidate(session, tenant_id, candidate_id)
    if source not in ("cv", "github") or source not in registry:
        raise _unprocessable(f"source {source!r} is not available yet")
    prepared = source_payload(session, candidate, source)
    if prepared is None:
        raise _unprocessable(f"candidate submitted nothing for {source}")
    payload, document_id = prepared
    try:
        run = run_source(
            session, registry, candidate, source, payload, actor=actor, document_id=document_id
        )
    except ConsentMissing as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    settle_source_step(
        session,
        candidate,
        source,
        run,
        registry.input_key(candidate_ref(candidate), source, payload),
    )
    request_rescore(session, queue, [candidate])
    return _run_out(run)


@router.delete("/candidates/{candidate_id}/consents/{source}")
def revoke_consent(
    candidate_id: uuid.UUID,
    source: EvidenceSource,
    session: SessionDep,
    tenant_id: TenantDep,
    actor: ActorDep,
    queue: QueueDep,
) -> CandidateOut:
    """Record that the candidate withdrew consent for a source. Nothing more is collected
    from it, what was derived from it (evidence, profile, run reports, fetched GitHub data) is
    deleted so it cannot be scored or shown, and a new score is queued. The uploaded file
    stays until the candidate's data is deleted (Phase 8.3/8.5)."""
    candidate = load_candidate(session, tenant_id, candidate_id)
    lock_candidate(session, candidate.id)  # a run in progress stores its evidence first, or not
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
    removed = purge_source_data(session, candidate.id, source)
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
            "evidence_removed": removed,
        },
    )
    session.commit()
    request_rescore(session, queue, [candidate])
    return _candidate_out(session, candidate)
