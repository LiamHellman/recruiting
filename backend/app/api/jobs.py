import uuid
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_actor, get_gateway, get_pipeline_queue, get_taxonomy, get_tenant_id
from app.audit.log import record, sha256_hex, stable_hash
from app.db.models import Candidate, Job
from app.db.session import get_session
from app.extraction.posting import parse_posting, spec_problems
from app.llm.gateway import CallContext, Gateway
from app.pipeline.graph import PipelineQueue, request_rescore
from app.schemas.job import (
    CreateJobRequest,
    JobOut,
    JobSpec,
    ParseReport,
    UpdateJobSkillsRequest,
)
from app.taxonomy.loader import Taxonomy

router = APIRouter(prefix="/jobs", tags=["jobs"])

SessionDep = Annotated[Session, Depends(get_session)]
TenantDep = Annotated[uuid.UUID, Depends(get_tenant_id)]
ActorDep = Annotated[str, Depends(get_actor)]
GatewayDep = Annotated[Gateway, Depends(get_gateway)]
TaxonomyDep = Annotated[Taxonomy, Depends(get_taxonomy)]
QueueDep = Annotated[PipelineQueue, Depends(get_pipeline_queue)]


def _to_out(job: Job) -> JobOut:
    spec = JobSpec.model_validate(job.spec)
    return JobOut(
        id=str(job.id),
        status="confirmed" if job.status == "confirmed" else "draft",
        title=job.title,
        spec=spec,
        unmapped=spec.unmapped(),
        parse_report=ParseReport.model_validate(job.parse_report),
        created_at=job.created_at,
        updated_at=job.updated_at,
        confirmed_at=job.confirmed_at,
    )


def _skills_summary(spec: JobSpec) -> dict[str, Any]:
    """Non-personal summary of a spec for the audit log."""

    def compact(skills: list[Any]) -> list[dict[str, Any]]:
        return [
            {"raw_text": s.raw_text, "skill_id": s.skill_id, "weight": s.weight} for s in skills
        ]

    return {
        "must_have": compact(spec.must_have),
        "nice_to_have": compact(spec.nice_to_have),
        "seniority": spec.seniority,
        "domain": spec.domain,
    }


def load_job(session: Session, tenant_id: uuid.UUID, job_id: uuid.UUID) -> Job:
    job = session.get(Job, job_id)
    if job is None or job.tenant_id != tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
    return job


@router.post("", status_code=status.HTTP_201_CREATED)
def create_job(
    body: CreateJobRequest,
    session: SessionDep,
    tenant_id: TenantDep,
    actor: ActorDep,
    gateway: GatewayDep,
    taxonomy: TaxonomyDep,
) -> JobOut:
    """Parse a posting into a draft job. The recruiter must confirm it before scoring."""
    job_id = uuid.uuid4()
    parsed = parse_posting(
        gateway, body.posting_text, CallContext(tenant_id=tenant_id, job_id=job_id), taxonomy
    )
    job = Job(
        id=job_id,
        tenant_id=tenant_id,
        title=body.title or parsed.title,
        posting_text=body.posting_text,
        status="draft",
        spec=parsed.spec.model_dump(mode="json"),
        parse_report=parsed.report.model_dump(mode="json"),
    )
    session.add(job)
    record(
        session,
        tenant_id=tenant_id,
        actor=actor,
        action="job.drafted",
        entity_type="job",
        entity_id=job_id,
        payload={
            "posting_sha256": sha256_hex(body.posting_text),
            "prompt_ref": parsed.report.prompt_ref,
            "model": parsed.report.model,
            "model_served": parsed.report.model_served,
            "llm_cached": parsed.report.cached,
            "spec_hash": stable_hash(job.spec),
            "skills": _skills_summary(parsed.spec),
            "rejected_count": len(parsed.report.rejected),
        },
    )
    session.commit()
    return _to_out(job)


@router.get("/{job_id}")
def get_job(job_id: uuid.UUID, session: SessionDep, tenant_id: TenantDep) -> JobOut:
    return _to_out(load_job(session, tenant_id, job_id))


@router.put("/{job_id}/skills")
def update_job_skills(
    job_id: uuid.UUID,
    body: UpdateJobSkillsRequest,
    session: SessionDep,
    tenant_id: TenantDep,
    actor: ActorDep,
    taxonomy: TaxonomyDep,
    queue: QueueDep,
) -> JobOut:
    """Replace the job's skills. `confirm=true` confirms the job and queues a score for its
    candidates; `false` leaves or returns it to draft, so a confirmed job that is edited must
    be confirmed again."""
    job = load_job(session, tenant_id, job_id)
    spec = body.to_spec()
    problems = spec_problems(spec, job.posting_text, taxonomy, confirm=body.confirm)
    if problems:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, problems)

    before_hash = stable_hash(job.spec)
    job.spec = spec.model_dump(mode="json")
    job.status = "confirmed" if body.confirm else "draft"
    job.confirmed_at = datetime.now(UTC) if body.confirm else None
    record(
        session,
        tenant_id=tenant_id,
        actor=actor,
        action="job.confirmed" if body.confirm else "job.skills_updated",
        entity_type="job",
        entity_id=job.id,
        payload={
            "before_hash": before_hash,
            "after_hash": stable_hash(job.spec),
            "skills": _skills_summary(spec),
            "reason": body.reason,
        },
    )
    session.commit()
    if body.confirm:
        _rescore_all(session, queue, job)
    return _to_out(job)


def _rescore_all(session: Session, queue: PipelineQueue, job: Job) -> None:
    candidates = session.scalars(select(Candidate).where(Candidate.job_id == job.id))
    request_rescore(session, queue, candidates)
