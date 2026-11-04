import uuid
from collections import defaultdict
from decimal import Decimal
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.candidates import decision_out, load_candidate, override_out, step_out
from app.api.deps import get_actor, get_pipeline_queue, get_taxonomy, get_tenant_id
from app.api.jobs import load_job
from app.audit.log import record, sha256_hex, stable_hash
from app.db.models import (
    Candidate,
    CandidateDecision,
    CandidateOverride,
    CandidateScore,
    EvidenceRecord,
    Job,
    LLMUsage,
    PipelineStep,
    SourceRun,
)
from app.db.session import get_session
from app.pipeline.graph import (
    SOURCE_STEPS,
    PipelineQueue,
    request_rescore,
    resume_pipeline,
)
from app.pipeline.ingest import latest_profile
from app.schemas.github import GitHubProfile
from app.schemas.review import (
    DeadLetterOut,
    DecisionOut,
    DecisionRequest,
    EvidenceOverrideRequest,
    IdentityOverrideRequest,
    LevelOverrideRequest,
    NoteRequest,
    OverrideOut,
    OverrideRequest,
    PipelineStepOut,
    RankedCandidate,
    RankingsOut,
)
from app.schemas.score import ScoreResult
from app.scoring.inputs import ActiveOverrides, fold_overrides
from app.taxonomy.loader import Taxonomy

router = APIRouter(tags=["review"])

SessionDep = Annotated[Session, Depends(get_session)]
TenantDep = Annotated[uuid.UUID, Depends(get_tenant_id)]
ActorDep = Annotated[str, Depends(get_actor)]
QueueDep = Annotated[PipelineQueue, Depends(get_pipeline_queue)]
TaxonomyDep = Annotated[Taxonomy, Depends(get_taxonomy)]

IDENTITY_REJECTED_LABEL = "GitHub account rejected by recruiter: not counted"


def _require_confirmed(job: Job) -> None:
    if job.status != "confirmed":
        raise HTTPException(
            status.HTTP_409_CONFLICT, "confirm the job's skills before scoring or deciding"
        )


# --- Rankings -----------------------------------------------------------------------------


def _identity_label(profile: dict[str, Any] | None, overrides: ActiveOverrides) -> str | None:
    if profile is None or overrides.identity == "confirmed":
        return None
    if overrides.identity == "rejected":
        return IDENTITY_REJECTED_LABEL
    return GitHubProfile.model_validate(profile).identity.label


def _rank(rows: list[RankedCandidate]) -> list[RankedCandidate]:
    """Highest total first; equal totals share a rank. Unscored candidates stay in the list,
    after the scored ones. `rows` arrive in application order, which breaks ties."""
    scored = sorted((r for r in rows if r.total is not None), key=lambda r: -(r.total or 0))
    previous: float | None = None
    rank = 0
    for position, row in enumerate(scored, start=1):
        if row.total != previous:
            rank, previous = position, row.total
        row.rank = rank
    return [*scored, *(r for r in rows if r.total is None)]


@router.get("/jobs/{job_id}/rankings")
def get_rankings(job_id: uuid.UUID, session: SessionDep, tenant_id: TenantDep) -> RankingsOut:
    """Every candidate for a confirmed job, ranked by score, with coverage and flags. The
    list informs a recruiter's decision; it decides nothing, and nobody is filtered out."""
    job = load_job(session, tenant_id, job_id)
    _require_confirmed(job)
    spec_hash = stable_hash(job.spec)
    candidates = list(
        session.scalars(
            select(Candidate)
            .where(Candidate.job_id == job.id)
            .order_by(Candidate.created_at, Candidate.id)
        )
    )
    in_job = select(Candidate.id).where(Candidate.job_id == job.id)
    scores = {
        s.candidate_id: s
        for s in session.scalars(
            select(CandidateScore).where(CandidateScore.candidate_id.in_(in_job))
        )
    }
    steps: dict[uuid.UUID, dict[str, PipelineStep]] = defaultdict(dict)
    for step in session.scalars(select(PipelineStep).where(PipelineStep.candidate_id.in_(in_job))):
        steps[step.candidate_id][step.step] = step
    decisions: dict[uuid.UUID, CandidateDecision] = {}
    for row in session.scalars(
        select(CandidateDecision)
        .where(CandidateDecision.candidate_id.in_(in_job))
        .order_by(CandidateDecision.id)
    ):
        decisions[row.candidate_id] = row
    events: dict[uuid.UUID, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for override in session.scalars(
        select(CandidateOverride)
        .where(CandidateOverride.candidate_id.in_(in_job))
        .order_by(CandidateOverride.id)
    ):
        events[override.candidate_id].append((override.kind, override.target))
    profiles: dict[uuid.UUID, dict[str, Any] | None] = {}
    for run in session.scalars(
        select(SourceRun)
        .where(
            SourceRun.candidate_id.in_(in_job),
            SourceRun.source == "github",
            SourceRun.status == "succeeded",
        )
        .order_by(SourceRun.created_at)
    ):
        profiles[run.candidate_id] = run.profile
    costs = {
        candidate_id: Decimal(cost or 0)
        for candidate_id, cost in session.execute(
            select(LLMUsage.candidate_id, func.sum(LLMUsage.cost_usd))
            .where(LLMUsage.candidate_id.in_(in_job))
            .group_by(LLMUsage.candidate_id)
        )
    }

    rows: list[RankedCandidate] = []
    for candidate in candidates:
        own_steps = steps.get(candidate.id, {})
        score = scores.get(candidate.id)
        if score is not None and score.spec_hash != spec_hash:
            score = None  # scored against an earlier version of the skills
        result = ScoreResult.model_validate(score.result) if score else None
        score_step = own_steps.get("score")
        labels = list(result.labels) if result else []
        identity = _identity_label(
            profiles.get(candidate.id), fold_overrides(events.get(candidate.id, []))
        )
        if identity:
            labels.append(identity)
        labels.extend(
            f"{name} could not be processed: score may be incomplete"
            for name in SOURCE_STEPS
            if name in own_steps and own_steps[name].status in ("failed", "dead_letter")
        )
        decision = decisions.get(candidate.id)
        rows.append(
            RankedCandidate(
                candidate_id=str(candidate.id),
                full_name=candidate.full_name,
                rank=None,
                status=(
                    "scored"
                    if result
                    else "failed"
                    if score_step is not None and score_step.status == "dead_letter"
                    else "pending"
                ),
                stale=bool(
                    result
                    and score_step is not None
                    and score_step.status in ("pending", "running", "retrying")
                ),
                total=result.total if result else None,
                max_total=result.max_total if result else None,
                must_have_avg=result.must_have_avg if result else None,
                nice_to_have_avg=result.nice_to_have_avg if result else None,
                depth=result.depth if result else None,
                coverage=result.coverage if result else None,
                needs_human_review=bool(result and result.needs_human_review),
                needs_more_evidence=bool(result and result.needs_more_evidence),
                low_visibility=bool(score and not score.has_depth_signals),
                labels=labels,
                as_of=result.as_of if result else None,
                decision=decision_out(decision) if decision else None,
                llm_cost_usd=costs.get(candidate.id, Decimal(0)),
            )
        )
    return RankingsOut(job_id=str(job.id), candidates=_rank(rows))


@router.post("/jobs/{job_id}/rescore", status_code=status.HTTP_202_ACCEPTED)
def rescore_job(
    job_id: uuid.UUID, session: SessionDep, tenant_id: TenantDep, queue: QueueDep
) -> dict[str, str]:
    """Score everyone again as of today, so that recency is comparable across the list.
    Makes no LLM calls."""
    job = load_job(session, tenant_id, job_id)
    _require_confirmed(job)
    candidates = session.scalars(select(Candidate).where(Candidate.job_id == job.id))
    request_rescore(session, queue, candidates)
    return {"status": "queued"}


# --- Overrides and decisions --------------------------------------------------------------


def _override_target(
    session: Session, candidate: Candidate, body: OverrideRequest, taxonomy: Taxonomy
) -> dict[str, Any]:
    if isinstance(body, LevelOverrideRequest):
        if body.skill_id not in taxonomy:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT, f"unknown skill id {body.skill_id}"
            )
        return {"skill_id": body.skill_id, "level": body.level}
    if isinstance(body, EvidenceOverrideRequest):
        item = session.get(EvidenceRecord, body.evidence_id)
        if item is None or item.candidate_id != candidate.id:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "evidence item not found")
        return {
            "skill_id": item.skill_id,
            "source": item.source,
            "locator": item.locator,
            "wrong": body.wrong,
        }
    if isinstance(body, IdentityOverrideRequest):
        if latest_profile(session, candidate.id, "github") is None:
            raise HTTPException(status.HTTP_409_CONFLICT, "candidate has no GitHub profile")
        return {"decision": body.decision}
    assert isinstance(body, NoteRequest)
    return {}


@router.post("/candidates/{candidate_id}/overrides", status_code=status.HTTP_201_CREATED)
def add_override(
    candidate_id: uuid.UUID,
    body: OverrideRequest,
    session: SessionDep,
    tenant_id: TenantDep,
    actor: ActorDep,
    queue: QueueDep,
    taxonomy: TaxonomyDep,
) -> OverrideOut:
    """Record a recruiter's correction and re-score. A reason is required."""
    candidate = load_candidate(session, tenant_id, candidate_id)
    target = _override_target(session, candidate, body, taxonomy)
    override = CandidateOverride(
        tenant_id=tenant_id,
        candidate_id=candidate.id,
        kind=body.kind,
        target=target,
        reason=body.reason,
        actor=actor,
    )
    session.add(override)
    session.flush()
    record(
        session,
        tenant_id=tenant_id,
        actor=actor,
        action="override.added",
        entity_type="candidate",
        entity_id=candidate.id,
        payload={
            "override_id": override.id,
            "kind": body.kind,
            # Hashes: a candidate's skills and a recruiter's words about them are personal data.
            "target_sha256": stable_hash(target),
            "reason_sha256": sha256_hex(body.reason),
        },
    )
    session.commit()
    if body.kind != "note":
        request_rescore(session, queue, [candidate])
    return override_out(override)


@router.post("/candidates/{candidate_id}/decisions", status_code=status.HTTP_201_CREATED)
def add_decision(
    candidate_id: uuid.UUID,
    body: DecisionRequest,
    session: SessionDep,
    tenant_id: TenantDep,
    actor: ActorDep,
) -> DecisionOut:
    """Record a recruiter's decision, with the score it was made against. Only a person can
    do this; nothing in the system advances, holds or rejects a candidate by itself."""
    candidate = load_candidate(session, tenant_id, candidate_id)
    _require_confirmed(load_job(session, tenant_id, candidate.job_id))
    score = session.get(CandidateScore, candidate.id)
    decision = CandidateDecision(
        tenant_id=tenant_id,
        candidate_id=candidate.id,
        decision=body.decision,
        reason=body.reason,
        actor=actor,
        score_total=score.total if score else None,
        score_input_hash=score.input_hash if score else None,
    )
    session.add(decision)
    session.flush()
    record(
        session,
        tenant_id=tenant_id,
        actor=actor,
        action="decision.recorded",
        entity_type="candidate",
        entity_id=candidate.id,
        payload={
            "decision_id": decision.id,
            "decision": body.decision,
            "job_id": str(candidate.job_id),
            "reason_sha256": sha256_hex(body.reason),
            "score_total": decision.score_total,
            "score_input_hash": decision.score_input_hash,
            "score_version": score.version if score else None,
        },
    )
    session.commit()
    return decision_out(decision)


# --- Pipeline control ---------------------------------------------------------------------


@router.post("/candidates/{candidate_id}/pipeline", status_code=status.HTTP_202_ACCEPTED)
def resume_candidate_pipeline(
    candidate_id: uuid.UUID, session: SessionDep, tenant_id: TenantDep, queue: QueueDep
) -> list[PipelineStepOut]:
    """Queue a candidate's pipeline again, e.g. after a dead letter or a queue outage. Steps
    that already succeeded on the same input don't run again."""
    candidate = load_candidate(session, tenant_id, candidate_id)
    if not resume_pipeline(session, queue, candidate):
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "the queue is unavailable")
    session.expire_all()
    rows = session.scalars(select(PipelineStep).where(PipelineStep.candidate_id == candidate.id))
    return [step_out(row) for row in rows]


@router.get("/pipeline/dead-letters")
def list_dead_letters(session: SessionDep, tenant_id: TenantDep) -> list[DeadLetterOut]:
    """Steps that kept failing, for an operator to look into and re-queue."""
    rows = session.execute(
        select(PipelineStep, Candidate.job_id)
        .join(Candidate, Candidate.id == PipelineStep.candidate_id)
        .where(PipelineStep.tenant_id == tenant_id, PipelineStep.status == "dead_letter")
        .order_by(PipelineStep.updated_at)
    )
    return [
        DeadLetterOut(
            **step_out(step).model_dump(), candidate_id=str(step.candidate_id), job_id=str(job_id)
        )
        for step, job_id in rows
    ]
