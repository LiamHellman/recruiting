"""The score step: read a candidate's inputs, run the pure scorer, store and audit the result."""

import uuid
from datetime import UTC, date, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.audit.log import record, stable_hash
from app.db.models import Candidate, CandidateOverride, CandidateScore, EvidenceRecord, Job
from app.pipeline.ingest import latest_profile, lock_candidate
from app.schemas.evidence import EvidenceItem
from app.schemas.github import GitHubProfile
from app.schemas.job import JobSpec
from app.scoring.inputs import ActiveOverrides, fold_overrides, prepare_inputs
from app.scoring.policy import POLICY_V1, ScoringPolicy
from app.scoring.scorer import score_candidate
from app.taxonomy.loader import Taxonomy

PIPELINE_ACTOR = "system:pipeline"


def load_overrides(session: Session, candidate_id: uuid.UUID) -> ActiveOverrides:
    rows = session.execute(
        select(CandidateOverride.kind, CandidateOverride.target)
        .where(CandidateOverride.candidate_id == candidate_id)
        .order_by(CandidateOverride.id)
    )
    return fold_overrides((kind, target) for kind, target in rows)


def _stored_evidence(session: Session, candidate_id: uuid.UUID) -> list[EvidenceItem]:
    rows = session.scalars(
        select(EvidenceRecord).where(EvidenceRecord.candidate_id == candidate_id)
    )
    return [
        EvidenceItem(
            skill_id=row.skill_id,
            level=row.level,  # a check constraint guards it
            source=row.source,
            locator=row.locator,
            quote=row.quote,
            date_last_seen=row.date_last_seen,
            ownership=row.ownership,
            confidence=row.confidence,
        )
        for row in rows
    ]


def score_one(
    session: Session,
    candidate: Candidate,
    job: Job,
    taxonomy: Taxonomy,
    as_of: date,
    policy: ScoringPolicy = POLICY_V1,
) -> bool:
    """Score a candidate for a confirmed job and commit. False if the inputs are unchanged.

    The candidate's row is locked while inputs are read and the result written, so the last
    of two concurrent scorings is the one that sees the latest override. Revoking consent
    takes the same lock.
    """
    lock_candidate(session, candidate.id)
    spec = JobSpec.model_validate(job.spec)
    overrides = load_overrides(session, candidate.id)
    profile = latest_profile(session, candidate.id, "github")
    signals = GitHubProfile.model_validate(profile).depth if profile else None
    inputs = prepare_inputs(_stored_evidence(session, candidate.id), signals, overrides, policy)

    spec_hash = stable_hash(job.spec)
    input_hash = stable_hash(
        {
            "version": policy.version,
            "as_of": as_of,
            "spec": spec_hash,
            "evidence": sorted(
                (e.model_dump(mode="json") for e in inputs.evidence),
                key=lambda e: (e["skill_id"], e["source"], e["locator"]),
            ),
            "depth": inputs.depth,
            "levels": overrides.levels,
            # Editing the taxonomy changes scores without changing a candidate's evidence.
            "adjacency": sorted((a, b, sim) for (a, b), sim in taxonomy.adjacency.items()),
            "groups": sorted((s.id, s.parent or "") for s in taxonomy.skills.values()),
        }
    )
    current = session.get(CandidateScore, candidate.id)
    if current is not None and current.input_hash == input_hash:
        session.commit()  # releases the lock
        return False

    result = score_candidate(
        spec,
        inputs.evidence,
        inputs.depth,
        taxonomy,
        as_of,
        overrides=overrides.levels,
        policy=policy,
    )
    if current is None:
        current = CandidateScore(candidate_id=candidate.id, tenant_id=candidate.tenant_id)
        session.add(current)
    current.job_id = job.id
    current.version = result.version
    current.spec_hash = spec_hash
    current.input_hash = input_hash
    current.total = result.total
    current.has_depth_signals = inputs.has_depth_signals
    current.result = result.model_dump(mode="json")
    current.scored_at = datetime.now(UTC)
    record(
        session,
        tenant_id=candidate.tenant_id,
        actor=PIPELINE_ACTOR,
        action="candidate.scored",
        entity_type="candidate",
        entity_id=candidate.id,
        payload={
            "job_id": str(job.id),
            "version": result.version,
            "as_of": as_of.isoformat(),
            "total": result.total,
            "depth": result.depth,
            "spec_hash": spec_hash,
            "input_hash": input_hash,
            "evidence_count": len(inputs.evidence),
            "overridden_skills": len(overrides.levels),
            "needs_human_review": result.needs_human_review,
            "needs_more_evidence": result.needs_more_evidence,
        },
    )
    session.commit()
    return True
