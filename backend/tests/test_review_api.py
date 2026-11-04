import json
import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.config import DEFAULT_TENANT_ID
from app.db.models import (
    AuditLog,
    Candidate,
    CandidateDecision,
    CandidateOverride,
    CandidateScore,
    LLMUsage,
    PipelineStep,
)
from app.llm.fake import FakeGateway
from tests.review_world import (
    WORKED_EXAMPLE_SKILLS,
    apply_amira,
    apply_ben,
    apply_with_cv,
    confirm_job,
    create_job,
    rankings,
    row_for,
    scanning_cv_reply,
)


@pytest.fixture(autouse=True)
def any_cv(gateway: FakeGateway) -> None:
    gateway.backend.replies["cv_extractor"] = scanning_cv_reply


@pytest.fixture
def job_id(client: TestClient) -> str:
    job = create_job(client)
    confirm_job(client, job)
    return job


def override(client: TestClient, candidate: dict[str, Any], **body: Any) -> dict[str, Any]:
    body.setdefault("reason", "checked in a technical screen")
    response = client.post(f"/candidates/{candidate['id']}/overrides", json=body)
    assert response.status_code == 201, response.text
    data: dict[str, Any] = response.json()
    return data


def score_of(client: TestClient, candidate: dict[str, Any]) -> dict[str, Any]:
    score: dict[str, Any] = client.get(f"/candidates/{candidate['id']}/evidence").json()["score"]
    return score


def skill_score(client: TestClient, candidate: dict[str, Any], skill_id: str) -> dict[str, Any]:
    return next(s for s in score_of(client, candidate)["skills"] if s["skill_id"] == skill_id)


def audit_entries(sessions: sessionmaker[Session], action: str) -> list[AuditLog]:
    with sessions() as session:
        return list(session.scalars(select(AuditLog).where(AuditLog.action == action)))


# --- Rankings ------------------------------------------------------------------------------


def test_rankings_need_a_confirmed_job(client: TestClient):
    draft = create_job(client)
    assert client.get(f"/jobs/{draft}/rankings").status_code == 409


def test_rankings_of_an_unknown_job_404(client: TestClient):
    assert client.get(f"/jobs/{uuid.uuid4()}/rankings").status_code == 404


def test_rankings_order_by_score_and_equal_totals_share_a_rank(client: TestClient, job_id: str):
    amira = apply_amira(client, job_id)
    first = apply_with_cv(client, job_id, "03-carmen-rojas", "Carmen Rojas")
    second = apply_with_cv(client, job_id, "03-carmen-rojas", "Carmen Rojas Bis")
    rows = rankings(client, job_id)
    assert rows[0]["candidate_id"] == amira["id"]
    assert [r["rank"] for r in rows] == [1, 2, 2]
    # Application order breaks the tie.
    assert [r["candidate_id"] for r in rows[1:]] == [first["id"], second["id"]]


def test_a_candidate_with_a_missing_must_have_stays_in_the_list_with_a_label(
    client: TestClient, job_id: str
):
    apply_amira(client, job_id)
    carmen = apply_with_cv(client, job_id, "03-carmen-rojas", "Carmen Rojas")
    row = row_for(rankings(client, job_id), carmen)
    assert row["rank"] == 2
    assert row["needs_human_review"] is True
    assert "missing evidence: needs human review" in row["labels"]
    assert "needs more evidence" in row["labels"]
    assert "skill:python" in row["coverage"]["missing"]


def test_low_visibility_marks_candidates_without_public_code(client: TestClient, job_id: str):
    amira = apply_amira(client, job_id)
    carmen = apply_with_cv(client, job_id, "03-carmen-rojas", "Carmen Rojas")
    rows = rankings(client, job_id)
    assert row_for(rows, amira)["low_visibility"] is False
    assert row_for(rows, carmen)["low_visibility"] is True
    assert row_for(rows, carmen)["depth"] == 0


def test_scores_trace_back_to_evidence_locators(client: TestClient, job_id: str):
    amira = apply_amira(client, job_id)
    python = skill_score(client, amira, "skill:python")
    assert python["credit_kind"] == "evidence"
    assert python["best_item"]["level"] == "deep"
    assert python["best_item"]["locator"].startswith("pr:tablekit-org/django-tablekit#")
    fastapi = skill_score(client, amira, "skill:fastapi")
    assert (fastapi["credit_kind"], fastapi["credit_skill"]) == ("adjacent", "skill:django")
    assert fastapi["best_item"]["locator"]


def test_identity_needing_confirmation_is_flagged_but_counts(client: TestClient, job_id: str):
    ben = apply_ben(client, job_id)  # nothing links this account to the CV
    row = row_for(rankings(client, job_id), ben)
    assert "possible match: confirm?" in row["labels"]
    assert row["depth"] > 0


# --- Level overrides -----------------------------------------------------------------------


def test_level_override_rescores_and_is_shown(client: TestClient, job_id: str):
    ben = apply_ben(client, job_id)
    before = row_for(rankings(client, job_id), ben)["total"]
    override(client, ben, kind="level", skill_id="skill:postgresql", level="deep")
    row = row_for(rankings(client, job_id), ben)
    assert row["total"] == pytest.approx(before + 60 * (1.0 - 0.4) / 3, abs=0.01)
    assert row["coverage"]["demonstrated"] == [
        "skill:python",
        "skill:fastapi",
        "skill:postgresql",
    ]
    postgres = skill_score(client, ben, "skill:postgresql")
    assert (postgres["credit_kind"], postgres["override_level"]) == ("override", "deep")


def test_override_to_none_raises_the_gate_and_removing_it_restores(client: TestClient, job_id: str):
    ben = apply_ben(client, job_id)
    before = row_for(rankings(client, job_id), ben)
    override(client, ben, kind="level", skill_id="skill:python", level="none")
    lowered = row_for(rankings(client, job_id), ben)
    assert lowered["needs_human_review"] is True
    assert lowered["total"] < before["total"]
    override(client, ben, kind="level", skill_id="skill:python", level=None)
    assert row_for(rankings(client, job_id), ben)["total"] == before["total"]


@pytest.mark.parametrize(
    ("body", "code"),
    [
        ({"kind": "level", "skill_id": "skill:python", "level": "deep"}, 422),  # no reason
        ({"kind": "level", "skill_id": "skill:python", "level": "deep", "reason": "   "}, 422),
        ({"kind": "level", "skill_id": "skill:nope", "level": "deep", "reason": "x"}, 422),
        ({"kind": "level", "skill_id": "skill:python", "level": "expert", "reason": "x"}, 422),
        ({"kind": "laser", "reason": "x"}, 422),
        ({"kind": "evidence", "evidence_id": str(uuid.uuid4()), "reason": "x"}, 404),
    ],
)
def test_bad_overrides_are_refused(
    client: TestClient, job_id: str, body: dict[str, Any], code: int
):
    ben = apply_ben(client, job_id)
    assert client.post(f"/candidates/{ben['id']}/overrides", json=body).status_code == code


def test_override_for_an_unknown_candidate_404(client: TestClient):
    body = {"kind": "note", "reason": "x"}
    assert client.post(f"/candidates/{uuid.uuid4()}/overrides", json=body).status_code == 404


# --- Marking evidence wrong ----------------------------------------------------------------


def _evidence(client: TestClient, candidate: dict[str, Any], skill_id: str) -> list[dict[str, Any]]:
    rows = client.get(f"/candidates/{candidate['id']}/evidence").json()["evidence"]
    return [e for e in rows if e["skill_id"] == skill_id]


def test_marking_evidence_wrong_stops_it_counting_and_can_be_undone(
    client: TestClient, job_id: str
):
    ben = apply_ben(client, job_id)
    before = row_for(rankings(client, job_id), ben)["total"]
    [claim] = _evidence(client, ben, "skill:aws")
    override(client, ben, kind="evidence", evidence_id=claim["id"])
    assert _evidence(client, ben, "skill:aws")[0]["marked_wrong"] is True  # kept, for the record
    assert row_for(rankings(client, job_id), ben)["total"] == pytest.approx(
        before - 25 * 0.4 / 3, abs=0.01
    )
    override(client, ben, kind="evidence", evidence_id=claim["id"], wrong=False)
    assert row_for(rankings(client, job_id), ben)["total"] == before


def test_a_mark_survives_a_source_rerun(client: TestClient, job_id: str):
    ben = apply_ben(client, job_id)
    [claim] = _evidence(client, ben, "skill:aws")
    override(client, ben, kind="evidence", evidence_id=claim["id"])
    marked = row_for(rankings(client, job_id), ben)["total"]
    assert client.post(f"/candidates/{ben['id']}/sources/cv/runs").status_code == 201
    [after] = _evidence(client, ben, "skill:aws")
    assert after["id"] != claim["id"]  # the row was replaced, the mark still applies
    assert after["marked_wrong"] is True
    assert row_for(rankings(client, job_id), ben)["total"] == marked


def test_other_candidates_evidence_cannot_be_marked(client: TestClient, job_id: str):
    amira, ben = apply_amira(client, job_id), apply_ben(client, job_id)
    [claim] = _evidence(client, ben, "skill:aws")
    body = {"kind": "evidence", "evidence_id": claim["id"], "reason": "x"}
    assert client.post(f"/candidates/{amira['id']}/overrides", json=body).status_code == 404


# --- Notes ---------------------------------------------------------------------------------


def test_a_note_is_kept_and_does_not_rescore(
    client: TestClient, job_id: str, sessions: sessionmaker[Session]
):
    ben = apply_ben(client, job_id)
    scored = len(audit_entries(sessions, "candidate.scored"))
    override(client, ben, kind="note", reason="Prefers remote; will discuss the role")
    body = client.get(f"/candidates/{ben['id']}/evidence").json()
    assert [o["reason"] for o in body["overrides"]] == ["Prefers remote; will discuss the role"]
    assert len(audit_entries(sessions, "candidate.scored")) == scored


# --- Identity ------------------------------------------------------------------------------


def test_rejecting_a_github_account_removes_its_evidence_and_depth(client: TestClient, job_id: str):
    ben = apply_ben(client, job_id)
    before = row_for(rankings(client, job_id), ben)
    override(client, ben, kind="identity", decision="rejected", reason="not the same person")
    row = row_for(rankings(client, job_id), ben)
    assert row["depth"] == 0
    assert row["low_visibility"] is True
    assert row["total"] < before["total"]
    assert "GitHub account rejected by recruiter: not counted" in row["labels"]
    assert "possible match: confirm?" not in row["labels"]
    # Only claims from the CV remain.
    assert skill_score(client, ben, "skill:python")["strength"] == 0.4


def test_confirming_a_github_account_clears_the_flag_and_keeps_the_score(
    client: TestClient, job_id: str
):
    ben = apply_ben(client, job_id)
    before = row_for(rankings(client, job_id), ben)
    override(client, ben, kind="identity", decision="confirmed", reason="asked the candidate")
    row = row_for(rankings(client, job_id), ben)
    assert row["total"] == before["total"]
    assert "possible match: confirm?" not in row["labels"]


def test_identity_needs_a_github_profile(client: TestClient, job_id: str):
    carmen = apply_with_cv(client, job_id, "03-carmen-rojas", "Carmen Rojas")
    body = {"kind": "identity", "decision": "confirmed", "reason": "x"}
    assert client.post(f"/candidates/{carmen['id']}/overrides", json=body).status_code == 409


# --- Decisions -----------------------------------------------------------------------------


def test_a_decision_is_recorded_with_the_score_it_was_made_against(client: TestClient, job_id: str):
    ben = apply_ben(client, job_id)
    total = row_for(rankings(client, job_id), ben)["total"]
    response = client.post(
        f"/candidates/{ben['id']}/decisions", json={"decision": "advance", "reason": "strong"}
    )
    assert response.status_code == 201, response.text
    assert response.json()["score_total"] == total
    assert row_for(rankings(client, job_id), ben)["decision"]["decision"] == "advance"


def test_the_latest_decision_is_current_and_history_is_kept(client: TestClient, job_id: str):
    ben = apply_ben(client, job_id)
    for decision in ("hold", "reject"):
        client.post(
            f"/candidates/{ben['id']}/decisions", json={"decision": decision, "reason": "because"}
        )
    assert row_for(rankings(client, job_id), ben)["decision"]["decision"] == "reject"
    history = client.get(f"/candidates/{ben['id']}/evidence").json()["decisions"]
    assert [d["decision"] for d in history] == ["hold", "reject"]


def test_a_rejected_candidate_stays_in_the_list(client: TestClient, job_id: str):
    ben = apply_ben(client, job_id)
    client.post(f"/candidates/{ben['id']}/decisions", json={"decision": "reject", "reason": "no"})
    assert row_for(rankings(client, job_id), ben)["rank"] == 1


@pytest.mark.parametrize(
    "body",
    [
        {"decision": "advance"},
        {"decision": "advance", "reason": "  "},
        {"decision": "auto-reject", "reason": "x"},
    ],
)
def test_bad_decisions_are_refused(client: TestClient, job_id: str, body: dict[str, Any]):
    ben = apply_ben(client, job_id)
    assert client.post(f"/candidates/{ben['id']}/decisions", json=body).status_code == 422


def test_no_decision_on_a_draft_job(client: TestClient):
    draft = create_job(client)
    ben = apply_ben(client, draft)
    body = {"decision": "advance", "reason": "x"}
    assert client.post(f"/candidates/{ben['id']}/decisions", json=body).status_code == 409


def test_deleting_a_candidate_deletes_review_data_but_not_the_audit_log(
    client: TestClient, job_id: str, sessions: sessionmaker[Session]
):
    ben = apply_ben(client, job_id)
    identifier = uuid.UUID(ben["id"])
    override(client, ben, kind="note", reason="a note")
    client.post(f"/candidates/{ben['id']}/decisions", json={"decision": "hold", "reason": "why"})
    with sessions.begin() as session:
        session.add(
            LLMUsage(
                tenant_id=DEFAULT_TENANT_ID,
                candidate_id=identifier,
                prompt_ref="cv_extractor/v1",
                model_requested="m",
                cached=False,
                input_tokens=1,
                output_tokens=1,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
                latency_ms=1,
            )
        )
    audited = len(audit_entries(sessions, "decision.recorded"))
    with sessions.begin() as session:
        candidate = session.get(Candidate, identifier)
        assert candidate is not None
        session.delete(candidate)
    with sessions() as session:
        for table in (CandidateScore, CandidateOverride, CandidateDecision, PipelineStep):
            assert not session.scalars(select(table)).all(), table.__name__
        # What was spent stays on the books; whom it was spent on does not.
        assert [u.candidate_id for u in session.scalars(select(LLMUsage))] == [None]
    assert len(audit_entries(sessions, "decision.recorded")) == audited


def test_nothing_decides_without_a_recruiter(client: TestClient, job_id: str):
    apply_ben(client, job_id)
    assert rankings(client, job_id)[0]["decision"] is None


# --- Audit log -----------------------------------------------------------------------------


def test_audit_log_holds_hashes_of_reasons_and_targets(
    client: TestClient, job_id: str, sessions: sessionmaker[Session]
):
    ben = apply_ben(client, job_id)
    secret = "Talked to their former manager at Lumen"
    [claim] = _evidence(client, ben, "skill:aws")
    override(client, ben, kind="level", skill_id="skill:postgresql", level="deep", reason=secret)
    override(client, ben, kind="evidence", evidence_id=claim["id"], reason=secret)
    override(client, ben, kind="note", reason=secret)
    override(client, ben, kind="identity", decision="confirmed", reason=secret)
    client.post(f"/candidates/{ben['id']}/decisions", json={"decision": "hold", "reason": secret})

    added = audit_entries(sessions, "override.added")
    assert [e.payload["kind"] for e in added] == ["level", "evidence", "note", "identity"]
    [decided] = audit_entries(sessions, "decision.recorded")
    assert decided.payload["decision"] == "hold"
    assert decided.payload["score_input_hash"]
    scored = audit_entries(sessions, "candidate.scored")
    assert scored and all(e.payload["total"] is not None for e in scored)

    payloads = json.dumps([e.payload for e in [*added, decided, *scored]]).casefold()
    for leaked in ("lumen", "manager", "postgresql", "aws", "ben", "okafor", "example.com"):
        assert leaked not in payloads


def test_every_score_change_is_audited_with_its_inputs(
    client: TestClient, job_id: str, sessions: sessionmaker[Session]
):
    ben = apply_ben(client, job_id)
    override(client, ben, kind="level", skill_id="skill:redis", level="deep")
    scored = audit_entries(sessions, "candidate.scored")
    assert len(scored) == 2
    assert scored[0].payload["input_hash"] != scored[1].payload["input_hash"]
    assert scored[0].payload["version"] == "v1"
    assert scored[1].payload["overridden_skills"] == 1


# --- What triggers a re-score --------------------------------------------------------------


def test_revoking_consent_rescores_and_removes_overrides_about_that_source(
    client: TestClient, job_id: str, sessions: sessionmaker[Session]
):
    amira = apply_amira(client, job_id)
    before = row_for(rankings(client, job_id), amira)
    [repo_item, *_] = [
        e for e in _evidence(client, amira, "skill:docker") if e["source"] == "github"
    ]
    override(client, amira, kind="evidence", evidence_id=repo_item["id"])
    override(client, amira, kind="identity", decision="confirmed", reason="asked")
    override(client, amira, kind="level", skill_id="skill:redis", level="demonstrated")

    assert client.delete(f"/candidates/{amira['id']}/consents/github").status_code == 200
    row = row_for(rankings(client, job_id), amira)
    assert row["total"] < before["total"]
    assert row["depth"] == 0
    assert row["low_visibility"] is True
    with sessions() as session:
        kinds = session.scalars(select(CandidateOverride.kind)).all()
    assert kinds == ["level"]  # not derived from the source, so it stays


def test_a_source_rerun_requests_a_new_score(
    client: TestClient, job_id: str, sessions: sessionmaker[Session]
):
    ben = apply_ben(client, job_id)
    with sessions.begin() as session:
        row = session.get(CandidateScore, uuid.UUID(ben["id"]))
        assert row is not None
        row.input_hash = "out-of-date"
    assert client.post(f"/candidates/{ben['id']}/sources/cv/runs").status_code == 201
    with sessions() as session:
        row = session.get(CandidateScore, uuid.UUID(ben["id"]))
        assert row is not None
        assert row.input_hash != "out-of-date"


def test_editing_weights_rescores_everyone_after_confirmation(client: TestClient, job_id: str):
    amira, ben = apply_amira(client, job_id), apply_ben(client, job_id)
    before = {r["candidate_id"]: r["total"] for r in rankings(client, job_id)}
    weighted = json.loads(json.dumps(WORKED_EXAMPLE_SKILLS))
    weighted["must_have"][1]["weight"] = 5.0  # FastAPI matters much more now
    confirm_job(client, job_id, must_have=weighted["must_have"])
    after = {r["candidate_id"]: r["total"] for r in rankings(client, job_id)}
    assert after[amira["id"]] != before[amira["id"]]
    assert after[ben["id"]] != before[ben["id"]]


def test_a_score_for_an_earlier_spec_is_not_shown(
    client: TestClient, job_id: str, sessions: sessionmaker[Session]
):
    ben = apply_ben(client, job_id)
    with sessions.begin() as session:
        row = session.get(CandidateScore, uuid.UUID(ben["id"]))
        assert row is not None
        row.spec_hash = "an-earlier-spec"
    row_out = row_for(rankings(client, job_id), ben)
    assert (row_out["status"], row_out["total"], row_out["rank"]) == ("pending", None, None)


def test_rescore_endpoint_needs_a_confirmed_job_and_makes_no_llm_calls(
    client: TestClient, job_id: str, gateway: FakeGateway
):
    apply_ben(client, job_id)
    calls = len(gateway.calls)
    assert client.post(f"/jobs/{job_id}/rescore").status_code == 202
    assert len(gateway.calls) == calls
    draft = create_job(client)
    assert client.post(f"/jobs/{draft}/rescore").status_code == 409
