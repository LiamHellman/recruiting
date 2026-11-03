import uuid
from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import AuditLog, Job
from app.llm.fake import FakeGateway
from tests.conftest import POSTING


def audit(sessions: sessionmaker[Session]) -> list[AuditLog]:
    with sessions() as session:
        return list(session.scalars(select(AuditLog).order_by(AuditLog.id)))


def create(client: TestClient) -> dict[str, Any]:
    response = client.post("/jobs", json={"posting_text": POSTING})
    assert response.status_code == 201, response.text
    data: dict[str, Any] = response.json()
    return data


def skills_body(job: dict[str, Any], **changes: Any) -> dict[str, Any]:
    body = {
        "must_have": job["spec"]["must_have"],
        "nice_to_have": job["spec"]["nice_to_have"],
        "seniority": job["spec"]["seniority"],
        "domain": job["spec"]["domain"],
    }
    body.update(changes)
    return body


def test_health(client: TestClient):
    assert client.get("/health").json() == {"status": "ok"}


def test_create_job_returns_a_draft_with_unmapped_terms_visible(client: TestClient):
    job = create(client)
    assert job["status"] == "draft"
    assert job["title"] == "Backend Engineer (Python)"
    assert [s["skill_id"] for s in job["spec"]["must_have"]] == [
        "skill:python",
        "skill:fastapi",
        "skill:postgresql",
        None,
    ]
    assert job["unmapped"] == ["Temporal"]
    assert job["parse_report"]["prompt_ref"] == "posting_parser/v1"
    assert [r["skill"] for r in job["parse_report"]["rejected"]] == ["Kubernetes"]


def test_create_job_is_audit_logged_without_the_posting_text(
    client: TestClient, sessions: sessionmaker[Session]
):
    job = create(client)
    [entry] = audit(sessions)
    assert entry.action == "job.drafted"
    assert entry.entity_id == job["id"]
    assert entry.payload["prompt_ref"] == "posting_parser/v1"
    assert entry.payload["model"] == "claude-opus-5-5"
    assert len(entry.payload["posting_sha256"]) == 64
    assert "Northwind" not in str(entry.payload)


def test_title_from_the_request_wins(client: TestClient):
    response = client.post("/jobs", json={"posting_text": POSTING, "title": "Platform Eng"})
    assert response.json()["title"] == "Platform Eng"


def test_reposting_the_same_text_reuses_the_cached_parse(client: TestClient, gateway: FakeGateway):
    first = create(client)
    second = create(client)
    assert len(gateway.calls) == 1
    assert second["parse_report"]["cached"]
    assert second["spec"] == first["spec"]
    assert second["id"] != first["id"]


def test_get_job(client: TestClient):
    job = create(client)
    assert client.get(f"/jobs/{job['id']}").json() == job


def test_unknown_job_is_404(client: TestClient):
    assert client.get("/jobs/00000000-0000-0000-0000-00000000dead").status_code == 404


def test_cannot_confirm_with_unmapped_skills(client: TestClient):
    job = create(client)
    response = client.put(f"/jobs/{job['id']}/skills", json=skills_body(job, confirm=True))
    assert response.status_code == 422
    assert "unmapped" in str(response.json()["detail"])
    assert client.get(f"/jobs/{job['id']}").json()["status"] == "draft"


def test_recruiter_maps_edits_and_confirms(client: TestClient, sessions: sessionmaker[Session]):
    job = create(client)
    must = [s for s in job["spec"]["must_have"] if s["skill_id"] is not None]
    must[0]["weight"] = 2.0  # Python matters more for this role
    nice = [*job["spec"]["nice_to_have"], {"raw_text": "Kafka", "skill_id": "skill:kafka"}]
    body = skills_body(job, must_have=must, nice_to_have=nice, confirm=True, reason="Drop Temporal")

    response = client.put(f"/jobs/{job['id']}/skills", json=body)

    assert response.status_code == 200, response.text
    confirmed = response.json()
    assert confirmed["status"] == "confirmed"
    assert confirmed["confirmed_at"] is not None
    assert confirmed["unmapped"] == []
    assert confirmed["spec"]["must_have"][0]["weight"] == 2.0
    entry = audit(sessions)[-1]
    assert entry.action == "job.confirmed"
    assert entry.payload["reason"] == "Drop Temporal"
    assert entry.payload["before_hash"] != entry.payload["after_hash"]
    assert {"raw_text": "Kafka", "skill_id": "skill:kafka", "weight": 1.0} in entry.payload[
        "skills"
    ]["nice_to_have"]


def test_editing_a_confirmed_job_returns_it_to_draft(
    client: TestClient, sessions: sessionmaker[Session]
):
    job = create(client)
    must = [s for s in job["spec"]["must_have"] if s["skill_id"] is not None]
    client.put(f"/jobs/{job['id']}/skills", json=skills_body(job, must_have=must, confirm=True))

    response = client.put(f"/jobs/{job['id']}/skills", json=skills_body(job, must_have=must[:1]))

    assert response.json()["status"] == "draft"
    assert response.json()["confirmed_at"] is None
    assert [e.action for e in audit(sessions)] == [
        "job.drafted",
        "job.confirmed",
        "job.skills_updated",
    ]


def test_invalid_edits_are_rejected_and_not_logged(
    client: TestClient, sessions: sessionmaker[Session]
):
    job = create(client)
    bad_id = skills_body(job, must_have=[{"raw_text": "Snake", "skill_id": "skill:snake"}])
    forged = skills_body(
        job,
        must_have=[
            {
                "raw_text": "Python",
                "skill_id": "skill:python",
                "source_quote": "Python guru",
                "source_span": [0, 11],
            }
        ],
    )
    malformed = skills_body(job, must_have=[{"raw_text": "Python", "skill_id": "Python"}])
    for body in (bad_id, forged, malformed):
        assert client.put(f"/jobs/{job['id']}/skills", json=body).status_code == 422
    assert [e.action for e in audit(sessions)] == ["job.drafted"]
    with sessions() as session:
        stored = session.get(Job, uuid.UUID(job["id"]))
        assert stored is not None
        assert stored.spec == job["spec"]


def test_llm_refusal_becomes_a_clean_error(client: TestClient, gateway: FakeGateway):
    gateway.backend.stop_reason = "refusal"
    response = client.post("/jobs", json={"posting_text": POSTING})
    assert response.status_code == 422
    assert "declined" in response.json()["detail"]


def test_empty_posting_is_rejected(client: TestClient):
    assert client.post("/jobs", json={"posting_text": ""}).status_code == 422
