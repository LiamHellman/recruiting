import json
import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.api.deps import build_registry, get_pipeline_queue
from app.db.models import AuditLog, Candidate, Document, EvidenceRecord, GitHubCacheEntry
from app.extraction.text import DOCX, MAX_BYTES
from app.github.client import InMemoryHttpCache
from app.llm.fake import FakeGateway
from app.pipeline.graph import InlineQueue, PipelineContext
from app.sources.github import GitHubClientFactory
from app.storage import InMemoryFileStore
from app.taxonomy.loader import load_taxonomy
from tests.conftest import POSTING, fixture_cv
from tests.docgen import make_docx, make_pdf
from tests.github_fake import FakeGitHub

AMIRA = fixture_cv("01-amira-haddad")
CV_FILE = ("Amira_Haddad_CV.docx", make_docx(AMIRA.splitlines()), DOCX)
BEN_CV = ("cv.docx", make_docx(fixture_cv("02-ben-okafor").splitlines()), DOCX)


@pytest.fixture
def job_id(client: TestClient) -> str:
    response = client.post("/jobs", json={"posting_text": POSTING})
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


def apply(client: TestClient, job_id: str, cv: tuple[str, bytes, str] = CV_FILE, **fields: Any):
    data = {
        "full_name": "Amira Haddad",
        "email": "amira.haddad@example.com",
        "notice_version": "2026-10-draft",
        "consent_cv": "true",
    }
    data.update({k: v for k, v in fields.items() if v is not None})
    for key in [k for k, v in fields.items() if v is None]:
        data.pop(key, None)
    return client.post(f"/jobs/{job_id}/candidates", data=data, files={"cv": cv})


def applied(client: TestClient, job_id: str, **fields: Any) -> dict[str, Any]:
    response = apply(client, job_id, **fields)
    assert response.status_code == 201, response.text
    data: dict[str, Any] = response.json()
    return data


def audit(sessions: sessionmaker[Session]) -> list[AuditLog]:
    with sessions() as session:
        return list(session.scalars(select(AuditLog).order_by(AuditLog.id)))


def test_applying_stores_consent_document_and_cv_evidence(
    client: TestClient, job_id: str, file_store: InMemoryFileStore
):
    candidate = applied(client, job_id)
    assert candidate["job_id"] == job_id
    assert [(c["source"], c["notice_version"], c["revoked_at"]) for c in candidate["consents"]] == [
        ("cv", "2026-10-draft", None)
    ]
    [document] = candidate["documents"]
    assert document["content_type"] == DOCX
    assert set(document["links"]) == {"https://github.com/amira-h", "https://amira.dev/"}
    [run] = candidate["runs"]
    assert run["status"] == "succeeded"
    assert run["prompt_ref"] == "cv_extractor/v1"
    assert len(run["rejected"]) == 3
    assert run["unmapped_terms"] == ["Django REST Framework"]
    assert run["redactions"]["name"] == 1
    assert list(file_store.files.values()) == [CV_FILE[1]]


def test_evidence_locators_point_into_the_stored_cv_text(
    client: TestClient, job_id: str, sessions: sessionmaker[Session]
):
    candidate = applied(client, job_id)
    response = client.get(f"/candidates/{candidate['id']}/evidence")
    assert response.status_code == 200
    body = response.json()
    with sessions() as session:
        text = session.scalars(select(Document.text)).one()
    assert body["evidence"]
    for item in body["evidence"]:
        doc_id, offsets = item["locator"].removeprefix("cv:").split("#")
        assert doc_id == candidate["documents"][0]["id"]
        start, end = map(int, offsets.split("-"))
        assert text[start:end] == item["quote"]
        assert item["level"] == "claimed"
    assert [e["organization"] for e in body["cv_profile"]["entries"]] == [
        "Lumen Freight",
        "Pixelbay Studio",
        "django-tablekit",
    ]


def test_audit_log_and_stored_rows_hold_no_name_contact_or_cv_text(
    client: TestClient, job_id: str, sessions: sessionmaker[Session]
):
    applied(client, job_id)
    entries = audit(sessions)
    assert [e.action for e in entries] == ["job.drafted", "candidate.applied", "source.ingested"]
    payloads = json.dumps([e.payload for e in entries[1:]]).casefold()
    for secret in ("amira", "haddad", "example.com", "555", "django", "postgresql", "lumen"):
        assert secret not in payloads
    assert "Amira_Haddad_CV" not in json.dumps([e.payload for e in entries])
    with sessions() as session:
        stored = session.scalars(select(Document)).one()
        assert not hasattr(stored, "filename")


def test_cv_consent_is_required(client: TestClient, job_id: str, sessions: sessionmaker[Session]):
    response = apply(client, job_id, consent_cv="false")
    assert response.status_code == 422
    assert "consent_cv" in response.text
    with sessions() as session:
        assert session.scalars(select(Candidate)).first() is None


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"github_handle": "amira-h"}, "consent_github"),
        ({"site_url": "https://amira.dev"}, "consent_site"),
        ({"site_url": "file:///etc/passwd", "consent_site": "true"}, "http(s) URL"),
        ({"github_handle": "-bad-", "consent_github": "true"}, "github_handle"),
        ({"email": "not-an-email"}, "email"),
        ({"notice_version": "../../etc"}, "notice_version"),
    ],
)
def test_sources_need_matching_consent_and_valid_values(
    client: TestClient, job_id: str, fields: dict[str, Any], message: str
):
    response = apply(client, job_id, **fields)
    assert response.status_code == 422
    assert message in response.text


def test_consented_github_is_collected_and_site_is_recorded(
    client: TestClient, job_id: str, sessions: sessionmaker[Session]
):
    candidate = applied(
        client,
        job_id,
        github_handle="amira-h",
        consent_github="true",
        site_url="https://amira.dev",
        consent_site="true",
    )
    assert {c["source"] for c in candidate["consents"]} == {"cv", "github", "site"}
    assert (candidate["github_handle"], candidate["github_handle_from"]) == ("amira-h", "form")
    runs = {r["source"]: r for r in candidate["runs"]}
    assert set(runs) == {"cv", "github"}  # the site fetcher comes in Phase 6
    github_run = runs["github"]
    assert github_run["status"] == "succeeded"
    assert github_run["prompt_ref"] == "readme_extractor/v1"
    assert github_run["fetch"]["requests"] > 0
    assert github_run["unmapped_terms"] == ["Django REST Framework", "Stripe"]

    body = client.get(f"/candidates/{candidate['id']}/evidence").json()
    github = [e for e in body["evidence"] if e["source"] == "github"]
    assert {(e["skill_id"], e["level"]) for e in github} >= {
        ("skill:python", "deep"),
        ("skill:django", "deep"),
        ("skill:postgresql", "demonstrated"),
    }
    assert all(e["locator"].startswith(("repo:", "pr:")) for e in github)
    profile = body["github_profile"]
    assert profile["identity"]["status"] == "corroborated"
    assert profile["depth"]["merged_external_prs"] == 2
    response = client.post(f"/candidates/{candidate['id']}/sources/site/runs")
    assert response.status_code == 422
    assert "not available yet" in response.text

    # The audit log records hashes and counts, not the handle, repositories or skills.
    payloads = json.dumps([e.payload for e in audit(sessions)])
    for secret in ("amira-h", "shop-api", "tablekit", "django"):
        assert secret not in payloads.casefold()


def test_the_github_handle_can_come_from_the_cv(client: TestClient, job_id: str):
    candidate = applied(client, job_id, consent_github="true")
    assert (candidate["github_handle"], candidate["github_handle_from"]) == ("amira-h", "cv")
    [github_run] = [r for r in candidate["runs"] if r["source"] == "github"]
    assert github_run["status"] == "succeeded"
    identity = client.get(f"/candidates/{candidate['id']}/evidence").json()["github_profile"][
        "identity"
    ]
    assert identity["handle_from"] == "cv"


def test_github_consent_without_a_handle_anywhere_is_refused(
    client: TestClient, job_id: str, sessions: sessionmaker[Session]
):
    response = apply(client, job_id, cv=BEN_CV, consent_github="true")
    assert response.status_code == 422
    assert "github_handle" in response.text
    with sessions() as session:
        assert session.scalars(select(Candidate)).first() is None


def test_ben_gets_a_confirm_prompt_and_no_credit_for_forks(client: TestClient, job_id: str):
    candidate = applied(
        client,
        job_id,
        cv=BEN_CV,
        full_name="Ben Okafor",
        email="ben.okafor@example.com",
        github_handle="ben-okafor",
        consent_github="true",
    )
    body = client.get(f"/candidates/{candidate['id']}/evidence").json()
    assert body["github_profile"]["identity"]["label"] == "possible match: confirm?"
    credited = {e["locator"].split("@")[0] for e in body["evidence"] if e["ownership"] > 0}
    assert credited <= {"repo:ben-okafor/shelflife-api"}


def test_a_github_rerun_is_served_from_cache(
    client: TestClient, job_id: str, github: FakeGitHub, gateway: FakeGateway
):
    candidate = applied(client, job_id, github_handle="amira-h", consent_github="true")
    before = client.get(f"/candidates/{candidate['id']}/evidence").json()["evidence"]
    seen, calls = len(github.requests), len(gateway.backend.calls)
    response = client.post(f"/candidates/{candidate['id']}/sources/github/runs")
    assert response.status_code == 201
    run = response.json()
    assert run["status"] == "succeeded" and run["cached"] is True
    assert run["fetch"]["not_modified"] == run["fetch"]["requests"] > 0
    assert all(r.extensions["fake_status"] == 304 for r in github.requests[seen:])
    assert len(gateway.backend.calls) == calls
    after = client.get(f"/candidates/{candidate['id']}/evidence").json()["evidence"]
    assert sorted((e["skill_id"], e["locator"]) for e in after) == sorted(
        (e["skill_id"], e["locator"]) for e in before
    )


def test_an_unknown_github_account_is_a_failed_run(client: TestClient, job_id: str):
    candidate = applied(client, job_id, github_handle="nobody-here", consent_github="true")
    [github_run] = [r for r in candidate["runs"] if r["source"] == "github"]
    assert github_run["status"] == "failed"
    assert "'nobody-here' not found" in github_run["error"]
    assert "test-token" not in github_run["error"]


def test_revoking_github_consent_removes_its_evidence_and_fetched_data(
    client: TestClient, job_id: str, sessions: sessionmaker[Session]
):
    candidate = applied(client, job_id, github_handle="amira-h", consent_github="true")
    with sessions() as session:
        assert session.scalars(select(GitHubCacheEntry)).first() is not None
    response = client.delete(f"/candidates/{candidate['id']}/consents/github")
    assert response.status_code == 200
    body = client.get(f"/candidates/{candidate['id']}/evidence").json()
    assert {e["source"] for e in body["evidence"]} == {"cv"}
    assert body["github_profile"] is None
    with sessions() as session:
        assert session.scalars(select(GitHubCacheEntry)).first() is None
    assert client.post(f"/candidates/{candidate['id']}/sources/github/runs").status_code == 409


def test_pdf_cvs_are_accepted(client: TestClient, job_id: str):
    pdf = make_pdf([AMIRA.splitlines()])
    candidate = applied(client, job_id, cv=("cv.pdf", pdf, "application/pdf"))
    assert candidate["documents"][0]["content_type"] == "application/pdf"
    assert candidate["runs"][0]["status"] == "succeeded"


@pytest.mark.parametrize(
    ("cv", "status_code"),
    [
        (("cv.txt", b"plain text CV", "text/plain"), 422),
        (("cv.pdf", b"%PDF-" + b"\0" * MAX_BYTES, "application/pdf"), 413),
    ],
)
def test_unreadable_or_oversized_files_are_refused(
    client: TestClient, job_id: str, cv: tuple[str, bytes, str], status_code: int
):
    assert apply(client, job_id, cv=cv).status_code == status_code


def test_unknown_job_or_candidate_is_404(client: TestClient):
    assert apply(client, str(uuid.uuid4())).status_code == 404
    assert client.get(f"/candidates/{uuid.uuid4()}").status_code == 404
    assert client.get(f"/candidates/{uuid.uuid4()}/evidence").status_code == 404


def test_revoking_consent_removes_evidence_and_blocks_further_runs(
    client: TestClient, job_id: str, gateway: FakeGateway, sessions: sessionmaker[Session]
):
    candidate = applied(client, job_id)
    calls = len(gateway.calls)

    response = client.delete(f"/candidates/{candidate['id']}/consents/cv")
    assert response.status_code == 200
    assert response.json()["consents"][0]["revoked_at"] is not None
    evidence = client.get(f"/candidates/{candidate['id']}/evidence").json()
    assert evidence["evidence"] == []
    assert evidence["cv_profile"] is None
    [run] = client.get(f"/candidates/{candidate['id']}").json()["runs"]
    assert (run["status"], run["rejected"], run["unmapped_terms"]) == ("succeeded", [], [])

    response = client.post(f"/candidates/{candidate['id']}/sources/cv/runs")
    assert response.status_code == 409
    assert len(gateway.calls) == calls  # the model never saw the CV again
    with sessions() as session:
        assert session.scalars(select(EvidenceRecord)).first() is None
    actions = [e.action for e in audit(sessions)]
    assert actions[-2:] == ["consent.revoked", "source.refused"]
    assert client.delete(f"/candidates/{candidate['id']}/consents/cv").status_code == 404


def test_a_failed_model_call_keeps_the_candidate_and_can_be_retried(
    client: TestClient, job_id: str, gateway: FakeGateway, sessions: sessionmaker[Session]
):
    gateway.backend.stop_reason = "refusal"
    candidate = applied(client, job_id)
    [run] = candidate["runs"]
    assert run["status"] == "failed"
    assert "LLMRefusal" in run["error"]
    assert "source.failed" in [e.action for e in audit(sessions)]

    gateway.backend.stop_reason = "end_turn"
    response = client.post(f"/candidates/{candidate['id']}/sources/cv/runs")
    assert response.status_code == 201
    assert response.json()["status"] == "succeeded"
    runs = client.get(f"/candidates/{candidate['id']}").json()["runs"]
    assert [r["status"] for r in runs] == ["succeeded", "failed"]


def test_a_rerun_replaces_the_sources_evidence(
    client: TestClient, job_id: str, sessions: sessionmaker[Session]
):
    candidate = applied(client, job_id)
    before = client.get(f"/candidates/{candidate['id']}/evidence").json()["evidence"]
    response = client.post(f"/candidates/{candidate['id']}/sources/cv/runs")
    assert response.status_code == 201
    assert response.json()["cached"] is True
    after = client.get(f"/candidates/{candidate['id']}/evidence").json()["evidence"]
    assert len(after) == len(before)
    assert {e["run_id"] for e in after} == {response.json()["id"]}


def test_without_a_github_token_the_run_fails_and_can_be_retried_later(
    client: TestClient, job_id: str, sessions: sessionmaker[Session], gateway: FakeGateway
):
    no_token = GitHubClientFactory(None, lambda c: InMemoryHttpCache())
    taxonomy = load_taxonomy()
    queue = InlineQueue(
        PipelineContext(sessions, build_registry(gateway, taxonomy, no_token), taxonomy)
    )
    client.app.dependency_overrides[get_pipeline_queue] = lambda: queue  # type: ignore[attr-defined]
    candidate = applied(client, job_id, github_handle="amira-h", consent_github="true")
    [github_run] = [r for r in candidate["runs"] if r["source"] == "github"]
    assert github_run["status"] == "failed"
    assert "GITHUB_TOKEN" in github_run["error"]
    assert [r["status"] for r in candidate["runs"] if r["source"] == "cv"] == ["succeeded"]
