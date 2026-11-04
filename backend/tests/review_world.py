"""Helpers for tests that take candidates through the whole pipeline, offline."""

import re
from typing import Any

from fastapi.testclient import TestClient

from app.extraction.text import DOCX
from app.schemas.evidence import CvSkillMention, ParsedCv
from app.taxonomy.loader import load_taxonomy
from tests.conftest import POSTING, fixture_cv
from tests.docgen import make_docx

# The worked example's posting, with the skills a recruiter would confirm.
WORKED_EXAMPLE_SKILLS: dict[str, Any] = {
    "must_have": [
        {"raw_text": "Python", "skill_id": "skill:python"},
        {"raw_text": "FastAPI", "skill_id": "skill:fastapi"},
        {"raw_text": "PostgreSQL", "skill_id": "skill:postgresql"},
    ],
    "nice_to_have": [
        {"raw_text": "Docker", "skill_id": "skill:docker"},
        {"raw_text": "AWS", "skill_id": "skill:aws"},
        {"raw_text": "Redis", "skill_id": "skill:redis"},
    ],
    "confirm": True,
}

_ALIASES = sorted(load_taxonomy().aliases, key=len, reverse=True)
_ALIAS = re.compile(
    r"(?<![\w+#.-])(" + "|".join(map(re.escape, _ALIASES)) + r")(?![\w+#-])", re.IGNORECASE
)


def scanning_cv_reply(content: str) -> ParsedCv:
    """A stand-in for cv_extractor that works on any CV: every taxonomy term on a line is a
    claim, quoted by that line. Its output is real input for the real validator."""
    mentions: dict[str, CvSkillMention] = {}
    for raw in content.splitlines():
        line = raw.strip()
        if len(line) > 900:
            continue
        for match in _ALIAS.finditer(line):
            mentions.setdefault(match[1].casefold(), CvSkillMention(skill=match[1], quote=line))
    return ParsedCv(entries=[], education=[], other_skills=list(mentions.values()))


def create_job(client: TestClient) -> str:
    response = client.post("/jobs", json={"posting_text": POSTING})
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


def confirm_job(client: TestClient, job_id: str, **changes: Any) -> None:
    response = client.put(f"/jobs/{job_id}/skills", json={**WORKED_EXAMPLE_SKILLS, **changes})
    assert response.status_code == 200, response.text


def apply_with_cv(
    client: TestClient, job_id: str, slug: str, name: str, **fields: Any
) -> dict[str, Any]:
    response = client.post(
        f"/jobs/{job_id}/candidates",
        data={
            "full_name": name,
            "notice_version": "2026-10-draft",
            "consent_cv": "true",
            **fields,
        },
        files={"cv": ("cv.docx", make_docx(fixture_cv(slug).splitlines()), DOCX)},
    )
    assert response.status_code == 201, response.text
    data: dict[str, Any] = response.json()
    return data


def apply_amira(client: TestClient, job_id: str) -> dict[str, Any]:
    return apply_with_cv(
        client,
        job_id,
        "01-amira-haddad",
        "Amira Haddad",
        email="amira.haddad@example.com",
        github_handle="amira-h",
        consent_github="true",
    )


def apply_ben(client: TestClient, job_id: str) -> dict[str, Any]:
    return apply_with_cv(
        client,
        job_id,
        "02-ben-okafor",
        "Ben Okafor",
        email="ben.okafor@example.com",
        github_handle="ben-okafor",
        consent_github="true",
    )


def rankings(client: TestClient, job_id: str) -> list[dict[str, Any]]:
    response = client.get(f"/jobs/{job_id}/rankings")
    assert response.status_code == 200, response.text
    rows: list[dict[str, Any]] = response.json()["candidates"]
    return rows


def row_for(rows: list[dict[str, Any]], candidate: dict[str, Any]) -> dict[str, Any]:
    return next(r for r in rows if r["candidate_id"] == candidate["id"])
