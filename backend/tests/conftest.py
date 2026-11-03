from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.deps import get_file_store, get_gateway, get_github_clients
from app.config import DEFAULT_TENANT_ID
from app.db.base import Base
from app.db.models import Tenant
from app.db.session import get_session
from app.github.cache import SqlHttpCache
from app.llm.fake import FakeGateway
from app.main import create_app
from app.schemas.evidence import CvEducation, CvEntry, CvSkillMention, ParsedCv
from app.schemas.job import ParsedPosting, PostingRequirement
from app.sources.base import CandidateRef
from app.sources.github import GitHubClientFactory
from app.storage import InMemoryFileStore
from tests.github_fake import FakeGitHub
from tests.github_world import fake_github, readme_reply

CVS = Path(__file__).parents[2] / "evals" / "cvs"


def fixture_cv(slug: str) -> str:
    """A fictional CV from evals/cvs/, as text."""
    return (CVS / f"{slug}.txt").read_text(encoding="utf-8")


# Fictional posting from the worked example in CLAUDE.md.
POSTING = """\
Backend Engineer (Python) - Northwind Robotics

We build fleet software for warehouse robots.

What you'll need
- Strong Python skills and experience building APIs with FastAPI.
- Experience designing schemas in PostgreSQL.
- Comfortable with Temporal workflows.

Nice to have
- Docker and AWS experience.
- Familiarity with Redis.

5+ years of experience and a CS degree preferred.
"""


def req(skill: str, quote: str) -> PostingRequirement:
    return PostingRequirement(skill=skill, quote=quote)


def posting_reply() -> ParsedPosting:
    """What a well-behaved posting_parser would return for POSTING, plus two bad items."""
    return ParsedPosting(
        title="Backend Engineer (Python)",
        must_have=[
            req("Python", "Strong Python skills"),
            req("FastAPI", "experience building APIs with FastAPI"),
            req("PostgreSQL", "Experience designing schemas in PostgreSQL"),
            req("Temporal", "Comfortable with Temporal workflows"),
            req("Kubernetes", "Deep Kubernetes expertise"),  # not in the posting
        ],
        nice_to_have=[
            req("Docker", "Docker and AWS experience"),
            req("AWS", "Docker and AWS experience"),
            req("Redis", "Familiarity with Redis"),
            req("Python", "Strong Python skills"),  # duplicate of a must-have
        ],
        seniority="unspecified",
        domain="backend",
    )


def mention(skill: str, quote: str) -> CvSkillMention:
    return CvSkillMention(skill=skill, quote=quote)


def amira_cv_reply() -> ParsedCv:
    """What a well-behaved cv_extractor would return for 01-amira-haddad, plus three items
    the validator must reject (marked)."""
    return ParsedCv(
        entries=[
            CvEntry(
                kind="job",
                title="Senior Backend Developer",
                organization="Lumen Freight",
                start="Jan 2022",
                end="Present",
                heading_quote="Senior Backend Developer — Lumen Freight, Montréal",
                skills=[
                    mention("Django", "shipment-tracking API in Django"),
                    mention("Django REST Framework", "Django REST Framework"),
                    mention("PostgreSQL", "Modelled billing schemas in PostgreSQL"),
                    mention("Docker", "Containerised services with Docker"),
                    mention("GitHub Actions", "set up GitHub Actions pipelines"),
                ],
            ),
            CvEntry(
                kind="job",
                title="Backend Developer",
                organization="Pixelbay Studio",
                start="Jun 2019",
                end="Dec 2021",
                heading_quote="Backend Developer — Pixelbay Studio, Montréal",
                skills=[
                    mention("Python 3", "features in Python 3"),
                    mention("Celery", "Moved background jobs from cron to Celery"),
                    mention("Redis", "with Redis as the broker"),
                    mention("pytest", "Wrote unit tests with pytest"),
                    mention("Kubernetes", "Deployed services to Kubernetes"),  # not in CV
                    mention("Docker", "Containerised services with Docker"),  # other entry
                    mention("MySQL", "Wrote unit tests with pytest"),  # quote lacks skill
                ],
            ),
            CvEntry(
                kind="project",
                title="Contributor",
                organization="django-tablekit",
                start="2021",
                end="Present",
                heading_quote="Contributor, django-tablekit",
                skills=[mention("PostgreSQL", "query optimisation for PostgreSQL JSON fields")],
            ),
        ],
        education=[
            CvEducation(
                credential="B.Sc.",
                field="Computer Science",
                institution="Université de Montréal",
                quote="B.Sc. Computer Science — Université de Montréal",
            )
        ],
        other_skills=[
            mention("Python", "Languages: Python, SQL, Bash"),
            mention("SQL", "Languages: Python, SQL, Bash"),
            mention("Bash", "Languages: Python, SQL, Bash"),
            mention("Django", "Frameworks: Django, Django REST Framework, Celery"),
            mention("Linux", "Tools: Docker, Git, GitHub Actions, Linux"),
        ],
    )


@pytest.fixture
def sessions() -> Iterator[sessionmaker[Session]]:
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    event.listen(engine, "connect", lambda conn, _rec: conn.execute("PRAGMA foreign_keys=ON"))
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory.begin() as session:
        session.add(Tenant(id=DEFAULT_TENANT_ID, name="default"))
    yield factory
    engine.dispose()


@pytest.fixture
def gateway() -> FakeGateway:
    return FakeGateway(
        {
            "posting_parser": posting_reply(),
            "cv_extractor": amira_cv_reply(),
            "readme_extractor": readme_reply,
        }
    )


@pytest.fixture
def github() -> FakeGitHub:
    """A fake api.github.com with the fictional Amira and Ben accounts."""
    return fake_github()


@pytest.fixture
def file_store() -> InMemoryFileStore:
    return InMemoryFileStore()


@pytest.fixture
def client(
    sessions: sessionmaker[Session],
    gateway: FakeGateway,
    file_store: InMemoryFileStore,
    github: FakeGitHub,
) -> Iterator[TestClient]:
    app = create_app()

    def cache_for(candidate: CandidateRef) -> SqlHttpCache:
        return SqlHttpCache(sessions, candidate.tenant_id, candidate.id)

    github_clients = GitHubClientFactory("test-token", cache_for, transport=github.transport())

    def session_override() -> Iterator[Session]:
        with sessions() as session:
            yield session

    app.dependency_overrides[get_session] = session_override
    app.dependency_overrides[get_gateway] = lambda: gateway
    app.dependency_overrides[get_file_store] = lambda: file_store
    app.dependency_overrides[get_github_clients] = lambda: github_clients
    with TestClient(app) as test_client:
        yield test_client
