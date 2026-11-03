"""Fictional GitHub accounts for Amira and Ben, following the worked example in CLAUDE.md.

Amira (amira-h): an original Django + PostgreSQL shop API (140 commits over 8 months,
tests, CI, Docker Compose, last commit 2026-08-14), dotfiles, an untouched fork, and two
merged PRs to an open-source Django library, plus PRs that must not count (her own repo,
self-merged, docs-only).
Ben (ben-okafor): 9 untouched forks, 2 tutorial clones with a single commit, and one small
original FastAPI app last updated in June 2022.
"""

from datetime import UTC, datetime

from app.schemas.github import ParsedReadme, ReadmeSkillMention
from tests.github_fake import Commit, FakeGitHub, Pull, Repo, User, iso


def spread(n: int, first: str, last: str, email: str, **kw: bool) -> list[Commit]:
    start = datetime.fromisoformat(first).replace(tzinfo=UTC)
    end = datetime.fromisoformat(last).replace(tzinfo=UTC)
    step = (end - start) / max(n - 1, 1)
    return [Commit(iso(start + step * i), email, **kw) for i in range(n)]


SHOP_README = """\
# shop-api

A small online-shop API built with Django and Django REST Framework, backed by PostgreSQL.
Maintained by Amira Haddad (amira.haddad@example.com), see https://amira.dev.

## Running locally

Start everything with Docker Compose: `docker compose up`.
Payments go through Stripe in test mode. Questions: ping @amira-h.
"""

SHOP_FILES = {
    "manage.py": "#!/usr/bin/env python\n" + "x" * 800,
    "shop/models.py": "from django.db import models\n" + "x" * 9000,
    "shop/views.py": "x" * 6000,
    "shop/tests/test_orders.py": "x" * 12000,
    "shop/templates/base.html": "<html></html>" + "x" * 3000,
    "requirements.txt": "Django==5.0.6\ndjangorestframework==3.15.1\n"
    "psycopg[binary]==3.1.19\ngunicorn==22.0.0  # app server\n",
    "requirements-dev.txt": "-r requirements.txt\npytest==8.2.0\npytest-django==4.8.0\n",
    "Dockerfile": "FROM python:3.12-slim\nWORKDIR /app\nCOPY . .\n",
    "docker-compose.yml": "services:\n  web:\n    build: .\n  db:\n    image: postgres:16\n",
    ".github/workflows/ci.yml": "name: CI\non: [push]\n",
    "node_modules/left-pad/package.json": '{"name": "left-pad"}',
    "README.md": SHOP_README,
}

TABLEKIT_PYPROJECT = """\
[project]
name = "django-tablekit"
dependencies = ["django>=4.2"]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-django"]
"""

AMIRA_PR_41 = Pull(
    repo="tablekit-org/django-tablekit",
    number=41,
    title="Add JSONField filters for PostgreSQL",
    merged_at="2026-07-02T15:00:00Z",
    merged_by="tablekit-maint",
    files=[
        ("django_tablekit/filters.py", 40, 5),
        ("tests/test_filters.py", 30, 0),
        ("docs/filters.md", 10, 0),
    ],
)


def amira() -> User:
    email = "amira.haddad@example.com"
    return User(
        login="amira-h",
        id=1001,
        created_at="2016-03-02T10:00:00Z",
        blog="https://amira.dev",
        name="Amira Haddad",
        location="Montréal",
        bio="Backend developer",
        repos=[
            Repo(
                owner="amira-h",
                name="shop-api",
                created_at="2025-12-10T09:00:00Z",
                pushed_at="2026-08-14T18:00:00Z",
                stars=14,
                commits=spread(140, "2025-12-16T10:00:00", "2026-08-14T17:00:00", email),
                languages={"Python": 182_000, "HTML": 12_000, "Dockerfile": 600},
                files=SHOP_FILES,
            ),
            Repo(
                owner="amira-h",
                name="dotfiles",
                created_at="2024-01-05T09:00:00Z",
                pushed_at="2024-11-20T09:00:00Z",
                commits=spread(
                    12,
                    "2024-01-05T10:00:00",
                    "2024-11-20T10:00:00",
                    "amira-h@users.noreply.github.com",
                ),
                languages={"Shell": 4_200},
                files={"install.sh": "#!/bin/sh\n" + "x" * 3000, ".bashrc": "x" * 900},
            ),
            Repo(
                owner="amira-h",
                name="django-tablekit",
                fork=True,
                parent="tablekit-org/django-tablekit",
                created_at="2024-02-01T09:00:00Z",
                pushed_at="2024-02-01T09:00:00Z",
                commits=[],
                others={"tablekit-maint": 300},
                languages={"Python": 90_000},
                files={"pyproject.toml": TABLEKIT_PYPROJECT},
            ),
        ],
        pulls=[
            AMIRA_PR_41,
            Pull(
                repo="tablekit-org/django-tablekit",
                number=37,
                title="Fix column ordering",
                merged_at="2026-03-10T12:00:00Z",
                merged_by="tablekit-maint",
                files=[("django_tablekit/columns.py", 12, 3)],
            ),
            Pull(
                repo="amira-h/shop-api",
                number=3,
                title="Add order export",
                merged_at="2026-05-01T12:00:00Z",
                merged_by="amira-h",
                files=[("shop/export.py", 80, 0)],
            ),
            Pull(
                repo="lumen-oss/devtools",
                number=5,
                title="Add lint helper",
                merged_at="2026-04-01T12:00:00Z",
                merged_by="amira-h",
                files=[("tools/lint.py", 50, 0)],
            ),
            Pull(
                repo="psf-example/docs",
                number=9,
                title="Fix typo",
                merged_at="2025-09-01T12:00:00Z",
                merged_by="docs-maint",
                files=[("docs/index.md", 1, 1)],
            ),
        ],
    )


AMIRA_TREES = {
    ("tablekit-org/django-tablekit", AMIRA_PR_41.merge_sha): {
        "pyproject.toml": TABLEKIT_PYPROJECT,
        "django_tablekit/filters.py": "x" * 4000,
    }
}

SHELF_README = """\
# ShelfLife

Inventory API built with FastAPI and SQLAlchemy. Deployed to Heroku.
"""

BEN_FORKS = [
    ("vinta", "awesome-python"),
    ("tiangolo", "fastapi"),
    ("django", "django"),
    ("pallets", "flask"),
    ("psf", "requests"),
    ("python", "cpython"),
    ("facebook", "react"),
    ("microsoft", "vscode"),
    ("public-apis", "public-apis"),
]


def ben() -> User:
    noreply = "ben-okafor@users.noreply.github.com"
    forks = [
        Repo(
            owner="ben-okafor",
            name=name,
            fork=True,
            parent=f"{upstream}/{name}",
            created_at="2021-01-10T09:00:00Z",
            pushed_at=f"2021-01-{10 + i:02d}T09:00:00Z",
            others={"maintainer": 5000},
            languages={"Python": 1_000_000},
        )
        for i, (upstream, name) in enumerate(BEN_FORKS)
    ]
    return User(
        login="ben-okafor",
        id=2002,
        created_at="2019-05-01T09:00:00Z",
        name="Ben Okafor",
        location="Toronto",
        repos=[
            *forks,
            Repo(
                owner="ben-okafor",
                name="flask-mega-tutorial",
                created_at="2020-11-03T09:00:00Z",
                pushed_at="2020-11-03T09:00:00Z",
                commits=[Commit("2020-11-03T10:00:00Z", noreply, additions=4210)],
                languages={"Python": 30_000, "HTML": 5_000},
                files={"requirements.txt": "Flask==1.1.2\n", "app/routes.py": "x" * 3000},
            ),
            Repo(
                owner="ben-okafor",
                name="django-polls-tutorial",
                created_at="2020-12-12T09:00:00Z",
                pushed_at="2020-12-12T09:00:00Z",
                commits=[Commit("2020-12-12T10:00:00Z", noreply, additions=1800)],
                languages={"Python": 20_000},
                files={"requirements.txt": "Django==3.1\n", "polls/views.py": "x" * 2000},
            ),
            Repo(
                owner="ben-okafor",
                name="shelflife-api",
                created_at="2021-03-01T09:00:00Z",
                pushed_at="2022-06-18T09:00:00Z",
                commits=spread(23, "2021-03-02T10:00:00", "2022-06-18T10:00:00", noreply),
                languages={"Python": 41_000},
                files={
                    "requirements.txt": "fastapi==0.68.0\nuvicorn==0.15.0\n"
                    "SQLAlchemy==1.4.22\npydantic==1.8.2\n",
                    "app/main.py": "x" * 7000,
                    "app/models.py": "x" * 3000,
                    "tests/test_items.py": "x" * 2500,
                    "README.md": SHELF_README,
                },
            ),
        ],
    )


def fake_github() -> FakeGitHub:
    return FakeGitHub([amira(), ben()], trees=AMIRA_TREES)


def m(skill: str, quote: str) -> ReadmeSkillMention:
    return ReadmeSkillMention(skill=skill, quote=quote)


def readme_reply(content: str) -> ParsedReadme:
    """A well-behaved readme_extractor for the READMEs above, plus one bad quote."""
    if "online-shop API" in content:
        return ParsedReadme(
            skills=[
                m("Django", "built with Django"),
                m("Django REST Framework", "Django REST Framework"),
                m("PostgreSQL", "backed by PostgreSQL"),
                m("Docker Compose", "Docker Compose"),
                m("Stripe", "Payments go through Stripe"),
                m("Redis", "cached in Redis"),  # not in the README: rejected
            ]
        )
    if "ShelfLife" in content:
        return ParsedReadme(
            skills=[
                m("FastAPI", "built with FastAPI"),
                m("SQLAlchemy", "FastAPI and SQLAlchemy"),
                m("Heroku", "Deployed to Heroku"),
            ]
        )
    return ParsedReadme(skills=[])


AMIRA_TODAY = datetime(2026, 10, 15, tzinfo=UTC).date()
