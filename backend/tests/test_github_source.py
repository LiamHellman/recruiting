"""GitHubSource end to end against the fake api.github.com (tests/github_fake.py) and the
fictional Amira and Ben accounts (tests/github_world.py). No network, no real LLM."""

import json
import re
import uuid
from dataclasses import dataclass
from datetime import date
from typing import Any

import pytest

from app.config import DEFAULT_TENANT_ID
from app.github.client import GitHubError, GitHubNotConfigured, GitHubNotFound, InMemoryHttpCache
from app.github.collect import Limits
from app.llm.fake import FakeGateway
from app.schemas.evidence import EvidenceItem
from app.schemas.github import GitHubProfile
from app.sources.base import CandidateRef, IngestResult, SourceError
from app.sources.github import GitHubClientFactory, GitHubPayload, GitHubSource
from app.taxonomy.loader import load_taxonomy
from tests.github_fake import FakeGitHub, User
from tests.github_world import fake_github, readme_reply

TODAY = date(2026, 10, 15)
AMIRA = CandidateRef(
    id=uuid.uuid4(),
    tenant_id=DEFAULT_TENANT_ID,
    job_id=None,
    names=("Amira Haddad",),
    emails=("amira.haddad@example.com",),
)
BEN = CandidateRef(
    id=uuid.uuid4(),
    tenant_id=DEFAULT_TENANT_ID,
    job_id=None,
    names=("Ben Okafor",),
    emails=("ben.okafor@example.com",),
)
AMIRA_PAYLOAD = GitHubPayload(
    handle="amira-h",
    handle_from="form",
    collected_on=TODAY,
    site_url="https://amira.dev",
    cv_links=("https://github.com/amira-h", "https://amira.dev/"),
)
BEN_PAYLOAD = GitHubPayload(handle="ben-okafor", handle_from="form", collected_on=TODAY)


@dataclass
class World:
    github: FakeGitHub
    gateway: FakeGateway
    caches: dict[uuid.UUID, InMemoryHttpCache]
    source: GitHubSource

    def ingest(self, candidate: CandidateRef, payload: GitHubPayload) -> IngestResult:
        return self.source.ingest(candidate, payload)


def make_world(github: FakeGitHub | None = None, **source_options: Any) -> World:
    github = github or fake_github()
    gateway = FakeGateway({"readme_extractor": readme_reply})
    caches: dict[uuid.UUID, InMemoryHttpCache] = {}
    clients = GitHubClientFactory(
        "test-token",
        lambda c: caches.setdefault(c.id, InMemoryHttpCache()),
        transport=github.transport(),
    )
    source = GitHubSource(clients, gateway, load_taxonomy(), **source_options)
    return World(github, gateway, caches, source)


@pytest.fixture
def world() -> World:
    return make_world()


def summary(evidence: list[EvidenceItem]) -> set[tuple[str, str, float, str]]:
    """(skill, level, ownership, repository) per item."""
    return {
        (e.skill_id, e.level, e.ownership, re.split(r"[@#]", e.locator.split(":", 1)[1])[0])
        for e in evidence
    }


def profile(result: IngestResult) -> GitHubProfile:
    return GitHubProfile.model_validate(result.profile)


# --- Amira --------------------------------------------------------------------------------


def test_amira_owned_repo_is_demonstrated_and_merged_prs_are_deep(world: World):
    result = world.ingest(AMIRA, AMIRA_PAYLOAD)
    assert summary(result.evidence) == {
        ("skill:django", "demonstrated", 1.0, "amira-h/shop-api"),
        ("skill:postgresql", "demonstrated", 1.0, "amira-h/shop-api"),
        ("skill:pytest", "demonstrated", 1.0, "amira-h/shop-api"),
        ("skill:docker", "demonstrated", 1.0, "amira-h/shop-api"),
        ("skill:github-actions", "demonstrated", 1.0, "amira-h/shop-api"),
        ("skill:python", "demonstrated", 1.0, "amira-h/shop-api"),
        ("skill:bash", "demonstrated", 1.0, "amira-h/dotfiles"),
        ("skill:python", "deep", 1.0, "tablekit-org/django-tablekit"),
        ("skill:django", "deep", 1.0, "tablekit-org/django-tablekit"),
    }
    by_key = {(e.skill_id, e.level): e for e in result.evidence}
    postgres = by_key[("skill:postgresql", "demonstrated")]
    assert re.fullmatch(r"repo:amira-h/shop-api@[0-9a-f]{40}:requirements\.txt", postgres.locator)
    assert postgres.quote == "psycopg[binary]==3.1.19"  # the dependency, before the image
    assert postgres.date_last_seen == date(2026, 8, 14)  # her last commit in the repo
    django_deep = by_key[("skill:django", "deep")]
    assert re.fullmatch(
        r"pr:tablekit-org/django-tablekit#41@[0-9a-f]{40}:pyproject\.toml", django_deep.locator
    )
    assert django_deep.quote == "django>=4.2"
    assert django_deep.date_last_seen == date(2026, 7, 2)
    python_deep = by_key[("skill:python", "deep")]
    assert python_deep.quote == "django_tablekit/filters.py"  # the most-changed .py file
    assert all(e.source == "github" and e.confidence == 1.0 for e in result.evidence)


def test_prs_that_are_not_peer_reviewed_external_work_do_not_count(world: World):
    result = world.ingest(AMIRA, AMIRA_PAYLOAD)
    cards = {(c.repo, c.number): (c.counts, c.reason) for c in profile(result).pulls}
    assert cards == {
        ("tablekit-org/django-tablekit", 41): (True, None),
        ("tablekit-org/django-tablekit", 37): (True, None),
        ("amira-h/shop-api", 3): (False, "pull request to the candidate's own repository"),
        ("lumen-oss/devtools", 5): (False, "merged by the candidate"),
        ("psf-example/docs", 9): (False, "fewer than 10 changed lines of code"),
    }
    # pytest is a dev dependency of the library: not credited through her PRs.
    assert not any(e.level == "deep" and e.skill_id == "skill:pytest" for e in result.evidence)
    # Only the qualifying PR's repository was read, at its merge commit.
    tree_reads = [r.url.path for r in world.github.requests if "/git/trees/" in r.url.path]
    assert not any("lumen-oss" in p or "psf-example" in p for p in tree_reads)


def test_readme_claims_are_validated_redacted_and_merged(world: World):
    result = world.ingest(AMIRA, AMIRA_PAYLOAD)
    assert result.report.prompt_ref == "readme_extractor/v1"
    # Django/PostgreSQL/Docker claims are outranked by demonstrated items in the same repo.
    assert not any(e.level == "claimed" for e in result.evidence)
    assert result.report.unmapped_terms == ["Django REST Framework", "Stripe"]
    assert [(r.item, r.reason) for r in result.report.rejected] == [
        ("Redis", "quote not found in source")
    ]
    # Only her one credited repo with a README went to the model, redacted.
    [call] = world.gateway.backend.calls
    assert "online-shop API" in call.user_content
    for secret in ("Amira", "Haddad", "amira-h", "amira.haddad@example.com", "amira.dev"):
        assert secret not in call.user_content
    assert result.report.redactions == {"name": 2, "email": 1, "link": 1}


def test_readme_only_claims_become_claimed_evidence_with_readme_offsets():
    github = fake_github()
    shelf = next(r for r in github.repos.values() if r.name == "shelflife-api")
    readme = shelf.files["README.md"] + "Built by Ben Okafor. Caching with Redis.\n"
    shelf.files["README.md"] = readme
    world = make_world(github)
    world.gateway.backend.replies["readme_extractor"] = {
        "skills": [
            {"skill": "Redis", "quote": "Caching with Redis"},
            {"skill": "SQL", "quote": "SQLAlchemy"},  # not a whole word: rejected
        ]
    }
    result = world.ingest(BEN, BEN_PAYLOAD)
    [redis] = [e for e in result.evidence if e.skill_id == "skill:redis"]
    assert redis.level == "claimed" and redis.ownership == 1.0
    match = re.fullmatch(
        r"repo:ben-okafor/shelflife-api@[0-9a-f]{40}:README\.md#(\d+)-(\d+)", redis.locator
    )
    assert match is not None
    start, end = int(match.group(1)), int(match.group(2))
    assert readme[start:end] == redis.quote == "Caching with Redis"
    assert ("SQL", "quote does not mention the skill") in [
        (r.item, r.reason) for r in result.report.rejected
    ]
    assert "Okafor" not in world.gateway.backend.calls[0].user_content


def test_amira_identity_is_corroborated(world: World):
    identity = profile(world.ingest(AMIRA, AMIRA_PAYLOAD)).identity
    assert identity.status == "corroborated" and not identity.needs_confirmation
    matched = {s.name for s in identity.signals if s.outcome == "match"}
    assert matched == {"cv_link", "profile_website", "commit_email"}


def test_amira_depth_signals(world: World):
    depth = profile(world.ingest(AMIRA, AMIRA_PAYLOAD)).depth
    assert depth.credited_repos == 2
    assert (
        depth.repos_with_tests
        == depth.repos_with_ci
        == depth.repos_with_docker
        == ["amira-h/shop-api"]
    )
    assert depth.merged_external_prs == 2
    assert depth.external_repos == ["tablekit-org/django-tablekit"]
    assert depth.max_stars == 14
    assert depth.longest_active_days == 320


# --- Ben ----------------------------------------------------------------------------------


def test_ben_forks_and_tutorial_clones_earn_no_credit(world: World):
    result = world.ingest(BEN, BEN_PAYLOAD)
    credited = {(s, lvl, repo) for s, lvl, own, repo in summary(result.evidence) if own > 0}
    assert credited == {
        ("skill:fastapi", "demonstrated", "ben-okafor/shelflife-api"),
        ("skill:sqlalchemy", "demonstrated", "ben-okafor/shelflife-api"),
        ("skill:pydantic", "demonstrated", "ben-okafor/shelflife-api"),
        ("skill:python", "demonstrated", "ben-okafor/shelflife-api"),
    }
    assert all(e.date_last_seen == date(2022, 6, 18) for e in result.evidence if e.ownership > 0)
    # Learning projects are shown with ownership 0; untouched forks give nothing at all.
    zero = {(s, repo) for s, _, own, repo in summary(result.evidence) if own == 0}
    assert zero == {
        ("skill:flask", "ben-okafor/flask-mega-tutorial"),
        ("skill:python", "ben-okafor/flask-mega-tutorial"),
        ("skill:django", "ben-okafor/django-polls-tutorial"),
        ("skill:python", "ben-okafor/django-polls-tutorial"),
    }
    cards = {c.repo.split("/")[1]: c for c in profile(result).repos}
    assert len(cards) == 12
    assert sum(c.label == "fork_untouched" for c in cards.values()) == 9
    assert cards["flask-mega-tutorial"].learning and cards["flask-mega-tutorial"].ownership == 0
    assert cards["shelflife-api"].label == "owned" and cards["shelflife-api"].ownership == 1.0
    assert profile(result).depth.credited_repos == 1


def test_untouched_forks_are_not_read_beyond_their_commits(world: World):
    world.ingest(BEN, BEN_PAYLOAD)
    fork_paths = [r.url.path for r in world.github.requests if "/ben-okafor/cpython" in r.url.path]
    assert fork_paths == ["/repos/ben-okafor/cpython", "/repos/ben-okafor/cpython/commits"]
    [commits] = [r for r in world.github.requests if r.url.path.endswith("/cpython/commits")]
    assert commits.url.params["since"] == "2021-01-10T09:00:00Z"  # only since forking


def test_ben_identity_needs_confirmation(world: World):
    identity = profile(world.ingest(BEN, BEN_PAYLOAD)).identity
    assert identity.status == "unverified"
    assert identity.label == "possible match: confirm?"


# --- caching ------------------------------------------------------------------------------


def test_a_second_run_is_served_from_cache(world: World):
    first = world.ingest(AMIRA, AMIRA_PAYLOAD)
    model_calls = len(world.gateway.backend.calls)
    seen = len(world.github.requests)
    second = world.ingest(AMIRA, AMIRA_PAYLOAD)

    again = world.github.requests[seen:]
    assert again and all(r.extensions["fake_status"] == 304 for r in again)
    assert not any(r.url.path == "/graphql" for r in again)  # within the GraphQL TTL
    assert not any("/contents/" in r.url.path or "/git/trees/" in r.url.path for r in again)
    assert second.report.fetch["not_modified"] == len(again)
    assert second.report.fetch["cache_hits"] > 0
    assert len(world.gateway.backend.calls) == model_calls  # README answer cached too
    assert second.report.cached is True
    assert second.evidence == first.evidence


def test_the_cache_keeps_no_more_personal_data_than_needed(world: World):
    world.ingest(AMIRA, AMIRA_PAYLOAD)
    entries = world.caches[AMIRA.id].entries.values()
    # File contents are kept as fetched: they are what README locators point into, like the
    # stored CV text. Everything else is API metadata, reduced to the fields used.
    stored = json.dumps([e.data for e in entries if not isinstance(e.data, str)])
    for dropped in ("avatar", "Montréal", "Backend developer", "Amira Haddad", "followers",
                    "amira.haddad@example.com", "noreply", "Someone", '"email"'):  # fmt: skip
        assert dropped not in stored, dropped
    # Profile: also no name, location, bio or account creation date.
    stored_profile = json.dumps(world.ingest(AMIRA, AMIRA_PAYLOAD).profile)
    for dropped in ("Montréal", "Amira Haddad", "2016", "avatar"):
        assert dropped not in stored_profile, dropped


def test_caches_are_per_candidate(world: World):
    world.ingest(AMIRA, AMIRA_PAYLOAD)
    other = CandidateRef(id=uuid.uuid4(), tenant_id=DEFAULT_TENANT_ID, job_id=None)
    seen = len(world.github.requests)
    world.ingest(other, AMIRA_PAYLOAD)
    assert all(r.extensions["fake_status"] != 304 for r in world.github.requests[seen:])


# --- failures and limits ------------------------------------------------------------------


def test_unknown_or_organization_accounts_fail_cleanly():
    world = make_world(
        FakeGitHub([User("acme-corp", 9, "2015-01-01T00:00:00Z", type="Organization")])
    )
    with pytest.raises(GitHubNotFound, match="'nobody-here' not found"):
        world.ingest(BEN, GitHubPayload("nobody-here", "form", TODAY))
    with pytest.raises(GitHubError, match="not a personal account"):
        world.ingest(BEN, GitHubPayload("acme-corp", "form", TODAY))
    assert issubclass(GitHubError, SourceError)  # recorded as a failed run


def test_no_token_means_no_collection():
    clients = GitHubClientFactory(None, lambda c: InMemoryHttpCache())
    source = GitHubSource(clients, FakeGateway(), load_taxonomy())
    with pytest.raises(GitHubNotConfigured):
        source.ingest(BEN, BEN_PAYLOAD)


def test_a_refused_readme_is_skipped_not_fatal(world: World):
    world.gateway.backend.stop_reason = "refusal"
    result = world.ingest(AMIRA, AMIRA_PAYLOAD)
    assert [r.reason for r in result.report.rejected] == ["not read (LLMRefusal)"]
    assert any(e.skill_id == "skill:django" for e in result.evidence)


def test_repository_limit_takes_the_most_recently_pushed():
    world = make_world(limits=Limits(max_repos=1))
    result = world.ingest(BEN, BEN_PAYLOAD)
    assert [c.repo for c in profile(result).repos] == ["ben-okafor/shelflife-api"]
    assert profile(result).repos_total == 12 and profile(result).repos_analysed == 1


def test_unlinked_commits_with_the_candidates_email_count_as_theirs():
    github = fake_github()
    shelf = next(r for r in github.repos.values() if r.name == "shelflife-api")
    for commit in shelf.commits:
        commit.linked = False
        commit.email = "Ben.Okafor@example.com"
    result = make_world(github).ingest(BEN, BEN_PAYLOAD)
    card = next(c for c in profile(result).repos if c.repo.endswith("shelflife-api"))
    assert (card.label, card.ownership, card.candidate_commits) == ("owned", 1.0, 23)
    assert card.last_commit is None  # dates come from linked commits only
