from dataclasses import replace

import pytest

from app.github.ownership import (
    RepoFacts,
    assess_ownership,
    pull_disqualifier,
    tutorial_marker,
)
from app.identity.github import CONFIRM_LABEL, assess_identity, github_accounts, handle_from_cv

OWNED = RepoFacts(
    name="shop-api",
    fork=False,
    template_from=None,
    topics=(),
    candidate_commits=140,
    total_commits=140,
)


@pytest.mark.parametrize(
    ("changes", "label", "factor"),
    [
        ({}, "owned", 1.0),
        ({"candidate_commits": 30, "total_commits": 120}, "owned", 0.25),
        ({"fork": True, "candidate_commits": 0, "total_commits": 5000}, "fork_untouched", 0.0),
        ({"fork": True, "candidate_commits": 3, "total_commits": 300}, "fork_with_commits", 0.01),
        ({"template_from": "org/starter"}, "template", 0.0),
        ({"name": "netflix-clone"}, "tutorial", 0.0),
        ({"name": "todo-app"}, "tutorial", 0.0),
        ({"topics": ("Tutorial",)}, "tutorial", 0.0),
        ({"candidate_commits": 0, "total_commits": 50}, "no_linked_commits", 0.0),
        ({"candidate_commits": 1, "total_commits": 1}, "single_commit", 0.0),
        # More commits counted for the candidate than GitHub's total never exceeds 1.0.
        ({"candidate_commits": 12, "total_commits": 10}, "owned", 1.0),
    ],
)
def test_ownership_rules(changes: dict[str, object], label: str, factor: float):
    own = assess_ownership(replace(OWNED, **changes))  # type: ignore[arg-type]
    assert (own.label, own.factor) == (label, factor)


def test_rules_apply_in_order_and_explain_themselves():
    fork_named_like_a_tutorial = replace(OWNED, name="todo-app", fork=True, candidate_commits=0)
    assert assess_ownership(fork_named_like_a_tutorial).label == "fork_untouched"
    dump = assess_ownership(replace(OWNED, candidate_commits=1, single_commit_additions=4210))
    assert dump.reason == "learning project: a single commit (+4,210 lines)"
    assert dump.learning
    assert not assess_ownership(OWNED).learning
    assert assess_ownership(OWNED).reason == "140 of 140 commits by the candidate"


@pytest.mark.parametrize(
    "name",
    ["todo", "react-todo-list", "my_todos", "airbnb-clone", "clone-of-x", "flask-mega-tutorial",
     "python-course", "cs50-pset", "hello-world", "learn-go", "100-days-of-code", "exercises"],
)  # fmt: skip
def test_tutorial_style_names(name: str):
    assert tutorial_marker(name) is not None


@pytest.mark.parametrize(
    "name",
    ["machine-learning-pipeline", "course-planner", "shelflife-api", "git-clone-helper",
     "todoist-sync", "deep-learning-notes"],
)  # fmt: skip
def test_real_project_names_are_not_tutorials(name: str):
    assert tutorial_marker(name) is None


def test_which_merged_prs_count():
    assert pull_disqualifier("tablekit-org", "maintainer", "amira-h") is None
    assert pull_disqualifier("Amira-H", "maintainer", "amira-h") == (
        "pull request to the candidate's own repository"
    )
    assert pull_disqualifier("org", "AMIRA-H", "amira-h") == "merged by the candidate"
    assert pull_disqualifier("org", None, "amira-h") == "merged by an unknown account"


# --- identity -----------------------------------------------------------------------------


def test_github_links_in_a_cv():
    links = [
        "https://github.com/amira-h",
        "github.com/Amira-H/",  # same account, other case
        "https://github.com/tablekit-org/django-tablekit",
        "https://github.com/features/actions",  # a GitHub page, not an account
        "https://gist.github.com/amira-h",
        "https://amira.dev",
    ]
    assert github_accounts(links) == (["amira-h"], ["tablekit-org"])
    assert handle_from_cv(links) == "amira-h"
    assert handle_from_cv(["https://github.com/a", "https://github.com/b"]) is None
    assert handle_from_cv(["https://github.com/org/repo"]) is None  # repo links don't name one
    assert handle_from_cv([]) is None


def identity(**changes: object):
    args: dict[str, object] = {
        "login": "amira-h",
        "handle_from": "form",
        "cv_links": [],
        "site_url": None,
        "blog": None,
        "profile_email_match": False,
        "commit_email_match": False,
    }
    args.update(changes)
    return assess_identity(**args)  # type: ignore[arg-type]


def outcomes(result: object) -> dict[str, str]:
    return {s.name: s.outcome for s in result.signals}  # type: ignore[attr-defined]


def test_no_corroboration_asks_the_recruiter_to_confirm():
    result = identity()
    assert result.status == "unverified"
    assert result.needs_confirmation and result.label == CONFIRM_LABEL
    assert outcomes(result)["site_links_account"] == "not_checked"


@pytest.mark.parametrize(
    "changes",
    [
        {"cv_links": ["https://github.com/amira-h"]},
        {"cv_links": ["https://github.com/AMIRA-H/shop-api"]},
        {"blog": "amira.dev", "site_url": "https://www.amira.dev/"},
        {"blog": "https://amira.dev/blog", "cv_links": ["amira.dev/"]},
        {"commit_email_match": True},
        {"profile_email_match": True},
    ],
)
def test_any_deterministic_match_corroborates(changes: dict[str, object]):
    result = identity(**changes)
    assert result.status == "corroborated"
    assert not result.needs_confirmation and result.label is None


def test_a_cv_linking_another_account_is_a_conflict():
    result = identity(cv_links=["https://github.com/someone-else"], commit_email_match=True)
    assert result.status == "conflict"
    assert result.needs_confirmation
    assert outcomes(result)["cv_link"] == "conflict"


def test_a_handle_taken_from_the_cv_does_not_corroborate_itself():
    result = identity(handle_from="cv", cv_links=["https://github.com/amira-h"])
    assert result.status == "unverified"
    assert (
        identity(
            handle_from="cv", cv_links=["https://github.com/amira-h"], commit_email_match=True
        ).status
        == "corroborated"
    )


def test_profile_websites_on_other_hosts_or_github_itself_do_not_match():
    assert identity(blog="https://medium.com/@amira", site_url="https://amira.dev").status == (
        "unverified"
    )
    assert identity(
        blog="https://github.com/amira-h", cv_links=["https://github.com/x/y"]
    ).status == ("unverified")
