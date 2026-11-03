"""How much of a repository is the candidate's own work. Pure functions, no I/O.

Rules, in order (the first that applies decides):
- a fork with no commits by the candidate since it was forked -> untouched fork, 0;
- created from a template repository -> learning project, 0;
- a tutorial-style name or topic (todo-app, *-clone, ...) -> learning project, 0;
- no commits linked to the candidate's account -> 0 (GitHub links a commit to an account
  only through a verified email; unlinked commits whose email matches the candidate's CV
  or form email count as theirs);
- exactly one commit by the candidate (a single-commit dump) -> 0;
- otherwise the candidate's share of the commits on the default branch.

Learning projects and dumps still produce evidence cards (with ownership 0) so the recruiter
sees why they earn no credit; untouched forks produce none.
"""

import re
from dataclasses import dataclass
from typing import Literal

OwnershipLabel = Literal[
    "owned",
    "fork_with_commits",
    "fork_untouched",
    "template",
    "tutorial",
    "no_linked_commits",
    "single_commit",
]

_TUTORIAL_NAMES = tuple(
    re.compile(p)
    for p in (
        r"(^|[-_.])to-?dos?([-_.]|$)",
        r"(^|[-_.])todo-?list([-_.]|$)",
        r"[-_.]clone$",
        r"^clone[-_.]",
        r"tutorial",
        r"(^|[-_.])course(work)?$",
        r"bootcamp",
        r"udemy",
        r"freecodecamp",
        r"odin-?project",
        r"(^|[-_.])cs50([-_.]|$)",
        r"hello-?world",
        r"(^|[-_.])exercises?$",
        r"100-?days",
        r"^learn(ing)?[-_.]",
        r"(^|[-_.])practice([-_.]|$)",
        r"(^|[-_.])homework([-_.]|$)",
    )
)
_TUTORIAL_TOPICS = frozenset(
    {"tutorial", "course", "learning", "bootcamp", "exercise", "exercises", "homework"}
)


@dataclass(frozen=True)
class RepoFacts:
    name: str
    fork: bool
    template_from: str | None
    topics: tuple[str, ...]
    candidate_commits: int  # since the fork was created, for forks
    total_commits: int  # on the default branch, everyone
    single_commit_additions: int | None = None  # size of the one commit, when there is one


@dataclass(frozen=True)
class Ownership:
    label: OwnershipLabel
    factor: float  # 0-1, the scorer's ownership input
    reason: str  # shown to the recruiter

    @property
    def learning(self) -> bool:
        return self.label in ("template", "tutorial", "single_commit")


def tutorial_marker(name: str, topics: tuple[str, ...] = ()) -> str | None:
    """What makes a repo look like a tutorial or course exercise, or None."""
    lower = name.lower()
    for pattern in _TUTORIAL_NAMES:
        if pattern.search(lower):
            return f"tutorial-style name '{name}'"
    topic = next((t for t in topics if t.lower() in _TUTORIAL_TOPICS), None)
    return f"topic '{topic}'" if topic else None


def assess_ownership(repo: RepoFacts) -> Ownership:
    if repo.fork and repo.candidate_commits == 0:
        return Ownership("fork_untouched", 0.0, "fork with no commits by the candidate")
    if repo.template_from:
        return Ownership(
            "template", 0.0, f"learning project: generated from template {repo.template_from}"
        )
    if marker := tutorial_marker(repo.name, repo.topics):
        return Ownership("tutorial", 0.0, f"learning project: {marker}")
    if repo.candidate_commits == 0:
        return Ownership("no_linked_commits", 0.0, "no commits linked to the candidate")
    if repo.candidate_commits == 1:
        size = (
            f" (+{repo.single_commit_additions:,} lines)"
            if repo.single_commit_additions is not None
            else ""
        )
        return Ownership("single_commit", 0.0, f"learning project: a single commit{size}")
    total = max(repo.total_commits, repo.candidate_commits)
    share = round(repo.candidate_commits / total, 2)
    label: OwnershipLabel = "fork_with_commits" if repo.fork else "owned"
    return Ownership(label, share, f"{repo.candidate_commits} of {total} commits by the candidate")


def pull_disqualifier(repo_owner: str, merged_by: str | None, login: str) -> str | None:
    """Why a merged pull request is not peer-reviewed work in someone else's repository, or
    None if it is (it then counts as `deep` evidence)."""
    if repo_owner.casefold() == login.casefold():
        return "pull request to the candidate's own repository"
    if merged_by is None:
        return "merged by an unknown account"
    if merged_by.casefold() == login.casefold():
        return "merged by the candidate"
    return None
