from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

# --- Identity -----------------------------------------------------------------------------

SignalOutcome = Literal["match", "conflict", "none", "not_checked"]
IdentityStatus = Literal["corroborated", "unverified", "conflict"]
HandleSource = Literal["form", "cv"]


class IdentitySignal(BaseModel):
    name: str
    outcome: SignalOutcome
    detail: str


class IdentityAssessment(BaseModel):
    """Whether the GitHub account is the candidate's, from deterministic signals only.
    Never from names or other people-search; never merged automatically."""

    handle: str
    handle_from: HandleSource
    status: IdentityStatus
    needs_confirmation: bool
    label: str | None  # "possible match: confirm?" when the recruiter should check
    signals: list[IdentitySignal]


# --- Stored profile -----------------------------------------------------------------------


class RepoCard(BaseModel):
    repo: str
    locator: str | None  # repo:<owner>/<name>@<sha>, None when not analysed
    fork_of: str | None
    template_from: str | None
    label: str  # OwnershipLabel, or "not analysed"
    learning: bool
    ownership: float
    reason: str
    candidate_commits: int
    total_commits: int
    active_days: int | None  # first to last commit by the candidate
    last_commit: date | None
    languages: list[str]  # skill ids
    stars: int
    tests: bool
    ci: bool
    docker: bool


class PullCard(BaseModel):
    repo: str
    number: int
    title: str
    merged_on: date
    counts: bool  # peer-reviewed work in someone else's repository
    reason: str | None  # why it does not count


class DepthSignals(BaseModel):
    """Deterministic inputs to the scorer's depth score, from repos with ownership > 0 and
    qualifying merged PRs. Turning these into 0-1 is the scorer's job (an open decision)."""

    credited_repos: int
    longest_active_days: int
    longest_active_repo: str | None
    repos_with_tests: list[str]
    repos_with_ci: list[str]
    repos_with_docker: list[str]
    merged_external_prs: int
    external_repos: list[str]
    max_stars: int  # raw; a weak signal the scorer must cap


class GitHubProfile(BaseModel):
    handle: str
    collected_on: date
    # Account age is a proxy for the candidate's age, so only "is it brand new" is kept.
    new_account: bool
    identity: IdentityAssessment
    repos_total: int
    repos_analysed: int
    repos: list[RepoCard]
    pulls_total: int
    pulls: list[PullCard]
    depth: DepthSignals


# --- LLM output schema (readme_extractor) -------------------------------------------------


class ReadmeSkillMention(BaseModel):
    skill: str = Field(
        description=(
            "One technology, language, framework, library, database, platform or tool the "
            "project uses, named exactly as the README writes it."
        )
    )
    quote: str = Field(
        description=(
            "The shortest excerpt of the README, copied character for character, that says "
            "the project uses this technology. It must contain the name as written."
        )
    )


class ParsedReadme(BaseModel):
    skills: list[ReadmeSkillMention]
