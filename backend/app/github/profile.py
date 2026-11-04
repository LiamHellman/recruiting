"""The recruiter-facing GitHub profile: repository and pull-request cards, and the depth
signals behind the 0-1 depth score (app/scoring/depth.py). Pure functions over a snapshot.

Only repositories with ownership > 0 and pull requests that count feed the depth signals,
so forks, learning projects and trivial PRs add nothing. Stars are reported raw for the
recruiter; the depth score leaves them out.
"""

from collections.abc import Callable, Mapping
from datetime import date, timedelta

from app.extraction.github import pull_status, repo_languages
from app.extraction.manifests import is_ci_path, is_docker_path, is_test_path
from app.github.collect import GitHubSnapshot, PullSnapshot, RepoSnapshot
from app.github.ownership import Ownership
from app.schemas.github import (
    DepthSignals,
    GitHubProfile,
    IdentityAssessment,
    PullCard,
    RepoActivity,
    RepoCard,
)
from app.taxonomy.loader import Taxonomy
from app.taxonomy.packages import Catalog

NEW_ACCOUNT_DAYS = 180


def _active_days(repo: RepoSnapshot) -> int | None:
    if repo.first_commit is None or repo.last_commit is None:
        return None
    return (repo.last_commit - repo.first_commit).days


def repo_card(repo: RepoSnapshot, own: Ownership, taxonomy: Taxonomy) -> RepoCard:
    paths = [f.path for f in repo.files]
    return RepoCard(
        repo=repo.full_name,
        locator=f"repo:{repo.full_name}@{repo.head_sha}" if repo.head_sha else None,
        fork_of=repo.parent,
        template_from=repo.template_from,
        label=own.label,
        learning=own.learning,
        ownership=own.factor,
        reason=own.reason if not repo.skipped or own.label == "fork_untouched" else repo.skipped,
        candidate_commits=repo.candidate_commits,
        total_commits=repo.total_commits,
        active_days=_active_days(repo),
        last_commit=repo.last_commit,
        languages=repo_languages(repo, taxonomy) if not repo.skipped else [],
        stars=repo.stars,
        tests=any(map(is_test_path, paths)),
        ci=any(map(is_ci_path, paths)),
        docker=any(map(is_docker_path, paths)),
    )


def pull_card(pull: PullSnapshot, login: str, catalog: Catalog) -> PullCard:
    reason = pull_status(pull, login, catalog)
    return PullCard(
        repo=pull.repo,
        number=pull.number,
        title=pull.title[:300],
        merged_on=pull.merged_at.date(),
        counts=reason is None,
        reason=reason,
    )


def depth_signals(
    snapshot: GitHubSnapshot, ownership: Mapping[str, Ownership], catalog: Catalog
) -> DepthSignals:
    credited = [r for r in snapshot.repos if not r.skipped and ownership[r.full_name].factor > 0]
    spans = [(days, r.full_name) for r in credited if (days := _active_days(r)) is not None]
    longest = max(spans, default=(0, None))
    pulls = [p for p in snapshot.pulls if pull_status(p, snapshot.account.login, catalog) is None]

    def having(test: Callable[[str], bool]) -> list[str]:
        return sorted(r.full_name for r in credited if any(test(f.path) for f in r.files))

    return DepthSignals(
        credited_repos=len(credited),
        longest_active_days=longest[0],
        longest_active_repo=longest[1],
        activity=[
            RepoActivity(days=days, commits=r.candidate_commits)
            for r in credited
            if (days := _active_days(r)) is not None
        ],
        repos_with_tests=having(is_test_path),
        repos_with_ci=having(is_ci_path),
        repos_with_docker=having(is_docker_path),
        merged_external_prs=len(pulls),
        external_repos=sorted({p.repo for p in pulls}),
        max_stars=max((r.stars for r in credited), default=0),
    )


def build_profile(
    snapshot: GitHubSnapshot,
    ownership: Mapping[str, Ownership],
    identity: IdentityAssessment,
    taxonomy: Taxonomy,
    catalog: Catalog,
    collected_on: date,
) -> GitHubProfile:
    account = snapshot.account
    return GitHubProfile(
        handle=account.login,
        collected_on=collected_on,
        new_account=collected_on - account.created_at.date() < timedelta(days=NEW_ACCOUNT_DAYS),
        identity=identity,
        repos_total=snapshot.repos_total,
        repos_analysed=len(snapshot.repos),
        repos=[repo_card(r, ownership[r.full_name], taxonomy) for r in snapshot.repos],
        pulls_total=snapshot.pulls_total,
        pulls=[pull_card(p, account.login, catalog) for p in snapshot.pulls],
        depth=depth_signals(snapshot, ownership, catalog),
    )
