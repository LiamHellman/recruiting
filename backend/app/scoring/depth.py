"""Depth rubric: deterministic GitHub signals -> 0-1. Pure."""

from app.schemas.github import DepthSignals
from app.scoring.policy import POLICY_V1, ScoringPolicy


def depth_score(signals: DepthSignals | None, policy: ScoringPolicy = POLICY_V1) -> float:
    """0 when there are no signals. The candidate stays comparable on everything else, and
    `low_visibility` in the rankings says that this 0 means "not shown", not "not there"."""
    if signals is None:
        return 0.0
    sustained = max(
        (
            min(repo.days / policy.sustained_full_days, 1.0)
            * min(repo.commits / policy.sustained_full_commits, 1.0)
            for repo in signals.activity
        ),
        default=0.0,
    )
    prs = min(signals.merged_external_prs / policy.external_prs_full, 1.0)
    return (
        policy.depth_sustained * sustained
        + policy.depth_tests * bool(signals.repos_with_tests)
        + policy.depth_ci * bool(signals.repos_with_ci)
        + policy.depth_docker * bool(signals.repos_with_docker)
        + policy.depth_external_prs * prs
    )
