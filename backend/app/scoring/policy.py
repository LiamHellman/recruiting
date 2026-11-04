"""Every number in the published scoring formula, in one versioned place.

The weights are illustrative until calibrated with recruiters and evals. Changing any of
them means a new version and a re-run of the bias tests (Phase 8.6).
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

from app.schemas.evidence import EvidenceLevel


def _levels() -> Mapping[EvidenceLevel, float]:
    return {"claimed": 0.4, "demonstrated": 0.75, "deep": 1.0}


@dataclass(frozen=True)
class ScoringPolicy:
    version: str = "v1"
    level_values: Mapping[EvidenceLevel, float] = field(default_factory=_levels)

    # Recency: up to `recent_years` old, then up to `aging_years`, then older; and no date.
    # Older evidence takes no further penalty, because steep ones hurt older and returning
    # workers. The date stays visible to the recruiter on the evidence card.
    recent_years: int = 2
    recent_factor: float = 1.0
    aging_years: int = 5
    aging_factor: float = 0.8
    older_factor: float = 0.8
    undated_factor: float = 0.8

    must_have_points: float = 60.0
    nice_to_have_points: float = 25.0
    depth_points: float = 15.0

    # Depth rubric. Stars are left out: they measure popularity, not skill.
    depth_sustained: float = 0.25
    sustained_full_days: int = 365  # both needed in one repository for full credit
    sustained_full_commits: int = 100
    depth_tests: float = 0.20
    depth_ci: float = 0.15
    depth_docker: float = 0.10
    depth_external_prs: float = 0.30
    external_prs_full: int = 3


POLICY_V1 = ScoringPolicy()
