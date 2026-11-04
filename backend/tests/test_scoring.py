import itertools
import random
from datetime import date
from typing import Any

import pytest

from app.schemas.evidence import EVIDENCE_SOURCES, EvidenceItem
from app.schemas.github import DepthSignals, RepoActivity
from app.schemas.job import JobSkill, JobSpec
from app.schemas.score import ItemCredit, ScoreResult, SkillScore
from app.scoring.depth import depth_score
from app.scoring.policy import POLICY_V1
from app.scoring.scorer import (
    LABEL_MISSING,
    LABEL_MORE_EVIDENCE,
    UnscorableSpec,
    recency_factor,
    score_candidate,
)
from app.taxonomy.loader import load_taxonomy

TODAY = date(2026, 10, 15)
RECENT = date(2026, 8, 1)
OLD_4Y = date(2022, 6, 18)

TAXONOMY = load_taxonomy()


def skill(name: str, weight: float = 1.0) -> JobSkill:
    return JobSkill(raw_text=name, skill_id=f"skill:{name}", weight=weight)


def item(name: str, level: str, seen: date | None = RECENT, **kw: Any) -> EvidenceItem:
    fields: dict[str, Any] = {
        "skill_id": f"skill:{name}",
        "level": level,
        "quote": name,
        "date_last_seen": seen,
        "source": "cv",
        "locator": f"cv:doc#{name}",
        "ownership": 1.0,
        "confidence": 1.0,
    }
    return EvidenceItem(**{**fields, **kw})


POSTING = JobSpec(
    must_have=[skill("python"), skill("fastapi"), skill("postgresql")],
    nice_to_have=[skill("docker"), skill("aws"), skill("redis")],
)
AMIRA = [
    item("python", "deep", source="github", locator="pr:lib#41"),
    item("django", "deep", source="github", locator="pr:lib#41"),
    item("postgresql", "demonstrated", source="github", locator="repo:amira/shop"),
    item("docker", "demonstrated", source="github", locator="repo:amira/shop"),
]
BEN = [
    item("python", "demonstrated", OLD_4Y, source="github", locator="repo:ben/app"),
    item("fastapi", "demonstrated", OLD_4Y, source="github", locator="repo:ben/app"),
    item("postgresql", "claimed"),
    item("docker", "claimed"),
    item("aws", "claimed"),
    item("redis", "claimed"),
    # Forks and tutorial clones count 0.
    item("django", "demonstrated", source="github", locator="repo:ben/polls", ownership=0.0),
    item("flask", "demonstrated", source="github", locator="repo:ben/mega", ownership=0.0),
]
BEN_AFTER = [
    *BEN,
    item("postgresql", "demonstrated", source="github", locator="repo:ben/new"),
    item("aws", "demonstrated", source="github", locator="repo:ben/new"),
]


def score(
    evidence: list[EvidenceItem], depth: float = 0.0, spec: JobSpec = POSTING, **kw: Any
) -> ScoreResult:
    return score_candidate(spec, evidence, depth, TAXONOMY, TODAY, **kw)


def by_skill(result: ScoreResult) -> dict[str, SkillScore]:
    return {s.skill_id.removeprefix("skill:"): s for s in result.skills}


def best(skill_score: SkillScore) -> ItemCredit:
    assert skill_score.best_item is not None
    return skill_score.best_item


# --- Regression fixtures -------------------------------------------------------------------


def test_worked_example_amira():
    result = score(AMIRA, 0.8)
    assert result.total == 67.25
    skills = by_skill(result)
    assert skills["python"].strength == 1.0
    assert skills["fastapi"].strength == pytest.approx(0.7)
    assert skills["postgresql"].strength == 0.75
    assert skills["docker"].strength == 0.75
    assert skills["aws"].strength == 0 and skills["redis"].strength == 0


def test_worked_example_ben():
    result = score(BEN, 0.3)
    assert result.total == 46.5
    skills = by_skill(result)
    assert skills["python"].strength == pytest.approx(0.6)
    assert skills["fastapi"].strength == pytest.approx(0.6)  # the zero-credit Django adds nothing
    assert skills["postgresql"].strength == 0.4


def test_worked_example_ben_after_more_evidence():
    assert score(BEN_AFTER, 0.3).total == 56.42


def test_formula_components():
    result = score(AMIRA, 0.8)
    assert result.must_have_avg == pytest.approx((1.0 + 0.7 + 0.75) / 3)
    assert result.nice_to_have_avg == pytest.approx(0.75 / 3)
    assert result.depth == 0.8
    assert result.max_total == 100
    assert result.version == POLICY_V1.version


# --- Edge cases named in the build plan ----------------------------------------------------


def test_empty_nice_to_have_list():
    spec = JobSpec(must_have=[skill("python")], nice_to_have=[])
    result = score([item("python", "deep")], 1.0, spec=spec)
    assert result.nice_to_have_avg == 0
    assert result.total == 75
    assert result.max_total == 75


def test_adjacency_only_skill_counts_as_covered_with_its_source_named():
    spec = JobSpec(must_have=[skill("fastapi")], nice_to_have=[])
    result = score([item("django", "demonstrated", source="github", locator="repo:a/b")], spec=spec)
    fastapi = by_skill(result)["fastapi"]
    assert fastapi.direct == 0
    assert fastapi.strength == pytest.approx(0.75 * 0.7)
    assert (fastapi.credit_kind, fastapi.credit_skill, fastapi.similarity) == (
        "adjacent",
        "skill:django",
        0.7,
    )
    assert best(fastapi).locator == "repo:a/b"
    assert result.coverage.demonstrated == ["skill:fastapi"]
    assert result.labels == []


def test_adjacency_is_one_hop_and_directional():
    # django -> fastapi exists; fastapi -> django does not.
    spec = JobSpec(must_have=[skill("django")], nice_to_have=[])
    assert by_skill(score([item("fastapi", "deep")], spec=spec))["django"].strength == 0


def test_direct_evidence_beats_weaker_adjacency_and_the_other_way_round():
    spec = JobSpec(must_have=[skill("fastapi")], nice_to_have=[])
    both = [item("fastapi", "claimed"), item("django", "deep")]
    fastapi = by_skill(score(both, spec=spec))["fastapi"]
    assert fastapi.strength == pytest.approx(0.7)  # not 0.4: max, per the 2026-10-29 decision
    assert fastapi.direct == 0.4
    assert fastapi.coverage == "demonstrated"
    tied = [item("fastapi", "demonstrated"), item("django", "demonstrated")]
    assert by_skill(score(tied, spec=spec))["fastapi"].credit_kind == "evidence"


def test_undated_items_get_the_undated_factor():
    result = score([item("python", "demonstrated", None)])
    python = by_skill(result)["python"]
    assert python.strength == pytest.approx(0.75 * 0.8)
    assert best(python).recency == 0.8


def test_all_must_haves_missing_keeps_the_candidate_and_raises_both_labels():
    result = score([item("docker", "deep")], 0.5)
    assert result.total == pytest.approx(25 / 3 + 7.5, abs=0.01)
    assert result.needs_human_review and result.needs_more_evidence
    assert result.labels == [LABEL_MISSING, LABEL_MORE_EVIDENCE]
    assert result.coverage.missing == ["skill:python", "skill:fastapi", "skill:postgresql"]


def test_no_evidence_at_all_is_still_a_score():
    result = score([])
    assert result.total == 0
    assert result.needs_human_review


# --- Recency -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("seen", "factor"),
    [
        (date(2026, 10, 15), 1.0),
        (date(2024, 10, 15), 1.0),  # exactly 2 years
        (date(2024, 10, 14), 0.8),
        (date(2021, 10, 15), 0.8),  # exactly 5 years
        (date(2021, 10, 14), 0.8),  # older: no further penalty
        (date(2005, 1, 1), 0.8),
        (date(2027, 1, 1), 1.0),  # a future end date is not penalised
        (None, 0.8),
    ],
)
def test_recency_bands(seen, factor):
    assert recency_factor(seen, TODAY) == factor


def test_recency_on_a_leap_day():
    assert recency_factor(date(2022, 2, 28), date(2024, 2, 29)) == 1.0
    assert recency_factor(date(2022, 2, 27), date(2024, 2, 29)) == 0.8


# --- Ownership and coverage ----------------------------------------------------------------


def test_ownership_scales_a_repo_item_and_zero_proves_nothing():
    spec = JobSpec(must_have=[skill("python")], nice_to_have=[])
    half = score([item("python", "demonstrated", ownership=0.5)], spec=spec)
    assert by_skill(half)["python"].strength == 0.375
    assert half.coverage.demonstrated == ["skill:python"]
    zero = score([item("python", "deep", ownership=0.0)], spec=spec)
    assert zero.coverage.missing == ["skill:python"]
    assert zero.needs_human_review


def test_coverage_uses_the_highest_level_even_when_a_claim_is_worth_more():
    # Demonstrated at 0.2 ownership (0.15) is below a claim (0.4), but it was still shown.
    spec = JobSpec(must_have=[skill("python")], nice_to_have=[])
    both = [item("python", "demonstrated", ownership=0.2), item("python", "claimed")]
    python = by_skill(score(both, spec=spec))["python"]
    assert python.strength == 0.4
    assert best(python).level == "claimed"
    assert python.coverage == "demonstrated"


def test_claimed_only_must_have_adds_needs_more_evidence_but_not_the_gate():
    spec = JobSpec(must_have=[skill("python"), skill("sql")], nice_to_have=[])
    result = score([item("python", "deep"), item("sql", "claimed")], spec=spec)
    assert result.coverage.claimed_only == ["skill:sql"]
    assert not result.needs_human_review
    assert result.labels == [LABEL_MORE_EVIDENCE]


def test_adjacent_claim_is_claimed_only():
    spec = JobSpec(must_have=[skill("fastapi")], nice_to_have=[])
    result = score([item("django", "claimed")], spec=spec)
    assert by_skill(result)["fastapi"].strength == pytest.approx(0.28)
    assert result.coverage.claimed_only == ["skill:fastapi"]
    assert result.labels == [LABEL_MORE_EVIDENCE]


def test_nice_to_haves_do_not_affect_coverage_or_gates():
    result = score(AMIRA)
    assert result.coverage.demonstrated == ["skill:python", "skill:fastapi", "skill:postgresql"]
    assert not result.needs_human_review and not result.needs_more_evidence


# --- Weights -------------------------------------------------------------------------------


def test_averages_use_skill_weights():
    spec = JobSpec(must_have=[skill("python", 3.0), skill("sql", 1.0)], nice_to_have=[])
    result = score([item("python", "deep")], spec=spec)
    assert result.must_have_avg == pytest.approx(0.75)
    assert result.total == 45


# --- Umbrella skills -----------------------------------------------------------------------


def test_group_skill_takes_child_evidence_in_full():
    spec = JobSpec(must_have=[skill("relational-database")], nice_to_have=[])
    result = score([item("postgresql", "deep", locator="pr:x")], spec=spec)
    group = by_skill(result)["relational-database"]
    assert group.strength == 1.0
    assert (group.credit_kind, group.credit_skill) == ("child", "skill:postgresql")
    assert best(group).locator == "pr:x"
    assert result.coverage.demonstrated == ["skill:relational-database"]


def test_group_evidence_does_not_credit_children_and_siblings_do_not_mix():
    spec = JobSpec(must_have=[skill("postgresql")], nice_to_have=[])
    for other in ("relational-database", "sqlite"):
        assert by_skill(score([item(other, "deep")], spec=spec))["postgresql"].strength == 0


def test_group_takes_the_best_of_its_own_and_its_children_and_the_top_level():
    spec = JobSpec(must_have=[skill("relational-database")], nice_to_have=[])
    evidence = [item("relational-database", "claimed"), item("mysql", "demonstrated", OLD_4Y)]
    group = by_skill(score(evidence, spec=spec))["relational-database"]
    assert group.strength == pytest.approx(0.6)
    assert group.coverage == "demonstrated"


def test_siblings_credit_each_other_only_through_an_adjacency_edge():
    spec = JobSpec(must_have=[skill("postgresql")], nice_to_have=[])
    credited = by_skill(score([item("mysql", "demonstrated")], spec=spec))["postgresql"]
    assert credited.credit_kind == "adjacent"
    assert credited.strength == pytest.approx(0.75 * 0.8)


# --- Overrides -----------------------------------------------------------------------------


def test_override_replaces_the_level_and_ignores_recency_and_adjacency():
    spec = JobSpec(must_have=[skill("fastapi")], nice_to_have=[])
    evidence = [item("fastapi", "claimed", OLD_4Y), item("django", "deep")]
    raised = by_skill(score(evidence, spec=spec, overrides={"skill:fastapi": "deep"}))["fastapi"]
    assert raised.strength == 1.0
    assert raised.credit_kind == "override"
    assert raised.override_level == "deep"
    assert raised.best_item is None
    lowered = by_skill(score(evidence, spec=spec, overrides={"skill:fastapi": "claimed"}))
    assert lowered["fastapi"].strength == 0.4  # not the adjacent 0.7: the recruiter decided


def test_override_none_is_missing_and_raises_the_gate():
    spec = JobSpec(must_have=[skill("python")], nice_to_have=[])
    result = score([item("python", "deep")], spec=spec, overrides={"skill:python": "none"})
    assert by_skill(result)["python"].strength == 0
    assert result.needs_human_review


def test_override_on_a_skill_credits_the_skills_adjacent_to_it():
    spec = JobSpec(must_have=[skill("fastapi")], nice_to_have=[])
    result = score([], spec=spec, overrides={"skill:django": "demonstrated"})
    assert by_skill(result)["fastapi"].strength == pytest.approx(0.525)
    assert result.coverage.demonstrated == ["skill:fastapi"]


# --- Properties ----------------------------------------------------------------------------


@pytest.mark.parametrize("source", EVIDENCE_SOURCES)
def test_scorer_never_branches_on_source(source):
    baseline = score(AMIRA, 0.8)
    moved = [e.model_copy(update={"source": source}) for e in AMIRA]
    assert score(moved, 0.8).total == baseline.total


def test_result_does_not_depend_on_evidence_order():
    shuffled = list(BEN_AFTER)
    baseline = score(BEN_AFTER, 0.3)
    for seed in range(5):
        random.Random(seed).shuffle(shuffled)
        assert score(shuffled, 0.3) == baseline


def test_more_evidence_never_lowers_a_score():
    pool = [item("python", "claimed"), item("fastapi", "demonstrated"), item("docker", "deep")]
    for count in range(len(pool)):
        for subset in itertools.combinations(pool, count):
            assert score(list(subset)).total <= score(pool).total


def test_input_types_guarded():
    with pytest.raises(UnscorableSpec):
        score([], spec=JobSpec(must_have=[JobSkill(raw_text="Temporal")], nice_to_have=[]))
    with pytest.raises(ValueError, match="depth"):
        score([], 1.5)


# --- Depth rubric --------------------------------------------------------------------------


def signals(**kw: Any) -> DepthSignals:
    defaults: dict[str, Any] = {
        "credited_repos": 1,
        "longest_active_days": 0,
        "longest_active_repo": None,
        "repos_with_tests": [],
        "repos_with_ci": [],
        "repos_with_docker": [],
        "merged_external_prs": 0,
        "external_repos": [],
        "max_stars": 0,
    }
    return DepthSignals(**{**defaults, **kw})


def test_depth_without_signals_is_zero():
    assert depth_score(None) == 0
    assert depth_score(signals()) == 0


def test_depth_full_marks_add_up_to_one():
    full = signals(
        activity=[RepoActivity(days=400, commits=300)],
        repos_with_tests=["a/b"],
        repos_with_ci=["a/b"],
        repos_with_docker=["a/b"],
        merged_external_prs=7,
    )
    assert depth_score(full) == pytest.approx(1.0)


def test_sustained_history_needs_both_time_and_commits_in_one_repository():
    assert depth_score(signals(activity=[RepoActivity(days=365, commits=100)])) == 0.25
    # Long span, few commits, and many commits in a burst, in separate repositories.
    split = [RepoActivity(days=1000, commits=10), RepoActivity(days=30, commits=500)]
    assert depth_score(signals(activity=split)) == pytest.approx(0.25 * 0.1)
    # The best repository wins, however the longest-span one looks.
    best = [RepoActivity(days=1000, commits=12), RepoActivity(days=300, commits=500)]
    assert depth_score(signals(activity=best)) == pytest.approx(0.25 * 300 / 365)


def test_depth_ignores_stars():
    assert depth_score(signals(max_stars=50_000)) == 0


def test_external_prs_scale_to_three():
    assert depth_score(signals(merged_external_prs=1)) == pytest.approx(0.1)
    assert depth_score(signals(merged_external_prs=3)) == pytest.approx(0.3)
    assert depth_score(signals(merged_external_prs=30)) == pytest.approx(0.3)
