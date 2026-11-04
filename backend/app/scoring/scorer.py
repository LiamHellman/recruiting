"""The published scoring formula, plus coverage, gates and labels. Pure: no I/O, no LLM.

    value(item)     = level x recency x ownership
    direct(skill)   = max value of the skill's items; a group skill also takes its children's
    strength(skill) = max(direct(skill), max over adjacent a of direct(a) x similarity(a->skill))
    total           = 60 x avg(must-haves) + 25 x avg(nice-to-haves) + 15 x depth

Averages are weighted by skill weight. Adjacency is one hop. A recruiter's level override
replaces a skill's direct value and its strength, and still credits skills adjacent to it.

The scorer sees the confirmed spec, evidence items, a depth score and overrides. It never
branches on an item's source or on anything about the person, and nothing here rejects
anyone: a must-have with no credit raises a label, and the candidate stays in the list.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date

from app.schemas.evidence import EvidenceItem, EvidenceLevel
from app.schemas.job import JobSkill, JobSpec, SkillList
from app.schemas.score import (
    Coverage,
    CoverageStatus,
    CreditKind,
    ItemCredit,
    OverrideLevel,
    ScoreResult,
    SkillScore,
)
from app.scoring.policy import POLICY_V1, ScoringPolicy
from app.taxonomy.loader import Taxonomy

LABEL_MISSING = "missing evidence: needs human review"
LABEL_MORE_EVIDENCE = "needs more evidence"

_LEVEL_RANK: dict[EvidenceLevel, int] = {"claimed": 1, "demonstrated": 2, "deep": 3}


class UnscorableSpec(ValueError):
    """The spec has unmapped skills. Only confirmed jobs, which have none, can be scored."""


def _years_before(day: date, years: int) -> date:
    try:
        return day.replace(year=day.year - years)
    except ValueError:  # 29 February
        return day.replace(year=day.year - years, day=28)


def recency_factor(seen: date | None, as_of: date, policy: ScoringPolicy = POLICY_V1) -> float:
    if seen is None:
        return policy.undated_factor
    if seen >= _years_before(as_of, policy.recent_years):
        return policy.recent_factor
    if seen >= _years_before(as_of, policy.aging_years):
        return policy.aging_factor
    return policy.older_factor


def _top_level(levels: Iterable[EvidenceLevel | None]) -> EvidenceLevel | None:
    return max(
        (lvl for lvl in levels if lvl is not None),
        key=lambda lvl: _LEVEL_RANK[lvl],
        default=None,
    )


@dataclass(frozen=True)
class _Credit:
    value: float
    level: EvidenceLevel | None  # the highest level with any credit; drives coverage
    kind: CreditKind
    skill: str | None = None
    similarity: float | None = None
    item: ItemCredit | None = None


_NO_CREDIT = _Credit(0.0, None, "none")


class _Scoring:
    def __init__(
        self,
        evidence: Iterable[EvidenceItem],
        taxonomy: Taxonomy,
        as_of: date,
        overrides: Mapping[str, OverrideLevel],
        policy: ScoringPolicy,
    ) -> None:
        self._taxonomy = taxonomy
        self.overrides = overrides
        self._policy = policy
        self._items: dict[str, list[ItemCredit]] = {}
        for item in evidence:
            recency = recency_factor(item.date_last_seen, as_of, policy)
            value = policy.level_values[item.level] * recency * item.ownership
            if value > 0:  # forks, tutorials and the like earn nothing, so prove nothing
                self._items.setdefault(item.skill_id, []).append(
                    ItemCredit(
                        source=item.source,
                        locator=item.locator,
                        level=item.level,
                        date_last_seen=item.date_last_seen,
                        recency=recency,
                        ownership=item.ownership,
                        value=value,
                    )
                )

    def _override(self, skill_id: str) -> _Credit | None:
        level = self.overrides.get(skill_id)
        if level is None:
            return None
        if level == "none":
            return _Credit(0.0, None, "override")
        return _Credit(self._policy.level_values[level], level, "override")

    def _own(self, skill_id: str) -> _Credit:
        override = self._override(skill_id)
        if override is not None:
            return override
        items = self._items.get(skill_id)
        if not items:
            return _NO_CREDIT
        # The full key makes the result independent of the order of the evidence.
        best = max(items, key=lambda i: (i.value, _LEVEL_RANK[i.level], i.locator, i.source))
        return _Credit(best.value, _top_level(i.level for i in items), "evidence", item=best)

    def direct(self, skill_id: str) -> _Credit:
        """A group skill also takes its children's credit, in full: PostgreSQL experience is
        relational-database experience. An override on the group itself wins."""
        own = self._own(skill_id)
        skill = self._taxonomy.skills.get(skill_id)
        if skill is None or not skill.group or own.kind == "override":
            return own
        best, levels = own, [own.level]
        for child in sorted(self._taxonomy.descendants(skill_id)):
            credit = self._own(child)
            levels.append(credit.level)
            if credit.value > best.value:
                best = _Credit(credit.value, None, "child", skill=child, item=credit.item)
        return _replace_level(best, _top_level(levels))

    def strength(self, skill_id: str) -> tuple[_Credit, _Credit]:
        """(direct, strength). One hop of adjacency; an override on the skill is final."""
        direct = self.direct(skill_id)
        if direct.kind == "override":
            return direct, direct
        best, levels = direct, [direct.level]
        for source, similarity in sorted(self._taxonomy.adjacent_sources(skill_id).items()):
            credit = self.direct(source)
            value = credit.value * similarity
            if value <= 0:
                continue
            levels.append(credit.level)
            if value > best.value:
                best = _Credit(value, None, "adjacent", source, similarity, credit.item)
        return direct, _replace_level(best, _top_level(levels))


def _replace_level(credit: _Credit, level: EvidenceLevel | None) -> _Credit:
    return _Credit(credit.value, level, credit.kind, credit.skill, credit.similarity, credit.item)


def _coverage(level: EvidenceLevel | None) -> CoverageStatus:
    if level is None:
        return "missing"
    return "claimed_only" if level == "claimed" else "demonstrated"


def _weighted_average(skills: list[SkillScore]) -> float:
    weight = sum(s.weight for s in skills)
    return sum(s.strength * s.weight for s in skills) / weight if weight else 0.0


def _score_skill(scoring: _Scoring, skill: JobSkill, bucket: SkillList) -> SkillScore:
    if skill.skill_id is None:
        raise UnscorableSpec(f"unmapped skill {skill.raw_text!r}")
    direct, strength = scoring.strength(skill.skill_id)
    return SkillScore(
        skill_id=skill.skill_id,
        raw_text=skill.raw_text,
        bucket=bucket,
        weight=skill.weight,
        direct=direct.value,
        strength=strength.value,
        credit_kind=strength.kind,
        credit_skill=strength.skill,
        similarity=strength.similarity,
        best_item=strength.item,
        coverage=_coverage(strength.level),
        override_level=scoring.overrides.get(skill.skill_id),
    )


def score_candidate(
    spec: JobSpec,
    evidence: Iterable[EvidenceItem],
    depth: float,
    taxonomy: Taxonomy,
    as_of: date,
    *,
    overrides: Mapping[str, OverrideLevel] | None = None,
    policy: ScoringPolicy = POLICY_V1,
) -> ScoreResult:
    if not 0 <= depth <= 1:
        raise ValueError(f"depth must be within [0, 1], got {depth}")
    scoring = _Scoring(evidence, taxonomy, as_of, overrides or {}, policy)
    must = [_score_skill(scoring, s, "must_have") for s in spec.must_have]
    nice = [_score_skill(scoring, s, "nice_to_have") for s in spec.nice_to_have]
    must_avg, nice_avg = _weighted_average(must), _weighted_average(nice)
    total = (
        policy.must_have_points * must_avg
        + policy.nice_to_have_points * nice_avg
        + policy.depth_points * depth
    )

    def having(status: CoverageStatus) -> list[str]:
        return [s.skill_id for s in must if s.coverage == status]

    coverage = Coverage(
        demonstrated=having("demonstrated"),
        claimed_only=having("claimed_only"),
        missing=having("missing"),
    )
    needs_review = bool(coverage.missing)
    needs_more = needs_review or bool(coverage.claimed_only)
    labels = [LABEL_MISSING] if needs_review else []
    if needs_more:
        labels.append(LABEL_MORE_EVIDENCE)
    max_total = policy.must_have_points + policy.depth_points
    if nice:  # with none, nice-to-haves add nothing for anyone, so the ranking is unchanged
        max_total += policy.nice_to_have_points
    return ScoreResult(
        version=policy.version,
        as_of=as_of,
        total=round(total, 2),
        max_total=max_total,
        must_have_avg=must_avg,
        nice_to_have_avg=nice_avg,
        depth=depth,
        skills=[*must, *nice],
        coverage=coverage,
        needs_human_review=needs_review,
        needs_more_evidence=needs_more,
        labels=labels,
    )
