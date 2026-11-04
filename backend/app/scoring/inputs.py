"""From stored evidence and recruiter overrides to the scorer's inputs. Pure.

This is the one place source-specific rules live, so that the scorer itself never branches
on `source`: rejecting a GitHub account removes its evidence and depth signals here.

Override targets, by kind:
    level:    {"skill_id": ..., "level": "claimed" | "demonstrated" | "deep" | "none" | None}
              None removes the override.
    evidence: {"skill_id": ..., "source": ..., "locator": ..., "wrong": bool}
    identity: {"decision": "confirmed" | "rejected"}
    note:     {} (the text is the override's reason; notes never affect a score)
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from app.schemas.evidence import EvidenceItem
from app.schemas.github import DepthSignals
from app.schemas.score import OverrideLevel
from app.scoring.depth import depth_score
from app.scoring.policy import POLICY_V1, ScoringPolicy

IdentityDecision = Literal["confirmed", "rejected"]
EvidenceKey = tuple[str, str, str]  # skill_id, source, locator: stable across re-runs


def evidence_key(item: EvidenceItem) -> EvidenceKey:
    return (item.skill_id, item.source, item.locator)


@dataclass(frozen=True)
class ActiveOverrides:
    levels: Mapping[str, OverrideLevel]
    wrong: frozenset[EvidenceKey]
    identity: IdentityDecision | None


def fold_overrides(events: Iterable[tuple[str, Mapping[str, Any]]]) -> ActiveOverrides:
    """Reduce (kind, target) events, oldest first: the latest for each target wins."""
    levels: dict[str, OverrideLevel] = {}
    wrong: set[EvidenceKey] = set()
    identity: IdentityDecision | None = None
    for kind, target in events:
        if kind == "level":
            if target["level"] is None:
                levels.pop(target["skill_id"], None)
            else:
                levels[target["skill_id"]] = target["level"]
        elif kind == "evidence":
            key = (target["skill_id"], target["source"], target["locator"])
            if target["wrong"]:
                wrong.add(key)
            else:
                wrong.discard(key)
        elif kind == "identity":
            identity = cast(IdentityDecision, target["decision"])
    return ActiveOverrides(levels, frozenset(wrong), identity)


@dataclass(frozen=True)
class ScoreInputs:
    evidence: list[EvidenceItem]
    depth: float
    has_depth_signals: bool


def prepare_inputs(
    evidence: Iterable[EvidenceItem],
    signals: DepthSignals | None,
    overrides: ActiveOverrides,
    policy: ScoringPolicy = POLICY_V1,
) -> ScoreInputs:
    """Drop evidence a recruiter marked wrong. A rejected GitHub account contributes nothing.
    An account awaiting confirmation does count: the candidate gave the handle, which is the
    same standing as a CV claim, and it is flagged for the recruiter."""
    rejected = overrides.identity == "rejected"
    kept = [
        item
        for item in evidence
        if evidence_key(item) not in overrides.wrong and not (rejected and item.source == "github")
    ]
    usable = None if rejected else signals
    return ScoreInputs(kept, depth_score(usable, policy), usable is not None)
