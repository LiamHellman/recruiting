"""Load and validate the hand-curated taxonomy: skills, aliases, adjacency. No network, no DB."""

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

TAXONOMY_DIR = Path(__file__).parent
SKILL_ID_RE = re.compile(r"^skill:[a-z0-9][a-z0-9-]*$")


@dataclass(frozen=True)
class Skill:
    id: str
    name: str
    parent: str | None = None
    group: bool = False


@dataclass(frozen=True)
class Taxonomy:
    skills: Mapping[str, Skill]
    aliases: Mapping[str, str]  # term_key(alias) -> skill id; includes names and slugs
    adjacency: Mapping[tuple[str, str], float]  # (from, to) -> similarity

    def __contains__(self, skill_id: object) -> bool:
        return skill_id in self.skills

    def lookup(self, term: str) -> str | None:
        """Exact alias match after `term_key` normalization."""
        return self.aliases.get(term_key(term))

    def adjacent_sources(self, skill_id: str) -> dict[str, float]:
        """Skills whose evidence gives partial credit for `skill_id`, with similarity."""
        return {src: sim for (src, dst), sim in self.adjacency.items() if dst == skill_id}


class TaxonomyError(ValueError):
    def __init__(self, issues: list[str]) -> None:
        super().__init__("invalid taxonomy:\n  " + "\n  ".join(issues))
        self.issues = issues


def term_key(text: str) -> str:
    """Case-, width-, whitespace- and hyphen-insensitive key for matching skill names."""
    s = unicodedata.normalize("NFKC", text).casefold()
    s = s.replace("-", " ")
    s = re.sub(r"\s*/\s*", "/", s)
    s = re.sub(r"\s+", " ", s).strip()
    while True:
        trimmed = s.strip("\"'`()[]{} ").rstrip(".,;:!?")
        if trimmed == s:
            return s
        s = trimmed


def build_taxonomy(
    skills_doc: Mapping[str, Any],
    aliases_doc: Mapping[str, Any],
    adjacency_doc: Mapping[str, Any],
) -> Taxonomy:
    """Validate the three documents together and build a Taxonomy. Reports every issue."""
    issues: list[str] = []

    skills: dict[str, Skill] = {}
    for raw in skills_doc.get("skills") or []:
        sid = raw.get("id")
        name = raw.get("name")
        if not isinstance(sid, str) or not SKILL_ID_RE.match(sid):
            issues.append(f"bad skill id {sid!r}")
            continue
        if sid in skills:
            issues.append(f"duplicate skill id {sid}")
            continue
        if not isinstance(name, str) or not name.strip():
            issues.append(f"{sid}: missing name")
            continue
        skills[sid] = Skill(
            id=sid, name=name, parent=raw.get("parent"), group=bool(raw.get("group", False))
        )
    if not skills:
        issues.append("no skills defined")

    for skill in skills.values():
        if skill.parent is None:
            continue
        if skill.parent == skill.id:
            issues.append(f"{skill.id}: is its own parent")
        elif skill.parent not in skills:
            issues.append(f"{skill.id}: dangling parent {skill.parent}")
    for skill in skills.values():
        seen = {skill.id}
        node = skill
        while node.parent is not None and node.parent in skills:
            if node.parent in seen:
                issues.append(f"{skill.id}: parent cycle via {node.parent}")
                break
            seen.add(node.parent)
            node = skills[node.parent]

    aliases: dict[str, str] = {}

    def add_alias(alias: str, sid: str) -> None:
        key = term_key(alias)
        if not key:
            issues.append(f"{sid}: empty alias {alias!r}")
        elif aliases.get(key, sid) != sid:
            issues.append(f"alias {alias!r} maps to both {aliases[key]} and {sid}")
        else:
            aliases[key] = sid

    for skill in skills.values():
        add_alias(skill.name, skill.id)
        add_alias(skill.id.removeprefix("skill:"), skill.id)
    for sid, alias_list in (aliases_doc.get("aliases") or {}).items():
        if sid not in skills:
            issues.append(f"aliases for unknown skill {sid}")
            continue
        for alias in alias_list or []:
            add_alias(str(alias), sid)

    adjacency: dict[tuple[str, str], float] = {}
    for edge in adjacency_doc.get("adjacency") or []:
        src, dst, sim = edge.get("from"), edge.get("to"), edge.get("similarity")
        label = f"adjacency {src} -> {dst}"
        if src not in skills or dst not in skills:
            issues.append(f"{label}: dangling skill id")
            continue
        if src == dst:
            issues.append(f"{label}: self-loop")
            continue
        if isinstance(sim, bool) or not isinstance(sim, int | float) or not 0 <= sim <= 1:
            issues.append(f"{label}: similarity {sim!r} not within [0, 1]")
            continue
        if (src, dst) in adjacency:
            issues.append(f"{label}: duplicate edge")
            continue
        adjacency[(src, dst)] = float(sim)

    if issues:
        raise TaxonomyError(issues)
    return Taxonomy(skills=skills, aliases=aliases, adjacency=adjacency)


def _read_yaml(path: Path) -> Mapping[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, Mapping):
        raise TaxonomyError([f"{path.name}: top level must be a mapping"])
    return data


@lru_cache
def load_taxonomy(directory: Path = TAXONOMY_DIR) -> Taxonomy:
    return build_taxonomy(
        _read_yaml(directory / "skills.yaml"),
        _read_yaml(directory / "aliases.yaml"),
        _read_yaml(directory / "adjacency.yaml"),
    )
