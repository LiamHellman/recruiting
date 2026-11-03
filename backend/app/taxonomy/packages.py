"""Load and query packages.yaml: dependency, image and file-extension names -> skill ids."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml

from app.taxonomy.loader import TAXONOMY_DIR, Taxonomy, TaxonomyError, load_taxonomy

Ecosystem = Literal["pypi", "npm", "go", "maven"]
SECTIONS = ("pypi", "npm", "go", "maven", "images", "extensions")


def pypi_name(name: str) -> str:
    """PEP 503 normalization."""
    return re.sub(r"[-_.]+", "-", name).lower()


def image_name(ref: str) -> str:
    """`docker.io/library/postgres:16@sha256:...` -> `postgres`; `bitnami/kafka:3` ->
    `bitnami/kafka`. Registry hosts (a first segment with a dot, a colon or `localhost`)
    and `library/` are dropped."""
    ref = ref.split("@", 1)[0]
    parts = ref.split("/")
    if len(parts) > 1 and ("." in parts[0] or ":" in parts[0] or parts[0] == "localhost"):
        parts = parts[1:]
    if parts and parts[0] == "library" and len(parts) > 1:
        parts = parts[1:]
    last = parts[-1].split(":", 1)[0] if parts else ""
    return "/".join([*parts[:-1], last]).lower()


@dataclass(frozen=True)
class Catalog:
    sections: Mapping[str, Mapping[str, str]]

    def package(self, ecosystem: Ecosystem, name: str) -> str | None:
        table = self.sections[ecosystem]
        if ecosystem == "pypi":
            key = pypi_name(name)
            if key in table:
                return table[key]
            return next(
                (
                    sid
                    for pat, sid in table.items()
                    if pat.endswith("*") and key.startswith(pat[:-1])
                ),
                None,
            )
        if ecosystem == "npm":
            key = name.lower()
            if key in table:
                return table[key]
            if key.startswith("@") and "/" in key:
                return table.get(key.split("/", 1)[0] + "/*")
            return None
        if ecosystem == "go":
            # Longest matching module prefix, on path-segment boundaries.
            matches = [p for p in table if name == p or name.startswith(p + "/")]
            return table[max(matches, key=len)] if matches else None
        group = name.split(":", 1)[0]
        return table.get(name) or table.get(f"{group}:*")

    def image(self, ref: str) -> str | None:
        return self.sections["images"].get(image_name(ref))

    def extension(self, path: str) -> str | None:
        name = path.rsplit("/", 1)[-1]
        if "." not in name.lstrip("."):
            return None
        return self.sections["extensions"].get(name.rsplit(".", 1)[-1].lower())


def build_catalog(doc: Mapping[str, Any], taxonomy: Taxonomy) -> Catalog:
    issues: list[str] = []
    sections: dict[str, dict[str, str]] = {}
    for section in SECTIONS:
        table = doc.get(section) or {}
        if not isinstance(table, Mapping):
            issues.append(f"{section}: must be a mapping")
            continue
        sections[section] = {}
        for raw_name, skill_id in table.items():
            name = str(raw_name)
            if skill_id not in taxonomy:
                issues.append(f"{section}: {name} -> unknown skill {skill_id!r}")
                continue
            key = pypi_name(name) if section == "pypi" else name
            key = key if section in ("go", "maven") else key.lower()
            if key in sections[section]:
                issues.append(f"{section}: {name} listed twice")
            sections[section][key] = str(skill_id)
    for extra in set(doc) - set(SECTIONS):
        issues.append(f"unknown section {extra!r}")
    if issues:
        raise TaxonomyError(issues)
    return Catalog(sections=sections)


@lru_cache
def load_catalog(directory: Path = TAXONOMY_DIR) -> Catalog:
    doc = yaml.safe_load((directory / "packages.yaml").read_text(encoding="utf-8")) or {}
    return build_catalog(doc, load_taxonomy(directory))
