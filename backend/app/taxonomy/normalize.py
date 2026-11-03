"""Deterministic mapping from a free-text skill term to a canonical skill id."""

import re

from app.taxonomy.loader import Taxonomy, term_key

# "Python 3.12", "Java 17+", "PHP 8.x", "Vue v3"
_VERSION_SUFFIX = re.compile(r"\s+v?\d+(?:\.\d+)*(?:\.x|\+)?$")
# Generic words people append to a name: "PostgreSQL database", "Go programming language".
_GENERIC_SUFFIXES = (
    " programming language",
    " language",
    " databases",
    " database",
    " db",
    " framework",
)


def normalize_term(term: str, taxonomy: Taxonomy) -> str | None:
    """Return the canonical skill id for `term`, or None if it is unknown.

    Tries, in order: the exact alias; the term without a trailing version number; the term
    without a generic trailing word. Nothing fuzzy: an unknown term stays unmapped for a
    human to resolve.
    """
    key = term_key(term)
    if not key:
        return None
    candidates = [key]
    unversioned = _VERSION_SUFFIX.sub("", key).strip()
    candidates.append(unversioned)
    for suffix in _GENERIC_SUFFIXES:
        if unversioned.endswith(suffix):
            candidates.append(unversioned.removesuffix(suffix).strip())
            break
    for candidate in candidates:
        if candidate and (skill_id := taxonomy.aliases.get(candidate)):
            return skill_id
    return None
