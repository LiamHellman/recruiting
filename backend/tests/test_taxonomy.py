from pathlib import Path
from typing import Any

import pytest
import yaml

from app.taxonomy.loader import TaxonomyError, build_taxonomy, load_taxonomy, term_key
from app.taxonomy.normalize import normalize_term

TAXONOMY_DIR = Path(__file__).parents[1] / "app" / "taxonomy"


def docs(skills: Any = None, aliases: Any = None, adjacency: Any = None) -> tuple[Any, Any, Any]:
    return (
        {
            "skills": skills
            if skills is not None
            else [
                {"id": "skill:python", "name": "Python"},
                {"id": "skill:django", "name": "Django"},
                {"id": "skill:fastapi", "name": "FastAPI"},
            ]
        },
        {"aliases": aliases or {}},
        {"adjacency": adjacency or []},
    )


def issues_of(*args) -> list[str]:
    with pytest.raises(TaxonomyError) as err:
        build_taxonomy(*args)
    return err.value.issues


# --- The shipped seed ---------------------------------------------------------------------


def test_seed_loads_without_issues():
    taxonomy = load_taxonomy(TAXONOMY_DIR)
    assert "skill:postgresql" in taxonomy
    assert len(taxonomy.skills) > 50


def test_seed_has_the_spec_adjacency_edges():
    adjacency = load_taxonomy(TAXONOMY_DIR).adjacency
    assert adjacency[("skill:django", "skill:fastapi")] == 0.7
    assert adjacency[("skill:mysql", "skill:postgresql")] == 0.8
    assert adjacency[("skill:gcp", "skill:aws")] == 0.6


def test_seed_adjacency_is_directional():
    adjacency = load_taxonomy(TAXONOMY_DIR).adjacency
    assert ("skill:fastapi", "skill:django") not in adjacency


def test_seed_aliases_from_the_build_plan():
    taxonomy = load_taxonomy(TAXONOMY_DIR)
    for alias in ("postgres", "psql", "PostgreSQL"):
        assert taxonomy.lookup(alias) == "skill:postgresql"
    assert taxonomy.lookup("k8s") == "skill:kubernetes"


def test_seed_parent_example():
    skills = load_taxonomy(TAXONOMY_DIR).skills
    assert skills["skill:fastapi"].parent == "skill:python-web-framework"
    assert skills["skill:python-web-framework"].group


def test_seed_every_alias_target_exists():
    taxonomy = load_taxonomy(TAXONOMY_DIR)
    assert set(taxonomy.aliases.values()) <= set(taxonomy.skills)


def test_seed_files_are_plain_yaml_mappings():
    for name in ("skills.yaml", "aliases.yaml", "adjacency.yaml"):
        assert isinstance(yaml.safe_load((TAXONOMY_DIR / name).read_text()), dict)


# --- Validation ---------------------------------------------------------------------------


def test_valid_minimal_taxonomy():
    taxonomy = build_taxonomy(
        *docs(
            aliases={"skill:python": ["py3"]},
            adjacency=[{"from": "skill:django", "to": "skill:fastapi", "similarity": 0.7}],
        )
    )
    assert taxonomy.lookup("PY3") == "skill:python"
    assert taxonomy.adjacent_sources("skill:fastapi") == {"skill:django": 0.7}


def test_dangling_parent():
    issues = issues_of(*docs(skills=[{"id": "skill:a", "name": "A", "parent": "skill:nope"}]))
    assert any("dangling parent" in i for i in issues)


def test_parent_cycle():
    skills = [
        {"id": "skill:a", "name": "A", "parent": "skill:b"},
        {"id": "skill:b", "name": "B", "parent": "skill:a"},
    ]
    assert any("cycle" in i for i in issues_of(*docs(skills=skills)))


def test_bad_and_duplicate_ids():
    skills = [
        {"id": "Skill:Python", "name": "Python"},
        {"id": "skill:a", "name": "A"},
        {"id": "skill:a", "name": "A again"},
    ]
    issues = issues_of(*docs(skills=skills))
    assert any("bad skill id" in i for i in issues)
    assert any("duplicate skill id" in i for i in issues)


def test_alias_for_unknown_skill():
    issues = issues_of(*docs(aliases={"skill:nope": ["x"]}))
    assert any("unknown skill" in i for i in issues)


def test_alias_conflict():
    issues = issues_of(*docs(aliases={"skill:python": ["fast"], "skill:fastapi": ["Fast"]}))
    assert any("maps to both" in i for i in issues)


@pytest.mark.parametrize(
    ("edge", "message"),
    [
        ({"from": "skill:django", "to": "skill:nope", "similarity": 0.5}, "dangling"),
        ({"from": "skill:nope", "to": "skill:django", "similarity": 0.5}, "dangling"),
        ({"from": "skill:django", "to": "skill:django", "similarity": 0.5}, "self-loop"),
        ({"from": "skill:django", "to": "skill:fastapi", "similarity": 1.2}, "within [0, 1]"),
        ({"from": "skill:django", "to": "skill:fastapi", "similarity": -0.1}, "within [0, 1]"),
        ({"from": "skill:django", "to": "skill:fastapi", "similarity": "0.7"}, "within [0, 1]"),
        ({"from": "skill:django", "to": "skill:fastapi", "similarity": True}, "within [0, 1]"),
    ],
)
def test_bad_adjacency(edge, message):
    assert any(message in i for i in issues_of(*docs(adjacency=[edge])))


def test_duplicate_edge():
    edge = {"from": "skill:django", "to": "skill:fastapi", "similarity": 0.7}
    assert any("duplicate edge" in i for i in issues_of(*docs(adjacency=[edge, edge])))


def test_all_issues_reported_at_once():
    issues = issues_of(
        *docs(
            aliases={"skill:nope": ["x"]},
            adjacency=[{"from": "skill:django", "to": "skill:fastapi", "similarity": 2}],
        )
    )
    assert len(issues) == 2


# --- Normalization ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("term", "expected"),
    [
        ("Postgres", "skill:postgresql"),
        ("  PSQL ", "skill:postgresql"),
        ("PostgreSQL database", "skill:postgresql"),
        ("Python 3.12", "skill:python"),
        ("Java 17+", "skill:java"),
        ("K8s", "skill:kubernetes"),
        ("CI / CD", "skill:ci-cd"),
        ("Node.js", "skill:nodejs"),
        ("C#", "skill:csharp"),
        ("C++", "skill:cpp"),
        ("C", "skill:c"),
        ("Go programming language", "skill:go"),
        ("Golang", "skill:go"),
        ("RESTful APIs", "skill:rest-api"),
        ("Test-driven development", "skill:automated-testing"),
        ("Ruby on Rails", "skill:rails"),
        ('"Docker".', "skill:docker"),
        ("Spring Boot", "skill:spring-boot"),
        ("relational databases", "skill:relational-database"),
    ],
)
def test_normalize_known_terms(term, expected):
    assert normalize_term(term, load_taxonomy(TAXONOMY_DIR)) == expected


@pytest.mark.parametrize("term", ["Temporal", "AWS or GCP", "Vue.js", "", "   ", "S3", "R"])
def test_unknown_terms_stay_unmapped(term):
    assert normalize_term(term, load_taxonomy(TAXONOMY_DIR)) is None


def test_term_key():
    assert term_key("  Test-Driven   Development. ") == "test driven development"
    assert term_key("CI / CD") == "ci/cd"
    assert term_key(".NET") == ".net"
