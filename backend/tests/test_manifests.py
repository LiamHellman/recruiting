import pytest

from app.extraction.manifests import (
    config_skill,
    images,
    is_excluded,
    is_test_path,
    manifest_kind,
    parse_dependencies,
)
from app.taxonomy.loader import TaxonomyError, load_taxonomy
from app.taxonomy.packages import build_catalog, image_name, load_catalog, pypi_name


def deps(path: str, text: str) -> list[tuple[str, bool]]:
    found = parse_dependencies(path, text)
    for dep in found:
        assert dep.quote in text, dep  # every quote is verbatim
    return [(d.name, d.dev) for d in found]


def test_requirements_files():
    text = (
        "# web\nDjango==5.0.6  # pinned\npsycopg[binary]>=3.1\n-r base.txt\n"
        "-e git+https://example.com/x.git#egg=x\nhttps://example.com/pkg.whl\n\nrequests\n"
    )
    assert deps("requirements.txt", text) == [
        ("Django", False),
        ("psycopg", False),
        ("requests", False),
    ]
    assert parse_dependencies("requirements.txt", text)[0].quote == "Django==5.0.6"
    assert deps("requirements-dev.txt", "pytest\n") == [("pytest", True)]
    assert deps("requirements/test.txt", "pytest\n") == [("pytest", True)]


def test_pyproject_dependencies_and_groups():
    text = """\
[project]
name = "x"
dependencies = ["fastapi>=0.110", 'sqlalchemy[asyncio]~=2.0']

[project.optional-dependencies]
dev = ["pytest>=8"]
postgres = ["asyncpg"]

[dependency-groups]
lint = ["ruff"]

[tool.poetry.dependencies]
python = "^3.12"
django = "^5.0"

[tool.poetry.group.test.dependencies]
pytest-django = "*"
"""
    assert deps("pyproject.toml", text) == [
        ("fastapi", False),
        ("sqlalchemy", False),
        ("pytest", True),
        ("asyncpg", False),
        ("ruff", True),
        ("django", False),
        ("pytest-django", True),
    ]
    assert deps("pyproject.toml", "not [valid toml") == []


def test_package_json():
    text = (
        '{"name": "x", "dependencies": {"express": "^4.18.2", "@nestjs/core": "10.0.0"},'
        ' "devDependencies": {"typescript": "5.4.0"}}'
    )
    assert deps("web/package.json", text) == [
        ("express", False),
        ("@nestjs/core", False),
        ("typescript", True),
    ]
    assert parse_dependencies("package.json", text)[0].quote == '"express": "^4.18.2"'
    assert deps("package.json", "[1, 2]") == []


def test_go_mod_skips_indirect_dependencies():
    text = """\
module example.com/svc

go 1.22

require github.com/lib/pq v1.10.9

require (
\tgithub.com/jackc/pgx/v5 v5.5.0
\tgolang.org/x/text v0.14.0 // indirect
)
"""
    assert deps("go.mod", text) == [
        ("github.com/lib/pq", False),
        ("github.com/jackc/pgx/v5", False),
    ]


def test_pom_xml():
    text = """\
<project>
  <parent>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-parent</artifactId>
  </parent>
  <dependencies>
    <dependency>
      <groupId>org.postgresql</groupId>
      <artifactId>postgresql</artifactId>
    </dependency>
    <dependency>
      <groupId>org.junit.jupiter</groupId>
      <artifactId>junit-jupiter</artifactId>
      <scope>test</scope>
    </dependency>
  </dependencies>
</project>
"""
    assert deps("pom.xml", text) == [
        ("org.springframework.boot:spring-boot-starter-parent", False),
        ("org.postgresql:postgresql", False),
        ("org.junit.jupiter:junit-jupiter", True),
    ]


def test_images_from_compose_files_and_dockerfiles():
    compose = 'services:\n  db:\n    image: "postgres:16"\n  cache:\n    image: redis:7 # x\n'
    assert [(i.ref, i.quote) for i in images("docker-compose.yml", compose)] == [
        ("postgres:16", 'image: "postgres:16"'),
        ("redis:7", "image: redis:7 # x"),
    ]
    dockerfile = "FROM --platform=linux/amd64 python:3.12-slim AS base\nFROM base\n"
    assert [i.ref for i in images("Dockerfile", dockerfile)] == ["python:3.12-slim", "base"]
    assert images("README.md", compose) == []


@pytest.mark.parametrize(
    ("ref", "name"),
    [
        ("postgres:16", "postgres"),
        ("docker.io/library/postgres:16-alpine", "postgres"),
        ("bitnami/kafka:3.7@sha256:abc", "bitnami/kafka"),
        ("mcr.microsoft.com/mssql/server:2022-latest", "mssql/server"),
        ("localhost:5000/team/app:1", "team/app"),
    ],
)
def test_image_names(ref: str, name: str):
    assert image_name(ref) == name


@pytest.mark.parametrize(
    ("path", "kind", "skill"),
    [
        ("requirements.txt", "requirements", None),
        ("backend/requirements-dev.txt", "requirements", None),
        ("pyproject.toml", "pyproject", None),
        ("web/package.json", "package_json", None),
        ("go.mod", "go_mod", None),
        ("pom.xml", "pom", None),
        ("Dockerfile", "dockerfile", "skill:docker"),
        ("deploy/api.Dockerfile", "dockerfile", "skill:docker"),
        ("docker-compose.prod.yml", "compose", "skill:docker"),
        ("compose.yaml", "compose", "skill:docker"),
        (".github/workflows/ci.yml", None, "skill:github-actions"),
        (".gitlab-ci.yml", None, "skill:gitlab-ci"),
        ("Jenkinsfile", None, "skill:jenkins"),
        ("infra/main.tf", None, "skill:terraform"),
        ("README.md", None, None),
        ("requirements.md", None, None),
    ],
)
def test_file_kinds(path: str, kind: str | None, skill: str | None):
    assert manifest_kind(path) == kind
    assert config_skill(path) == skill
    if skill is not None:
        assert skill in load_taxonomy()


def test_vendored_and_test_paths():
    assert is_excluded("node_modules/left-pad/package.json")
    assert is_excluded("app/.venv/lib/site.py")
    assert not is_excluded("app/vendor.py")
    for path in (
        "tests/test_api.py",
        "pkg/api_test.go",
        "src/app.test.ts",
        "src/test/FooTest.java",
    ):
        assert is_test_path(path), path
    assert not is_test_path("app/testing_utils.py")


def test_catalog_lookups():
    catalog = load_catalog()
    assert catalog.package("pypi", "Psycopg2_Binary") == "skill:postgresql"
    assert catalog.package("pypi", "google-cloud-storage") == "skill:gcp"
    assert catalog.package("pypi", "djangorestframework") is None  # unlisted: ignored
    assert catalog.package("npm", "@aws-sdk/client-s3") == "skill:aws"
    assert catalog.package("npm", "rest") is None  # no guessing from the alias table
    assert catalog.package("go", "github.com/jackc/pgx/v5") == "skill:postgresql"
    assert catalog.package("go", "github.com/jackc/pgxpool") is None  # segment boundary
    assert catalog.package("maven", "org.springframework.boot:spring-boot-starter-web") == (
        "skill:spring-boot"
    )
    assert catalog.image("docker.io/library/redis:7") == "skill:redis"
    assert catalog.extension("src/app.TSX") == "skill:typescript"
    assert catalog.extension("Makefile") is None
    assert catalog.extension(".bashrc") is None


def test_catalog_rejects_unknown_skills_and_sections():
    taxonomy = load_taxonomy()
    with pytest.raises(TaxonomyError) as err:
        build_catalog({"pypi": {"x": "skill:nope"}, "crates": {}}, taxonomy)
    assert any("unknown skill" in i for i in err.value.issues)
    assert any("unknown section" in i for i in err.value.issues)
    assert pypi_name("Foo_Bar.baz") == "foo-bar-baz"
