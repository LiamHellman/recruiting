"""Dependency manifests and config files -> dependencies and Docker images, each with a quote
copied verbatim from the file. Pure: text in, facts out; no I/O.

Manifests: requirements*.txt, pyproject.toml, package.json, go.mod, pom.xml.
Config files: Dockerfile, docker-compose / compose files (images), CI and Terraform files,
which count by their presence.
"""

import json
import re
import tomllib
from dataclasses import dataclass
from typing import Literal

from app.taxonomy.packages import Ecosystem

ManifestKind = Literal[
    "requirements", "pyproject", "package_json", "go_mod", "pom", "dockerfile", "compose"
]

# Vendored, generated or environment directories: not the candidate's work.
EXCLUDED_DIRS = frozenset(
    {
        "node_modules",
        "vendor",
        "third_party",
        "site-packages",
        ".venv",
        "venv",
        "env",
        ".tox",
        "dist",
        "build",
        "bower_components",
        "__pycache__",
    }
)

# Files whose presence is the evidence (no need to read them).
_CONFIG_FILES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(^|/)\.github/workflows/[^/]+\.ya?ml$"), "skill:github-actions"),
    (re.compile(r"(^|/)\.gitlab-ci\.ya?ml$"), "skill:gitlab-ci"),
    (re.compile(r"(^|/)Jenkinsfile$"), "skill:jenkins"),
    (re.compile(r"\.tf$"), "skill:terraform"),
)
DOCKER_SKILL = "skill:docker"

_DEV_WORDS = re.compile(r"dev|test|lint|doc|typing", re.IGNORECASE)
_DEV_GROUPS = frozenset({"dev", "test", "tests", "testing", "lint", "docs", "doc", "typing"})
_REQ_NAME = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)")


@dataclass(frozen=True)
class Dependency:
    ecosystem: Ecosystem
    name: str
    quote: str  # verbatim substring of the manifest
    dev: bool  # development/test-only dependency


@dataclass(frozen=True)
class ImageRef:
    ref: str  # e.g. "postgres:16"
    quote: str  # the verbatim line


def is_excluded(path: str) -> bool:
    return any(part in EXCLUDED_DIRS for part in path.split("/")[:-1])


def manifest_kind(path: str) -> ManifestKind | None:
    parts = path.split("/")
    name = parts[-1]
    lower = name.lower()
    if re.fullmatch(r"requirements([-_.][\w.-]+)?\.txt", lower) or (
        len(parts) > 1 and parts[-2] == "requirements" and lower.endswith(".txt")
    ):
        return "requirements"
    if name == "pyproject.toml":
        return "pyproject"
    if name == "package.json":
        return "package_json"
    if name == "go.mod":
        return "go_mod"
    if name == "pom.xml":
        return "pom"
    if lower == "dockerfile" or lower.startswith("dockerfile.") or lower.endswith(".dockerfile"):
        return "dockerfile"
    if re.fullmatch(r"(docker-)?compose([-.][\w.-]+)?\.ya?ml", lower):
        return "compose"
    return None


def config_skill(path: str) -> str | None:
    """The skill a file's presence demonstrates (CI, Terraform, Docker), or None."""
    if manifest_kind(path) in ("dockerfile", "compose"):
        return DOCKER_SKILL
    return next((skill for pattern, skill in _CONFIG_FILES if pattern.search(path)), None)


def is_test_path(path: str) -> bool:
    parts = path.split("/")
    name = parts[-1]
    return (
        any(p in ("tests", "test", "__tests__", "spec") for p in parts[:-1])
        or re.fullmatch(
            r"test_.+\.py|.+_test\.(py|go)|.+\.(test|spec)\.[jt]sx?|.+Tests?\.java", name
        )
        is not None
    )


def is_ci_path(path: str) -> bool:
    return config_skill(path) in ("skill:github-actions", "skill:gitlab-ci", "skill:jenkins") or (
        path.endswith(".circleci/config.yml")
    )


def is_docker_path(path: str) -> bool:
    return manifest_kind(path) in ("dockerfile", "compose")


def parse_dependencies(path: str, text: str) -> list[Dependency]:
    kind = manifest_kind(path)
    if kind == "requirements":
        return _requirements(text, dev=bool(_DEV_WORDS.search(path)))
    if kind == "pyproject":
        return _pyproject(text)
    if kind == "package_json":
        return _package_json(text)
    if kind == "go_mod":
        return _go_mod(text)
    if kind == "pom":
        return _pom(text)
    return []


def images(path: str, text: str) -> list[ImageRef]:
    kind = manifest_kind(path)
    if kind == "dockerfile":
        pattern = r"^[ \t]*FROM[ \t]+(?:--platform=\S+[ \t]+)?(\S+).*$"
    elif kind == "compose":
        pattern = r"""^[ \t]*image:[ \t]*["']?([^\s"'#]+).*$"""
    else:
        return []
    return [
        ImageRef(ref=m.group(1), quote=m.group(0).strip())
        for m in re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)
    ]


# --- parsers ------------------------------------------------------------------------------


def _requirements(text: str, *, dev: bool) -> list[Dependency]:
    deps = []
    for line in text.splitlines():
        body = re.split(r"\s#", line, maxsplit=1)[0].strip()
        if not body or body.startswith(("#", "-")) or "://" in body:
            continue
        if match := _REQ_NAME.match(body):
            deps.append(Dependency("pypi", match.group(1), body, dev))
    return deps


def _pep508(spec: str, text: str, dev: bool) -> Dependency | None:
    match = _REQ_NAME.match(spec.strip())
    if not match:
        return None
    if spec in text:
        quote = spec
    elif re.search(rf"\b{re.escape(match.group(1))}\b", text):
        quote = match.group(1)
    else:
        return None
    return Dependency("pypi", match.group(1), quote, dev)


def _poetry_key(name: str, text: str, dev: bool) -> Dependency | None:
    line = re.search(rf"""^[ \t]*["']?{re.escape(name)}["']?[ \t]*=.*$""", text, re.MULTILINE)
    if line is None:
        return None
    return Dependency("pypi", name, line.group(0).strip(), dev)


def _pyproject(text: str) -> list[Dependency]:
    try:
        doc = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return []
    specs: list[tuple[str, bool]] = []
    project = doc.get("project") or {}
    specs += [(s, False) for s in project.get("dependencies") or []]
    for group, items in (project.get("optional-dependencies") or {}).items():
        specs += [(s, group.lower() in _DEV_GROUPS) for s in items or []]
    for items in (doc.get("dependency-groups") or {}).values():
        specs += [(s, True) for s in items or []]
    tool = doc.get("tool") or {}
    specs += [(s, True) for s in (tool.get("uv") or {}).get("dev-dependencies") or []]

    deps = [d for s, dev in specs if isinstance(s, str) and (d := _pep508(s, text, dev))]

    poetry = tool.get("poetry") or {}
    keyed: list[tuple[str, bool]] = [(k, False) for k in poetry.get("dependencies") or {}]
    keyed += [(k, True) for k in poetry.get("dev-dependencies") or {}]
    for group, body in (poetry.get("group") or {}).items():
        keyed += [(k, group != "main") for k in (body or {}).get("dependencies") or {}]
    deps += [d for k, dev in keyed if k.lower() != "python" and (d := _poetry_key(k, text, dev))]
    return deps


def _package_json(text: str) -> list[Dependency]:
    try:
        doc = json.loads(text)
    except ValueError:
        return []
    if not isinstance(doc, dict):
        return []
    deps = []
    for section, dev in (
        ("dependencies", False),
        ("peerDependencies", False),
        ("optionalDependencies", False),
        ("devDependencies", True),
    ):
        table = doc.get(section)
        if not isinstance(table, dict):
            continue
        for name in table:
            match = re.search(rf'"{re.escape(name)}"\s*:\s*"[^"\n]*"', text)
            if match:
                deps.append(Dependency("npm", name, match.group(0), dev))
    return deps


def _go_mod(text: str) -> list[Dependency]:
    deps = []
    in_block = False
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"require\s*\($", stripped):
            in_block = True
            continue
        if in_block and stripped == ")":
            in_block = False
            continue
        if not in_block and not stripped.startswith("require "):
            continue
        body = stripped.removeprefix("require ").strip() if not in_block else stripped
        if not body or body.startswith("//") or "// indirect" in body:
            continue
        parts = body.split()
        if len(parts) >= 2:
            deps.append(Dependency("go", parts[0], stripped, False))
    return deps


_POM_BLOCK = re.compile(r"<(dependency|parent|plugin)>(.*?)</\1>", re.DOTALL)


def _pom(text: str) -> list[Dependency]:
    deps = []
    for block in _POM_BLOCK.finditer(text):
        body = block.group(2)
        group = re.search(r"<groupId>\s*([^<\s]+)\s*</groupId>", body)
        artifact = re.search(r"<artifactId>\s*([^<\s]+)\s*</artifactId>", body)
        if not group or not artifact:
            continue
        scope = re.search(r"<scope>\s*([^<\s]+)\s*</scope>", body)
        deps.append(
            Dependency(
                "maven",
                f"{group.group(1)}:{artifact.group(1)}",
                artifact.group(0),
                scope is not None and scope.group(1) == "test",
            )
        )
    return deps
