"""GitHub snapshot -> evidence items. Pure: no I/O, no LLM (README mentions arrive already
extracted and validated).

Per analysed repository (untouched forks and empty repos give nothing), `demonstrated`:
- each dependency in a manifest, and each Docker image in a compose file or Dockerfile,
  that packages.yaml maps to a skill (quote: the dependency or image line);
- config files whose presence is the evidence: Dockerfile/compose (Docker), CI workflows,
  Terraform (quote: the path);
- each programming language with at least 10% or 20 KB of the repo's code by GitHub's count
  (quote: the path of the largest file in that language);
then `claimed`: technologies the README says the project uses.
All with the repo's ownership factor (0 for learning projects) and the candidate's last
commit date in it. Locators: `repo:<owner>/<name>@<sha>:<path>` (+ `#start-end` in a README).

Per merged pull request into someone else's repository, merged by someone else, `deep`:
- each language with at least 10 changed lines in the PR (quote: the changed file's path);
- the base repository's runtime dependencies (not dev/test) in those languages' ecosystems,
  read at the merge commit (quote: the dependency line).
Ownership 1.0, dated at the merge. Locators: `pr:<owner>/<name>#<number>@<sha>:<path>`.

One item per skill per repository: the strongest level, then the first found. Every quote
goes through the shared validator against the text it was taken from.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date

from app.extraction.manifests import config_skill, images, is_test_path, parse_dependencies
from app.extraction.quotes import check_quote
from app.extraction.readme import ReadmeBuild
from app.github.collect import GitHubSnapshot, PullSnapshot, RepoSnapshot
from app.github.ownership import Ownership, pull_disqualifier
from app.schemas.evidence import EvidenceItem, EvidenceLevel, RejectedItem
from app.taxonomy.loader import Taxonomy
from app.taxonomy.packages import Catalog, Ecosystem

LANGUAGE_MIN_SHARE = 0.10
LANGUAGE_MIN_BYTES = 20_000
PR_MIN_CODE_LINES = 10
DETERMINISTIC_CONFIDENCE = 1.0
README_CONFIDENCE = 1.0  # validated quote that names the skill, as for CVs

_LANGUAGE_ECOSYSTEM: Mapping[str, Ecosystem] = {
    "skill:python": "pypi",
    "skill:javascript": "npm",
    "skill:typescript": "npm",
    "skill:go": "go",
    "skill:java": "maven",
    "skill:kotlin": "maven",
    "skill:scala": "maven",
}
_RANK: Mapping[EvidenceLevel, int] = {"claimed": 0, "demonstrated": 1, "deep": 2}


def repo_locator(full_name: str, sha: str, path: str, span: tuple[int, int] | None = None) -> str:
    suffix = f"#{span[0]}-{span[1]}" if span else ""
    return f"repo:{full_name}@{sha}:{path}{suffix}"


def pr_locator(full_name: str, number: int, sha: str, path: str) -> str:
    return f"pr:{full_name}#{number}@{sha}:{path}"


def language_skill(language: str, taxonomy: Taxonomy) -> str | None:
    """A GitHub (linguist) language name -> a programming-language skill, or None."""
    skill_id = taxonomy.lookup(language)
    if skill_id is None or taxonomy.skills[skill_id].parent != "skill:programming-language":
        return None
    return skill_id


def repo_languages(repo: RepoSnapshot, taxonomy: Taxonomy) -> list[str]:
    total = sum(repo.languages.values())
    found = []
    for language, size in sorted(repo.languages.items(), key=lambda kv: -kv[1]):
        skill_id = language_skill(language, taxonomy)
        if (
            skill_id
            and total
            and (size / total >= LANGUAGE_MIN_SHARE or size >= LANGUAGE_MIN_BYTES)
        ):
            found.append(skill_id)
    return found


def pull_code_lines(pull: PullSnapshot, catalog: Catalog) -> dict[str, tuple[int, str]]:
    """Changed lines per language in a PR, with the most-changed file in that language."""
    lines: dict[str, tuple[int, str, int]] = {}
    for f in pull.files:
        skill_id = catalog.extension(f.path)
        if skill_id is None:
            continue
        changed = f.additions + f.deletions
        total, best, best_changed = lines.get(skill_id, (0, f.path, -1))
        if changed > best_changed:
            best, best_changed = f.path, changed
        lines[skill_id] = (total + changed, best, best_changed)
    return {sid: (total, best) for sid, (total, best, _) in lines.items()}


def pull_status(pull: PullSnapshot, login: str, catalog: Catalog) -> str | None:
    """None if the PR counts as peer-reviewed work; otherwise why it does not."""
    if reason := pull_disqualifier(pull.repo_owner, pull.merged_by, login):
        return reason
    if pull.merge_sha is None:
        return "no merge commit"
    if not any(n >= PR_MIN_CODE_LINES for n, _ in pull_code_lines(pull, catalog).values()):
        return f"fewer than {PR_MIN_CODE_LINES} changed lines of code"
    return None


@dataclass
class GitHubBuild:
    evidence: list[EvidenceItem] = field(default_factory=list)
    checked: int = 0
    rejected: list[RejectedItem] = field(default_factory=list)
    unmapped_terms: list[str] = field(default_factory=list)
    duplicates_merged: int = 0


@dataclass(frozen=True)
class _Where:
    bucket: str  # one item per skill per bucket (a repository, or PRs to one repository)
    last_seen: date | None
    ownership: float


class _Builder:
    def __init__(self) -> None:
        self.build = GitHubBuild()
        self._items: dict[tuple[str, str], EvidenceItem] = {}

    def add(
        self,
        where: _Where,
        *,
        skill_id: str,
        level: EvidenceLevel,
        locator: str,
        quote: str,
        source_text: str,
        confidence: float = DETERMINISTIC_CONFIDENCE,
        validated: bool = False,
    ) -> None:
        if not validated:
            self.build.checked += 1
            check = check_quote(source_text, quote)
            if check.span is None:
                self.build.rejected.append(
                    RejectedItem(item=skill_id, quote=quote, reason=str(check.reason))
                )
                return
        key = (skill_id, where.bucket)
        current = self._items.get(key)
        if current is not None:
            self.build.duplicates_merged += 1
            if _RANK[level] <= _RANK[current.level]:
                return
        self._items[key] = EvidenceItem(
            skill_id=skill_id,
            level=level,
            source="github",
            locator=locator,
            quote=quote,
            date_last_seen=where.last_seen,
            ownership=where.ownership,
            confidence=confidence,
        )

    def finish(self) -> GitHubBuild:
        self.build.evidence = list(self._items.values())
        return self.build


@dataclass(frozen=True)
class RepoReadme:
    path: str
    build: ReadmeBuild


def build_github_evidence(
    snapshot: GitHubSnapshot,
    ownership: Mapping[str, Ownership],
    readmes: Mapping[str, RepoReadme],
    taxonomy: Taxonomy,
    catalog: Catalog,
) -> GitHubBuild:
    b = _Builder()
    unmapped: dict[str, str] = {}

    for repo in snapshot.repos:
        own = ownership[repo.full_name]
        if repo.skipped or repo.head_sha is None or own.label == "fork_untouched":
            continue
        sha, full = repo.head_sha, repo.full_name
        where = _Where(full, repo.last_commit, own.factor)
        listing = "\n".join(f.path for f in repo.files)

        readme = readmes.get(full)
        manifests = sorted(
            (path, text)
            for path, text in repo.contents.items()
            if readme is None or path != readme.path
        )
        for path, text in manifests:  # dependencies first: the most specific citation
            for dep in parse_dependencies(path, text):
                if skill_id := catalog.package(dep.ecosystem, dep.name):
                    b.add(
                        where,
                        skill_id=skill_id,
                        level="demonstrated",
                        locator=repo_locator(full, sha, path),
                        quote=dep.quote,
                        source_text=text,
                    )
        for path, text in manifests:
            for image in images(path, text):
                if skill_id := catalog.image(image.ref):
                    b.add(
                        where,
                        skill_id=skill_id,
                        level="demonstrated",
                        locator=repo_locator(full, sha, path),
                        quote=image.quote,
                        source_text=text,
                    )
        for f in repo.files:
            if skill_id := config_skill(f.path):
                b.add(
                    where,
                    skill_id=skill_id,
                    level="demonstrated",
                    locator=repo_locator(full, sha, f.path),
                    quote=f.path,
                    source_text=listing,
                )
        for skill_id in repo_languages(repo, taxonomy):
            in_language = [f for f in repo.files if catalog.extension(f.path) == skill_id]
            if not in_language:
                continue
            cited = max(in_language, key=lambda f: (not is_test_path(f.path), f.size))
            b.add(
                where,
                skill_id=skill_id,
                level="demonstrated",
                locator=repo_locator(full, sha, cited.path),
                quote=cited.path,
                source_text=listing,
            )
        if readme is not None:
            b.build.checked += readme.build.checked
            b.build.rejected += readme.build.rejected
            for term in readme.build.unmapped_terms:
                unmapped.setdefault(term.casefold(), term)
            for mention in readme.build.mentions:
                b.add(
                    where,
                    skill_id=mention.skill_id,
                    level="claimed",
                    locator=repo_locator(full, sha, readme.path, mention.span),
                    quote=mention.quote,
                    source_text="",
                    confidence=README_CONFIDENCE,
                    validated=True,
                )

    login = snapshot.account.login
    for pull in sorted(snapshot.pulls, key=lambda p: p.merged_at, reverse=True):
        if pull_status(pull, login, catalog) is not None or pull.merge_sha is None:
            continue
        where = _Where(f"pr:{pull.repo}", pull.merged_at.date(), 1.0)
        changed = "\n".join(f.path for f in pull.files)
        ecosystems: set[Ecosystem] = set()
        for skill_id, (lines, path) in sorted(pull_code_lines(pull, catalog).items()):
            if lines < PR_MIN_CODE_LINES:
                continue
            if skill_id in _LANGUAGE_ECOSYSTEM:
                ecosystems.add(_LANGUAGE_ECOSYSTEM[skill_id])
            b.add(
                where,
                skill_id=skill_id,
                level="deep",
                locator=pr_locator(pull.repo, pull.number, pull.merge_sha, path),
                quote=path,
                source_text=changed,
            )
        for path, text in sorted(pull.manifests.items()):
            for dep in parse_dependencies(path, text):
                if dep.dev or dep.ecosystem not in ecosystems:
                    continue
                if skill_id := catalog.package(dep.ecosystem, dep.name):
                    b.add(
                        where,
                        skill_id=skill_id,
                        level="deep",
                        locator=pr_locator(pull.repo, pull.number, pull.merge_sha, path),
                        quote=dep.quote,
                        source_text=text,
                    )

    build = b.finish()
    build.unmapped_terms = sorted(unmapped.values(), key=str.casefold)
    return build
