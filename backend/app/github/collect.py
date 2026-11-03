"""Collect what GitHub says about one candidate-submitted account into a `GitHubSnapshot`.

Only the account the candidate gave is read: its profile, the repositories it owns, the
candidate's commits in them, and the candidate's merged pull requests (plus, for those
merged into someone else's repository, that repository's dependency files at the merge
commit). No search, no other users.

Each answer is reduced to the fields used below before it is cached (see `GitHubClient`).
Emails are compared, never kept: commit and contributor emails become a yes/no "matches the
candidate's email".
"""

import asyncio
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from typing import Any
from urllib.parse import parse_qs, quote, urlsplit

from app.extraction.manifests import is_excluded, manifest_kind
from app.github.client import GitHubClient, GitHubError, GitHubNotFound, api_path, repo_path
from app.github.ownership import RepoFacts, pull_disqualifier

README = re.compile(r"readme(\.(md|markdown|rst|txt))?", re.IGNORECASE)

PULLS_QUERY = """
query($login: String!, $after: String) {
  user(login: $login) {
    pullRequests(states: [MERGED], first: 50, after: $after,
                 orderBy: {field: CREATED_AT, direction: DESC}) {
      totalCount
      pageInfo { hasNextPage endCursor }
      nodes {
        number
        title
        mergedAt
        mergedBy { login }
        mergeCommit { oid }
        repository { nameWithOwner isPrivate owner { login } }
        files(first: 100) { totalCount nodes { path additions deletions } }
      }
    }
  }
}
"""


@dataclass(frozen=True)
class Limits:
    max_repos: int = 30  # most recently pushed owned repos
    max_files_per_repo: int = 12  # manifests, compose/Dockerfiles and the README
    max_file_bytes: int = 256_000
    max_path_depth: int = 4  # manifests deeper than this are ignored
    max_tree_entries: int = 20_000
    max_contributor_pages: int = 5
    max_pulls: int = 100
    max_pull_repos: int = 10  # external repos whose manifests are read
    max_readme_chars: int = 30_000


@dataclass(frozen=True)
class TreeFile:
    path: str
    size: int


@dataclass(frozen=True)
class AccountSnapshot:
    login: str
    id: int
    created_at: datetime
    blog: str | None
    email_match: bool  # the public profile email is one of the candidate's emails
    public_repos: int


@dataclass(frozen=True)
class RepoSnapshot:
    full_name: str
    fork: bool
    parent: str | None
    template_from: str | None
    topics: tuple[str, ...]
    created_at: datetime
    stars: int
    default_branch: str
    head_sha: str | None = None
    linked_commits: int = 0  # by the account (since forking, for forks)
    email_commits: int = 0  # unlinked, but with one of the candidate's emails
    total_commits: int = 0
    first_commit: date | None = None
    last_commit: date | None = None
    commit_email_match: bool = False
    single_commit_additions: int | None = None
    languages: dict[str, int] = field(default_factory=dict)
    files: tuple[TreeFile, ...] = ()  # blobs outside vendored/generated directories
    tree_truncated: bool = False
    contents: dict[str, str] = field(default_factory=dict)  # path -> text
    skipped: str | None = None  # why it was not analysed (empty, untouched fork)

    @property
    def name(self) -> str:
        return self.full_name.split("/", 1)[1]

    @property
    def candidate_commits(self) -> int:
        return self.linked_commits + self.email_commits

    def facts(self) -> RepoFacts:
        return RepoFacts(
            name=self.name,
            fork=self.fork,
            template_from=self.template_from,
            topics=self.topics,
            candidate_commits=self.candidate_commits,
            total_commits=self.total_commits,
            single_commit_additions=self.single_commit_additions,
        )


@dataclass(frozen=True)
class PullFile:
    path: str
    additions: int
    deletions: int


@dataclass(frozen=True)
class PullSnapshot:
    repo: str
    repo_owner: str
    number: int
    title: str
    merged_at: datetime
    merged_by: str | None
    merge_sha: str | None
    files: tuple[PullFile, ...]
    files_total: int
    # The base repository's dependency files at the merge commit (latest PR per repo only).
    manifests: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class GitHubSnapshot:
    account: AccountSnapshot
    repos: tuple[RepoSnapshot, ...]
    repos_total: int
    pulls: tuple[PullSnapshot, ...]
    pulls_total: int


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class Collector:
    def __init__(
        self,
        client: GitHubClient,
        handle: str,
        emails: Iterable[str],
        limits: Limits,
        *,
        pull_counts: Callable[["PullSnapshot", str], bool] | None = None,
    ) -> None:
        self._client = client
        self._handle = handle
        self._emails = frozenset(e.casefold() for e in emails if e)
        self._limits = limits
        # Whether a merged PR counts as the candidate's peer-reviewed work; only those have
        # their base repository's manifests read.
        self._pull_counts = pull_counts or (
            lambda p, login: pull_disqualifier(p.repo_owner, p.merged_by, login) is None
        )

    def _mine(self, email: Any) -> bool:
        return isinstance(email, str) and email.casefold() in self._emails

    # --- projections: keep only what is used ----------------------------------------

    def _project_user(self, d: dict[str, Any]) -> dict[str, Any]:
        return {
            "login": d["login"],
            "id": d["id"],
            "type": d.get("type"),
            "created_at": d["created_at"],
            "blog": d.get("blog") or None,
            "email_match": self._mine(d.get("email")),
            "public_repos": d.get("public_repos", 0),
        }

    @staticmethod
    def _project_repo_list(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{"full_name": r["full_name"], "fork": r["fork"]} for r in items]

    @staticmethod
    def _project_repo(d: dict[str, Any]) -> dict[str, Any]:
        return {
            "full_name": d["full_name"],
            "fork": d["fork"],
            "parent": (d.get("parent") or {}).get("full_name"),
            "template_from": (d.get("template_repository") or {}).get("full_name"),
            "topics": d.get("topics") or [],
            "created_at": d["created_at"],
            "stars": d.get("stargazers_count", 0),
            "default_branch": d.get("default_branch") or "main",
        }

    def _project_commits(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "sha": c["sha"],
                "date": c["commit"]["author"]["date"],
                "email_match": self._mine(c["commit"]["author"].get("email")),
            }
            for c in items
        ]

    def _project_contributors(self, items: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        handle = self._handle.casefold()
        return [
            {
                "contributions": c.get("contributions", 0),
                "linked": str(c.get("login") or "").casefold() == handle,
                "email_match": c.get("type") == "Anonymous" and self._mine(c.get("email")),
            }
            for c in items or []
        ]

    def _project_tree(self, d: dict[str, Any]) -> dict[str, Any]:
        files = [
            [e["path"], e.get("size", 0)]
            for e in d.get("tree") or []
            if e.get("type") == "blob" and not is_excluded(e["path"])
        ]
        return {
            "truncated": bool(d.get("truncated")) or len(files) > self._limits.max_tree_entries,
            "files": files[: self._limits.max_tree_entries],
        }

    @staticmethod
    def _project_pulls(d: dict[str, Any]) -> dict[str, Any]:
        conn = ((d or {}).get("user") or {}).get("pullRequests") or {}
        return {
            "total": conn.get("totalCount", 0),
            "next": (conn.get("pageInfo") or {}).get("endCursor")
            if (conn.get("pageInfo") or {}).get("hasNextPage")
            else None,
            "pulls": [
                {
                    "repo": n["repository"]["nameWithOwner"],
                    "repo_owner": n["repository"]["owner"]["login"],
                    "private": n["repository"].get("isPrivate", False),
                    "number": n["number"],
                    "title": n["title"],
                    "merged_at": n["mergedAt"],
                    "merged_by": (n.get("mergedBy") or {}).get("login"),
                    "merge_sha": (n.get("mergeCommit") or {}).get("oid"),
                    "files_total": (n.get("files") or {}).get("totalCount", 0),
                    "files": [
                        [f["path"], f.get("additions", 0), f.get("deletions", 0)]
                        for f in (n.get("files") or {}).get("nodes") or []
                    ],
                }
                for n in conn.get("nodes") or []
            ],
        }

    # --- collection -----------------------------------------------------------------

    async def collect(self) -> GitHubSnapshot:
        account = await self._account()
        listing = await self._client.get_json(
            f"/users/{quote(account.login, safe='')}/repos",
            {"type": "owner", "sort": "pushed", "direction": "desc", "per_page": 100},
            project=self._project_repo_list,
        )
        items = list(listing.data or [])
        found = await asyncio.gather(
            *(self._repo(item["full_name"]) for item in items[: self._limits.max_repos])
        )
        repos = tuple(r for r in found if r is not None)
        pulls, pulls_total = await self._pulls(account.login)
        return GitHubSnapshot(
            account=account,
            repos=repos,
            repos_total=max(account.public_repos, len(items)),
            pulls=pulls,
            pulls_total=pulls_total,
        )

    async def _account(self) -> AccountSnapshot:
        try:
            entry = await self._client.get_json(
                f"/users/{quote(self._handle, safe='')}", project=self._project_user
            )
        except GitHubNotFound as exc:
            raise GitHubNotFound(f"GitHub account '{self._handle}' not found") from exc
        d = entry.data
        if d["type"] != "User":
            raise GitHubError(f"GitHub account '{d['login']}' is not a personal account")
        return AccountSnapshot(
            login=d["login"],
            id=d["id"],
            created_at=_dt(d["created_at"]),
            blog=d["blog"],
            email_match=d["email_match"],
            public_repos=d["public_repos"],
        )

    async def _repo(self, full_name: str) -> RepoSnapshot | None:
        try:
            entry = await self._client.get_json(repo_path(full_name), project=self._project_repo)
        except GitHubNotFound:  # deleted or renamed since the listing
            return None
        d = entry.data
        repo = RepoSnapshot(
            full_name=d["full_name"],
            fork=d["fork"],
            parent=d["parent"],
            template_from=d["template_from"],
            topics=tuple(d["topics"]),
            created_at=_dt(d["created_at"]),
            stars=d["stars"],
            default_branch=d["default_branch"],
        )
        try:
            commits = await self._commits(repo)
        except GitHubNotFound:  # 409: empty repository
            return replace(repo, skipped="empty repository")
        linked, first, last, email_match, only_sha = commits
        if repo.fork and linked == 0:
            return replace(repo, skipped="untouched fork")

        total, email_commits = await self._contributors(repo)
        try:
            ref = await self._client.get_json(
                repo_path(full_name, "git", "ref", "heads")
                + "/"
                + quote(repo.default_branch, safe="/"),
                project=lambda r: {"sha": r["object"]["sha"]},
            )
        except GitHubNotFound:
            return replace(repo, skipped="empty repository")
        head_sha = ref.data["sha"]
        tree = await self._client.get_json(
            repo_path(full_name, "git", "trees", head_sha),
            {"recursive": 1},
            project=self._project_tree,
            immutable=True,
        )
        files = tuple(TreeFile(path, size) for path, size in tree.data["files"])
        languages = await self._client.get_json(repo_path(full_name, "languages"))
        single_additions = None
        if linked + email_commits == 1 and only_sha:
            single = await self._client.get_json(
                repo_path(full_name, "commits", only_sha),
                project=lambda c: {"additions": (c.get("stats") or {}).get("additions")},
                immutable=True,
            )
            single_additions = single.data["additions"]
        contents = await self._contents(full_name, head_sha, files, readme=True)
        return replace(
            repo,
            head_sha=head_sha,
            linked_commits=linked,
            email_commits=0 if repo.fork else email_commits,
            total_commits=total,
            first_commit=first,
            last_commit=last,
            commit_email_match=email_match,
            single_commit_additions=single_additions,
            languages=dict(languages.data or {}),
            files=files,
            tree_truncated=tree.data["truncated"],
            contents=contents,
        )

    async def _commits(
        self, repo: RepoSnapshot
    ) -> tuple[int, date | None, date | None, bool, str | None]:
        """The account's commits on the default branch: count, first and last date, whether
        any uses one of the candidate's emails, and the sha when there is exactly one."""
        params: dict[str, str | int] = {
            "author": self._handle,
            "sha": repo.default_branch,
            "per_page": 100,
        }
        if repo.fork:
            params["since"] = repo.created_at.isoformat().replace("+00:00", "Z")
        path = repo_path(repo.full_name, "commits")
        first_page = await self._client.get_json(path, params, project=self._project_commits)
        commits = list(first_page.data or [])
        count, oldest = len(commits), commits[-1] if commits else None
        if "last" in first_page.links:
            last_url = first_page.links["last"]
            page = int(parse_qs(urlsplit(last_url).query).get("page", ["1"])[0])
            last_page = await self._client.get_json(
                api_path(last_url), project=self._project_commits
            )
            tail = list(last_page.data or [])
            commits += tail
            count = (page - 1) * 100 + len(tail)
            oldest = tail[-1] if tail else oldest
        if not commits:
            return 0, None, None, False, None
        return (
            count,
            _dt(oldest["date"]).date() if oldest else None,
            _dt(commits[0]["date"]).date(),
            any(c["email_match"] for c in commits),
            commits[0]["sha"] if count == 1 else None,
        )

    async def _contributors(self, repo: RepoSnapshot) -> tuple[int, int]:
        """Total commits on the default branch, and those by unlinked authors whose email is
        the candidate's."""
        total = email_commits = 0
        path: str | None = repo_path(repo.full_name, "contributors")
        params: dict[str, str | int] | None = {"anon": 1, "per_page": 100}
        for _ in range(self._limits.max_contributor_pages):
            if path is None:
                break
            page = await self._client.get_json(path, params, project=self._project_contributors)
            for c in page.data or []:
                total += c["contributions"]
                if c["email_match"] and not c["linked"]:
                    email_commits += c["contributions"]
            path = api_path(page.links["next"]) if "next" in page.links else None
            params = None
        return total, email_commits

    def _wanted_files(self, files: tuple[TreeFile, ...], *, readme: bool) -> list[TreeFile]:
        limits = self._limits
        wanted = [
            f
            for f in files
            if f.size <= limits.max_file_bytes
            and f.path.count("/") < limits.max_path_depth
            and manifest_kind(f.path) is not None
        ]
        wanted.sort(key=lambda f: (f.path.count("/"), f.path))
        if readme:
            top = next((f for f in files if "/" not in f.path and README.fullmatch(f.path)), None)
            if top is not None:
                wanted.insert(0, top)
        return wanted[: limits.max_files_per_repo]

    async def _contents(
        self, full_name: str, sha: str, files: tuple[TreeFile, ...], *, readme: bool
    ) -> dict[str, str]:
        wanted = self._wanted_files(files, readme=readme)
        texts = await asyncio.gather(
            *(
                self._client.get_raw(
                    repo_path(full_name, "contents", *f.path.split("/")), {"ref": sha}
                )
                for f in wanted
            )
        )
        return {f.path: text for f, text in zip(wanted, texts, strict=True)}

    async def _pulls(self, login: str) -> tuple[tuple[PullSnapshot, ...], int]:
        raw: list[dict[str, Any]] = []
        after: str | None = None
        total = 0
        while len(raw) < self._limits.max_pulls:
            page = await self._client.graphql(
                PULLS_QUERY, {"login": login, "after": after}, project=self._project_pulls
            )
            total = page["total"]
            raw += page["pulls"]
            after = page["next"]
            if after is None:
                break
        raw = [p for p in raw[: self._limits.max_pulls] if not p["private"]]
        pulls = [
            PullSnapshot(
                repo=p["repo"],
                repo_owner=p["repo_owner"],
                number=p["number"],
                title=p["title"],
                merged_at=_dt(p["merged_at"]),
                merged_by=p["merged_by"],
                merge_sha=p["merge_sha"],
                files=tuple(PullFile(path, a, d) for path, a, d in p["files"]),
                files_total=p["files_total"],
            )
            for p in raw
        ]

        # The base repository's manifests, at the merge commit of the latest PR that counts.
        latest: dict[str, int] = {}
        for i in sorted(range(len(pulls)), key=lambda i: pulls[i].merged_at, reverse=True):
            if pulls[i].merge_sha and self._pull_counts(pulls[i], login):
                latest.setdefault(pulls[i].repo, i)
        targets = list(latest.values())[: self._limits.max_pull_repos]
        manifests = await asyncio.gather(*(self._pull_manifests(pulls[i]) for i in targets))
        for i, found in zip(targets, manifests, strict=True):
            pulls[i] = replace(pulls[i], manifests=found)
        return tuple(pulls), total

    async def _pull_manifests(self, pull: PullSnapshot) -> dict[str, str]:
        assert pull.merge_sha is not None
        try:
            tree = await self._client.get_json(
                repo_path(pull.repo, "git", "trees", pull.merge_sha),
                {"recursive": 1},
                project=self._project_tree,
                immutable=True,
            )
        except GitHubNotFound:  # the repository was deleted or made private since
            return {}
        files = tuple(TreeFile(path, size) for path, size in tree.data["files"])
        return await self._contents(pull.repo, pull.merge_sha, files, readme=False)
