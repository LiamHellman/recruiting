"""A fake api.github.com for unit tests, served through `httpx.MockTransport`.

The answers are hand-built in the shapes GitHub's REST and GraphQL documentation gives
(trimmed, but including fields the collector must drop, such as `avatar_url`, `location`,
`bio`, `name` and emails), for fictional accounts only. It supports what the client relies
on: ETags and 304s, rate-limit headers, Link pagination, raw file contents and empty repos.
"""

import base64
import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import parse_qs, unquote, urlencode

import httpx

API = "https://api.github.com"


def _sha(*parts: object) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()


@dataclass
class Commit:
    date: str  # ISO 8601, e.g. "2026-08-14T10:00:00Z"
    email: str
    additions: int = 25
    linked: bool = True  # GitHub linked the commit to the account (verified email)

    @property
    def sha(self) -> str:
        return _sha(self.date, self.email, self.additions)


@dataclass
class Repo:
    owner: str
    name: str
    created_at: str = "2020-01-01T00:00:00Z"
    pushed_at: str = "2026-01-01T00:00:00Z"
    fork: bool = False
    parent: str | None = None
    template: str | None = None
    topics: list[str] = field(default_factory=list)
    stars: int = 0
    default_branch: str = "main"
    commits: list[Commit] = field(default_factory=list)  # the account owner's
    others: dict[str, int] = field(default_factory=dict)  # other logins -> commit count
    languages: dict[str, int] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)
    empty: bool = False

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.name}"

    @property
    def head_sha(self) -> str:
        return _sha(self.full_name, "head", len(self.commits), sorted(self.files))


@dataclass
class Pull:
    repo: str  # base repository, owner/name
    number: int
    title: str
    merged_at: str
    merged_by: str | None
    files: list[tuple[str, int, int]]  # path, additions, deletions
    merge_sha: str = ""

    def __post_init__(self) -> None:
        self.merge_sha = self.merge_sha or _sha(self.repo, self.number)


@dataclass
class User:
    login: str
    id: int
    created_at: str
    blog: str = ""
    email: str | None = None
    name: str = ""
    location: str = ""
    bio: str = ""
    type: str = "User"
    repos: list[Repo] = field(default_factory=list)
    pulls: list[Pull] = field(default_factory=list)


class FakeGitHub:
    def __init__(
        self,
        users: list[User],
        trees: dict[tuple[str, str], dict[str, str]] | None = None,
    ) -> None:
        self.users = {u.login.casefold(): u for u in users}
        self.repos = {r.full_name.casefold(): r for u in users for r in u.repos}
        self.trees = trees or {}  # (repo, sha) -> files, for other people's repos
        self.requests: list[httpx.Request] = []
        self.injected: list[httpx.Response] = []  # returned, in order, before routing
        self.remaining = 5000
        self.reset_at = 1_900_000_000

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def answered(self, status: int | None = None) -> list[httpx.Request]:
        """Requests seen (all, or those answered with `status`)."""
        if status is None:
            return list(self.requests)
        return [r for r in self.requests if r.extensions.get("fake_status") == status]

    # --- routing ----------------------------------------------------------------------

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        response = self.injected.pop(0) if self.injected else self._route(request)
        request.extensions["fake_status"] = response.status_code
        return response

    def _route(self, request: httpx.Request) -> httpx.Response:
        assert request.url.host == "api.github.com", request.url
        assert request.headers["authorization"].startswith("Bearer ")
        if request.method == "POST" and request.url.path == "/graphql":
            return self._graphql(json.loads(request.content))
        path = unquote(request.url.raw_path.decode().split("?", 1)[0])
        query = {k: v[0] for k, v in parse_qs(request.url.query.decode()).items()}
        routes: list[tuple[str, Callable[..., httpx.Response]]] = [
            (r"/users/([^/]+)", self._user),
            (r"/users/([^/]+)/repos", self._user_repos),
            (r"/repos/([^/]+/[^/]+)", self._repo),
            (r"/repos/([^/]+/[^/]+)/commits", self._commits),
            (r"/repos/([^/]+/[^/]+)/commits/([0-9a-f]+)", self._commit),
            (r"/repos/([^/]+/[^/]+)/contributors", self._contributors),
            (r"/repos/([^/]+/[^/]+)/languages", self._languages),
            (r"/repos/([^/]+/[^/]+)/git/ref/heads/(.+)", self._ref),
            (r"/repos/([^/]+/[^/]+)/git/trees/([0-9a-f]+)", self._tree),
            (r"/repos/([^/]+/[^/]+)/contents/(.+)", self._contents),
        ]
        for pattern, route in routes:
            if match := re.fullmatch(pattern, path):
                return route(request, query, *match.groups())
        return self._json(request, {"message": "Not Found"}, 404)

    # --- helpers ----------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "x-ratelimit-limit": "5000",
            "x-ratelimit-remaining": str(self.remaining),
            "x-ratelimit-reset": str(self.reset_at),
        }

    def _json(
        self,
        request: httpx.Request,
        body: object,
        status: int = 200,
        *,
        link: str | None = None,
    ) -> httpx.Response:
        text = json.dumps(body)
        etag = f'W/"{hashlib.sha1(text.encode()).hexdigest()}"'
        headers = self._headers()
        if status == 200 and request.headers.get("if-none-match") == etag:
            return httpx.Response(304, headers={**headers, "etag": etag})
        self.remaining -= 1
        headers["etag"] = etag
        if link:
            headers["link"] = link
        return httpx.Response(status, text=text, headers=headers)

    def _repo_of(self, full_name: str) -> Repo | None:
        return self.repos.get(full_name.casefold())

    def _user_json(self, user: User) -> dict[str, object]:
        return {
            "login": user.login,
            "id": user.id,
            "node_id": f"U_{user.id}",
            "avatar_url": f"https://avatars.githubusercontent.com/u/{user.id}",
            "type": user.type,
            "name": user.name,
            "company": "@example",
            "blog": user.blog,
            "location": user.location,
            "email": user.email,
            "bio": user.bio,
            "public_repos": len(user.repos),
            "followers": 12,
            "created_at": user.created_at,
        }

    def _repo_json(self, repo: Repo, *, full: bool) -> dict[str, object]:
        owner = self.users.get(repo.owner.casefold())
        body: dict[str, object] = {
            "id": int(_sha(repo.full_name)[:6], 16),
            "name": repo.name,
            "full_name": repo.full_name,
            "owner": self._user_json(owner) if owner else {"login": repo.owner},
            "fork": repo.fork,
            "created_at": repo.created_at,
            "pushed_at": repo.pushed_at,
            "size": 0 if repo.empty else 120,
            "stargazers_count": repo.stars,
            "forks_count": 0,
            "topics": repo.topics,
            "default_branch": repo.default_branch,
            "is_template": False,
        }
        if full:
            if repo.parent:
                body["parent"] = {"full_name": repo.parent}
                body["source"] = {"full_name": repo.parent}
            if repo.template:
                body["template_repository"] = {"full_name": repo.template}
        return body

    # --- REST -------------------------------------------------------------------------

    def _user(self, request: httpx.Request, query: dict[str, str], login: str) -> httpx.Response:
        user = self.users.get(login.casefold())
        if user is None:
            return self._json(request, {"message": "Not Found"}, 404)
        return self._json(request, self._user_json(user))

    def _user_repos(
        self, request: httpx.Request, query: dict[str, str], login: str
    ) -> httpx.Response:
        user = self.users[login.casefold()]
        repos = sorted(user.repos, key=lambda r: r.pushed_at, reverse=True)
        return self._json(request, [self._repo_json(r, full=False) for r in repos])

    def _repo(self, request: httpx.Request, query: dict[str, str], name: str) -> httpx.Response:
        repo = self._repo_of(name)
        if repo is None:
            return self._json(request, {"message": "Not Found"}, 404)
        return self._json(request, self._repo_json(repo, full=True))

    def _commits(self, request: httpx.Request, query: dict[str, str], name: str) -> httpx.Response:
        repo = self._repo_of(name)
        assert repo is not None
        if repo.empty:
            return self._json(request, {"message": "Git Repository is empty."}, 409)
        assert query["author"].casefold() == repo.owner.casefold()
        commits = [c for c in repo.commits if c.linked]
        if "since" in query:
            commits = [c for c in commits if c.date >= query["since"]]
        commits.sort(key=lambda c: c.date, reverse=True)
        per_page, page = int(query.get("per_page", 30)), int(query.get("page", 1))
        pages = max(1, -(-len(commits) // per_page))
        chunk = commits[(page - 1) * per_page : page * per_page]
        link = None
        if pages > 1:
            base = f"{API}/repos/{repo.full_name}/commits"
            q = {k: v for k, v in query.items() if k != "page"}
            rels = []
            if page < pages:
                rels.append(f'<{base}?{urlencode({**q, "page": page + 1})}>; rel="next"')
            rels.append(f'<{base}?{urlencode({**q, "page": pages})}>; rel="last"')
            link = ", ".join(rels)
        body = [
            {
                "sha": c.sha,
                "commit": {
                    "author": {"name": "Someone", "email": c.email, "date": c.date},
                    "message": "change",
                },
                "author": {"login": repo.owner},
            }
            for c in chunk
        ]
        return self._json(request, body, link=link)

    def _commit(
        self, request: httpx.Request, query: dict[str, str], name: str, sha: str
    ) -> httpx.Response:
        repo = self._repo_of(name)
        assert repo is not None
        commit = next(c for c in repo.commits if c.sha == sha)
        stats = {"additions": commit.additions, "deletions": 0, "total": commit.additions}
        return self._json(request, {"sha": sha, "stats": stats, "files": []})

    def _contributors(
        self, request: httpx.Request, query: dict[str, str], name: str
    ) -> httpx.Response:
        repo = self._repo_of(name)
        assert repo is not None
        body: list[dict[str, object]] = []
        linked = sum(c.linked for c in repo.commits)
        if linked:
            body.append({"login": repo.owner, "type": "User", "contributions": linked})
        body += [{"login": k, "type": "User", "contributions": v} for k, v in repo.others.items()]
        unlinked: dict[str, int] = {}
        for c in repo.commits:
            if not c.linked:
                unlinked[c.email] = unlinked.get(c.email, 0) + 1
        body += [
            {"email": e, "name": "Someone", "type": "Anonymous", "contributions": n}
            for e, n in unlinked.items()
        ]
        if not body:
            return httpx.Response(204, headers=self._headers())
        return self._json(request, body)

    def _languages(
        self, request: httpx.Request, query: dict[str, str], name: str
    ) -> httpx.Response:
        repo = self._repo_of(name)
        assert repo is not None
        return self._json(request, repo.languages)

    def _ref(
        self, request: httpx.Request, query: dict[str, str], name: str, branch: str
    ) -> httpx.Response:
        repo = self._repo_of(name)
        assert repo is not None
        if repo.empty:
            return self._json(request, {"message": "Git Repository is empty."}, 409)
        return self._json(
            request,
            {"ref": f"refs/heads/{branch}", "object": {"sha": repo.head_sha, "type": "commit"}},
        )

    def _files_at(self, name: str, sha: str) -> dict[str, str] | None:
        repo = self._repo_of(name)
        if repo is not None and repo.head_sha == sha:
            return repo.files
        return self.trees.get((name, sha))

    def _tree(
        self, request: httpx.Request, query: dict[str, str], name: str, sha: str
    ) -> httpx.Response:
        files = self._files_at(name, sha)
        if files is None:
            return self._json(request, {"message": "Not Found"}, 404)
        dirs = sorted({p.rsplit("/", 1)[0] for p in files if "/" in p})
        tree: list[dict[str, object]] = [
            {"path": d, "mode": "040000", "type": "tree", "sha": _sha(d)} for d in dirs
        ]
        tree += [
            {"path": p, "mode": "100644", "type": "blob", "sha": _sha(p, t), "size": len(t)}
            for p, t in sorted(files.items())
        ]
        return self._json(request, {"sha": sha, "tree": tree, "truncated": False})

    def _contents(
        self, request: httpx.Request, query: dict[str, str], name: str, path: str
    ) -> httpx.Response:
        files = self._files_at(name, query["ref"])
        if files is None or path not in files:
            return self._json(request, {"message": "Not Found"}, 404)
        if "raw" in request.headers.get("accept", ""):
            self.remaining -= 1
            return httpx.Response(200, text=files[path], headers=self._headers())
        encoded = base64.b64encode(files[path].encode()).decode()
        return self._json(request, {"path": path, "encoding": "base64", "content": encoded})

    # --- GraphQL ----------------------------------------------------------------------

    def _graphql(self, body: dict[str, object]) -> httpx.Response:
        self.remaining -= 1
        variables = body["variables"]
        assert isinstance(variables, dict)
        query = str(body["query"])
        assert "pullRequests" in query
        user = self.users.get(str(variables["login"]).casefold())
        if user is None:
            data = {"data": {"user": None}, "errors": [{"type": "NOT_FOUND"}]}
            return httpx.Response(200, json=data, headers=self._headers())
        pulls = sorted(user.pulls, key=lambda p: p.merged_at, reverse=True)
        start = int(str(variables.get("after") or "0"))
        page = pulls[start : start + 50]
        nodes = [
            {
                "number": p.number,
                "title": p.title,
                "mergedAt": p.merged_at,
                "mergedBy": {"login": p.merged_by} if p.merged_by else None,
                "mergeCommit": {"oid": p.merge_sha},
                "author": {"login": user.login, "avatarUrl": "https://avatars.example/x"},
                "repository": {
                    "nameWithOwner": p.repo,
                    "isPrivate": False,
                    "owner": {"login": p.repo.split("/")[0]},
                },
                "files": {
                    "totalCount": len(p.files),
                    "nodes": [{"path": f, "additions": a, "deletions": d} for f, a, d in p.files],
                },
            }
            for p in page
        ]
        more = start + 50 < len(pulls)
        data = {
            "data": {
                "user": {
                    "pullRequests": {
                        "totalCount": len(pulls),
                        "pageInfo": {
                            "hasNextPage": more,
                            "endCursor": str(start + 50) if more else None,
                        },
                        "nodes": nodes,
                    }
                }
            }
        }
        return httpx.Response(200, json=data, headers=self._headers())


def iso(day: datetime) -> str:
    return day.strftime("%Y-%m-%dT%H:%M:%SZ")
