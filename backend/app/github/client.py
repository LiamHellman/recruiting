"""Async client for GitHub's official REST and GraphQL APIs: the only code that calls GitHub.

- Fetches only from https://api.github.com. Paths are relative; pagination links and
  redirects are followed only when they point back at the API host.
- Honors the primary rate limit (`x-ratelimit-*` headers) and secondary limits
  (`retry-after`, or at least a minute), with backoff and jitter on 5xx and network errors.
  A wait longer than `max_wait` fails the request instead of blocking.
- Caches every answer per candidate (`HttpCache`). Mutable REST answers are revalidated with
  ETags (a 304 costs no rate limit); answers addressed by commit SHA never change and are
  served without a request; GraphQL answers are re-used for a TTL.
- Callers pass a `project` function that keeps only the fields they use, applied before
  anything is cached, so the cache holds no more personal data than the collector needs.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol
from urllib.parse import quote, urlencode, urlsplit

import httpx

from app.sources.base import SourceError

API_HOST = "api.github.com"
API = f"https://{API_HOST}"
API_VERSION = "2022-11-28"
USER_AGENT = "recruiting-evidence/0.1 (consent-based applicant assessment)"
JSON_MEDIA = "application/vnd.github+json"
RAW_MEDIA = "application/vnd.github.raw+json"

Params = Mapping[str, str | int]
Project = Callable[[Any], Any]


class GitHubError(SourceError):
    retryable = True  # unreachable, or kept failing: an outage until shown otherwise


class GitHubNotConfigured(GitHubError):
    retryable = False


class GitHubNotFound(GitHubError):
    """404, or 409 for an empty repository."""

    retryable = False


class GitHubNotPersonal(GitHubError):
    """An organization account: only a person's own account is collected."""

    retryable = False


class GitHubRateLimited(GitHubError):
    pass


@dataclass(frozen=True)
class CachedResponse:
    data: Any  # projected JSON, or text for raw file contents
    etag: str | None
    links: dict[str, str]
    immutable: bool
    fetched_at: datetime


class HttpCache(Protocol):
    def get(self, key: str) -> CachedResponse | None: ...

    def put(self, key: str, entry: CachedResponse) -> None: ...


@dataclass
class InMemoryHttpCache:
    entries: dict[str, CachedResponse] = field(default_factory=dict)

    def get(self, key: str) -> CachedResponse | None:
        return self.entries.get(key)

    def put(self, key: str, entry: CachedResponse) -> None:
        self.entries[key] = entry


@dataclass
class FetchStats:
    requests: int = 0  # sent to GitHub, including 304s and retries
    not_modified: int = 0  # 304: served from cache after revalidation
    cache_hits: int = 0  # served from cache without a request
    retries: int = 0
    waited_seconds: float = 0.0

    def as_dict(self) -> dict[str, int]:
        return {
            "requests": self.requests,
            "not_modified": self.not_modified,
            "cache_hits": self.cache_hits,
            "retries": self.retries,
            "waited_seconds": round(self.waited_seconds),
        }


_LINK = re.compile(r'<([^>]+)>\s*;\s*rel="([^"]+)"')


def parse_links(header: str | None) -> dict[str, str]:
    return {rel: url for url, rel in _LINK.findall(header or "")}


def api_path(url: str) -> str:
    """The path (and query) of an absolute API URL. Refuses any other host."""
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.netloc != API_HOST:
        raise GitHubError(f"refusing to follow a link off {API_HOST}")
    return parts.path + (f"?{parts.query}" if parts.query else "")


def repo_path(full_name: str, *rest: str) -> str:
    """`/repos/{owner}/{name}/...` with each segment URL-quoted."""
    owner, name = full_name.split("/", 1)
    segments = [owner, name, *rest]
    return "/repos/" + "/".join(quote(s, safe="") for s in segments)


def _cache_key(method: str, path: str, params: Params | None, extra: str) -> str:
    query = urlencode(sorted((params or {}).items()))
    return hashlib.sha256("\x1f".join((method, path, query, extra)).encode()).hexdigest()


class GitHubClient:
    def __init__(
        self,
        token: str,
        cache: HttpCache,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        concurrency: int = 4,
        max_attempts: int = 4,
        base_delay: float = 1.0,
        max_wait: float = 60.0,
        graphql_ttl: timedelta = timedelta(hours=24),
        timeout: float = 20.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.time,
        rng: random.Random | None = None,
    ) -> None:
        if not token:
            raise GitHubNotConfigured("GitHub is not configured (GITHUB_TOKEN is missing)")
        self._token = token
        self._cache = cache
        self._transport = transport
        self._concurrency = concurrency
        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._max_wait = max_wait
        self._graphql_ttl = graphql_ttl
        self._timeout = timeout
        self._sleep = sleep
        self._clock = clock
        self._rng = rng or random.Random()
        self._http: httpx.AsyncClient | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self._remaining: int | None = None
        self._reset_at: float | None = None
        self.stats = FetchStats()

    async def __aenter__(self) -> GitHubClient:
        self._http = httpx.AsyncClient(
            base_url=API,
            transport=self._transport,
            timeout=self._timeout,
            follow_redirects=False,
            headers={
                "Authorization": f"Bearer {self._token}",
                "User-Agent": USER_AGENT,
                "X-GitHub-Api-Version": API_VERSION,
            },
        )
        self._semaphore = asyncio.Semaphore(self._concurrency)
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    # --- public API -----------------------------------------------------------------

    async def get_json(
        self,
        path: str,
        params: Params | None = None,
        *,
        project: Project | None = None,
        immutable: bool = False,
    ) -> CachedResponse:
        """GET a JSON resource. `immutable` for resources addressed by commit SHA."""
        return await self._get(path, params, JSON_MEDIA, project, immutable)

    async def get_raw(self, path: str, params: Params | None = None) -> str:
        """GET file contents as text (always addressed by commit SHA, so cached forever)."""
        entry = await self._get(path, params, RAW_MEDIA, None, True)
        return str(entry.data)

    async def graphql(
        self, query: str, variables: Mapping[str, Any], *, project: Project | None = None
    ) -> Any:
        key = _cache_key("POST", "/graphql", None, query + json.dumps(variables, sort_keys=True))
        cached = self._cache.get(key)
        now = datetime.fromtimestamp(self._clock(), UTC)
        if cached is not None and now - cached.fetched_at < self._graphql_ttl:
            self.stats.cache_hits += 1
            return cached.data
        response = await self._send(
            "POST", "/graphql", json_body={"query": query, "variables": dict(variables)}
        )
        if response.status_code != 200:
            raise self._status_error(response, "/graphql")
        body = response.json()
        errors = body.get("errors") or []
        if errors:
            types = {str(error.get("type")) for error in errors}
            if "RATE_LIMITED" in types:
                raise GitHubRateLimited("GitHub's GraphQL rate limit is exhausted; retry later")
            if "NOT_FOUND" in types and not body.get("data"):
                raise GitHubNotFound("GitHub account not found")
            raise GitHubError(f"GitHub GraphQL error ({', '.join(sorted(types))})")
        data = project(body.get("data")) if project else body.get("data")
        self._cache.put(
            key,
            CachedResponse(data=data, etag=None, links={}, immutable=False, fetched_at=now),
        )
        return data

    # --- internals ------------------------------------------------------------------

    async def _get(
        self,
        path: str,
        params: Params | None,
        accept: str,
        project: Project | None,
        immutable: bool,
    ) -> CachedResponse:
        if not path.startswith("/") or "://" in path:
            raise GitHubError("GitHub paths must be relative to the API host")
        key = _cache_key("GET", path, params, accept)
        cached = self._cache.get(key)
        if cached is not None and cached.immutable:
            self.stats.cache_hits += 1
            return cached
        headers = {"Accept": accept}
        if cached is not None and cached.etag:
            headers["If-None-Match"] = cached.etag
        response = await self._send("GET", path, params=params, headers=headers)
        if response.status_code == 304 and cached is not None:
            self.stats.not_modified += 1
            return cached
        if response.status_code == 204:
            data: Any = None
        elif response.status_code == 200:
            data = response.text if accept == RAW_MEDIA else response.json()
        else:
            raise self._status_error(response, path)
        entry = CachedResponse(
            data=project(data) if project else data,
            etag=response.headers.get("etag"),
            links=parse_links(response.headers.get("link")),
            immutable=immutable,
            fetched_at=datetime.fromtimestamp(self._clock(), UTC),
        )
        self._cache.put(key, entry)
        return entry

    def _status_error(self, response: httpx.Response, path: str) -> GitHubError:
        status = response.status_code
        if status in (404, 409):
            return GitHubNotFound(f"GitHub returned {status} for {path}")
        if status == 401:
            return GitHubError("GitHub rejected the token (401)")
        return GitHubError(f"GitHub returned {status} for {path}")

    async def _send(
        self,
        method: str,
        path: str,
        *,
        params: Params | None = None,
        headers: Mapping[str, str] | None = None,
        json_body: Any = None,
    ) -> httpx.Response:
        assert self._http is not None and self._semaphore is not None, "use `async with`"
        redirects = 0
        attempt = 0
        while True:
            attempt += 1
            await self._respect_primary_limit()
            try:
                async with self._semaphore:
                    response = await self._http.request(
                        method, path, params=params, headers=headers, json=json_body
                    )
            except httpx.TransportError as exc:
                self.stats.requests += 1
                if attempt >= self._max_attempts:
                    raise GitHubError(f"could not reach GitHub ({type(exc).__name__})") from exc
                await self._backoff(attempt)
                continue
            self.stats.requests += 1
            self._note_rate_limit(response.headers)

            if response.status_code in (301, 302, 307, 308) and method == "GET":
                redirects += 1
                location = response.headers.get("location", "")
                if redirects > 3 or not location:
                    raise GitHubError(f"too many redirects for {path}")
                target = api_path(str(httpx.URL(API).join(location)))
                path, params = target, None
                continue
            if self._is_rate_limited(response):
                if attempt >= self._max_attempts:
                    raise GitHubRateLimited("GitHub rate limit; retry later")
                await self._wait(self._rate_limit_delay(response, attempt), rate_limited=True)
                continue
            if response.status_code >= 500:
                if attempt >= self._max_attempts:
                    raise GitHubError(f"GitHub returned {response.status_code} for {path}")
                await self._backoff(attempt)
                continue
            return response

    def _note_rate_limit(self, headers: httpx.Headers) -> None:
        remaining, reset = headers.get("x-ratelimit-remaining"), headers.get("x-ratelimit-reset")
        if remaining is not None and remaining.isdigit():
            self._remaining = int(remaining)
        if reset is not None and reset.isdigit():
            self._reset_at = float(reset)

    def _is_rate_limited(self, response: httpx.Response) -> bool:
        if response.status_code == 429:
            return True
        if response.status_code != 403:
            return False
        headers = response.headers
        return (
            headers.get("x-ratelimit-remaining") == "0"
            or "retry-after" in headers
            or "rate limit" in response.text.lower()
        )

    def _rate_limit_delay(self, response: httpx.Response, attempt: int) -> float:
        retry_after = response.headers.get("retry-after", "")
        if retry_after.isdigit():
            return float(retry_after)
        if response.headers.get("x-ratelimit-remaining") == "0" and self._reset_at is not None:
            return max(self._reset_at - self._clock(), 0.0) + 1.0
        # Secondary limit without a hint: GitHub asks for at least a minute, growing.
        return 60.0 * 2.0 ** (attempt - 1)

    async def _respect_primary_limit(self) -> None:
        if self._remaining == 0 and self._reset_at is not None:
            delay = self._reset_at - self._clock()
            if delay > 0:
                await self._wait(delay + 1.0, rate_limited=True)
            self._remaining = None

    async def _backoff(self, attempt: int) -> None:
        self.stats.retries += 1
        delay = min(self._base_delay * 2 ** (attempt - 1), 30.0) * self._rng.uniform(0.5, 1.0)
        await self._wait(delay, rate_limited=False)

    async def _wait(self, delay: float, *, rate_limited: bool) -> None:
        if self.stats.waited_seconds + delay > self._max_wait:
            if rate_limited:
                raise GitHubRateLimited(
                    f"GitHub asks to wait {delay:.0f}s, longer than allowed; retry later"
                )
            raise GitHubError("GitHub kept failing; retry later")
        if rate_limited:
            self.stats.retries += 1
        self.stats.waited_seconds += delay
        await self._sleep(delay)
