import asyncio
import json
from collections.abc import Callable
from datetime import timedelta
from typing import Any

import httpx
import pytest

from app.github.client import (
    GitHubClient,
    GitHubError,
    GitHubNotConfigured,
    GitHubNotFound,
    GitHubRateLimited,
    InMemoryHttpCache,
    api_path,
    parse_links,
    repo_path,
)

Handler = Callable[[httpx.Request], httpx.Response]


class Clock:
    def __init__(self, now: float = 1_000_000.0) -> None:
        self.now = now
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def run(
    handler: Handler,
    body: Callable[[GitHubClient], Any],
    *,
    cache: InMemoryHttpCache | None = None,
    clock: Clock | None = None,
    **options: Any,
) -> Any:
    clock = clock or Clock()

    async def main() -> Any:
        client = GitHubClient(
            "test-token",
            cache if cache is not None else InMemoryHttpCache(),
            transport=httpx.MockTransport(handler),
            sleep=clock.sleep,
            clock=clock.time,
            **options,
        )
        async with client:
            return await body(client), client.stats

    return asyncio.run(main())


def ok(body: object, **headers: str) -> httpx.Response:
    return httpx.Response(200, json=body, headers={"x-ratelimit-remaining": "4999", **headers})


def test_requests_carry_token_version_and_user_agent_and_stay_on_the_api_host():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return ok({"login": "amira-h"})

    run(handler, lambda c: c.get_json("/users/amira-h"))
    [request] = seen
    assert request.url == "https://api.github.com/users/amira-h"
    assert request.headers["authorization"] == "Bearer test-token"
    assert request.headers["x-github-api-version"] == "2022-11-28"
    assert "recruiting-evidence" in request.headers["user-agent"]


def test_absolute_urls_and_links_off_the_api_host_are_refused():
    with pytest.raises(GitHubError, match="relative"):
        run(lambda r: ok({}), lambda c: c.get_json("https://evil.example/x"))
    with pytest.raises(GitHubError, match="refusing"):
        api_path("https://evil.example/repos?page=2")
    with pytest.raises(GitHubError, match="refusing"):
        api_path("http://api.github.com/repos?page=2")  # not https
    assert api_path("https://api.github.com/user/1/repos?page=2") == "/user/1/repos?page=2"


def test_redirects_are_followed_only_on_the_api_host():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/repos/old/name":
            return httpx.Response(
                301, headers={"location": "https://api.github.com/repositories/7"}
            )
        if request.url.path == "/repos/evil/name":
            return httpx.Response(301, headers={"location": "https://evil.example/steal"})
        return ok({"full_name": "new/name"})

    entry, _ = run(handler, lambda c: c.get_json("/repos/old/name"))
    assert entry.data == {"full_name": "new/name"}
    with pytest.raises(GitHubError, match="refusing"):
        run(handler, lambda c: c.get_json("/repos/evil/name"))


def test_etag_revalidation_serves_the_cached_answer_on_304():
    calls: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.headers.get("if-none-match"))
        if request.headers.get("if-none-match") == '"v1"':
            return httpx.Response(304)
        return ok({"login": "amira-h", "bio": "dropped"}, etag='"v1"')

    cache = InMemoryHttpCache()
    keep_login: Callable[[Any], Any] = lambda d: {"login": d["login"]}  # noqa: E731
    first, _ = run(handler, lambda c: c.get_json("/users/amira-h", project=keep_login), cache=cache)
    second, stats = run(
        handler, lambda c: c.get_json("/users/amira-h", project=keep_login), cache=cache
    )
    assert calls == [None, '"v1"']
    assert first.data == second.data == {"login": "amira-h"}
    assert stats.not_modified == 1
    # Only the projected fields were ever cached.
    assert "dropped" not in json.dumps([e.data for e in cache.entries.values()])


def test_immutable_answers_are_served_without_a_request():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, text="Django==5.0\n")

    cache = InMemoryHttpCache()
    path = repo_path("amira-h/shop-api", "contents", "requirements.txt")
    run(handler, lambda c: c.get_raw(path, {"ref": "abc"}), cache=cache)
    text, stats = run(handler, lambda c: c.get_raw(path, {"ref": "abc"}), cache=cache)
    assert text == "Django==5.0\n"
    assert calls == 1
    assert stats.requests == 0 and stats.cache_hits == 1


def test_repo_paths_quote_each_segment():
    assert repo_path("a/b", "contents", "dir name", "x#y.py") == (
        "/repos/a/b/contents/dir%20name/x%23y.py"
    )


def test_not_found_and_empty_repos_raise_not_found():
    def handler(request: httpx.Request) -> httpx.Response:
        status = 404 if "missing" in request.url.path else 409
        return httpx.Response(status, json={"message": "nope"})

    with pytest.raises(GitHubNotFound):
        run(handler, lambda c: c.get_json("/users/missing"))
    with pytest.raises(GitHubNotFound):
        run(handler, lambda c: c.get_json("/repos/a/empty/commits"))


def test_an_exhausted_primary_limit_waits_until_reset():
    clock = Clock()
    replies = [
        ok(
            {"n": 1},
            **{"x-ratelimit-remaining": "0", "x-ratelimit-reset": str(int(clock.now) + 30)},
        ),
        ok({"n": 2}),
    ]

    async def body(c: GitHubClient) -> Any:
        await c.get_json("/a")
        return await c.get_json("/b")

    entry, stats = run(lambda r: replies.pop(0), body, clock=clock)
    assert entry.data == {"n": 2}
    assert clock.sleeps == [31.0]
    assert stats.waited_seconds == 31.0


def test_secondary_limits_honour_retry_after_then_retry():
    clock = Clock()
    replies = [
        httpx.Response(
            403, text="You have exceeded a secondary rate limit", headers={"retry-after": "5"}
        ),
        httpx.Response(429, text="slow down"),
        ok({"done": True}),
    ]
    entry, stats = run(
        lambda r: replies.pop(0), lambda c: c.get_json("/a"), clock=clock, max_wait=600
    )
    assert entry.data == {"done": True}
    assert clock.sleeps == [5.0, 120.0]  # retry-after, then at least a minute, doubling
    assert stats.retries == 2


def test_waits_longer_than_allowed_fail_instead_of_blocking():
    clock = Clock()
    reset = str(int(clock.now) + 3600)
    limited = httpx.Response(
        403,
        text="API rate limit exceeded",
        headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": reset},
    )
    with pytest.raises(GitHubRateLimited, match="retry later"):
        run(lambda r: limited, lambda c: c.get_json("/a"), clock=clock, max_wait=60)
    assert clock.sleeps == []


def test_server_errors_and_network_errors_back_off_with_jitter():
    clock = Clock()
    replies: list[httpx.Response | Exception] = [
        httpx.Response(502),
        httpx.ConnectError("boom"),
        ok({"fine": True}),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    entry, stats = run(handler, lambda c: c.get_json("/a"), clock=clock)
    assert entry.data == {"fine": True}
    assert len(clock.sleeps) == 2
    assert 0.5 <= clock.sleeps[0] <= 1.0 and 1.0 <= clock.sleeps[1] <= 2.0
    assert stats.retries == 2


def test_persistent_server_errors_give_up():
    with pytest.raises(GitHubError, match="503"):
        run(lambda r: httpx.Response(503), lambda c: c.get_json("/a"), max_attempts=3)


def test_graphql_answers_are_reused_for_the_ttl():
    clock = Clock()
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.method == "POST" and request.url.path == "/graphql"
        return ok({"data": {"user": {"login": "amira-h"}}})

    cache = InMemoryHttpCache()

    def query(c: GitHubClient) -> Any:
        return c.graphql("query { user }", {"login": "amira-h"})

    ttl = timedelta(hours=1)
    run(handler, query, cache=cache, clock=clock, graphql_ttl=ttl)
    data, stats = run(handler, query, cache=cache, clock=clock, graphql_ttl=ttl)
    assert data == {"user": {"login": "amira-h"}} and calls == 1 and stats.cache_hits == 1
    clock.now += 3601
    run(handler, query, cache=cache, clock=clock, graphql_ttl=ttl)
    assert calls == 2


@pytest.mark.parametrize(
    ("errors", "error"),
    [
        ([{"type": "RATE_LIMITED"}], GitHubRateLimited),
        ([{"type": "NOT_FOUND"}], GitHubNotFound),
        ([{"type": "SOMETHING"}], GitHubError),
    ],
)
def test_graphql_errors(errors: list[dict[str, str]], error: type[Exception]):
    with pytest.raises(error):
        run(lambda r: ok({"data": None, "errors": errors}), lambda c: c.graphql("q", {}))


def test_a_rejected_token_is_reported():
    with pytest.raises(GitHubError, match="401"):
        run(lambda r: httpx.Response(401), lambda c: c.get_json("/a"))


def test_no_token_no_client():
    with pytest.raises(GitHubNotConfigured):
        GitHubClient("", InMemoryHttpCache())


def test_link_header_parsing():
    header = (
        '<https://api.github.com/x?page=2>; rel="next", '
        '<https://api.github.com/x?page=5>; rel="last"'
    )
    assert parse_links(header) == {
        "next": "https://api.github.com/x?page=2",
        "last": "https://api.github.com/x?page=5",
    }
    assert parse_links(None) == {}
