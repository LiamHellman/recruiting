"""Is this GitHub account the candidate's? Pure functions over signals already collected.

The account is only ever one the candidate gave: typed into the form, or the single GitHub
profile link in their CV. Nothing here searches for people or compares names. Signals:

- the CV links to the account (a conflict if it links a different account instead);
- the account's profile website is the candidate's site (or another site their CV links);
- a commit linked to the account uses the candidate's email (GitHub links a commit to an
  account only through a verified email, so this is strong);
- the account's public email is the candidate's;
- the candidate's site links back to the account (needs the Phase 6 fetcher; not checked).

Any conflict -> "conflict"; else any match -> "corroborated"; else "unverified". Anything
but "corroborated" asks the recruiter to confirm ("possible match: confirm?").
"""

import re
from collections.abc import Iterable
from urllib.parse import urlsplit

from app.extraction.redact import normalize_url
from app.schemas.github import (
    HandleSource,
    IdentityAssessment,
    IdentitySignal,
    IdentityStatus,
    SignalOutcome,
)

CONFIRM_LABEL = "possible match: confirm?"
_GITHUB_HOSTS = frozenset({"github.com", "www.github.com"})
_HANDLE = re.compile(r"^[A-Za-z0-9](?:-?[A-Za-z0-9])*$")
# First path segments on github.com that are pages, not accounts.
_RESERVED = frozenset(
    {
        "about", "apps", "collections", "contact", "enterprise", "events", "explore",
        "features", "issues", "join", "login", "marketplace", "new", "notifications",
        "orgs", "pricing", "pulls", "search", "settings", "site", "sponsors", "topics",
        "trending",
    }
)  # fmt: skip


def _host(url: str) -> str:
    host = (urlsplit(normalize_url(url.strip())).hostname or "").lower()
    return host.removeprefix("www.")


def github_accounts(links: Iterable[str]) -> tuple[list[str], list[str]]:
    """GitHub accounts a list of links points at: (profile links, owners of repo links)."""
    profiles: dict[str, str] = {}
    owners: dict[str, str] = {}
    for link in links:
        parts = urlsplit(normalize_url(link.strip()))
        if (parts.hostname or "").lower() not in _GITHUB_HOSTS:
            continue
        segments = [s for s in parts.path.split("/") if s]
        if not segments:
            continue
        owner = segments[0]
        if owner.lower() in _RESERVED or len(owner) > 39 or not _HANDLE.match(owner):
            continue
        target = profiles if len(segments) == 1 else owners
        target.setdefault(owner.casefold(), owner)
    return list(profiles.values()), list(owners.values())


def handle_from_cv(links: Iterable[str]) -> str | None:
    """The one GitHub profile the CV links to, or None if it links none or several."""
    profiles, _ = github_accounts(links)
    return profiles[0] if len(profiles) == 1 else None


def assess_identity(
    *,
    login: str,
    handle_from: HandleSource,
    cv_links: Iterable[str],
    site_url: str | None,
    blog: str | None,
    profile_email_match: bool,
    commit_email_match: bool,
) -> IdentityAssessment:
    links = list(cv_links)
    profiles, owners = github_accounts(links)
    me = login.casefold()
    signals: list[IdentitySignal] = []

    def signal(name: str, outcome: SignalOutcome, detail: str) -> None:
        signals.append(IdentitySignal(name=name, outcome=outcome, detail=detail))

    if handle_from == "cv":
        signal("cv_link", "none", "handle taken from the CV")
    elif me in {a.casefold() for a in profiles + owners}:
        signal("cv_link", "match", "the CV links to this account")
    elif profiles:
        others = ", ".join(profiles)
        signal("cv_link", "conflict", f"the CV links a different GitHub account ({others})")
    else:
        signal("cv_link", "none", "the CV has no GitHub link")

    candidate_sites = [site_url] if site_url else []
    sites = {_host(u) for u in [*candidate_sites, *links]} - {"", "github.com", "linkedin.com"}
    if blog and _host(blog) in sites:
        signal(
            "profile_website", "match", f"the profile website is the candidate's ({_host(blog)})"
        )
    elif blog:
        signal("profile_website", "none", "the profile website is not one of the candidate's")
    else:
        signal("profile_website", "none", "no profile website")

    signal("site_links_account", "not_checked", "the candidate's site is not fetched yet")
    if commit_email_match:
        signal("commit_email", "match", "a commit by this account uses the candidate's email")
    else:
        signal("commit_email", "none", "no commit by this account uses the candidate's email")
    if profile_email_match:
        signal("profile_email", "match", "the public profile email is the candidate's")
    else:
        signal("profile_email", "none", "no public profile email, or not the candidate's")

    outcomes = {s.outcome for s in signals}
    status: IdentityStatus = (
        "conflict"
        if "conflict" in outcomes
        else "corroborated"
        if "match" in outcomes
        else "unverified"
    )
    needs = status != "corroborated"
    return IdentityAssessment(
        handle=login,
        handle_from=handle_from,
        status=status,
        needs_confirmation=needs,
        label=CONFIRM_LABEL if needs else None,
        signals=signals,
    )
