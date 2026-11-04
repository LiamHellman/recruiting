import asyncio
import hashlib
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any

from app.audit.log import stable_hash
from app.extraction.github import RepoReadme, build_github_evidence, pull_status
from app.extraction.readme import README_PROMPT, build_readme_mentions, readme_user_content
from app.extraction.redact import Redacted, find_emails, redact
from app.github.client import FetchStats, GitHubClient, GitHubNotConfigured, HttpCache
from app.github.collect import (
    README,
    Collector,
    GitHubSnapshot,
    Limits,
    PullSnapshot,
    RepoSnapshot,
)
from app.github.ownership import Ownership, assess_ownership
from app.github.profile import build_profile
from app.identity.github import assess_identity
from app.llm.gateway import CallContext, Gateway, LLMOutputError, LLMRefusal
from app.llm.prompts import Prompt, load_prompt
from app.schemas.evidence import EvidenceSource, IngestReport, RejectedItem
from app.schemas.github import HandleSource, ParsedReadme
from app.sources.base import CandidateRef, IngestResult
from app.taxonomy.loader import Taxonomy
from app.taxonomy.packages import Catalog, load_catalog


def redact_readme(
    text: str, names: tuple[str, ...], emails: tuple[str, ...], taxonomy: Taxonomy
) -> Redacted:
    """Exactly what the model sees of a README (also used by the README eval)."""
    return redact(
        text,
        names=names,
        emails=(*emails, *find_emails(text)),
        is_skill_term=lambda term: taxonomy.lookup(term) is not None,
    )


@dataclass(frozen=True)
class GitHubPayload:
    handle: str
    handle_from: HandleSource
    collected_on: date
    site_url: str | None = None
    cv_links: tuple[str, ...] = ()  # identity signals only; never fetched
    emails: tuple[str, ...] = ()  # from the CV; the form email comes with the CandidateRef


class GitHubClientFactory:
    """One client per run, with the candidate's own cache."""

    def __init__(
        self,
        token: str | None,
        cache_for: Callable[[CandidateRef], HttpCache],
        **client_options: Any,
    ) -> None:
        self._token = token
        self._cache_for = cache_for
        self._options = client_options

    def __call__(self, candidate: CandidateRef) -> GitHubClient:
        if not self._token:
            raise GitHubNotConfigured("GitHub is not configured (GITHUB_TOKEN is missing)")
        return GitHubClient(self._token, self._cache_for(candidate), **self._options)


@dataclass(frozen=True)
class _ReadmeRun:
    readmes: dict[str, RepoReadme]
    rejected: list[RejectedItem]
    prompt_ref: str | None
    model: str | None
    model_served: str | None
    cached: bool | None
    input_sha256: str | None
    redactions: dict[str, int]


class GitHubSource:
    """A candidate-submitted GitHub account -> `demonstrated`, `deep` and README `claimed`
    evidence, ownership labels, identity signals and depth signals."""

    name: EvidenceSource = "github"

    def __init__(
        self,
        clients: Callable[[CandidateRef], GitHubClient],
        gateway: Gateway,
        taxonomy: Taxonomy,
        catalog: Catalog | None = None,
        *,
        limits: Limits | None = None,
        prompt: Prompt | None = None,
        max_readmes: int = 10,
    ) -> None:
        self._clients = clients
        self._gateway = gateway
        self._taxonomy = taxonomy
        self._catalog = catalog or load_catalog()
        self._limits = limits or Limits()
        self._prompt = prompt or load_prompt(*README_PROMPT)
        self._max_readmes = max_readmes

    def input_key(self, candidate: CandidateRef, payload: GitHubPayload) -> str:
        """Not the date: a retried or re-queued pipeline does not refresh an account. An
        explicit re-run (POST /candidates/{id}/sources/github/runs) does."""
        return stable_hash(
            {
                "prompt": self._prompt.ref,
                "model": self._prompt.model,
                "handle": payload.handle,
                "handle_from": payload.handle_from,
                "site_url": payload.site_url,
                "cv_links": payload.cv_links,
                "names": candidate.names,
                "emails": [candidate.emails, payload.emails],
            }
        )

    def ingest(self, candidate: CandidateRef, payload: GitHubPayload) -> IngestResult:
        client = self._clients(candidate)
        emails = tuple(dict.fromkeys((*candidate.emails, *payload.emails)))
        # Sources are synchronous (FastAPI runs sync endpoints in a worker thread; Celery
        # tasks are sync), so the async collection gets its own event loop here.
        snapshot, stats = asyncio.run(self._collect(client, payload.handle, emails))

        ownership = {r.full_name: assess_ownership(r.facts()) for r in snapshot.repos}
        readme_run = self._readmes(candidate, snapshot, ownership, emails)
        build = build_github_evidence(
            snapshot, ownership, readme_run.readmes, self._taxonomy, self._catalog
        )
        identity = assess_identity(
            login=snapshot.account.login,
            handle_from=payload.handle_from,
            cv_links=payload.cv_links,
            site_url=payload.site_url,
            blog=snapshot.account.blog,
            profile_email_match=snapshot.account.email_match,
            commit_email_match=any(r.commit_email_match for r in snapshot.repos),
        )
        profile = build_profile(
            snapshot, ownership, identity, self._taxonomy, self._catalog, payload.collected_on
        )
        return IngestResult(
            evidence=build.evidence,
            profile=profile.model_dump(mode="json"),
            report=IngestReport(
                source="github",
                prompt_ref=readme_run.prompt_ref,
                model=readme_run.model,
                model_served=readme_run.model_served,
                cached=readme_run.cached,
                input_sha256=readme_run.input_sha256,
                checked=build.checked,
                rejected=[*build.rejected, *readme_run.rejected],
                accepted=len(build.evidence),
                duplicates_merged=build.duplicates_merged,
                unmapped_terms=build.unmapped_terms,
                redactions=readme_run.redactions,
                fetch=stats.as_dict(),
            ),
        )

    async def _collect(
        self, client: GitHubClient, handle: str, emails: tuple[str, ...]
    ) -> tuple[GitHubSnapshot, FetchStats]:
        def counts(pull: PullSnapshot, login: str) -> bool:
            return pull_status(pull, login, self._catalog) is None

        async with client:
            collector = Collector(client, handle, emails, self._limits, pull_counts=counts)
            snapshot = await collector.collect()
        return snapshot, client.stats

    def _readmes(
        self,
        candidate: CandidateRef,
        snapshot: GitHubSnapshot,
        ownership: Mapping[str, Ownership],
        emails: tuple[str, ...],
    ) -> _ReadmeRun:
        """README claims for repos that earn credit. One model call per README (cached by
        content); a README the model refuses is skipped, not fatal."""
        targets: list[tuple[RepoSnapshot, str]] = []
        for repo in snapshot.repos:
            path = next((p for p in repo.contents if "/" not in p and README.fullmatch(p)), None)
            if path and not repo.skipped and ownership[repo.full_name].factor > 0:
                targets.append((repo, path))
        targets = targets[: self._max_readmes]

        readmes: dict[str, RepoReadme] = {}
        rejected: list[RejectedItem] = []
        redactions: Counter[str] = Counter()
        sent = hashlib.sha256()
        served: str | None = None
        cached: list[bool] = []
        for repo, path in targets:
            text = repo.contents[path][: self._limits.max_readme_chars]
            names = (*candidate.names, snapshot.account.login)  # the handle identifies too
            redacted = redact_readme(text, names, emails, self._taxonomy)
            content = readme_user_content(redacted.text)
            sent.update(content.encode())
            try:
                result = self._gateway.parse(
                    self._prompt,
                    content,
                    ParsedReadme,
                    CallContext(
                        tenant_id=candidate.tenant_id,
                        job_id=candidate.job_id,
                        candidate_id=candidate.id,
                    ),
                )
            except (LLMRefusal, LLMOutputError) as exc:
                rejected.append(
                    RejectedItem(
                        item=f"README of {repo.full_name}",
                        quote="",
                        reason=f"not read ({type(exc).__name__})",
                    )
                )
                continue
            served = result.model_served or served
            cached.append(result.cached)
            redactions.update(redacted.counts())
            readmes[repo.full_name] = RepoReadme(
                path, build_readme_mentions(result.output, redacted, self._taxonomy)
            )
        called = bool(targets)
        return _ReadmeRun(
            readmes=readmes,
            rejected=rejected,
            prompt_ref=self._prompt.ref if called else None,
            model=self._prompt.model if called else None,
            model_served=served,
            cached=all(cached) if cached else None,
            input_sha256=sent.hexdigest() if called else None,
            redactions=dict(redactions),
        )
