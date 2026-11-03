import uuid
from datetime import timedelta
from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.config import get_settings
from app.db.session import get_sessionmaker
from app.github.cache import SqlHttpCache
from app.github.collect import Limits
from app.llm.gateway import AnthropicBackend, Budgets, Gateway
from app.llm.stores import SqlResultCache, SqlUsageLedger
from app.sources.base import CandidateRef, SourceRegistry
from app.sources.cv import CvSource
from app.sources.github import GitHubClientFactory, GitHubSource
from app.storage import FileStore, LocalFileStore
from app.taxonomy.loader import Taxonomy, load_taxonomy


def get_tenant_id() -> uuid.UUID:
    # One tenant until multi-tenancy lands.
    return get_settings().default_tenant_id


def get_actor() -> str:
    # Placeholder until recruiter login (Phase 5.1) supplies a real identity.
    return "recruiter:dev"


def get_taxonomy() -> Taxonomy:
    return load_taxonomy()


def anthropic_backend() -> AnthropicBackend:
    key = get_settings().anthropic_api_key
    return AnthropicBackend(api_key=key.get_secret_value() if key else None)


@lru_cache
def get_gateway() -> Gateway:
    settings = get_settings()
    sessions = get_sessionmaker()
    return Gateway(
        anthropic_backend(),
        SqlResultCache(sessions),
        SqlUsageLedger(sessions),
        Budgets(per_job=settings.llm_job_token_budget, per_tenant=settings.llm_tenant_token_budget),
    )


@lru_cache
def get_file_store() -> FileStore:
    return LocalFileStore(get_settings().storage_dir)


@lru_cache
def get_github_clients() -> GitHubClientFactory:
    settings = get_settings()
    sessions = get_sessionmaker()

    def cache_for(candidate: CandidateRef) -> SqlHttpCache:
        return SqlHttpCache(sessions, candidate.tenant_id, candidate.id)

    token = settings.github_token
    return GitHubClientFactory(
        token.get_secret_value() if token else None,
        cache_for,
        concurrency=settings.github_concurrency,
        max_wait=settings.github_max_wait_seconds,
        graphql_ttl=timedelta(hours=settings.github_graphql_ttl_hours),
    )


def get_registry(
    gateway: Annotated[Gateway, Depends(get_gateway)],
    taxonomy: Annotated[Taxonomy, Depends(get_taxonomy)],
    github_clients: Annotated[GitHubClientFactory, Depends(get_github_clients)],
) -> SourceRegistry:
    # Site (Phase 6) and LinkedIn uploads (Phase 7) register here.
    limits = Limits(max_repos=get_settings().github_max_repos)
    return SourceRegistry(
        [
            CvSource(gateway, taxonomy),
            GitHubSource(github_clients, gateway, taxonomy, limits=limits),
        ]
    )
