import uuid
from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/.env, found the same way whatever the working directory (API, Alembic, evals).
ENV_FILE = Path(__file__).resolve().parents[1] / ".env"

# The single tenant used until multi-tenancy lands. Seeded by the first migration.
DEFAULT_TENANT_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")


class Settings(BaseSettings):
    """Runtime settings, read from the environment and an optional backend/.env file."""

    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    database_url: str = "postgresql+psycopg://recruiting:recruiting@localhost:5432/recruiting"
    redis_url: str = "redis://localhost:6379/0"

    # Passed to the Anthropic SDK, which on its own only reads the process environment.
    # Unset: the SDK falls back to ANTHROPIC_API_KEY in the environment or `ant auth login`.
    anthropic_api_key: SecretStr | None = None

    default_tenant_id: uuid.UUID = DEFAULT_TENANT_ID

    # Uploaded files (CVs). Gitignored; an encrypted bucket replaces this in production.
    storage_dir: Path = ENV_FILE.parent / "storage"

    # GitHub (Phase 3). A token is required: GraphQL (merged PRs) needs one, and
    # unauthenticated REST allows only 60 requests an hour. Read-only public access is enough.
    github_token: SecretStr | None = None
    github_max_repos: int = 30  # most recently pushed owned repos analysed per candidate
    github_concurrency: int = 4  # GitHub asks for few concurrent requests per client
    github_max_wait_seconds: float = 60.0  # longer rate-limit waits fail the run instead
    github_graphql_ttl_hours: int = 24  # GraphQL has no ETags; re-use answers this long

    # Token budgets count input + output + cache tokens across all LLM calls.
    llm_job_token_budget: int = 500_000
    llm_tenant_token_budget: int = 20_000_000


@lru_cache
def get_settings() -> Settings:
    return Settings()
