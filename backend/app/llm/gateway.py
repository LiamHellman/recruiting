"""The only place in the codebase that calls Claude.

`Gateway` owns everything around a model call: the result cache keyed by
(prompt version, model, input hash, output schema), per-job and per-tenant token budgets,
retries with backoff and jitter, and a usage record (tokens, latency, cost) for every call.
The actual request goes through a `ModelBackend`: `AnthropicBackend` in production,
`FakeBackend` (see app/llm/fake.py) in tests.

The LLM extracts structure. It never scores, ranks or decides.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Protocol

import anthropic
import pydantic
from pydantic import BaseModel

from app.llm.pricing import cost_usd
from app.llm.prompts import Prompt

logger = logging.getLogger("app.llm")

# Server-side refusal fallback: on a policy decline the API re-runs the request on
# Anthropic's recommended fallback model inside the same call.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


# --- Errors -------------------------------------------------------------------------------


class LLMError(Exception):
    pass


class RetryableLLMError(LLMError):
    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class LLMRefusal(LLMError):
    pass


class LLMOutputError(LLMError):
    """The model's output was truncated or did not validate against the schema."""


class BudgetExceeded(LLMError):
    pass


# --- Value types --------------------------------------------------------------------------


@dataclass(frozen=True)
class CallContext:
    tenant_id: uuid.UUID
    job_id: uuid.UUID | None = None


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    @property
    def total(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_input_tokens
            + self.cache_creation_input_tokens
        )


@dataclass(frozen=True)
class BackendResponse[T: BaseModel]:
    output: T | None
    model_served: str
    usage: TokenUsage
    stop_reason: str | None


@dataclass(frozen=True)
class LLMResult[T: BaseModel]:
    output: T
    prompt_ref: str
    model: str
    model_served: str | None  # None when served from cache
    cached: bool
    usage: TokenUsage
    cost_usd: Decimal | None


@dataclass(frozen=True)
class UsageRecord:
    tenant_id: uuid.UUID
    job_id: uuid.UUID | None
    prompt_ref: str
    model_requested: str
    model_served: str | None
    cached: bool
    usage: TokenUsage
    cost_usd: Decimal | None
    latency_ms: int


@dataclass(frozen=True)
class Budgets:
    per_job: int
    per_tenant: int


# --- Ports --------------------------------------------------------------------------------


class ModelBackend(Protocol):
    def call[T: BaseModel](
        self, prompt: Prompt, user_content: str, output_model: type[T]
    ) -> BackendResponse[T]: ...


class ResultCache(Protocol):
    def get(self, tenant_id: uuid.UUID, key: str) -> dict[str, Any] | None: ...

    def put(
        self,
        tenant_id: uuid.UUID,
        key: str,
        *,
        prompt_ref: str,
        model: str,
        input_hash: str,
        output: dict[str, Any],
    ) -> None: ...


class UsageLedger(Protocol):
    def record(self, record: UsageRecord) -> None: ...

    def tokens_used(self, tenant_id: uuid.UUID, job_id: uuid.UUID | None = None) -> int: ...


@dataclass
class InMemoryResultCache:
    entries: dict[tuple[uuid.UUID, str], dict[str, Any]] = field(default_factory=dict)

    def get(self, tenant_id: uuid.UUID, key: str) -> dict[str, Any] | None:
        return self.entries.get((tenant_id, key))

    def put(
        self,
        tenant_id: uuid.UUID,
        key: str,
        *,
        prompt_ref: str,
        model: str,
        input_hash: str,
        output: dict[str, Any],
    ) -> None:
        self.entries[(tenant_id, key)] = output


@dataclass
class InMemoryUsageLedger:
    records: list[UsageRecord] = field(default_factory=list)

    def record(self, record: UsageRecord) -> None:
        self.records.append(record)

    def tokens_used(self, tenant_id: uuid.UUID, job_id: uuid.UUID | None = None) -> int:
        return sum(
            r.usage.total
            for r in self.records
            if r.tenant_id == tenant_id and (job_id is None or r.job_id == job_id)
        )


# --- Gateway ------------------------------------------------------------------------------


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def schema_hash(output_model: type[BaseModel]) -> str:
    return _sha256(json.dumps(output_model.model_json_schema(), sort_keys=True))


def cache_key(prompt_ref: str, model: str, input_hash: str, output_schema_hash: str) -> str:
    return _sha256("\x1f".join((prompt_ref, model, input_hash, output_schema_hash)))


class Gateway:
    def __init__(
        self,
        backend: ModelBackend,
        cache: ResultCache,
        ledger: UsageLedger,
        budgets: Budgets,
        *,
        max_attempts: int = 4,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self._backend = backend
        self._cache = cache
        self._ledger = ledger
        self._budgets = budgets
        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._sleep = sleep
        self._rng = rng or random.Random()

    def parse[T: BaseModel](
        self, prompt: Prompt, user_content: str, output_model: type[T], ctx: CallContext
    ) -> LLMResult[T]:
        """Run `prompt` on `user_content` and return a validated `output_model` instance."""
        input_hash = _sha256(user_content)
        key = cache_key(prompt.ref, prompt.model, input_hash, schema_hash(output_model))

        cached = self._from_cache(key, output_model, ctx)
        if cached is not None:
            self._ledger.record(
                UsageRecord(
                    tenant_id=ctx.tenant_id,
                    job_id=ctx.job_id,
                    prompt_ref=prompt.ref,
                    model_requested=prompt.model,
                    model_served=None,
                    cached=True,
                    usage=TokenUsage(),
                    cost_usd=Decimal(0),
                    latency_ms=0,
                )
            )
            return LLMResult(
                output=cached,
                prompt_ref=prompt.ref,
                model=prompt.model,
                model_served=None,
                cached=True,
                usage=TokenUsage(),
                cost_usd=Decimal(0),
            )

        self._check_budgets(ctx)

        started = time.monotonic()
        response = self._call_with_retries(prompt, user_content, output_model)
        latency_ms = int((time.monotonic() - started) * 1000)

        # Priced at the serving model's rates; a refusal fallback can change the model.
        cost = cost_usd(
            response.model_served,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read_input_tokens=response.usage.cache_read_input_tokens,
            cache_creation_input_tokens=response.usage.cache_creation_input_tokens,
        )
        self._ledger.record(
            UsageRecord(
                tenant_id=ctx.tenant_id,
                job_id=ctx.job_id,
                prompt_ref=prompt.ref,
                model_requested=prompt.model,
                model_served=response.model_served,
                cached=False,
                usage=response.usage,
                cost_usd=cost,
                latency_ms=latency_ms,
            )
        )
        logger.info(
            "llm call prompt=%s model=%s served_by=%s in=%d out=%d cache_read=%d "
            "cache_write=%d latency_ms=%d cost_usd=%s stop=%s",
            prompt.ref,
            prompt.model,
            response.model_served,
            response.usage.input_tokens,
            response.usage.output_tokens,
            response.usage.cache_read_input_tokens,
            response.usage.cache_creation_input_tokens,
            latency_ms,
            cost,
            response.stop_reason,
        )

        if response.stop_reason == "refusal":
            raise LLMRefusal(f"{prompt.ref}: the model declined the request")
        if response.stop_reason == "max_tokens":
            raise LLMOutputError(f"{prompt.ref}: output truncated at max_tokens")
        if response.output is None:
            raise LLMOutputError(f"{prompt.ref}: no parsed output")

        self._cache.put(
            ctx.tenant_id,
            key,
            prompt_ref=prompt.ref,
            model=prompt.model,
            input_hash=input_hash,
            output=response.output.model_dump(mode="json"),
        )
        return LLMResult(
            output=response.output,
            prompt_ref=prompt.ref,
            model=prompt.model,
            model_served=response.model_served,
            cached=False,
            usage=response.usage,
            cost_usd=cost,
        )

    def _from_cache[T: BaseModel](
        self, key: str, output_model: type[T], ctx: CallContext
    ) -> T | None:
        raw = self._cache.get(ctx.tenant_id, key)
        if raw is None:
            return None
        try:
            return output_model.model_validate(raw)
        except pydantic.ValidationError:
            logger.warning("llm cache entry %s no longer validates; calling the model", key)
            return None

    def _check_budgets(self, ctx: CallContext) -> None:
        # Checked before each call, so one call can overshoot a budget by its own size.
        used = self._ledger.tokens_used(ctx.tenant_id)
        if used >= self._budgets.per_tenant:
            raise BudgetExceeded(f"tenant {ctx.tenant_id} used {used} tokens")
        if ctx.job_id is not None:
            used = self._ledger.tokens_used(ctx.tenant_id, ctx.job_id)
            if used >= self._budgets.per_job:
                raise BudgetExceeded(f"job {ctx.job_id} used {used} tokens")

    def _call_with_retries[T: BaseModel](
        self, prompt: Prompt, user_content: str, output_model: type[T]
    ) -> BackendResponse[T]:
        for attempt in range(1, self._max_attempts + 1):
            try:
                return self._backend.call(prompt, user_content, output_model)
            except RetryableLLMError as exc:
                if attempt == self._max_attempts:
                    raise
                backoff = min(self._base_delay * 2 ** (attempt - 1), self._max_delay)
                delay = backoff * self._rng.uniform(0.5, 1.0)  # jitter
                if exc.retry_after is not None:
                    delay = max(delay, exc.retry_after)
                logger.warning(
                    "llm call %s failed (%s); retry %d/%d in %.1fs",
                    prompt.ref,
                    exc,
                    attempt,
                    self._max_attempts - 1,
                    delay,
                )
                self._sleep(delay)
        raise AssertionError("unreachable")


# --- Anthropic backend --------------------------------------------------------------------


def _retry_after(exc: anthropic.APIStatusError) -> float | None:
    value = exc.response.headers.get("retry-after")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


class AnthropicBackend:
    """Structured output via `messages.parse()` against the Pydantic schema.

    The SDK's own retries are off (`max_retries=0`); the Gateway retries so that every
    attempt is logged and backoff is testable.
    """

    def __init__(
        self, client: anthropic.Anthropic | None = None, *, api_key: str | None = None
    ) -> None:
        self._client = client or anthropic.Anthropic(api_key=api_key, max_retries=0)

    def call[T: BaseModel](
        self, prompt: Prompt, user_content: str, output_model: type[T]
    ) -> BackendResponse[T]:
        try:
            response = self._client.beta.messages.parse(
                model=prompt.model,
                max_tokens=prompt.max_tokens,
                # The system prompt is stable per prompt version, so it is the cache prefix.
                system=[
                    {
                        "type": "text",
                        "text": prompt.system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_content}],
                output_format=output_model,
                thinking={"type": "adaptive"},
                output_config={"effort": prompt.effort},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
        except anthropic.RateLimitError as exc:
            raise RetryableLLMError("rate limited", _retry_after(exc)) from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500 or exc.status_code in (408, 409):
                raise RetryableLLMError(f"HTTP {exc.status_code}", _retry_after(exc)) from exc
            raise LLMError(f"HTTP {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:  # includes timeouts
            raise RetryableLLMError(f"connection error: {exc}") from exc
        except pydantic.ValidationError as exc:
            raise LLMOutputError(f"{prompt.ref}: output failed validation: {exc}") from exc
        except anthropic.CredentialsError as exc:
            raise LLMError(f"Claude credentials could not be loaded: {exc}") from exc
        except TypeError as exc:
            # The SDK raises a bare TypeError when no credentials are configured at all.
            if "authentication" not in str(exc):
                raise
            raise LLMError(
                "Claude credentials are not configured (set ANTHROPIC_API_KEY or run "
                "`ant auth login`)"
            ) from exc

        usage = response.usage
        return BackendResponse(
            output=response.parsed_output,
            model_served=response.model,
            usage=TokenUsage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_input_tokens=usage.cache_read_input_tokens or 0,
                cache_creation_input_tokens=usage.cache_creation_input_tokens or 0,
            ),
            stop_reason=response.stop_reason,
        )
