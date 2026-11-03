import uuid
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest
from pydantic import BaseModel

from app.config import DEFAULT_TENANT_ID
from app.llm.fake import FakeBackend, FakeGateway
from app.llm.gateway import (
    FALLBACK_BETA,
    AnthropicBackend,
    BudgetExceeded,
    Budgets,
    CallContext,
    LLMError,
    LLMOutputError,
    LLMRefusal,
    RetryableLLMError,
)
from app.llm.pricing import cost_usd
from app.llm.prompts import Prompt, PromptError, load_prompt


class Echo(BaseModel):
    text: str
    count: int


PROMPT = Prompt(
    name="echo", version=1, model="claude-opus-5-5", max_tokens=1000, effort="low", system="Echo."
)
PROMPT_V2 = Prompt(
    name="echo", version=2, model="claude-opus-5-5", max_tokens=1000, effort="low", system="Echo!"
)
CTX = CallContext(tenant_id=DEFAULT_TENANT_ID, job_id=uuid.uuid4())


def echo_gateway(**kwargs: Any) -> FakeGateway:
    return FakeGateway({"echo": lambda content: {"text": content, "count": len(content)}}, **kwargs)


def test_fake_gateway_returns_a_validated_object():
    result = echo_gateway().parse(PROMPT, "hello", Echo, CTX)
    assert isinstance(result.output, Echo)
    assert result.output == Echo(text="hello", count=5)
    assert result.prompt_ref == "echo/v1"
    assert not result.cached


def test_fake_gateway_rejects_replies_that_do_not_match_the_schema():
    gateway = FakeGateway({"echo": {"text": "hi", "count": "many"}})
    with pytest.raises(ValueError):
        gateway.parse(PROMPT, "hello", Echo, CTX)


def test_second_identical_call_is_served_from_cache():
    gateway = echo_gateway()
    first = gateway.parse(PROMPT, "hello", Echo, CTX)
    second = gateway.parse(PROMPT, "hello", Echo, CTX)
    assert len(gateway.calls) == 1
    assert second.cached
    assert second.output == first.output
    assert [r.cached for r in gateway.ledger.records] == [False, True]
    assert gateway.ledger.records[1].usage.total == 0


def test_cache_key_includes_input_prompt_version_and_tenant():
    gateway = echo_gateway()
    gateway.parse(PROMPT, "hello", Echo, CTX)
    gateway.parse(PROMPT, "hello!", Echo, CTX)
    gateway.parse(PROMPT_V2, "hello", Echo, CTX)
    gateway.parse(PROMPT, "hello", Echo, CallContext(tenant_id=uuid.uuid4()))
    assert len(gateway.calls) == 4


def test_cache_entry_that_no_longer_validates_is_ignored():
    gateway = echo_gateway()
    gateway.parse(PROMPT, "hello", Echo, CTX)
    for key in gateway.cache.entries:
        gateway.cache.entries[key] = {"text": "stale"}
    result = gateway.parse(PROMPT, "hello", Echo, CTX)
    assert not result.cached
    assert len(gateway.calls) == 2


def test_retries_with_backoff_then_succeeds():
    backend = FakeBackend(
        replies={"echo": {"text": "ok", "count": 1}},
        errors=[RetryableLLMError("503"), RetryableLLMError("429", retry_after=7.0)],
    )
    gateway = FakeGateway(backend=backend)
    sleeps: list[float] = []
    gateway._sleep = sleeps.append
    result = gateway.parse(PROMPT, "hello", Echo, CTX)
    assert result.output.text == "ok"
    assert len(backend.calls) == 3
    assert 0.5 <= sleeps[0] <= 1.0  # first backoff: 1s with jitter
    assert sleeps[1] == 7.0  # honours retry-after when it is longer


def test_gives_up_after_max_attempts():
    backend = FakeBackend(
        replies={"echo": {"text": "ok", "count": 1}},
        errors=[RetryableLLMError("503") for _ in range(4)],
    )
    with pytest.raises(RetryableLLMError):
        FakeGateway(backend=backend).parse(PROMPT, "hello", Echo, CTX)
    assert len(backend.calls) == 4


def test_non_retryable_errors_are_not_retried():
    backend = FakeBackend(errors=[LLMError("HTTP 400")])
    with pytest.raises(LLMError):
        FakeGateway(backend=backend).parse(PROMPT, "hello", Echo, CTX)
    assert len(backend.calls) == 1


@pytest.mark.parametrize(
    ("stop_reason", "error"), [("refusal", LLMRefusal), ("max_tokens", LLMOutputError)]
)
def test_refusal_and_truncation_raise_and_are_not_cached(stop_reason, error):
    backend = FakeBackend(replies={"echo": {"text": "x", "count": 1}}, stop_reason=stop_reason)
    gateway = FakeGateway(backend=backend)
    with pytest.raises(error):
        gateway.parse(PROMPT, "hello", Echo, CTX)
    assert gateway.cache.entries == {}
    assert gateway.ledger.records[0].usage.total > 0  # tokens were still spent


def test_job_budget():
    gateway = echo_gateway(budgets=Budgets(per_job=60, per_tenant=10**6))
    gateway.parse(PROMPT, "first call", Echo, CTX)  # uses 53 tokens
    gateway.parse(PROMPT, "second call", Echo, CTX)  # 53 < 60, allowed; now over
    with pytest.raises(BudgetExceeded, match="job"):
        gateway.parse(PROMPT, "third call", Echo, CTX)
    # A different job of the same tenant still has budget.
    gateway.parse(PROMPT, "third call", Echo, CallContext(tenant_id=CTX.tenant_id))


def test_tenant_budget():
    gateway = echo_gateway(budgets=Budgets(per_job=10**6, per_tenant=50))
    gateway.parse(PROMPT, "first call", Echo, CTX)
    with pytest.raises(BudgetExceeded, match="tenant"):
        gateway.parse(PROMPT, "x", Echo, CallContext(tenant_id=CTX.tenant_id))


def test_cache_hits_do_not_need_budget():
    gateway = echo_gateway(budgets=Budgets(per_job=10, per_tenant=10))
    gateway.parse(PROMPT, "hello", Echo, CTX)
    assert gateway.parse(PROMPT, "hello", Echo, CTX).cached


def test_usage_is_priced():
    gateway = echo_gateway()
    result = gateway.parse(PROMPT, "hello", Echo, CTX)
    # FakeBackend: input = len//4 + 1 = 2, output = 50; Opus 5.5 is $4 / $20 per MTok.
    assert result.cost_usd == Decimal("0.001008")


def test_pricing():
    assert cost_usd(
        "claude-opus-5-5",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_input_tokens=1_000_000,
        cache_creation_input_tokens=1_000_000,
    ) == Decimal("29.200000")
    zero = {
        "input_tokens": 1,
        "output_tokens": 1,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    assert cost_usd("some-unknown-model", **zero) is None


# --- Prompt files -------------------------------------------------------------------------


def test_posting_parser_v1_loads():
    prompt = load_prompt("posting_parser", 1)
    assert prompt.ref == "posting_parser/v1"
    assert prompt.model == "claude-opus-5-5"
    assert "<posting>" in prompt.system


def test_prompt_without_front_matter_is_rejected(tmp_path: Path):
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "v1.md").write_text("Just a prompt.")
    with pytest.raises(PromptError):
        load_prompt("bad", 1, tmp_path)


def test_prompt_with_unknown_effort_is_rejected(tmp_path: Path):
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "v1.md").write_text(
        "---\nmodel: m\nmax_tokens: 10\neffort: extreme\n---\nHi."
    )
    with pytest.raises(PromptError, match="effort"):
        load_prompt("bad", 1, tmp_path)


# --- AnthropicBackend, against a stub client (no network) ---------------------------------


class StubMessages:
    def __init__(self, outcome: Any) -> None:
        self.outcome = outcome
        self.kwargs: dict[str, Any] = {}

    def parse(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def stub_client(outcome: Any) -> tuple[Any, StubMessages]:
    messages = StubMessages(outcome)
    return SimpleNamespace(beta=SimpleNamespace(messages=messages)), messages


def http_error(cls: type[anthropic.APIStatusError], code: int, headers: dict[str, str]) -> Any:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(code, headers=headers, request=request)
    return cls("error", response=response, body=None)


def test_anthropic_backend_request_shape_and_response_mapping():
    reply = SimpleNamespace(
        parsed_output=Echo(text="hi", count=2),
        model="claude-opus-5-5",
        stop_reason="end_turn",
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_read_input_tokens=None,
            cache_creation_input_tokens=3,
        ),
    )
    client, messages = stub_client(reply)
    response = AnthropicBackend(client).call(PROMPT, "hello", Echo)

    assert response.output == Echo(text="hi", count=2)
    assert response.usage.cache_read_input_tokens == 0
    assert response.usage.cache_creation_input_tokens == 3
    sent = messages.kwargs
    assert sent["model"] == "claude-opus-5-5"
    assert sent["output_format"] is Echo
    assert sent["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert sent["messages"] == [{"role": "user", "content": "hello"}]
    assert sent["output_config"] == {"effort": "low"}
    assert sent["fallbacks"] == "default"
    assert sent["betas"] == [FALLBACK_BETA]


@pytest.mark.parametrize(
    ("error", "expected", "retry_after"),
    [
        (http_error(anthropic.RateLimitError, 429, {"retry-after": "3"}), RetryableLLMError, 3.0),
        (http_error(anthropic.InternalServerError, 500, {}), RetryableLLMError, None),
        (http_error(anthropic.APIStatusError, 529, {}), RetryableLLMError, None),
        (http_error(anthropic.BadRequestError, 400, {}), LLMError, None),
        (http_error(anthropic.AuthenticationError, 401, {}), LLMError, None),
    ],
)
def test_anthropic_backend_error_mapping(error, expected, retry_after):
    client, _ = stub_client(error)
    with pytest.raises(expected) as caught:
        AnthropicBackend(client).call(PROMPT, "hello", Echo)
    if expected is RetryableLLMError:
        assert caught.value.retry_after == retry_after
    else:
        assert not isinstance(caught.value, RetryableLLMError)


def test_anthropic_backend_connection_errors_are_retryable():
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    client, _ = stub_client(anthropic.APIConnectionError(request=request))
    with pytest.raises(RetryableLLMError):
        AnthropicBackend(client).call(PROMPT, "hello", Echo)


def test_anthropic_backend_reports_missing_credentials_clearly():
    client, _ = stub_client(TypeError('"Could not resolve authentication method. Expected ..."'))
    with pytest.raises(LLMError, match="credentials are not configured"):
        AnthropicBackend(client).call(PROMPT, "hello", Echo)


def test_anthropic_backend_does_not_swallow_other_type_errors():
    client, _ = stub_client(TypeError("unexpected keyword argument 'foo'"))
    with pytest.raises(TypeError):
        AnthropicBackend(client).call(PROMPT, "hello", Echo)
