"""Test doubles for the gateway. Unit tests never call a real LLM."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from app.llm.gateway import (
    BackendResponse,
    Budgets,
    Gateway,
    InMemoryResultCache,
    InMemoryUsageLedger,
    TokenUsage,
)
from app.llm.prompts import Prompt

# A scripted reply: a model instance, a plain dict, or a function of the user content.
Reply = BaseModel | dict[str, Any] | Callable[[str], BaseModel | dict[str, Any]]


@dataclass
class FakeCall:
    prompt_ref: str
    user_content: str


@dataclass
class FakeBackend:
    """Answers each prompt (by name) with a scripted reply, validated like structured output.

    `errors` are raised, in order, before any reply is returned (for retry tests).
    `stop_reason` lets a test simulate a refusal or truncation.
    """

    replies: dict[str, Reply] = field(default_factory=dict)
    errors: list[Exception] = field(default_factory=list)
    stop_reason: str = "end_turn"
    calls: list[FakeCall] = field(default_factory=list)

    def call[T: BaseModel](
        self, prompt: Prompt, user_content: str, output_model: type[T]
    ) -> BackendResponse[T]:
        self.calls.append(FakeCall(prompt.ref, user_content))
        if self.errors:
            raise self.errors.pop(0)
        if prompt.name not in self.replies:
            raise KeyError(f"FakeBackend has no reply for prompt {prompt.name!r}")
        reply = self.replies[prompt.name]
        if callable(reply) and not isinstance(reply, BaseModel):
            reply = reply(user_content)
        data = reply.model_dump(mode="json") if isinstance(reply, BaseModel) else reply
        output = output_model.model_validate(data) if self.stop_reason == "end_turn" else None
        return BackendResponse(
            output=output,
            model_served=prompt.model,
            usage=TokenUsage(input_tokens=len(user_content) // 4 + 1, output_tokens=50),
            stop_reason=self.stop_reason,
        )


class FakeGateway(Gateway):
    """A real Gateway (cache, budgets, retries, ledger) over a FakeBackend."""

    def __init__(
        self,
        replies: dict[str, Reply] | None = None,
        *,
        budgets: Budgets | None = None,
        backend: FakeBackend | None = None,
    ) -> None:
        self.backend = backend or FakeBackend(replies=dict(replies or {}))
        self.cache = InMemoryResultCache()
        self.ledger = InMemoryUsageLedger()
        super().__init__(
            self.backend,
            self.cache,
            self.ledger,
            budgets or Budgets(per_job=10**9, per_tenant=10**9),
            sleep=lambda _seconds: None,
        )

    @property
    def calls(self) -> list[FakeCall]:
        return self.backend.calls
