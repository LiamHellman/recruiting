from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class Price:
    """USD per million tokens."""

    input: Decimal
    output: Decimal
    cache_read: Decimal
    cache_write: Decimal  # 5-minute cache writes cost 1.25x input


# First-party API list prices as of 2026-10. Re-check when adding a model.
PRICES: dict[str, Price] = {
    "claude-opus-5-5": Price(Decimal("4"), Decimal("20"), Decimal("0.20"), Decimal("5")),
    "claude-sonnet-5-5": Price(Decimal("2"), Decimal("10"), Decimal("0.20"), Decimal("2.5")),
    # Haiku 5.5 prices are for prompts up to 100K tokens.
    "claude-haiku-5-5": Price(Decimal("0.10"), Decimal("0.50"), Decimal("0.01"), Decimal("0.125")),
}

_PER_TOKEN = Decimal(1_000_000)


def cost_usd(
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_input_tokens: int,
    cache_creation_input_tokens: int,
) -> Decimal | None:
    """Cost of one call, or None for a model without a known price."""
    price = PRICES.get(model)
    if price is None:
        return None
    total = (
        input_tokens * price.input
        + output_tokens * price.output
        + cache_read_input_tokens * price.cache_read
        + cache_creation_input_tokens * price.cache_write
    ) / _PER_TOKEN
    return total.quantize(Decimal("0.000001"))
