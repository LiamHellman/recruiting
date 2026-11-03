"""The shared quote validator: a model-supplied quote must appear in the exact text that was
sent to the model.

Matching is verbatim except for whitespace: any run of whitespace matches any other run.
Every source's extractor runs its quotes through `check_quote`; rejections carry a reason,
and the rejection rate per prompt version is an eval metric.
"""

from dataclasses import dataclass

MAX_QUOTE_CHARS = 500

NOT_FOUND = "quote not found in source"
OUTSIDE = "quote is outside its section"
EMPTY = "empty quote"
TOO_LONG = "quote too long"


def _collapse(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace runs to one space; map each output char to its source index."""
    chars: list[str] = []
    index: list[int] = []
    in_space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if not in_space:
                chars.append(" ")
                index.append(i)
            in_space = True
        else:
            chars.append(ch)
            index.append(i)
            in_space = False
    return "".join(chars), index


def locate_quote(
    text: str, quote: str, start: int = 0, end: int | None = None
) -> tuple[int, int] | None:
    """Return [start, end) offsets of `quote` in `text[start:end]` (as offsets into `text`),
    or None if it is not there. The first occurrence wins."""
    needle, _ = _collapse(quote.strip())
    if not needle:
        return None
    haystack, index = _collapse(text[start:end])
    pos = haystack.find(needle)
    if pos < 0:
        return None
    return start + index[pos], start + index[pos + len(needle) - 1] + 1


@dataclass(frozen=True)
class QuoteCheck:
    span: tuple[int, int] | None
    reason: str | None  # set when rejected

    @property
    def ok(self) -> bool:
        return self.span is not None


def check_quote(
    text: str,
    quote: str,
    *,
    within: tuple[int, int] | None = None,
    max_chars: int = MAX_QUOTE_CHARS,
) -> QuoteCheck:
    """Validate `quote` against `text`, optionally requiring it inside the `within` span
    (e.g. the CV entry a skill was attributed to)."""
    if not quote.strip():
        return QuoteCheck(None, EMPTY)
    if len(quote.strip()) > max_chars:
        return QuoteCheck(None, TOO_LONG)
    if within is None:
        span = locate_quote(text, quote)
        return QuoteCheck(span, None if span else NOT_FOUND)
    span = locate_quote(text, quote, *within)
    if span is not None:
        return QuoteCheck(span, None)
    return QuoteCheck(None, OUTSIDE if locate_quote(text, quote) else NOT_FOUND)
