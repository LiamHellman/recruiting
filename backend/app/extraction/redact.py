"""Deterministic contact extraction and redaction, run before any LLM call.

`find_links` and `find_emails` pull contact data out of source text without an LLM.
`redact` replaces the candidate's name, contact details and personal-detail fields
(date of birth, gender, nationality, ...) with placeholders such as "[name]", and keeps a
mapping so offsets in the redacted text map back to the original text. Validation runs
against the redacted text (what the model saw); locators use original offsets so the UI
can highlight the CV itself.

Pure functions: no I/O.
"""

import bisect
import re
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Literal

Category = Literal["name", "email", "phone", "link", "postal_code", "address", "personal_detail"]

# --- Contact extraction ---------------------------------------------------------------------

_EMAIL = re.compile(
    r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"
)
_PROFILE_HOSTS = (
    r"(?:github\.com|gitlab\.com|bitbucket\.org|linkedin\.com|[a-z]{2,3}\.linkedin\.com)"
)
# Bare "host/path" links need a common TLD, so "node.js/express" is not a link.
_TLDS = r"(?:com|org|net|io|dev|ca|co|uk|me|app|ai|tech|page|site|xyz|info|blog|eu|de|fr|us)"
_LINK = re.compile(
    r"(?:https?://|www\.)[^\s<>\"'|]+"  # scheme or www.
    rf"|(?<![\w@.]){_PROFILE_HOSTS}/[^\s<>\"'|]+"  # bare profile links
    rf"|(?<![\w@.])[a-z0-9][a-z0-9-]*(?:\.[a-z0-9-]+)*\.{_TLDS}/[^\s<>\"'|]*"  # host/path
)
_TRAILING = ".,;:!?)]}'\""


def _trim_link(text: str, start: int, end: int) -> int:
    """Drop trailing punctuation, keeping a ')' that closes a '(' inside the link."""
    while end > start and text[end - 1] in _TRAILING:
        if text[end - 1] == ")" and text[start:end].count("(") >= text[start:end].count(")"):
            break
        end -= 1
    return end


def _link_spans(text: str) -> Iterator[tuple[int, int]]:
    for match in _LINK.finditer(text):
        start = match.start()
        end = _trim_link(text, start, match.end())
        # "name@example.com/..." is an email, not a link host
        if end > start and "@" not in text[start:end].split("/", 1)[0]:
            yield start, end


def normalize_url(raw: str) -> str:
    return raw if re.match(r"(?i)https?://", raw) else f"https://{raw}"


def find_links(text: str) -> list[str]:
    """URLs written in the text, normalized to https:// when no scheme is given."""
    return list(dict.fromkeys(normalize_url(text[s:e]) for s, e in _link_spans(text)))


def find_emails(text: str) -> list[str]:
    return list(dict.fromkeys(m.group(0).lower() for m in _EMAIL.finditer(text)))


# --- Detectors ------------------------------------------------------------------------------

_PHONE = re.compile(r"(?<![\w+])\+?\(?\d[\d \t().-]{7,20}\d(?![\w])")
_CA_POSTAL = re.compile(r"\b[ABCEGHJ-NPRSTVXY]\d[ABCEGHJ-NPRSTV-Z][ -]?\d[ABCEGHJ-NPRSTV-Z]\d\b")
_UK_POSTAL = re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]? \d[A-Z]{2}\b")
_US_ZIP = re.compile(r"(?<=\b[A-Z]{2} )\d{5}(?:-\d{4})?\b")
_STREET = re.compile(
    r"\b\d{1,6}[A-Za-z]?,? (?:[A-Z][\w'.-]* ){1,4}"
    r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way|Place|Pl"
    r"|Terrace|Crescent|Cres|Square|Sq|Parkway|Pkwy|Highway|Hwy)\b\.?"
    r"(?:,? (?:Apt|Apartment|Suite|Unit|#) ?[\w-]+)?"
    r"|\b\d{1,6},? (?:rue|avenue|boulevard|chemin|place|Rue|Avenue|Boulevard|Chemin)"
    r"(?: [\w'-]+){1,4}"
)
# "Label: value" lines whose value we never send. Address and phone labels catch formats
# the patterns above miss.
_LABELLED = re.compile(
    r"(?im)^[ \t]*(?P<label>date of birth|birth ?date|d\.?o\.?b\.?|born|age|gender|sex"
    r"|marital status|civil status|family status|children|dependants|dependents"
    r"|nationality|citizenship|religion|place of birth|photo"
    r"|(?:home |postal |mailing )?address|phone|mobile|cell|tel(?:ephone)?)"
    r"[ \t]*[:\-\u2013][ \t]*(?P<value>\S[^\n]*)$"
)


def _labelled_category(label: str) -> Category:
    label = label.lower()
    if "address" in label:
        return "address"
    if label in ("phone", "mobile", "cell", "tel", "telephone"):
        return "phone"
    return "personal_detail"


def _is_date_like(value: str) -> bool:
    """ "2019.01 - 2021.12", "2018 2020": only years and 1-2 digit months or days."""
    groups = re.findall(r"\d+", value)
    years = [g for g in groups if len(g) == 4 and g[:2] in ("19", "20")]
    return bool(years) and all(len(g) <= 2 or g in years for g in groups)


def _phone_spans(text: str) -> Iterator[tuple[int, int]]:
    for match in _PHONE.finditer(text):
        digits = sum(ch.isdigit() for ch in match.group(0))
        if 9 <= digits <= 15 and not _is_date_like(match.group(0)):
            yield match.start(), match.end()


def _name_patterns(
    names: Iterable[str], is_skill_term: Callable[[str], bool]
) -> list[re.Pattern[str]]:
    """The full name (any case, flexible whitespace), plus each name part of 3+ letters as a
    whole word, except parts that are skill names (a "Ruby" stays when it is a skill)."""
    patterns: list[re.Pattern[str]] = []
    parts: set[str] = set()
    for name in names:
        words = name.split()
        if not words:
            continue
        patterns.append(
            re.compile(r"(?<!\w)" + r"\s+".join(map(re.escape, words)) + r"(?!\w)", re.IGNORECASE)
        )
        parts.update(w.strip(".,") for w in words)
    for part in sorted(parts, key=len, reverse=True):
        if len(part) >= 3 and not is_skill_term(part):
            patterns.append(re.compile(rf"(?<!\w){re.escape(part)}(?!\w)", re.IGNORECASE))
    return patterns


# --- Redaction ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Replacement:
    start: int  # original offsets
    end: int
    category: Category


@dataclass(frozen=True)
class Redacted:
    original: str
    text: str
    replacements: tuple[Replacement, ...]
    # Parallel arrays: segment i covers redacted [r_starts[i], r_ends[i]) and original
    # [o_starts[i], o_ends[i]); placeholder segments map as a whole.
    _r_starts: tuple[int, ...]
    _r_ends: tuple[int, ...]
    _o_starts: tuple[int, ...]
    _o_ends: tuple[int, ...]
    _is_placeholder: tuple[bool, ...]

    def counts(self) -> dict[str, int]:
        return dict(Counter(r.category for r in self.replacements))

    def to_original(self, start: int, end: int) -> tuple[int, int]:
        """Map a [start, end) span of the redacted text to the original text. A span that
        touches a placeholder widens to cover the whole redacted value."""
        if not 0 <= start < end <= len(self.text):
            raise ValueError(f"span {start}-{end} outside the redacted text")
        i = bisect.bisect_right(self._r_starts, start) - 1
        o_start = self._o_starts[i] + (0 if self._is_placeholder[i] else start - self._r_starts[i])
        j = bisect.bisect_left(self._r_ends, end)
        o_end = self._o_ends[j] - (0 if self._is_placeholder[j] else self._r_ends[j] - end)
        return o_start, o_end


def redact(
    text: str,
    *,
    names: Iterable[str] = (),
    emails: Iterable[str] = (),
    is_skill_term: Callable[[str], bool] = lambda _term: False,
) -> Redacted:
    """Replace names, emails, phone numbers, links, postal codes, street addresses and
    labelled personal details with "[category]" placeholders."""
    spans: list[tuple[int, int, Category]] = []
    for match in _LABELLED.finditer(text):
        category = _labelled_category(match["label"])
        # "Mobile: Swift, Kotlin" is a skills line, not a phone number.
        if category == "phone" and sum(ch.isdigit() for ch in match["value"]) < 7:
            continue
        spans.append((match.start("value"), match.end("value"), category))
    spans += [(m.start(), m.end(), "email") for m in _EMAIL.finditer(text)]
    for email in emails:
        spans += [
            (m.start(), m.end(), "email")
            for m in re.finditer(re.escape(email), text, re.IGNORECASE)
        ]
    spans += [(s, e, "link") for s, e in _link_spans(text)]
    spans += [(s, e, "phone") for s, e in _phone_spans(text)]
    for pattern, category in (
        (_STREET, "address"),
        (_CA_POSTAL, "postal_code"),
        (_UK_POSTAL, "postal_code"),
        (_US_ZIP, "postal_code"),
    ):
        spans += [(m.start(), m.end(), category) for m in pattern.finditer(text)]
    for pattern in _name_patterns(names, is_skill_term):
        spans += [(m.start(), m.end(), "name") for m in pattern.finditer(text)]

    # Merge overlaps; the earliest (then longest) span names the category.
    merged: list[Replacement] = []
    for start, end, category in sorted(spans, key=lambda s: (s[0], -(s[1] - s[0]))):
        if merged and start < merged[-1].end:
            last = merged[-1]
            merged[-1] = Replacement(last.start, max(last.end, end), last.category)
        else:
            merged.append(Replacement(start, end, category))

    out: list[str] = []
    r_starts: list[int] = []
    r_ends: list[int] = []
    o_starts: list[int] = []
    o_ends: list[int] = []
    flags: list[bool] = []
    position = 0  # in redacted text
    cursor = 0  # in original text

    def segment(chunk: str, o_start: int, o_end: int, placeholder: bool) -> None:
        nonlocal position
        if not chunk:
            return
        out.append(chunk)
        r_starts.append(position)
        position += len(chunk)
        r_ends.append(position)
        o_starts.append(o_start)
        o_ends.append(o_end)
        flags.append(placeholder)

    for rep in merged:
        segment(text[cursor : rep.start], cursor, rep.start, False)
        segment(f"[{rep.category}]", rep.start, rep.end, True)
        cursor = rep.end
    segment(text[cursor:], cursor, len(text), False)

    return Redacted(
        original=text,
        text="".join(out),
        replacements=tuple(merged),
        _r_starts=tuple(r_starts),
        _r_ends=tuple(r_ends),
        _o_starts=tuple(o_starts),
        _o_ends=tuple(o_ends),
        _is_placeholder=tuple(flags),
    )
