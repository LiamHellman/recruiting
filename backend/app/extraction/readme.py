"""README -> technologies the project says it uses, each with a validated quote.

The LLM reads a redacted README (the candidate's name, emails and links replaced, as for
CVs) and lists technologies with verbatim quotes. The checks after that are the CV's: the
quote must be in the text that was sent and must name the skill; terms map through the
alias table; unknown terms stay unmapped. A README is the candidate describing their own
project, so what it adds is `claimed` evidence (see app/extraction/github.py).
"""

from dataclasses import dataclass

from app.extraction.cv import NO_MENTION, mentions_skill
from app.extraction.quotes import check_quote
from app.extraction.redact import Redacted
from app.schemas.evidence import RejectedItem
from app.schemas.github import ParsedReadme
from app.taxonomy.loader import Taxonomy, term_key
from app.taxonomy.normalize import normalize_term

README_PROMPT = ("readme_extractor", 1)


def readme_user_content(redacted_text: str) -> str:
    return f"<readme>\n{redacted_text}\n</readme>"


@dataclass(frozen=True)
class ReadmeMention:
    skill_id: str
    quote: str  # as the model saw it (redacted)
    span: tuple[int, int]  # offsets into the original README


@dataclass(frozen=True)
class ReadmeBuild:
    mentions: list[ReadmeMention]
    checked: int
    rejected: list[RejectedItem]
    unmapped_terms: list[str]
    duplicates_merged: int


def build_readme_mentions(
    parsed: ParsedReadme, redacted: Redacted, taxonomy: Taxonomy
) -> ReadmeBuild:
    text = redacted.text
    found: dict[str, ReadmeMention] = {}
    rejected: list[RejectedItem] = []
    unmapped: dict[str, str] = {}
    duplicates = 0
    for item in parsed.skills:
        skill = " ".join(item.skill.split())
        if not skill or len(skill) > 200:
            rejected.append(
                RejectedItem(item=item.skill[:200], quote=item.quote, reason="bad skill name")
            )
            continue
        check = check_quote(text, item.quote)
        if check.span is None:
            rejected.append(RejectedItem(item=skill, quote=item.quote, reason=str(check.reason)))
            continue
        quote = text[check.span[0] : check.span[1]]
        skill_id = normalize_term(skill, taxonomy)
        if not mentions_skill(quote, skill, skill_id, taxonomy):
            rejected.append(RejectedItem(item=skill, quote=item.quote, reason=NO_MENTION))
            continue
        if skill_id is None:
            unmapped.setdefault(term_key(skill), skill)
            continue
        if skill_id in found:
            duplicates += 1
            continue
        found[skill_id] = ReadmeMention(skill_id, quote, redacted.to_original(*check.span))
    return ReadmeBuild(
        mentions=list(found.values()),
        checked=len(parsed.skills),
        rejected=rejected,
        unmapped_terms=sorted(unmapped.values(), key=str.casefold),
        duplicates_merged=duplicates,
    )
