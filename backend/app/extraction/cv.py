"""CV -> claimed-skill evidence.

The LLM reads the redacted CV and lists entries (jobs, projects, ...), education and skill
mentions, each with a verbatim quote. Everything after that is deterministic and pure:

- every quote is checked against the redacted text that was sent (the shared validator);
- a skill attributed to an entry must be quoted from inside that entry's section, and the
  quote must actually name the skill;
- terms map to canonical ids through the alias table; unknown terms stay unmapped;
- dates: an entry's skills are last seen at the entry's end ("present" = the day the CV was
  received); skills listed outside any entry (a skills section) are current claims, last
  seen the day the CV was received; an entry whose heading fails validation gives undated
  evidence;
- locators use offsets into the original text, so the UI highlights the CV itself.

Every CV item is level `claimed`: the CV is the candidate's own statement.
"""

import re
from dataclasses import dataclass
from datetime import date

from app.extraction.dates import period_end
from app.extraction.quotes import check_quote
from app.extraction.redact import Redacted
from app.schemas.evidence import (
    CvProfile,
    CvSkillMention,
    EvidenceItem,
    ParsedCv,
    ProfileEducation,
    ProfileEntry,
    RejectedItem,
)
from app.taxonomy.loader import Taxonomy, term_key
from app.taxonomy.normalize import normalize_term

CV_PROMPT = ("cv_extractor", 1)

# The quote is validated and names the skill. Lower confidence is for fuzzier evidence
# (e.g. LLM summaries of code), which CVs do not produce.
CLAIM_CONFIDENCE = 1.0
NO_MENTION = "quote does not mention the skill"


def cv_user_content(redacted_text: str) -> str:
    return f"<cv>\n{redacted_text}\n</cv>"


def cv_locator(document_id: str, span: tuple[int, int]) -> str:
    return f"cv:{document_id}#{span[0]}-{span[1]}"


def mentions_skill(quote: str, skill: str, skill_id: str | None, taxonomy: Taxonomy) -> bool:
    """Whether `quote` names the skill: as written, or by any alias of its canonical id."""
    haystack = term_key(quote)
    terms = {term_key(skill)}
    if skill_id is not None:
        terms |= {alias for alias, sid in taxonomy.aliases.items() if sid == skill_id}
    return any(
        re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", haystack)
        for term in terms
        if term
    )


@dataclass(frozen=True)
class CvBuild:
    evidence: list[EvidenceItem]
    profile: CvProfile
    checked: int
    rejected: list[RejectedItem]
    unmapped_terms: list[str]
    duplicates_merged: int


def build_cv_evidence(
    parsed: ParsedCv,
    redacted: Redacted,
    document_id: str,
    received_on: date,
    taxonomy: Taxonomy,
) -> CvBuild:
    text = redacted.text
    rejected: list[RejectedItem] = []
    checked = 0

    def reject(item: str, quote: str, reason: str) -> None:
        rejected.append(RejectedItem(item=item, quote=quote, reason=reason))

    # Locate entry headings and education lines; they bound each entry's section.
    entry_spans: list[tuple[int, int] | None] = []
    for entry in parsed.entries:
        checked += 1
        check = check_quote(text, entry.heading_quote)
        entry_spans.append(check.span)
        if check.reason:
            label = entry.title or entry.organization or entry.kind
            reject(f"entry: {label}", entry.heading_quote, f"heading {check.reason}")
    education: list[ProfileEducation] = []
    boundaries = sorted({span[0] for span in entry_spans if span} | {len(text)})
    for item in parsed.education:
        checked += 1
        check = check_quote(text, item.quote)
        if check.span is None:
            reject(
                f"education: {item.credential or item.institution}", item.quote, str(check.reason)
            )
            continue
        boundaries = sorted({*boundaries, check.span[0]})
        education.append(
            ProfileEducation(
                credential=_short(item.credential),
                field=_short(item.field),
                institution=_short(item.institution),
                span=redacted.to_original(*check.span),
            )
        )

    evidence: dict[tuple[str, int], EvidenceItem] = {}
    unmapped: dict[str, str] = {}
    duplicates = 0

    def accept(
        mention: CvSkillMention,
        bucket: int,
        within: tuple[int, int] | None,
        last_seen: date | None,
    ) -> None:
        nonlocal checked, duplicates
        checked += 1
        skill = " ".join(mention.skill.split())
        if not skill or len(skill) > 200:
            reject(mention.skill[:200], mention.quote, "bad skill name")
            return
        check = check_quote(text, mention.quote, within=within)
        if check.span is None:
            reject(skill, mention.quote, str(check.reason))
            return
        quote = text[check.span[0] : check.span[1]]
        skill_id = normalize_term(skill, taxonomy)
        if not mentions_skill(quote, skill, skill_id, taxonomy):
            reject(skill, mention.quote, NO_MENTION)
            return
        if skill_id is None:
            unmapped.setdefault(term_key(skill), skill)
            return
        if (skill_id, bucket) in evidence:
            duplicates += 1
            return
        evidence[(skill_id, bucket)] = EvidenceItem(
            skill_id=skill_id,
            level="claimed",
            source="cv",
            locator=cv_locator(document_id, redacted.to_original(*check.span)),
            quote=quote,
            date_last_seen=last_seen,
            ownership=1.0,
            confidence=CLAIM_CONFIDENCE,
        )

    entries: list[ProfileEntry] = []
    for index, (entry, span) in enumerate(zip(parsed.entries, entry_spans, strict=True)):
        if span is None:
            # Unverifiable heading: keep its skills, undated, searched anywhere in the CV.
            for mention in entry.skills:
                accept(mention, index, None, None)
            continue
        section = (span[0], next(b for b in boundaries if b > span[0]))
        last_seen = period_end(entry.end, today=received_on) or period_end(
            entry.start, today=received_on
        )
        for mention in entry.skills:
            accept(mention, index, section, last_seen)
        entries.append(
            ProfileEntry(
                kind=entry.kind,
                title=_short(entry.title),
                organization=_short(entry.organization),
                start=_short(entry.start, 50),
                end=_short(entry.end, 50),
                last_seen=last_seen,
                span=redacted.to_original(*span),
            )
        )
    for mention in parsed.other_skills:
        accept(mention, -1, None, received_on)

    return CvBuild(
        evidence=list(evidence.values()),
        profile=CvProfile(document_id=document_id, entries=entries, education=education),
        checked=checked,
        rejected=rejected,
        unmapped_terms=sorted(unmapped.values(), key=str.casefold),
        duplicates_merged=duplicates,
    )


def _short(value: str | None, limit: int = 200) -> str | None:
    if value is None:
        return None
    value = " ".join(value.split())[:limit]
    return value or None
