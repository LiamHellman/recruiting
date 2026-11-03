"""Job posting -> draft JobSpec.

The LLM lists the posting's technical requirements, each with a verbatim quote. Everything
after that is deterministic: quotes are checked against the posting, terms are mapped to
canonical skill ids through the alias table, and duplicates are merged. Unknown terms stay
unmapped for the recruiter; nothing is added to the taxonomy.
"""

from dataclasses import dataclass

from app.extraction.quotes import locate_quote
from app.llm.gateway import CallContext, Gateway
from app.llm.prompts import Prompt, load_prompt
from app.schemas.job import (
    JobSkill,
    JobSpec,
    ParsedPosting,
    ParseReport,
    PostingRequirement,
    RejectedRequirement,
    SkillList,
)
from app.taxonomy.loader import Taxonomy, term_key
from app.taxonomy.normalize import normalize_term

POSTING_PROMPT = ("posting_parser", 1)


@dataclass(frozen=True)
class SpecBuild:
    spec: JobSpec
    rejected: list[RejectedRequirement]
    duplicates_merged: int


def build_job_spec(parsed: ParsedPosting, posting_text: str, taxonomy: Taxonomy) -> SpecBuild:
    """Pure: turn the model's output into a draft JobSpec. Must-haves win over nice-to-haves."""
    lists: dict[SkillList, list[JobSkill]] = {"must_have": [], "nice_to_have": []}
    rejected: list[RejectedRequirement] = []
    seen: set[str] = set()
    duplicates = 0

    sources: tuple[tuple[SkillList, list[PostingRequirement]], ...] = (
        ("must_have", parsed.must_have),
        ("nice_to_have", parsed.nice_to_have),
    )
    for list_name, requirements in sources:
        for req in requirements:
            raw = " ".join(req.skill.split())
            if not raw:
                rejected.append(_reject(req.skill, req.quote, list_name, "empty skill name"))
                continue
            if len(raw) > 200:
                rejected.append(_reject(raw, req.quote, list_name, "skill name too long"))
                continue
            span = locate_quote(posting_text, req.quote)
            if span is None:
                rejected.append(_reject(raw, req.quote, list_name, "quote not found in posting"))
                continue
            skill_id = normalize_term(raw, taxonomy)
            dedup_key = skill_id or f"unmapped:{term_key(raw)}"
            if dedup_key in seen:
                duplicates += 1
                continue
            seen.add(dedup_key)
            lists[list_name].append(
                JobSkill(
                    raw_text=raw,
                    skill_id=skill_id,
                    weight=1.0,
                    source_quote=posting_text[span[0] : span[1]],
                    source_span=span,
                )
            )

    domain = parsed.domain.strip()[:100] if parsed.domain and parsed.domain.strip() else None
    spec = JobSpec(
        must_have=lists["must_have"],
        nice_to_have=lists["nice_to_have"],
        seniority=parsed.seniority,
        domain=domain,
    )
    return SpecBuild(spec=spec, rejected=rejected, duplicates_merged=duplicates)


def _reject(skill: str, quote: str, list_name: SkillList, reason: str) -> RejectedRequirement:
    return RejectedRequirement(skill=skill, quote=quote, list=list_name, reason=reason)


@dataclass(frozen=True)
class ParsedJob:
    title: str | None
    spec: JobSpec
    report: ParseReport


def posting_user_content(posting_text: str) -> str:
    return f"<posting>\n{posting_text}\n</posting>"


def parse_posting(
    gateway: Gateway,
    posting_text: str,
    ctx: CallContext,
    taxonomy: Taxonomy,
    prompt: Prompt | None = None,
) -> ParsedJob:
    prompt = prompt or load_prompt(*POSTING_PROMPT)
    result = gateway.parse(prompt, posting_user_content(posting_text), ParsedPosting, ctx)
    build = build_job_spec(result.output, posting_text, taxonomy)
    title = result.output.title.strip()[:300] if result.output.title else None
    return ParsedJob(
        title=title or None,
        spec=build.spec,
        report=ParseReport(
            prompt_ref=result.prompt_ref,
            model=result.model,
            model_served=result.model_served,
            cached=result.cached,
            rejected=build.rejected,
            duplicates_merged=build.duplicates_merged,
        ),
    )


def spec_problems(
    spec: JobSpec, posting_text: str, taxonomy: Taxonomy, *, confirm: bool
) -> list[str]:
    """Pure: everything wrong with a recruiter-edited spec. Empty means it can be saved."""
    problems: list[str] = []
    seen: set[str] = set()
    for skill in spec.must_have + spec.nice_to_have:
        if skill.skill_id is not None and skill.skill_id not in taxonomy:
            problems.append(f"unknown skill id {skill.skill_id}")
        key = skill.skill_id or f"unmapped:{term_key(skill.raw_text)}"
        if key in seen:
            problems.append(f"{skill.skill_id or skill.raw_text!r} is listed more than once")
        seen.add(key)
        if (skill.source_quote is None) != (skill.source_span is None):
            problems.append(f"{skill.raw_text!r}: source_quote and source_span go together")
        elif skill.source_span is not None:
            start, end = skill.source_span
            if not (0 <= start < end <= len(posting_text)) or (
                posting_text[start:end] != skill.source_quote
            ):
                problems.append(f"{skill.raw_text!r}: source_span does not match the posting")
    if confirm:
        if not spec.must_have:
            problems.append("a confirmed job needs at least one must-have skill")
        unmapped = spec.unmapped()
        if unmapped:
            problems.append(f"map or remove unmapped skills before confirming: {unmapped}")
    return problems
