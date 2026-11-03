"""README-extractor eval: compare the technologies extracted from a README with hand labels
in evals/readmes/.

Runs the production path (`redact_readme` -> `readme_extractor` -> `build_readme_mentions`),
so it measures the prompt, the quote validator and alias mapping together. Besides
precision/recall it reports the quote rejection rate and redaction leaks (fragments of the
author's name or email in the text sent to the model; must be zero).
"""

import uuid
from dataclasses import dataclass
from pathlib import Path

import yaml

from app.config import DEFAULT_TENANT_ID
from app.evals.cv import redaction_leaks
from app.evals.posting import Counts, compare, label_key
from app.extraction.readme import README_PROMPT, build_readme_mentions, readme_user_content
from app.llm.gateway import CallContext, Gateway
from app.llm.prompts import Prompt, load_prompt
from app.schemas.github import ParsedReadme
from app.sources.github import redact_readme
from app.taxonomy.loader import Taxonomy, term_key
from app.taxonomy.normalize import normalize_term


@dataclass(frozen=True)
class ReadmeCase:
    slug: str
    text: str
    name: str  # the author, as the candidate would be; "" if none
    emails: tuple[str, ...]
    uses: frozenset[str]
    acceptable: frozenset[str]


def load_readme_cases(directory: Path, taxonomy: Taxonomy) -> list[ReadmeCase]:
    cases = []
    for path in sorted(directory.glob("*.md")):
        labels_path = path.with_suffix(".labels.yaml")
        labels = yaml.safe_load(labels_path.read_text(encoding="utf-8")) or {}
        keys: dict[str, frozenset[str]] = {}
        for list_name in ("uses", "acceptable"):
            items = [str(item) for item in labels.get(list_name) or []]
            for item in items:
                if item.startswith("skill:") and item not in taxonomy:
                    raise ValueError(f"{labels_path.name}: unknown ids [{item!r}]")
                if not item.startswith("skill:") and (mapped := normalize_term(item, taxonomy)):
                    raise ValueError(f"{labels_path.name}: label {item!r} as {mapped}")
            keys[list_name] = frozenset(label_key(item) for item in items)
            if len(keys[list_name]) != len(items):
                raise ValueError(f"{labels_path.name}: labelled twice within {list_name}")
        if overlap := keys["uses"] & keys["acceptable"]:
            raise ValueError(f"{labels_path.name}: labelled twice: {sorted(overlap)}")
        cases.append(
            ReadmeCase(
                slug=path.stem,
                text=path.read_text(encoding="utf-8"),
                name=str(labels.get("name") or ""),
                emails=tuple(str(e) for e in labels.get("emails") or []),
                uses=keys["uses"],
                acceptable=keys["acceptable"],
            )
        )
    return cases


@dataclass(frozen=True)
class ReadmeCaseResult:
    slug: str
    counts: Counts
    false_positives: tuple[str, ...]
    misses: tuple[str, ...]
    checked: int
    rejected: int
    leaks: tuple[str, ...]


def evaluate_readmes(
    gateway: Gateway, cases: list[ReadmeCase], taxonomy: Taxonomy, prompt: Prompt | None = None
) -> list[ReadmeCaseResult]:
    prompt = prompt or load_prompt(*README_PROMPT)
    results = []
    for case in cases:
        names = (case.name,) if case.name else ()
        redacted = redact_readme(case.text, names, case.emails, taxonomy)
        output = gateway.parse(
            prompt,
            readme_user_content(redacted.text),
            ParsedReadme,
            CallContext(tenant_id=DEFAULT_TENANT_ID, job_id=uuid.UUID(int=0)),
        ).output
        build = build_readme_mentions(output, redacted, taxonomy)
        predicted = {m.skill_id for m in build.mentions} | {
            f"term:{term_key(t)}" for t in build.unmapped_terms
        }
        leaks = redaction_leaks(redacted.text, case.name, case.emails, taxonomy)
        results.append(
            ReadmeCaseResult(
                slug=case.slug,
                counts=compare(predicted, case.uses, case.acceptable),
                false_positives=tuple(sorted(predicted - case.uses - case.acceptable)),
                misses=tuple(sorted(case.uses - predicted)),
                checked=build.checked,
                rejected=len(build.rejected),
                leaks=tuple(leaks),
            )
        )
    return results


@dataclass(frozen=True)
class ReadmeSummary:
    cases: int
    counts: Counts
    quote_rejection_rate: float | None
    leaks: int


def summarize_readmes(results: list[ReadmeCaseResult]) -> ReadmeSummary:
    checked = sum(r.checked for r in results)
    return ReadmeSummary(
        cases=len(results),
        counts=sum((r.counts for r in results), Counts()),
        quote_rejection_rate=sum(r.rejected for r in results) / checked if checked else None,
        leaks=sum(len(r.leaks) for r in results),
    )
