import uuid

from app.config import DEFAULT_TENANT_ID
from app.extraction.posting import build_job_spec, parse_posting, spec_problems
from app.extraction.quotes import locate_quote
from app.llm.fake import FakeGateway
from app.llm.gateway import CallContext
from app.schemas.job import JobSkill, JobSpec, ParsedPosting
from app.taxonomy.loader import load_taxonomy
from tests.conftest import POSTING, posting_reply, req

TAXONOMY = load_taxonomy()


# --- Quote locator ------------------------------------------------------------------------


def test_locate_exact_quote():
    text = "Experience with PostgreSQL and Redis."
    start, end = locate_quote(text, "PostgreSQL and Redis") or (0, 0)
    assert text[start:end] == "PostgreSQL and Redis"


def test_locate_quote_ignores_whitespace_differences():
    text = "Strong   Python\n  skills required"
    span = locate_quote(text, "Strong Python skills")
    assert span is not None
    assert text[span[0] : span[1]] == "Strong   Python\n  skills"


def test_locate_quote_is_otherwise_verbatim():
    text = "Strong Python skills"
    assert locate_quote(text, "strong python skills") is None
    assert locate_quote(text, "Strong Python skill set") is None
    assert locate_quote(text, "   ") is None


# --- build_job_spec -----------------------------------------------------------------------


def test_build_maps_known_skills_and_keeps_unknown_ones_unmapped():
    build = build_job_spec(posting_reply(), POSTING, TAXONOMY)
    must = {s.raw_text: s.skill_id for s in build.spec.must_have}
    nice = {s.raw_text: s.skill_id for s in build.spec.nice_to_have}
    assert must == {
        "Python": "skill:python",
        "FastAPI": "skill:fastapi",
        "PostgreSQL": "skill:postgresql",
        "Temporal": None,
    }
    assert nice == {"Docker": "skill:docker", "AWS": "skill:aws", "Redis": "skill:redis"}
    assert build.spec.unmapped() == ["Temporal"]


def test_build_rejects_quotes_not_in_the_posting():
    build = build_job_spec(posting_reply(), POSTING, TAXONOMY)
    assert [(r.skill, r.reason) for r in build.rejected] == [
        ("Kubernetes", "quote not found in posting")
    ]
    assert "skill:kubernetes" not in {s.skill_id for s in build.spec.must_have}


def test_build_merges_duplicates_in_favour_of_must_have():
    build = build_job_spec(posting_reply(), POSTING, TAXONOMY)
    assert build.duplicates_merged == 1
    assert "skill:python" not in {s.skill_id for s in build.spec.nice_to_have}


def test_build_merges_aliases_of_the_same_skill():
    parsed = ParsedPosting(
        title=None,
        must_have=[req("PostgreSQL", "designing schemas in PostgreSQL")],
        nice_to_have=[req("Postgres", "PostgreSQL")],
        seniority="unspecified",
        domain=None,
    )
    build = build_job_spec(parsed, POSTING, TAXONOMY)
    assert len(build.spec.must_have) == 1
    assert build.spec.nice_to_have == []


def test_build_records_source_spans_and_default_weights():
    build = build_job_spec(posting_reply(), POSTING, TAXONOMY)
    for skill in build.spec.must_have + build.spec.nice_to_have:
        assert skill.weight == 1.0
        assert skill.source_span is not None
        start, end = skill.source_span
        assert POSTING[start:end] == skill.source_quote


def test_build_rejects_empty_skill_names():
    parsed = ParsedPosting(
        title=None,
        must_have=[req("  ", "Strong Python skills")],
        nice_to_have=[],
        seniority="senior",
        domain="  ",
    )
    build = build_job_spec(parsed, POSTING, TAXONOMY)
    assert build.spec.must_have == []
    assert build.rejected[0].reason == "empty skill name"
    assert build.spec.domain is None
    assert build.spec.seniority == "senior"


# --- parse_posting through the gateway ----------------------------------------------------


def test_parse_posting_uses_versioned_prompt_and_wraps_the_posting():
    gateway = FakeGateway({"posting_parser": posting_reply()})
    ctx = CallContext(tenant_id=DEFAULT_TENANT_ID, job_id=uuid.uuid4())
    parsed = parse_posting(gateway, POSTING, ctx, TAXONOMY)
    assert parsed.title == "Backend Engineer (Python)"
    assert parsed.report.prompt_ref == "posting_parser/v1"
    assert parsed.report.model == "claude-opus-5-5"
    assert len(parsed.report.rejected) == 1
    assert gateway.calls[0].user_content == f"<posting>\n{POSTING}\n</posting>"


# --- spec_problems ------------------------------------------------------------------------


def spec(*must: JobSkill, nice: tuple[JobSkill, ...] = ()) -> JobSpec:
    return JobSpec(must_have=list(must), nice_to_have=list(nice))


def test_valid_edit_has_no_problems():
    edited = spec(JobSkill(raw_text="Python", skill_id="skill:python", weight=2))
    assert spec_problems(edited, POSTING, TAXONOMY, confirm=True) == []


def test_confirm_requires_everything_mapped():
    edited = spec(JobSkill(raw_text="Temporal"))
    assert spec_problems(edited, POSTING, TAXONOMY, confirm=False) == []
    problems = spec_problems(edited, POSTING, TAXONOMY, confirm=True)
    assert any("unmapped" in p for p in problems)


def test_confirm_requires_a_must_have():
    edited = spec(nice=(JobSkill(raw_text="Docker", skill_id="skill:docker"),))
    assert any("must-have" in p for p in spec_problems(edited, POSTING, TAXONOMY, confirm=True))


def test_unknown_skill_id_and_duplicates():
    edited = spec(
        JobSkill(raw_text="Python", skill_id="skill:python"),
        JobSkill(raw_text="Snake", skill_id="skill:snake"),
        nice=(JobSkill(raw_text="python", skill_id="skill:python"),),
    )
    problems = spec_problems(edited, POSTING, TAXONOMY, confirm=False)
    assert any("unknown skill id skill:snake" in p for p in problems)
    assert any("more than once" in p for p in problems)


def test_forged_source_span_is_rejected():
    edited = spec(
        JobSkill(
            raw_text="Python",
            skill_id="skill:python",
            source_quote="Python expert required",
            source_span=(0, 22),
        )
    )
    problems = spec_problems(edited, POSTING, TAXONOMY, confirm=False)
    assert any("does not match the posting" in p for p in problems)
