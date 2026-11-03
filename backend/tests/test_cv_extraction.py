import uuid
from datetime import date

from app.config import DEFAULT_TENANT_ID
from app.extraction.cv import NO_MENTION, build_cv_evidence, mentions_skill
from app.extraction.quotes import NOT_FOUND, OUTSIDE
from app.extraction.redact import find_emails, redact
from app.llm.fake import FakeGateway
from app.schemas.evidence import CvEntry, EvidenceItem, ParsedCv
from app.sources.base import CandidateRef
from app.sources.cv import CvPayload, CvSource
from app.taxonomy.loader import load_taxonomy
from tests.conftest import amira_cv_reply, fixture_cv, mention

TAXONOMY = load_taxonomy()
RECEIVED = date(2026, 10, 15)
DOC_ID = "3f1c2b9e-0000-4000-8000-000000000001"
AMIRA = fixture_cv("01-amira-haddad")


def build(parsed: ParsedCv | None = None, text: str = AMIRA):
    redacted = redact(text, names=["Amira Haddad"])
    return build_cv_evidence(parsed or amira_cv_reply(), redacted, DOC_ID, RECEIVED, TAXONOMY)


def span(item: EvidenceItem) -> tuple[int, int]:
    prefix = f"cv:{DOC_ID}#"
    assert item.locator.startswith(prefix)
    start, end = item.locator.removeprefix(prefix).split("-")
    return int(start), int(end)


def line_at(text: str, start: int, end: int) -> str:
    return text[text.rfind("\n", 0, start) + 1 : text.find("\n", end)]


def test_every_item_is_claimed_cv_evidence_with_full_ownership():
    result = build()
    assert result.evidence
    for item in result.evidence:
        assert (item.level, item.source, item.ownership, item.confidence) == (
            "claimed",
            "cv",
            1.0,
            1.0,
        )


def test_locators_highlight_the_cv_line_that_names_the_skill():
    result = build()
    for item in result.evidence:
        start, end = span(item)
        assert AMIRA[start:end] == item.quote
        name = TAXONOMY.skills[item.skill_id].name
        assert name.casefold() in line_at(AMIRA, start, end).casefold()
    [pytest_item] = [i for i in result.evidence if i.skill_id == "skill:pytest"]
    start, end = span(pytest_item)
    assert (
        line_at(AMIRA, start, end)
        == "- Wrote unit tests with pytest; raised coverage from 40% to 85%."
    )


def test_skills_take_their_entrys_end_date_and_skills_sections_are_current():
    latest: dict[str, date] = {}
    for item in build().evidence:
        assert item.date_last_seen is not None
        latest[item.skill_id] = max(item.date_last_seen, latest.get(item.skill_id, date.min))
    assert latest["skill:pytest"] == date(2021, 12, 31)  # Pixelbay: Jun 2019 to Dec 2021
    assert latest["skill:celery"] == date(2021, 12, 31)
    assert latest["skill:docker"] == RECEIVED  # Lumen Freight: Jan 2022 to Present
    assert latest["skill:python"] == RECEIVED  # skills section


def test_invalid_quotes_are_rejected_and_counted_with_reasons():
    result = build()
    rejected = {(r.item, r.reason) for r in result.rejected}
    assert rejected == {
        ("Kubernetes", NOT_FOUND),
        ("Docker", OUTSIDE),  # quoted from the Lumen Freight entry, filed under Pixelbay
        ("MySQL", NO_MENTION),
    }
    reply = amira_cv_reply()
    mentions = sum(len(e.skills) for e in reply.entries) + len(reply.other_skills)
    assert result.checked == len(reply.entries) + len(reply.education) + mentions
    assert "skill:kubernetes" not in {i.skill_id for i in result.evidence}
    assert "skill:mysql" not in {i.skill_id for i in result.evidence}


def test_unknown_terms_stay_unmapped_and_are_not_evidence():
    result = build()
    assert result.unmapped_terms == ["Django REST Framework"]


def test_duplicates_within_an_entry_are_merged_but_kept_across_entries():
    reply = amira_cv_reply()
    reply.entries[0].skills.append(
        mention("Django", "Designed the shipment-tracking API in Django")
    )
    result = build(reply)
    assert result.duplicates_merged == 1
    postgres = [i for i in result.evidence if i.skill_id == "skill:postgresql"]
    assert len(postgres) == 2  # Lumen Freight and the django-tablekit project


def test_profile_has_entries_and_education_without_dates():
    profile = build().profile
    assert [(e.organization, e.start, e.end, e.last_seen) for e in profile.entries] == [
        ("Lumen Freight", "Jan 2022", "Present", RECEIVED),
        ("Pixelbay Studio", "Jun 2019", "Dec 2021", date(2021, 12, 31)),
        ("django-tablekit", "2021", "Present", RECEIVED),
    ]
    start, end = profile.entries[0].span
    assert AMIRA[start:end] == "Senior Backend Developer — Lumen Freight, Montréal"
    [education] = profile.education
    assert education.credential == "B.Sc."
    assert "start" not in education.model_dump()  # graduation years are an age proxy


def test_entry_with_an_unverifiable_heading_gives_undated_evidence():
    reply = ParsedCv(
        entries=[
            CvEntry(
                kind="job",
                title="CTO",
                organization="Invented Corp",
                start="2010",
                end="Present",
                heading_quote="CTO — Invented Corp",
                skills=[mention("Celery", "Moved background jobs from cron to Celery")],
            )
        ],
        education=[],
        other_skills=[],
    )
    result = build(reply)
    assert [r.reason for r in result.rejected] == [f"heading {NOT_FOUND}"]
    [item] = result.evidence
    assert (item.skill_id, item.date_last_seen) == ("skill:celery", None)
    assert result.profile.entries == []


def test_quote_must_name_the_skill_or_an_alias_as_a_whole_word():
    assert mentions_skill("Deployed on k8s clusters", "Kubernetes", "skill:kubernetes", TAXONOMY)
    assert mentions_skill("Python 3 and Django", "Python 3", "skill:python", TAXONOMY)
    assert not mentions_skill("Worked at Google", "Go", "skill:go", TAXONOMY)
    assert mentions_skill("Services in Go and Rust", "Go", "skill:go", TAXONOMY)
    assert mentions_skill("C/C++ firmware", "C", "skill:c", TAXONOMY)
    assert not mentions_skill("Wrote tests", "Zephyr", None, TAXONOMY)


def test_quotes_over_placeholders_map_to_the_original_but_never_contain_the_name():
    text = "Amira Haddad built Django apps.\nSkills: Python"
    reply = ParsedCv(
        entries=[],
        education=[],
        other_skills=[mention("Django", "[name] built Django apps.")],
    )
    result = build(reply, text)
    [item] = result.evidence
    assert item.quote == "[name] built Django apps."
    assert span(item) == (0, len("Amira Haddad built Django apps."))


def test_cv_source_sends_only_redacted_text_and_reports_the_run():
    gateway = FakeGateway({"cv_extractor": amira_cv_reply()})
    source = CvSource(gateway, TAXONOMY)
    candidate = CandidateRef(
        id=uuid.uuid4(),
        tenant_id=DEFAULT_TENANT_ID,
        job_id=uuid.uuid4(),
        names=("Amira Haddad",),
        emails=("amira.haddad@example.com",),
    )
    payload = CvPayload(
        document_id=uuid.UUID(DOC_ID),
        text=AMIRA,
        received_on=RECEIVED,
        emails=tuple(find_emails(AMIRA)),
    )
    result = source.ingest(candidate, payload)

    [call] = gateway.calls
    assert call.prompt_ref == "cv_extractor/v1"
    assert call.user_content.startswith("<cv>\n[name]\n")
    for secret in ("Amira", "AMIRA", "HADDAD", "amira.haddad@", "555-0142", "H2X"):
        assert secret not in call.user_content
    assert result.report.prompt_ref == "cv_extractor/v1"
    assert result.report.accepted == len(result.evidence)
    assert len(result.report.rejected) == 3
    assert result.report.rejection_rate == 3 / result.report.checked
    assert result.report.redactions["name"] == 1
    assert result.profile is not None and result.profile["document_id"] == DOC_ID

    # Same CV again: served from the cache, same evidence.
    again = source.ingest(candidate, payload)
    assert len(gateway.calls) == 1
    assert again.report.cached is True
    assert again.evidence == result.evidence
