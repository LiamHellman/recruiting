import pytest

from app.evals.cv import load_cv_cases, redaction_leaks
from app.extraction.redact import find_emails, find_links, redact
from app.taxonomy.loader import load_taxonomy
from tests.conftest import CVS

TAXONOMY = load_taxonomy()


def is_skill(term: str) -> bool:
    return TAXONOMY.lookup(term) is not None


def test_contacts_are_pulled_out_deterministically():
    text = (
        "AMIRA HADDAD · amira.haddad@Example.com · github.com/amira-h\n"
        "Site: https://amira.dev/blog). Built Node.js/Express apps; see (www.example.org/x)."
    )
    assert find_emails(text) == ["amira.haddad@example.com"]
    assert find_links(text) == [
        "https://github.com/amira-h",
        "https://amira.dev/blog",
        "https://www.example.org/x",
    ]


def test_redacts_name_contacts_and_personal_details():
    text = (
        "AMIRA HADDAD\n"
        "Montréal, QC H2X 1Y4 · +1 (514) 555-0142 · amira.haddad@example.com\n"
        "github.com/amira-h · amira.dev/\n"
        "Address: 14 Harbour Street, Halifax\n"
        "Date of birth: 12 March 1984\n"
        "Nationality: Swedish\n"
        "Mobile: Swift, Kotlin\n"
        "Amira led the Django migration in 2019 - 2021 and 2018.01 - 2020.12."
    )
    redacted = redact(text, names=["Amira Haddad"], is_skill_term=is_skill)
    assert redacted.text == (
        "[name]\n"
        "Montréal, QC [postal_code] · [phone] · [email]\n"
        "[link] · [link]\n"
        "Address: [address]\n"
        "Date of birth: [personal_detail]\n"
        "Nationality: [personal_detail]\n"
        "Mobile: Swift, Kotlin\n"
        "[name] led the Django migration in 2019 - 2021 and 2018.01 - 2020.12."
    )
    assert redacted.counts() == {
        "name": 2,
        "postal_code": 1,
        "phone": 1,
        "email": 1,
        "link": 2,
        "address": 1,
        "personal_detail": 2,
    }


def test_a_name_that_is_also_a_skill_is_only_redacted_as_a_full_name():
    text = "Ruby Castellanos\nRuby has been my main language. Castellanos wrote Ruby gems."
    redacted = redact(text, names=["Ruby Castellanos"], is_skill_term=is_skill)
    assert redacted.text == "[name]\nRuby has been my main language. [name] wrote Ruby gems."


def test_offsets_map_back_to_the_original_text():
    text = "Contact: amira@example.com. Python and Django at Lumen."
    redacted = redact(text)
    assert redacted.text == "Contact: [email]. Python and Django at Lumen."
    start = redacted.text.index("Python and Django")
    span = redacted.to_original(start, start + len("Python and Django"))
    assert text[span[0] : span[1]] == "Python and Django"
    # A span covering a placeholder widens to the whole redacted value.
    start = redacted.text.index("[email]")
    span = redacted.to_original(start + 1, start + 4)
    assert text[span[0] : span[1]] == "amira@example.com"
    span = redacted.to_original(0, len(redacted.text))
    assert span == (0, len(text))


def test_text_without_personal_data_is_unchanged():
    text = "Python, Django, PostgreSQL (2019 - 2021)\nDocker, AWS"
    redacted = redact(text, names=["Nobody Here"])
    assert redacted.text == text
    assert redacted.to_original(3, 10) == (3, 10)


def test_spans_outside_the_text_are_refused():
    with pytest.raises(ValueError, match="outside"):
        redact("abc").to_original(2, 9)


def test_no_fixture_cv_leaks_name_or_email_to_the_model():
    for case in load_cv_cases(CVS, TAXONOMY):
        redacted = redact(
            case.text,
            names=[case.name],
            emails=[*case.emails, *find_emails(case.text)],
            is_skill_term=is_skill,
        )
        assert redaction_leaks(redacted.text, case.name, case.emails, TAXONOMY) == [], case.slug
        assert not find_emails(redacted.text), case.slug
        assert not find_links(redacted.text), case.slug


def test_european_style_personal_details_never_reach_the_model():
    [case] = [c for c in load_cv_cases(CVS, TAXONOMY) if c.slug == "07-grace-lindqvist"]
    sent = redact(case.text, names=[case.name], is_skill_term=is_skill).text
    for detail in ("1984", "Swedish\n", "Married", "Harbour", "B3J", "555 0110"):
        assert detail not in sent
    assert "C# with ASP.NET Core" in sent
