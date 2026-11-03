from datetime import date

import pytest

from app.extraction.dates import period_end
from app.extraction.quotes import (
    EMPTY,
    NOT_FOUND,
    OUTSIDE,
    TOO_LONG,
    check_quote,
    locate_quote,
)

TEXT = "Lumen Freight\n- Built   APIs in Django.\n\nPixelbay\n- Wrote tests with pytest."


def test_quote_matches_verbatim_up_to_whitespace():
    span = locate_quote(TEXT, "Built APIs\nin Django.")
    assert span is not None
    assert TEXT[span[0] : span[1]] == "Built   APIs in Django."


def test_quote_search_can_be_limited_to_a_window():
    pixelbay = TEXT.index("Pixelbay")
    assert locate_quote(TEXT, "Built APIs", pixelbay) is None
    span = locate_quote(TEXT, "pytest", pixelbay, len(TEXT))
    assert span is not None
    assert TEXT[span[0] : span[1]] == "pytest"


@pytest.mark.parametrize(
    ("quote", "within", "reason"),
    [
        ("", None, EMPTY),
        ("   ", None, EMPTY),
        ("x" * 501, None, TOO_LONG),
        ("Built APIs in Flask.", None, NOT_FOUND),
        ("built apis in django.", None, NOT_FOUND),  # case matters
        ("Built APIs in Django.", (TEXT.index("Pixelbay"), len(TEXT)), OUTSIDE),
    ],
)
def test_rejections_carry_a_reason(quote: str, within: tuple[int, int] | None, reason: str):
    check = check_quote(TEXT, quote, within=within)
    assert not check.ok
    assert check.span is None
    assert check.reason == reason


def test_accepted_quote_has_a_span_and_no_reason():
    check = check_quote(
        TEXT, "Wrote tests with pytest.", within=(TEXT.index("Pixelbay"), len(TEXT))
    )
    assert check.ok
    assert check.reason is None


TODAY = date(2026, 10, 15)


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        ("2019", date(2019, 12, 31)),
        ("Mar 2021", date(2021, 3, 31)),
        ("March 2021", date(2021, 3, 31)),
        ("Sept. 2024", date(2024, 9, 30)),
        ("Feb 2024", date(2024, 2, 29)),
        ("03/2021", date(2021, 3, 31)),
        ("2021-03", date(2021, 3, 31)),
        ("Q2 2020", date(2020, 6, 30)),
        ("Summer 2023", date(2023, 8, 31)),
        ("Present", TODAY),
        ("currently", TODAY),
        ("2026", TODAY),  # never later than the day the CV arrived
        ("Dec 2030", TODAY),
        ("13/2021", None),
        ("expected April 2027", None),
        ("soon", None),
        ("", None),
        (None, None),
    ],
)
def test_period_end(written: str | None, expected: date | None):
    assert period_end(written, today=TODAY) == expected
