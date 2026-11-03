from datetime import date
from pathlib import Path

import pytest

from app.evals.cv import evaluate_cvs, load_cv_cases, summarize_cvs
from app.evals.results import append_result_row
from app.llm.fake import FakeGateway
from app.taxonomy.loader import load_taxonomy
from tests.conftest import CVS, amira_cv_reply

TAXONOMY = load_taxonomy()


def test_ten_fictional_cvs_load_with_valid_labels():
    cases = load_cv_cases(CVS, TAXONOMY)
    assert len(cases) == 10
    slugs = [c.slug for c in cases]
    assert slugs[:2] == ["01-amira-haddad", "02-ben-okafor"]
    for case in cases:
        assert case.claimed
        assert case.received_on == date(2026, 10, 15)
        assert all(e.endswith("@example.com") for e in case.emails)


def write_case(tmp_path: Path, labels: str) -> Path:
    (tmp_path / "x.txt").write_text("Python\n", encoding="utf-8")
    (tmp_path / "x.labels.yaml").write_text(labels, encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        ("name: X\nreceived_on: 2026-10-15\nclaimed: [skill:nope]\n", "unknown ids"),
        ("name: X\nreceived_on: 2026-10-15\nclaimed: [Postgres]\n", "as skill:postgresql"),
        ("name: X\nreceived_on: 2026-10-15\nclaimed: [CMake, cmake]\n", "twice within"),
        (
            "name: X\nreceived_on: 2026-10-15\nclaimed: [skill:go]\nacceptable: [skill:go]\n",
            "labelled twice",
        ),
        (
            "name: X\nreceived_on: 2026-10-15\nclaimed: [skill:go]\n"
            "last_seen: {skill:rust: 2026-10-15}\n",
            "unclaimed",
        ),
        ("claimed: [skill:go]\n", "needs name"),
    ],
)
def test_bad_labels_are_refused(tmp_path: Path, labels: str, message: str):
    with pytest.raises(ValueError, match=message):
        load_cv_cases(write_case(tmp_path, labels), TAXONOMY)


def test_evaluate_scores_precision_recall_dates_rejections_and_leaks():
    [amira] = [c for c in load_cv_cases(CVS, TAXONOMY) if c.slug == "01-amira-haddad"]
    gateway = FakeGateway({"cv_extractor": amira_cv_reply()})
    [result] = evaluate_cvs(gateway, [amira], TAXONOMY)
    # The scripted reply finds everything except Git; nothing unexpected.
    assert result.misses == ("skill:git",)
    assert result.false_positives == ()
    assert (result.dates_right, result.dates_total) == (2, 2)
    assert result.rejected == 3
    assert result.leaks == ()
    summary = summarize_cvs([result])
    assert summary.claimed.recall == 12 / 13
    assert summary.claimed.precision == 1.0
    assert summary.quote_rejection_rate == 3 / result.checked
    assert summary.leaks == 0


def test_result_rows_go_into_the_right_table(tmp_path: Path):
    results = tmp_path / "RESULTS.md"
    results.write_text(
        "# Eval results\n\n## A\n\n| x |\n|---|\n| 1 |\n\n## B\n\nText.\n\n| y |\n|---|\n",
        encoding="utf-8",
    )
    append_result_row(results, "## A", "| 2 |")
    append_result_row(results, "## B", "| 3 |\n")
    assert results.read_text(encoding="utf-8") == (
        "# Eval results\n\n## A\n\n| x |\n|---|\n| 1 |\n| 2 |\n\n## B\n\nText.\n\n"
        "| y |\n|---|\n| 3 |\n"
    )
    with pytest.raises(ValueError, match="no heading"):
        append_result_row(results, "## C", "| 4 |")
