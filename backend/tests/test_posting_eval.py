from pathlib import Path

import pytest

from app.config import DEFAULT_TENANT_ID
from app.evals.posting import (
    Counts,
    PostingCase,
    compare,
    evaluate,
    label_key,
    load_cases,
    summarize,
)
from app.llm.fake import FakeGateway
from app.llm.gateway import CallContext
from app.taxonomy.loader import load_taxonomy
from tests.conftest import POSTING, posting_reply

TAXONOMY = load_taxonomy()
GOLDEN = Path(__file__).parents[2] / "evals" / "postings"


def test_golden_postings_load_and_labels_use_known_ids():
    cases = load_cases(GOLDEN, TAXONOMY)
    assert len(cases) == 5
    for case in cases:
        assert case.posting_text.strip()
        assert case.must_have


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        ("must_have: [skill:python]\nnice_to_have: [skill:python]\n", "labelled twice"),
        ("must_have: [skill:python, skill:python]\n", "labelled twice within"),
        ("must_have: [Airflow]\nacceptable: [airflow]\n", "labelled twice"),
        ("must_have: [Postgres]\n", "label 'Postgres' as skill:postgresql"),
    ],
)
def test_bad_labels_are_rejected(tmp_path: Path, labels: str, message: str):
    (tmp_path / "a.md").write_text("posting")
    (tmp_path / "a.labels.yaml").write_text(labels)
    with pytest.raises(ValueError, match=message):
        load_cases(tmp_path, TAXONOMY)


def test_unknown_label_ids_are_rejected(tmp_path: Path):
    (tmp_path / "a.md").write_text("posting")
    (tmp_path / "a.labels.yaml").write_text("must_have: [skill:snake]\n")
    with pytest.raises(ValueError, match="unknown ids"):
        load_cases(tmp_path, TAXONOMY)


def test_label_keys():
    assert label_key("skill:python") == "skill:python"
    assert label_key("AWS or GCP") == "term:aws or gcp"


def test_compare_counts_and_acceptable_items():
    counts = compare({"a", "b", "x", "ok"}, {"a", "b", "c"}, {"ok"})
    assert counts == Counts(tp=2, fp=1, fn=1)
    assert counts.precision == pytest.approx(2 / 3)
    assert counts.recall == pytest.approx(2 / 3)
    assert Counts().precision is None


def test_evaluate_end_to_end_with_fake_gateway():
    case = PostingCase(
        slug="northwind",
        posting_text=POSTING,
        must_have=frozenset({"skill:python", "skill:fastapi", "skill:postgresql", "skill:docker"}),
        nice_to_have=frozenset({"skill:aws", "skill:redis", "term:temporal"}),
        acceptable=frozenset(),
        seniority="unspecified",
    )
    gateway = FakeGateway({"posting_parser": posting_reply()})
    [result] = evaluate(gateway, [case], TAXONOMY, CallContext(tenant_id=DEFAULT_TENANT_ID))

    # Predicted must: python, fastapi, postgresql, temporal; nice: docker, aws, redis.
    assert result.must == Counts(tp=3, fp=1, fn=1)
    assert result.nice == Counts(tp=2, fp=1, fn=1)
    assert result.either == Counts(tp=7, fp=0, fn=0)
    assert (result.placed_right, result.placed_total) == (5, 7)
    assert result.seniority_ok
    assert result.rejected_quotes == 1
    assert result.unmapped == ("Temporal",)

    summary = summarize([result])
    assert summary.quote_rejection_rate == pytest.approx(1 / 8)
    assert summary.placement == pytest.approx(5 / 7)
