from pathlib import Path

import pytest

from app.evals.ranking import evaluate_posting, kendall_tau_b, load_golden
from app.taxonomy.loader import load_taxonomy

GOLDEN = Path(__file__).parents[2] / "evals" / "golden"
TAXONOMY = load_taxonomy()


def test_tau_same_and_reversed_order():
    assert kendall_tau_b([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1)
    assert kendall_tau_b([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1)


def test_tau_b_handles_ties():
    # Worked by hand: 5 pairs concordant, 0 discordant, 1 tied in the second list.
    assert kendall_tau_b([1, 2, 3, 4], [1, 1, 2, 3]) == pytest.approx(5 / (6 * 5) ** 0.5)


def test_tau_with_no_differences_is_undefined():
    assert kendall_tau_b([1, 1, 1], [1, 2, 3]) is None


def test_tau_needs_equal_lengths():
    with pytest.raises(ValueError, match="same length"):
        kendall_tau_b([1, 2], [1])


def test_golden_set_is_ten_candidates_for_two_postings():
    golden = load_golden(GOLDEN, TAXONOMY)
    assert len(golden.candidates) == 10
    assert len(golden.postings) == 2


def test_golden_depth_follows_the_rubric():
    golden = load_golden(GOLDEN, TAXONOMY)
    assert golden.candidates["amira-haddad"].depth == pytest.approx(0.815, abs=0.001)
    assert golden.candidates["ben-okafor"].depth == pytest.approx(0.2575, abs=0.001)
    assert golden.candidates["grace-lindqvist"].depth == 0


def test_every_candidate_gets_a_score_for_every_posting():
    golden = load_golden(GOLDEN, TAXONOMY)
    for posting in golden.postings:
        result = evaluate_posting(golden, posting, TAXONOMY)
        assert len(result.rows) == 10
        assert result.tau is not None and -1 <= result.tau <= 1


def test_a_ranking_missing_a_candidate_is_refused(tmp_path: Path):
    for name in ("postings.yaml", "candidates.yaml"):
        (tmp_path / name).write_text((GOLDEN / name).read_text(encoding="utf-8"), encoding="utf-8")
    postings = tmp_path / "postings.yaml"
    postings.write_text(
        postings.read_text(encoding="utf-8").replace("[grace-lindqvist]  # no overlap", "[]"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="every candidate once"):
        load_golden(tmp_path, TAXONOMY)
