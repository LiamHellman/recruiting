from pathlib import Path

import pytest

from app.evals.cv import redaction_leaks
from app.evals.readme import evaluate_readmes, load_readme_cases, summarize_readmes
from app.llm.fake import FakeGateway
from app.schemas.github import ParsedReadme, ReadmeSkillMention
from app.sources.github import redact_readme
from app.taxonomy.loader import load_taxonomy

TAXONOMY = load_taxonomy()
READMES = Path(__file__).parents[2] / "evals" / "readmes"


def test_eight_fictional_readmes_load_with_valid_labels():
    cases = load_readme_cases(READMES, TAXONOMY)
    assert len(cases) == 8
    assert all(e.endswith("@example.com") for c in cases for e in c.emails)
    assert sum(1 for c in cases if not c.uses) == 1  # one README with nothing to find


def test_no_author_name_or_email_reaches_the_model():
    for case in load_readme_cases(READMES, TAXONOMY):
        names = (case.name,) if case.name else ()
        sent = redact_readme(case.text, names, case.emails, TAXONOMY).text
        assert redaction_leaks(sent, case.name, case.emails, TAXONOMY) == [], case.slug
        assert "@example.com" not in sent, case.slug


def test_a_faithful_extractor_scores_perfectly_and_a_bad_quote_is_counted():
    cases = [c for c in load_readme_cases(READMES, TAXONOMY) if c.slug == "03-go-shortener"]

    def reply(content: str) -> ParsedReadme:
        mentions = [
            ("Go", "A URL shortener in Go"),
            ("Redis", "stored in Redis"),
            ("gRPC", "speaks gRPC"),
            ("Prometheus", "exported for Prometheus"),
            ("Helm", "a Helm chart"),
            ("Kubernetes", "running it on Kubernetes"),
            ("Kafka", "click events to Kafka in production"),  # not in the README
        ]
        return ParsedReadme(skills=[ReadmeSkillMention(skill=s, quote=q) for s, q in mentions])

    [result] = evaluate_readmes(FakeGateway({"readme_extractor": reply}), cases, TAXONOMY)
    assert (result.counts.precision, result.counts.recall) == (1.0, 1.0)
    assert (result.rejected, result.checked) == (1, 7)
    summary = summarize_readmes([result])
    assert summary.quote_rejection_rate == pytest.approx(1 / 7)
    assert summary.leaks == 0


def test_labels_must_use_known_ids_and_unmapped_terms(tmp_path: Path):
    (tmp_path / "x.md").write_text("Uses Postgres.\n", encoding="utf-8")
    (tmp_path / "x.labels.yaml").write_text("uses: [Postgres]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="as skill:postgresql"):
        load_readme_cases(tmp_path, TAXONOMY)
