"""Data minimization guard: no table column or schema field may hold photos, dates of birth or
age, gender, marital status, nationality, religion, street addresses or postal codes.

The uploaded CV and its extracted text, and fetched GitHub file contents, are kept as the
source the locators point into; what the system extracts, stores as structured data, or sends
to the model must not include these. GitHub API answers are reduced to the fields the
collector uses before they are cached (tests/test_github_source.py checks that).
"""

import re
from typing import Any

from pydantic import BaseModel

from app.db.base import Base
from app.schemas.candidate import CandidateEvidenceOut, CandidateOut
from app.schemas.evidence import CvProfile, EvidenceItem, IngestReport, ParsedCv
from app.schemas.github import GitHubProfile, ParsedReadme
from app.schemas.review import OverrideOut, RankingsOut

SENSITIVE = re.compile(
    r"photo|picture|image|avatar|birth|dob|age$|^age|gender|sex$|marital|spouse|children"
    r"|nationality|citizenship|religion|ethnic|race|address|postal|zip|street",
    re.IGNORECASE,
)
# Reviewed by hand: names the pattern catches that hold nothing sensitive.
ALLOWED = {"coverage"}  # which must-haves have evidence


def field_names(schema: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for definition in [schema, *schema.get("$defs", {}).values()]:
        names |= set(definition.get("properties", {}))
    return names


def test_no_table_has_a_sensitive_column():
    columns = {f"{t.name}.{c.name}" for t in Base.metadata.tables.values() for c in t.columns}
    assert not {c for c in columns if SENSITIVE.search(c.split(".")[1])}


def test_no_schema_has_a_sensitive_field():
    models: list[type[BaseModel]] = [
        ParsedCv,  # what the model may return
        ParsedReadme,
        CvProfile,
        GitHubProfile,
        EvidenceItem,
        IngestReport,
        CandidateOut,
        CandidateEvidenceOut,
        RankingsOut,
        OverrideOut,
    ]
    for model in models:
        names = field_names(model.model_json_schema()) - ALLOWED
        flagged = {n for n in names if SENSITIVE.search(n)}
        assert not flagged, model.__name__
