import uuid
from dataclasses import dataclass, field
from datetime import date

import pytest

from app.config import DEFAULT_TENANT_ID
from app.schemas.evidence import EvidenceItem, EvidenceSource, IngestReport
from app.sources.base import (
    CandidateRef,
    ConsentMissing,
    IngestResult,
    SourceRegistry,
    UnknownSource,
)

CANDIDATE = CandidateRef(id=uuid.uuid4(), tenant_id=DEFAULT_TENANT_ID, job_id=None)


def item(source: EvidenceSource) -> EvidenceItem:
    return EvidenceItem(
        skill_id="skill:python",
        level="claimed",
        source=source,
        locator="cv:doc#0-6",
        quote="Python",
        date_last_seen=date(2026, 10, 15),
        ownership=1.0,
        confidence=1.0,
    )


@dataclass
class RecordingSource:
    name: EvidenceSource
    label: EvidenceSource | None = None
    payloads: list[object] = field(default_factory=list)

    def input_key(self, candidate: CandidateRef, payload: object) -> str:
        return f"{self.name}:{payload}"

    def ingest(self, candidate: CandidateRef, payload: object) -> IngestResult:
        self.payloads.append(payload)
        report = IngestReport(
            source=self.name, checked=1, rejected=[], accepted=1, unmapped_terms=[]
        )
        return IngestResult(evidence=[item(self.label or self.name)], report=report)


def test_a_source_without_active_consent_never_sees_the_payload():
    cv = RecordingSource("cv")
    registry = SourceRegistry([cv])
    with pytest.raises(ConsentMissing, match="'cv'"):
        registry.ingest(CANDIDATE, "cv", "the CV text", consented=frozenset({"github"}))
    assert cv.payloads == []


def test_a_consented_source_runs():
    cv = RecordingSource("cv")
    result = SourceRegistry([cv]).ingest(CANDIDATE, "cv", "text", consented={"cv"})
    assert cv.payloads == ["text"]
    assert [e.source for e in result.evidence] == ["cv"]


def test_consent_is_checked_before_the_source_is_looked_up():
    with pytest.raises(ConsentMissing):
        SourceRegistry([]).ingest(CANDIDATE, "github", "handle", consented=set())
    with pytest.raises(UnknownSource):
        SourceRegistry([]).ingest(CANDIDATE, "github", "handle", consented={"github"})


def test_a_source_cannot_label_evidence_as_another_source():
    registry = SourceRegistry([RecordingSource("site", label="github")])
    with pytest.raises(ValueError, match="labelled"):
        registry.ingest(CANDIDATE, "site", "url", consented={"site"})


def test_sources_register_once():
    with pytest.raises(ValueError, match="twice"):
        SourceRegistry([RecordingSource("cv"), RecordingSource("cv")])
