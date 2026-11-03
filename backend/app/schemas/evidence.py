from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.job import SkillId

EvidenceSource = Literal["cv", "github", "site", "linkedin_pdf", "linkedin_export", "work_sample"]
EvidenceLevel = Literal["claimed", "demonstrated", "deep"]

EVIDENCE_SOURCES: tuple[EvidenceSource, ...] = (
    "cv",
    "github",
    "site",
    "linkedin_pdf",
    "linkedin_export",
    "work_sample",
)


class EvidenceItem(BaseModel):
    """One cited piece of skill evidence. The scorer sees only these, never the source text,
    and never branches on `source`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    skill_id: SkillId
    level: EvidenceLevel
    source: EvidenceSource
    # Where the quote is, e.g. "cv:<document id>#1203-1260" (character offsets, [start, end)).
    locator: str = Field(min_length=1, max_length=500)
    # Verbatim text from what the extractor was given (after redaction, for CVs).
    quote: str = Field(min_length=1, max_length=1000)
    # None when the source gives no usable date.
    date_last_seen: date | None
    ownership: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)


class RejectedItem(BaseModel):
    """An extracted item that failed validation. Kept with the run, never in the audit log,
    because `quote` is source text."""

    item: str
    quote: str
    reason: str


class IngestReport(BaseModel):
    source: EvidenceSource
    prompt_ref: str | None = None
    model: str | None = None
    model_served: str | None = None
    cached: bool | None = None
    input_sha256: str | None = None  # hash of the exact (redacted) text sent to the model
    checked: int  # extracted items whose quote was validated
    rejected: list[RejectedItem]
    accepted: int  # evidence items produced
    duplicates_merged: int = 0
    unmapped_terms: list[str]  # skill terms with no taxonomy match; never auto-added
    redactions: dict[str, int] = Field(default_factory=dict)  # category -> count
    # Network sources: requests sent, 304s, answers served from cache, retries, seconds waited.
    fetch: dict[str, int] = Field(default_factory=dict)

    @property
    def rejection_rate(self) -> float | None:
        return len(self.rejected) / self.checked if self.checked else None


# --- LLM output schema (cv_extractor) -----------------------------------------------------


class CvSkillMention(BaseModel):
    skill: str = Field(
        description=(
            "One technology, language, framework, tool, platform or technical practice, "
            "named exactly as the CV writes it."
        )
    )
    quote: str = Field(
        description=(
            "The shortest excerpt of the CV, copied character for character, that mentions "
            "this skill. It must contain the skill name as written."
        )
    )


EntryKind = Literal["job", "internship", "project", "volunteering", "other"]


class CvEntry(BaseModel):
    kind: EntryKind
    title: str | None = Field(description="Role or project title as written, or null.")
    organization: str | None = Field(description="Employer or organization, or null.")
    start: str | None = Field(
        description="Start date exactly as written (e.g. 'Mar 2019', '2019'), or null."
    )
    end: str | None = Field(
        description=(
            "End date exactly as written; 'present' if the CV says it is ongoing "
            "(present, current, now, today); null if not stated."
        )
    )
    heading_quote: str = Field(
        description=(
            "The entry's heading copied character for character from one line of the CV "
            "(usually the title, organization and/or dates)."
        )
    )
    skills: list[CvSkillMention] = Field(
        description="Skills the CV mentions inside this entry's text."
    )


class CvEducation(BaseModel):
    credential: str | None = Field(description="Degree, diploma or certificate, or null.")
    field: str | None = Field(description="Field of study, or null.")
    institution: str | None = Field(description="School or provider, or null.")
    quote: str = Field(description="One line of the CV for this entry, copied exactly.")


class ParsedCv(BaseModel):
    entries: list[CvEntry] = Field(
        description="Work experience, internships, projects and volunteering, in CV order."
    )
    education: list[CvEducation]
    other_skills: list[CvSkillMention] = Field(
        description=(
            "Skills mentioned outside any entry: skills sections, summaries, certifications."
        )
    )


# --- Stored CV profile --------------------------------------------------------------------


class ProfileEntry(BaseModel):
    kind: EntryKind
    title: str | None
    organization: str | None
    start: str | None  # as written
    end: str | None  # as written
    last_seen: date | None  # parsed end (or start); "present" = the date the CV was received
    span: tuple[int, int]  # heading, in original-text offsets


class ProfileEducation(BaseModel):
    # No dates: graduation years are a proxy for age.
    credential: str | None
    field: str | None
    institution: str | None
    span: tuple[int, int]


class CvProfile(BaseModel):
    document_id: str
    entries: list[ProfileEntry]
    education: list[ProfileEducation]
