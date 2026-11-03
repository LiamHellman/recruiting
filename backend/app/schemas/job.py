from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

SkillId = Annotated[str, StringConstraints(pattern=r"^skill:[a-z0-9][a-z0-9-]*$")]
Seniority = Literal["intern", "junior", "mid", "senior", "staff", "principal", "unspecified"]
SkillList = Literal["must_have", "nice_to_have"]


# --- Stored contract ----------------------------------------------------------------------


class JobSkill(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_text: str = Field(min_length=1, max_length=200)
    # None until a human maps it. Never filled in by the LLM.
    skill_id: SkillId | None = None
    # Parsed skills start at 1.0; weights are a recruiter decision, not the LLM's.
    weight: float = Field(default=1.0, gt=0, le=10)
    # Where the parser found it: verbatim posting text and its [start, end) offsets.
    source_quote: str | None = None
    source_span: tuple[int, int] | None = None


class JobSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    must_have: list[JobSkill]
    nice_to_have: list[JobSkill]
    seniority: Seniority = "unspecified"
    domain: str | None = Field(default=None, max_length=100)

    def unmapped(self) -> list[str]:
        return [s.raw_text for s in self.must_have + self.nice_to_have if s.skill_id is None]


# --- LLM output schema (posting_parser) ---------------------------------------------------


class PostingRequirement(BaseModel):
    skill: str = Field(
        description=(
            "One technology, language, framework, tool, platform or technical practice, "
            "named as the posting names it (e.g. 'PostgreSQL', 'CI/CD')."
        )
    )
    quote: str = Field(
        description=(
            "The shortest excerpt of the posting, copied character for character, that "
            "states this requirement."
        )
    )


class ParsedPosting(BaseModel):
    title: str | None = Field(description="The job title as written, or null.")
    must_have: list[PostingRequirement] = Field(
        description="Technical skills the posting states are required."
    )
    nice_to_have: list[PostingRequirement] = Field(
        description="Technical skills the posting calls preferred, a bonus, or a plus."
    )
    seniority: Seniority = Field(
        description="Seniority the posting states; 'unspecified' if it does not say."
    )
    domain: str | None = Field(
        description="Area of work in a few words, e.g. 'backend', 'data platform', or null."
    )


# --- Parse report -------------------------------------------------------------------------


class RejectedRequirement(BaseModel):
    skill: str
    quote: str
    list: SkillList
    reason: str


class ParseReport(BaseModel):
    prompt_ref: str
    model: str
    model_served: str | None
    cached: bool
    rejected: list[RejectedRequirement]
    duplicates_merged: int


# --- Jobs API -----------------------------------------------------------------------------


class CreateJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    posting_text: str = Field(min_length=1, max_length=50_000)
    title: str | None = Field(default=None, max_length=300)


class UpdateJobSkillsRequest(BaseModel):
    """The full edited skill lists. `confirm=true` marks the job ready for scoring."""

    model_config = ConfigDict(extra="forbid")

    must_have: list[JobSkill]
    nice_to_have: list[JobSkill]
    seniority: Seniority = "unspecified"
    domain: str | None = Field(default=None, max_length=100)
    confirm: bool = False
    reason: str | None = Field(default=None, max_length=2000)

    def to_spec(self) -> JobSpec:
        return JobSpec(
            must_have=self.must_have,
            nice_to_have=self.nice_to_have,
            seniority=self.seniority,
            domain=self.domain,
        )


JobStatus = Literal["draft", "confirmed"]


class JobOut(BaseModel):
    id: str
    status: JobStatus
    title: str | None
    spec: JobSpec
    unmapped: list[str]
    parse_report: ParseReport
    created_at: datetime
    updated_at: datetime
    confirmed_at: datetime | None
