import uuid
from dataclasses import dataclass
from datetime import date

from app.audit.log import sha256_hex, stable_hash
from app.extraction.cv import CV_PROMPT, build_cv_evidence, cv_user_content
from app.extraction.redact import Redacted, redact
from app.llm.gateway import CallContext, Gateway
from app.llm.prompts import Prompt, load_prompt
from app.schemas.evidence import EvidenceSource, IngestReport, ParsedCv
from app.sources.base import CandidateRef, IngestResult
from app.taxonomy.loader import Taxonomy


@dataclass(frozen=True)
class CvPayload:
    document_id: uuid.UUID
    text: str  # the stored extracted text; locators are offsets into it
    received_on: date
    emails: tuple[str, ...] = ()  # pulled from the document, redacted as well


class CvSource:
    """The candidate's CV -> `claimed` evidence. Redacts identity and contact details first;
    the model only ever sees the redacted text."""

    name: EvidenceSource = "cv"

    def __init__(self, gateway: Gateway, taxonomy: Taxonomy, prompt: Prompt | None = None) -> None:
        self._gateway = gateway
        self._taxonomy = taxonomy
        self._prompt = prompt or load_prompt(*CV_PROMPT)

    def redact(self, candidate: CandidateRef, payload: CvPayload) -> Redacted:
        """Exactly what the model will see (also used by the eval's leak check)."""
        return redact(
            payload.text,
            names=candidate.names,
            emails=(*candidate.emails, *payload.emails),
            is_skill_term=lambda term: self._taxonomy.lookup(term) is not None,
        )

    def input_key(self, candidate: CandidateRef, payload: CvPayload) -> str:
        return stable_hash(
            {
                "prompt": self._prompt.ref,
                "model": self._prompt.model,
                "text": sha256_hex(payload.text),
                "document": payload.document_id,
                "received_on": payload.received_on,
                "names": candidate.names,
                "emails": [candidate.emails, payload.emails],
            }
        )

    def ingest(self, candidate: CandidateRef, payload: CvPayload) -> IngestResult:
        redacted = self.redact(candidate, payload)
        content = cv_user_content(redacted.text)
        result = self._gateway.parse(
            self._prompt,
            content,
            ParsedCv,
            CallContext(
                tenant_id=candidate.tenant_id, job_id=candidate.job_id, candidate_id=candidate.id
            ),
        )
        build = build_cv_evidence(
            result.output, redacted, str(payload.document_id), payload.received_on, self._taxonomy
        )
        return IngestResult(
            evidence=build.evidence,
            profile=build.profile.model_dump(mode="json"),
            report=IngestReport(
                source="cv",
                prompt_ref=result.prompt_ref,
                model=result.model,
                model_served=result.model_served,
                cached=result.cached,
                input_sha256=sha256_hex(content),
                checked=build.checked,
                rejected=build.rejected,
                accepted=len(build.evidence),
                duplicates_merged=build.duplicates_merged,
                unmapped_terms=build.unmapped_terms,
                redactions=redacted.counts(),
            ),
        )
