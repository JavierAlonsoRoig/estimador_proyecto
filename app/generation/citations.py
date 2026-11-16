from typing import Literal

import structlog
from pydantic import BaseModel

from app.generation.context import AssembledContext
from app.generation.schemas import Estimate, EstimateLineItem

logger = structlog.get_logger(__name__)

LineStatus = Literal[
    "grounded",             # every cited source exists, matches its document and contains the evidence
    "dangling_citation",    # cites a chunk_id never retrieved, or pairs it with the wrong document_id
    "unverified_evidence",  # sources exist, but the evidence is not found verbatim in the cited chunk
    "insufficient_data",    # correctly marked grounded=false: no hours, no sources
    "integrity_violation",  # breaks the grounded <-> sources/hours rule
]

# Statuses that reject a line: its hours must not count towards the estimate
REJECTED_STATUSES = {"dangling_citation", "unverified_evidence", "integrity_violation"}


class LineCheck(BaseModel):
    component: str
    status: LineStatus
    hours: float | None
    issues: list[str]


class CitationReport(BaseModel):
    lines: list[LineCheck]
    estimate_issues: list[str]

    def count(self, status: LineStatus) -> int:
        return sum(1 for line in self.lines if line.status == status)

    @property
    def is_clean(self) -> bool:
        return not self.estimate_issues and all(
            line.status not in REJECTED_STATUSES for line in self.lines
        )

    @property
    def verified_hours(self) -> float:
        # Only lines that passed verification count: a rejected line's hours are not trusted
        return sum(line.hours or 0.0 for line in self.lines if line.status == "grounded")


def _normalize(text: str) -> str:
    # Whitespace and case differences are not a fabrication; anything else is
    return " ".join(text.split()).lower()


def _check_line(item: EstimateLineItem, context: AssembledContext) -> LineCheck:
    def result(status: LineStatus, issues: list[str]) -> LineCheck:
        return LineCheck(component=item.component, status=status, hours=item.hours, issues=issues)

    if not item.grounded:
        issues = []
        if item.hours is not None:
            issues.append(f"ungrounded line carries hours ({item.hours})")
        if item.sources:
            issues.append("ungrounded line cites sources")
        return result("integrity_violation" if issues else "insufficient_data", issues)

    if not item.sources:
        return result("integrity_violation", ["grounded line without sources"])
    if item.hours is None:
        return result("integrity_violation", ["grounded line without hours"])

    dangling: list[str] = []
    unverified: list[str] = []
    for source in item.sources:
        expected_document = context.document_by_chunk_id.get(source.chunk_id)
        if expected_document is None:
            dangling.append(f"chunk_id {source.chunk_id} was not in the retrieved context")
        elif source.document_id != expected_document:
            # A real chunk attributed to the wrong budget is still a citation to a source the LLM never saw
            dangling.append(
                f"chunk_id {source.chunk_id} belongs to {expected_document}, not {source.document_id}"
            )
        elif not source.evidence.strip() or _normalize(source.evidence) not in _normalize(
            context.content_by_chunk_id[source.chunk_id]
        ):
            unverified.append(f"evidence not found verbatim in chunk {source.chunk_id}: {source.evidence!r}")

    if dangling:
        return result("dangling_citation", dangling + unverified)
    if unverified:
        return result("unverified_evidence", unverified)
    return result("grounded", [])


def verify_citations(estimate: Estimate, context: AssembledContext) -> CitationReport:
    """Flag any line whose citations are not backed by the context the LLM actually received."""
    lines = [_check_line(item, context) for item in estimate.line_items]

    estimate_issues: list[str] = []
    if estimate.insufficient_context and any(line.hours is not None for line in lines):
        estimate_issues.append("insufficient_context=true but some lines carry hours")
    if not estimate.insufficient_context and not lines:
        estimate_issues.append("no line items and insufficient_context=false")

    report = CitationReport(lines=lines, estimate_issues=estimate_issues)

    # request_id is not passed here: structlog merges it from contextvars (bound per request)
    for line in report.lines:
        if line.status in REJECTED_STATUSES:
            logger.warning("citation_rejected", component=line.component, status=line.status, issues=line.issues)
    log = logger.info if report.is_clean else logger.warning
    log(
        "citation_report",
        lines=len(lines),
        grounded=report.count("grounded"),
        dangling_citation=report.count("dangling_citation"),
        unverified_evidence=report.count("unverified_evidence"),
        insufficient_data=report.count("insufficient_data"),
        integrity_violation=report.count("integrity_violation"),
        estimate_issues=estimate_issues,
        verified_hours=report.verified_hours,
    )
    return report
