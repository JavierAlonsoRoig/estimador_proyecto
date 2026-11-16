from app.generation.citations import verify_citations
from app.generation.context import assemble_context
from app.generation.schemas import Estimate, EstimateLineItem, SourceReference
from app.retrieval.types import RetrievedChunk

CHUNK = RetrievedChunk(
    chunk_id=59,
    document_id=17,
    chunk_type="component",
    content="Component: Cart\\nDescription: Add to cart.\\nEstimated hours: 24.0",
    metadata={"budget_id": "BUD-2024-017", "client_sector": "ecommerce", "year": 2024},
)
CONTEXT = assemble_context([CHUNK])


def line(**overrides) -> EstimateLineItem:
    fields = {
        "component": "Shopping cart",
        "hours": 24.0,
        "rationale": "Same scope as the MVP cart.",
        "grounded": True,
        "sources": [SourceReference(chunk_id="59", document_id="BUD-2024-017", evidence="Estimated hours: 24.0")],
    }
    return EstimateLineItem(**{**fields, **overrides})


def report_for(*items: EstimateLineItem):
    estimate = Estimate(summary="test", insufficient_context=False, line_items=list(items))
    return verify_citations(estimate, CONTEXT)


def test_valid_citation_is_grounded():
    report = report_for(line())
    assert report.lines[0].status == "grounded"
    assert report.is_clean
    assert report.verified_hours == 24.0


def test_invented_chunk_id_is_dangling():
    fake = SourceReference(chunk_id="9999", document_id="BUD-2024-017", evidence="Estimated hours: 24.0")
    report = report_for(line(sources=[fake]))
    assert report.lines[0].status == "dangling_citation"
    assert not report.is_clean
    assert report.verified_hours == 0.0


def test_real_chunk_with_wrong_document_is_dangling():
    wrong = SourceReference(chunk_id="59", document_id="BUD-2024-005", evidence="Estimated hours: 24.0")
    assert report_for(line(sources=[wrong])).lines[0].status == "dangling_citation"


def test_evidence_not_in_chunk_is_unverified():
    made_up = SourceReference(chunk_id="59", document_id="BUD-2024-017", evidence="Estimated hours: 80.0")
    assert report_for(line(sources=[made_up])).lines[0].status == "unverified_evidence"


def test_ungrounded_line_without_hours_is_insufficient_data():
    report = report_for(line(grounded=False, hours=None, sources=[]))
    assert report.lines[0].status == "insufficient_data"
    assert report.is_clean


def test_ungrounded_line_with_hours_is_integrity_violation():
    assert report_for(line(grounded=False, hours=40.0, sources=[])).lines[0].status == "integrity_violation"


def test_grounded_line_without_sources_is_integrity_violation():
    assert report_for(line(sources=[])).lines[0].status == "integrity_violation"
