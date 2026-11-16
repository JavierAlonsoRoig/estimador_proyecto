"""Run retrieval + generation for one request and print the cited estimate.

Usage:
    uv run python scripts/generate_estimate.py --golden q1-fashion-store
    uv run python scripts/generate_estimate.py "Online shop for a clothing brand: ..."
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from uuid import uuid4

import structlog




# Running "python scripts/generate_estimate.py" puts scripts/ on sys.path, not the project root,
# so "import app..." would fail without this.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.generation.citations import CitationReport, verify_citations  # noqa: E402
from app.generation.schemas import SourceReference  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.db import SessionLocal, engine  # noqa: E402
from app.embedding_pipeline.embedder import OpenAIEmbedder  # noqa: E402
from app.generation.context import AssembledContext, assemble_context  # noqa: E402
from app.generation.generator import get_generator  # noqa: E402
from app.generation.schemas import Estimate  # noqa: E402
from app.retrieval.service import retrieve  # noqa: E402


async def run(request: str) -> tuple[AssembledContext, Estimate]:
    settings = get_settings()
    async with SessionLocal() as session:
        result = await retrieve(
            session,
            OpenAIEmbedder(),
            request,
            mode=settings.SEARCH_MODE,
            top_k=settings.CONTEXT_TOP_K,
            recall_k=settings.RECALL_K,
            rrf_k=settings.RRF_K,
        )
    await engine.dispose()
    context = assemble_context(result.chunks)
    return context, get_generator().generate(request, context)


def inject_dangling_citation(estimate: Estimate) -> Estimate:
    """Tamper with a copy of the estimate: add a citation to a chunk that was never retrieved."""
    tampered = estimate.model_copy(deep=True)
    target = next((item for item in tampered.line_items if item.grounded), None)
    if target is not None:
        target.sources.append(
            SourceReference(chunk_id="999999", document_id="BUD-2024-999", evidence="Estimated hours: 999.0")
        )
    return tampered


def print_report(report: CitationReport) -> None:
    print("\ncitation report:")
    for line in report.lines:
        hours = "-" if line.hours is None else f"{line.hours:g}h"
        print(f"  [{line.status:<20}] {line.component} ({hours})")
        for issue in line.issues:
            print(f"      ! {issue}")
    for issue in report.estimate_issues:
        print(f"  [estimate] ! {issue}")
    print(f"  clean: {report.is_clean} | verified hours: {report.verified_hours:g}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one cited estimate and verify its citations.")
    parser.add_argument("request", nargs="?", help="Free-text estimation request")
    parser.add_argument("--golden", help="Use the query of this golden set id instead")
    parser.add_argument("--golden-file", default="data/golden_set.json")
    parser.add_argument(
        "--inject-dangling", action="store_true", help="Add a fake citation to test the verifier"
    )
    args = parser.parse_args()

    if args.golden:
        golden = json.loads(Path(args.golden_file).read_text())
        request = next(item["query"] for item in golden if item["id"] == args.golden)
    elif args.request:
        request = args.request
    else:
        parser.error("pass a request or --golden <id>")

    # Same correlation as the API: every log line of this run shares one request_id
    request_id = uuid4().hex
    structlog.contextvars.bind_contextvars(request_id=request_id)

    context, estimate = asyncio.run(run(request))
    if args.inject_dangling:
        estimate = inject_dangling_citation(estimate)

    print(f"request_id: {request_id}\nrequest: {request}\n")
    print("context sources:")
    for chunk_id, document_id in context.document_by_chunk_id.items():
        print(f"  chunk_id={chunk_id:<4} document_id={document_id}")
    print("\nestimate:")
    print(estimate.model_dump_json(indent=2))

    print_report(verify_citations(estimate, context))


if __name__ == "__main__":
    main()
