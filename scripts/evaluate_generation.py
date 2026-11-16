"""Evaluate generation quality with RAGAS on the golden set (session 11 baseline).

For each golden query it runs the same pipeline as POST /rag/estimate
(retrieve -> assemble_context -> generate -> verify_citations) and scores the sample with:
  faithfulness       claims in the answer supported by the retrieved contexts
  answer_relevancy   the answer addresses the question
  context_precision  useful contexts (judged against the reference) ranked first
  context_recall     claims in the reference attributable to the retrieved contexts

Usage:
    uv run python scripts/evaluate_generation.py [--golden data/golden_set.json]
        [--output generation_evaluation.md] [--samples-output generation_samples.json]
"""

import argparse
import asyncio
import json
import logging
import math
import statistics
import sys
from pathlib import Path
from uuid import uuid4

import structlog
from openai import AsyncOpenAI
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

# Running "python scripts/evaluate_generation.py" puts scripts/ on sys.path, not the project root,
# so "import app..." would fail without this.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.db import SessionLocal, engine  # noqa: E402
from app.embedding_pipeline.embedder import OpenAIEmbedder  # noqa: E402
from app.generation.citations import CitationReport, verify_citations  # noqa: E402
from app.generation.context import assemble_context  # noqa: E402
from app.generation.generator import get_generator  # noqa: E402
from app.generation.schemas import Estimate  # noqa: E402
from app.retrieval.service import retrieve  # noqa: E402

METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

# Keep warnings (rejected citations) visible, hide the per-stage INFO logs
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))


def estimate_to_text(estimate: Estimate) -> str:
    """Render the estimate as the plain-text 'answer' RAGAS reads.

    The total is left out on purpose: it is computed in code, not stated by any source,
    so the judge would flag it as an unsupported claim.
    """
    lines = [estimate.summary]
    if estimate.insufficient_context:
        lines.append("The historical budgets are insufficient to estimate this project.")
    for item in estimate.line_items:
        if item.grounded and item.hours is not None:
            lines.append(f"- {item.component}: {item.hours:g} hours. {item.rationale}")
        else:
            lines.append(f"- {item.component}: not estimated, insufficient historical data.")
    return "\n".join(lines)


def citation_summary(report: CitationReport) -> str:
    parts = [f"{report.count('grounded')}/{len(report.lines)} grounded"]
    for status in ("insufficient_data", "dangling_citation", "unverified_evidence", "integrity_violation"):
        if report.count(status):
            parts.append(f"{report.count(status)} {status}")
    return ", ".join(parts)


async def score_sample(metrics: tuple, question: str, answer: str, contexts: list[str], reference: str) -> dict:
    faithfulness, answer_relevancy, context_precision, context_recall = metrics
    # The four metrics are independent: run them concurrently
    results = await asyncio.gather(
        faithfulness.ascore(user_input=question, response=answer, retrieved_contexts=contexts),
        answer_relevancy.ascore(user_input=question, response=answer),
        context_precision.ascore(user_input=question, reference=reference, retrieved_contexts=contexts),
        context_recall.ascore(user_input=question, retrieved_contexts=contexts, reference=reference),
    )
    return {name: float(result.value) for name, result in zip(METRIC_NAMES, results)}


async def evaluate(golden: list[dict]) -> list[dict]:
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    judge = llm_factory(settings.JUDGE_MODEL, client=client)
    # Same embedding model as the corpus, as the exercise requires
    embeddings = embedding_factory("openai", model="text-embedding-3-small", client=client)
    metrics = (
        Faithfulness(llm=judge),
        AnswerRelevancy(llm=judge, embeddings=embeddings),
        ContextPrecision(llm=judge),
        ContextRecall(llm=judge),
    )

    embedder = OpenAIEmbedder()
    generator = get_generator()
    rows: list[dict] = []
    async with SessionLocal() as session:
        for item in golden:
            structlog.contextvars.bind_contextvars(request_id=uuid4().hex, golden_id=item["id"])
            retrieval = await retrieve(
                session,
                embedder,
                item["query"],
                mode=settings.SEARCH_MODE,
                top_k=settings.CONTEXT_TOP_K,
                recall_k=settings.RECALL_K,
                rrf_k=settings.RRF_K,
            )
            context = assemble_context(retrieval.chunks)
            estimate = await asyncio.to_thread(generator.generate, item["query"], context)
            report = verify_citations(estimate, context)

            answer = estimate_to_text(estimate)
            # Exactly what the generator saw, in retrieval order (context_precision is rank-aware)
            contexts = list(context.content_by_chunk_id.values())
            scores = await score_sample(metrics, item["query"], answer, contexts, item["reference"])

            rows.append(
                {
                    "id": item["id"],
                    "scores": scores,
                    "citations": citation_summary(report),
                    "verified_hours": report.verified_hours,
                    "reference_hours": item.get("reference_hours"),
                    "question": item["query"],
                    "answer": answer,
                    "contexts": contexts,
                    "reference": item["reference"],
                }
            )
            print(f"done: {item['id']} {scores}", file=sys.stderr)

    await engine.dispose()
    return rows


def mean_ignoring_nan(values: list[float]) -> float:
    # A metric can return NaN when the judge output can't be parsed: don't let it poison the average
    valid = [value for value in values if not math.isnan(value)]
    return statistics.fmean(valid) if valid else math.nan


def render_report(rows: list[dict]) -> str:
    settings = get_settings()
    lines = [
        "## Generation evaluation (RAGAS)",
        "",
        f"Generator: {settings.GENERATION_MODEL} | judge: {settings.JUDGE_MODEL} | "
        f"embeddings: text-embedding-3-small | retrieval: {settings.SEARCH_MODE}, top_k={settings.CONTEXT_TOP_K}",
        "",
        "| Query | Faithfulness | Answer relevancy | Context precision | Context recall "
        "| Citations | Verified h | Reference h |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        s = row["scores"]
        lines.append(
            f"| {row['id']} | {s['faithfulness']:.2f} | {s['answer_relevancy']:.2f} "
            f"| {s['context_precision']:.2f} | {s['context_recall']:.2f} "
            f"| {row['citations']} | {row['verified_hours']:g} | {row['reference_hours']} |"
        )
    averages = {name: mean_ignoring_nan([row["scores"][name] for row in rows]) for name in METRIC_NAMES}
    lines.append(
        f"| **Average** | **{averages['faithfulness']:.2f}** | **{averages['answer_relevancy']:.2f}** "
        f"| **{averages['context_precision']:.2f}** | **{averages['context_recall']:.2f}** | | | |"
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS evaluation of the generation stage on the golden set.")
    parser.add_argument("--golden", default="data/golden_set.json", help="Golden set JSON file")
    parser.add_argument("--output", help="Also write the Markdown table to this file")
    parser.add_argument(
        "--samples-output",
        default="generation_samples.json",
        help="Per-query question/answer/contexts/reference/scores, to inspect low scores",
    )
    args = parser.parse_args()

    golden = json.loads(Path(args.golden).read_text())
    rows = asyncio.run(evaluate(golden))

    report = render_report(rows)
    print(report)
    if args.output:
        Path(args.output).write_text(report)
    Path(args.samples_output).write_text(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
