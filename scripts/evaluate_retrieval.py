"""Evaluate the four retrieval configurations (A-D) against the golden set through POST /search.

Going through the API measures what a real client gets: HTTP + FastAPI + retrieval, with the
cross-encoder running inside the server process. Per configuration it reports precision@5
(relevance annotated per budget) and latency: end-to-end as seen by the client, plus the
server-side breakdown by stage (embed / search / rerank) returned in timings_ms.

Start the API WITHOUT --reload before measuring:
    uv run uvicorn app.main:app

Usage:
    uv run python scripts/evaluate_retrieval.py [--golden data/golden_set.json] [--runs 5] [--output results.md]
"""

import argparse
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import httpx

# Running "python scripts/evaluate_retrieval.py" puts scripts/ on sys.path, not the project root,
# so "import app..." would fail without this.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402

TOP_K = 5

# (name, search mode, rerank)
CONFIGURATIONS = [
    ("A", "vector", False),
    ("B", "hybrid", False),
    ("C", "vector", True),
    ("D", "hybrid", True),
]


@dataclass
class ConfigurationResult:
    name: str
    mode: str
    rerank: bool
    precision_by_query: dict[str, float] = field(default_factory=dict)
    top_budgets_by_query: dict[str, list[str]] = field(default_factory=dict)
    # "client" = end-to-end wall time; the rest come from the server's timings_ms
    timings_ms: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    @property
    def mean_precision(self) -> float:
        return statistics.mean(self.precision_by_query.values())


def precision_at_k(budget_ids: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the first k results that belong to a relevant budget (always divided by k)."""
    return sum(1 for budget_id in budget_ids[:k] if budget_id in relevant) / k


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile: no interpolation, always an observed value."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(pct / 100 * len(ordered)) - 1)]


def search(client: httpx.Client, query: str, mode: str, rerank: bool) -> tuple[dict, float]:
    """One POST /search; returns the response body and the client-side latency in ms."""
    started_at = time.perf_counter()
    response = client.post(
        "/search", json={"query": query, "k": TOP_K, "mode": mode, "rerank": rerank}
    )
    elapsed_ms = (time.perf_counter() - started_at) * 1000
    response.raise_for_status()
    return response.json(), elapsed_ms


def evaluate(client: httpx.Client, golden: list[dict], runs: int) -> list[ConfigurationResult]:
    # Warm-up (not measured): OpenAI TLS connection, DB pool, reranker loaded in the server
    for _, mode, rerank in CONFIGURATIONS:
        for item in golden:
            search(client, item["query"], mode, rerank)

    results: list[ConfigurationResult] = []
    for name, mode, rerank in CONFIGURATIONS:
        config = ConfigurationResult(name=name, mode=mode, rerank=rerank)
        for item in golden:
            relevant = set(item["relevant_budget_ids"])
            for run in range(runs):
                data, client_ms = search(client, item["query"], mode, rerank)
                config.timings_ms["client"].append(client_ms)
                for stage, elapsed_ms in data["timings_ms"].items():
                    config.timings_ms[stage].append(elapsed_ms)
                if run == 0:
                    # Rankings are deterministic, so precision only needs one run
                    budget_ids = [result["metadata"]["budget_id"] for result in data["results"]]
                    config.precision_by_query[item["id"]] = precision_at_k(budget_ids, relevant, TOP_K)
                    config.top_budgets_by_query[item["id"]] = budget_ids
        print(f"done: configuration {name} ({mode}, rerank={rerank})", file=sys.stderr)
        results.append(config)
    return results


def render_report(results: list[ConfigurationResult], golden: list[dict], runs: int) -> str:
    settings = get_settings()
    lines = [
        "## Retrieval evaluation",
        "",
        f"Golden set: {len(golden)} queries | top_k={TOP_K} | recall_k={settings.RECALL_K} | "
        f"rrf_k={settings.RRF_K} | reranker={settings.RERANKER_MODEL} | runs per query={runs}",
        "",
        "Latency = end-to-end as seen by the client (HTTP + server). "
        "Embed / Search / Rerank = server-side stage medians.",
        "",
        "| Config | Search | Rerank | Precision@5 (mean) | Latency p50 (ms) | Latency p95 (ms) "
        "| Embed p50 | Search p50 | Rerank p50 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for config in results:
        timings = config.timings_ms
        rerank_p50 = f"{statistics.median(timings['rerank']):.0f}" if timings.get("rerank") else "-"
        lines.append(
            f"| {config.name} | {config.mode} | {'yes' if config.rerank else 'no'} "
            f"| {config.mean_precision:.2f} "
            f"| {statistics.median(timings['client']):.0f} | {percentile(timings['client'], 95):.0f} "
            f"| {statistics.median(timings['embed']):.0f} | {statistics.median(timings['search']):.1f} "
            f"| {rerank_p50} |"
        )

    lines += ["", "### Precision@5 per query", "", "| Query | " + " | ".join(c.name for c in results) + " |"]
    lines.append("|---|" + "---|" * len(results))
    for item in golden:
        cells = " | ".join(f"{c.precision_by_query[item['id']]:.1f}" for c in results)
        lines.append(f"| {item['id']} | {cells} |")

    lines += ["", "### Top-5 budgets per query (relevant ones in bold)", ""]
    for item in golden:
        relevant = set(item["relevant_budget_ids"])
        lines.append(f"**{item['id']}** (relevant: {', '.join(sorted(relevant))})")
        for config in results:
            shown = [
                f"**{b[-3:]}**" if b in relevant else b[-3:]
                for b in config.top_budgets_by_query[item["id"]]
            ]
            lines.append(f"- {config.name}: {' '.join(shown)}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval configurations A-D on the golden set.")
    parser.add_argument("--golden", default="data/golden_set.json", help="Golden set JSON file")
    parser.add_argument("--runs", type=int, default=5, help="Measured runs per query and configuration")
    parser.add_argument("--api-url", default=get_settings().API_URL, help="Base URL of the API")
    parser.add_argument("--output", help="Also write the Markdown report to this file")
    args = parser.parse_args()

    golden = json.loads(Path(args.golden).read_text())
    # Reranking 50 candidates on CPU can take seconds: allow far more than the default 5 s
    with httpx.Client(base_url=args.api_url, timeout=60.0) as client:
        try:
            results = evaluate(client, golden, args.runs)
        except httpx.ConnectError:
            sys.exit(f"Cannot connect to the API at {args.api_url}. Is uvicorn running?")
    report = render_report(results, golden, args.runs)
    print(report)
    if args.output:
        Path(args.output).write_text(report)


if __name__ == "__main__":
    main()
