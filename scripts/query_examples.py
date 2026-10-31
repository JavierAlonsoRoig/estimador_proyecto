"""Run five representative queries against POST /search and print the top results.

Usage:
    python scripts/query_examples.py [--k 5] [--api-url http://localhost:8000]
"""

import argparse
import sys
from pathlib import Path

import httpx

# Running "python scripts/query_examples.py" puts scripts/ on sys.path, not the project root,
# so "import app..." would fail without this.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402

CONTENT_PREVIEW_CHARS = 120

# (angle, query): each one exercises the corpus from a different direction
QUERIES = [
    ("Known component", "REST API development with JWT authentication for financial sector"),
    ("Semantic rephrasing", "secure backend service with token-based access control for banking applications"),
    ("Out of domain", "mobile application for restaurant reservations"),
    ("Ambiguous", "integration with external system"),
    ("Very specific", "migration from monolith to microservices architecture using Kubernetes"),
]


def search(client: httpx.Client, query: str, k: int) -> dict:
    response = client.post("/search", json={"query": query, "k": k})
    response.raise_for_status()
    return response.json()


def print_results(angle: str, data: dict) -> None:
    print("=" * 100)
    print(f"[{angle}] {data['query']}")
    print(f"k={data['k']} | {len(data['results'])} results | {data['search_time_ms']} ms")
    print("-" * 100)
    for rank, result in enumerate(data["results"], start=1):
        # Chunks contain newlines: flatten them so each result fits on one line
        preview = " ".join(result["content"].split())[:CONTENT_PREVIEW_CHARS]
        print(
            f"{rank}. chunk_id={result['chunk_id']:<5} distance={result['distance']:.4f}  "
            f"type={result['chunk_type']}"
        )
        print(f"   {preview}...")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Representative queries against POST /search.")
    parser.add_argument("--k", type=int, default=5, help="Results per query")
    parser.add_argument("--api-url", default=get_settings().API_URL, help="Base URL of the API")
    args = parser.parse_args()

    # Each search embeds the query with OpenAI first, so allow more than the default 5 s
    with httpx.Client(base_url=args.api_url, timeout=30.0) as client:
        for angle, query in QUERIES:
            try:
                data = search(client, query, args.k)
            except httpx.ConnectError:
                sys.exit(f"Cannot connect to the API at {args.api_url}. Is uvicorn running?")
            print_results(angle, data)


if __name__ == "__main__":
    main()
