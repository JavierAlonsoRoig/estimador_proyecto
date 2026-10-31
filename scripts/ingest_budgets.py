"""Ingest every budget in a JSON file (a list of budgets) through POST /embeddings/ingest.

Usage:
    python scripts/ingest_budgets.py [--file data/budgets_sample.json] [--api-url http://localhost:8000]
"""

import argparse
import json
import sys
from pathlib import Path

import httpx

# Running "python scripts/ingest_budgets.py" puts scripts/ on sys.path, not the project root,
# so "import app..." would fail without this.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402

DOCUMENT_TYPE = "historical_budget"


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a list of budgets into the vector store.")
    parser.add_argument("--file", default="data/budgets_sample.json", help="JSON file with a list of budgets")
    parser.add_argument("--api-url", default=get_settings().API_URL, help="Base URL of the API")
    args = parser.parse_args()

    budgets = json.loads(Path(args.file).read_text())
    ingested = skipped = failed = total_chunks = 0

    # Each ingest embeds all the budget's chunks with OpenAI, so allow more than the default 5 s
    with httpx.Client(base_url=args.api_url, timeout=60.0) as client:
        for budget in budgets:
            budget_id = budget["budget_id"]
            # One document per budget: the file path alone would collide (source_path is unique)
            payload = {
                "source_path": f"{args.file}#{budget_id}",
                "document_type": DOCUMENT_TYPE,
                "content": budget,
            }
            try:
                response = client.post("/embeddings/ingest", json=payload)
            except httpx.ConnectError:
                sys.exit(f"Cannot connect to the API at {args.api_url}. Is uvicorn running?")

            if response.status_code == 200:
                data = response.json()
                ingested += 1
                total_chunks += data["chunks_created"]
                print(f"OK    {budget_id}: document_id={data['document_id']} "
                      f"chunks={data['chunks_created']} {data['ingestion_time_ms']} ms")
            elif response.status_code == 409:
                skipped += 1
                print(f"SKIP  {budget_id}: already ingested (document_id={response.json()['document_id']})")
            else:
                # Keep going: one bad budget shouldn't stop the rest
                failed += 1
                print(f"FAIL  {budget_id}: HTTP {response.status_code} {response.text}")

    print(f"\nIngested: {ingested} ({total_chunks} chunks) | Skipped: {skipped} | Failed: {failed}")


if __name__ == "__main__":
    main()
