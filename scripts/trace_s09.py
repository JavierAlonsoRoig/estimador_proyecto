"""Trace a raw transcript through the current system (Session 09 pre-work).

Step 1: embed the whole transcript with the same embedder used at ingestion.
Step 2: send the whole transcript to POST /search and print the raw response.

Usage:
    python scripts/trace_s09.py [--file transcripts/02_ambiguous.txt] [--k 5]
"""

import argparse
import json
import math
import sys
from pathlib import Path

import httpx
import tiktoken

# Running "python scripts/trace_s09.py" puts scripts/ on sys.path, not the project root,
# so "import app..." would fail without this.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.embedding_pipeline.embedder import OpenAIEmbedder  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace a transcript through embed + search.")
    parser.add_argument("--file", default="transcripts/02_ambiguous.txt", help="Transcript to trace")
    parser.add_argument("--k", type=int, default=5, help="Results to request from /search")
    parser.add_argument("--api-url", default=get_settings().API_URL, help="Base URL of the API")
    args = parser.parse_args()

    transcript = Path(args.file).read_text()
    embedder = OpenAIEmbedder()
    # Same tokenizer as the chunker: lets us compare query size with chunk size (token_count)
    tokens = len(tiktoken.encoding_for_model(embedder.model_name).encode(transcript))

    print("=== Step 1: embed the full transcript ===")
    vector = embedder.embed_one(transcript)
    print(f"file: {args.file}")
    print(f"model: {embedder.model_name}")
    print(f"transcript_tokens: {tokens}")
    print(f"dimension: {len(vector)}")
    print(f"norm: {math.sqrt(math.fsum(x * x for x in vector)):.6f}")
    print(f"first_component: {vector[0]}")
    print(f"last_component: {vector[-1]}")

    print(f"\n=== Step 2: POST /search (k={args.k}) with the full transcript as query ===")
    # Each search embeds the query with OpenAI first, so allow more than the default 5 s
    response = httpx.post(
        f"{args.api_url}/search", json={"query": transcript, "k": args.k}, timeout=30.0
    )
    response.raise_for_status()
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
