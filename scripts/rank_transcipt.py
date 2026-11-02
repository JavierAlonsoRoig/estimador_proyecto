"""Rank the whole corpus against a transcript, to see what falls outside the top-k.

Usage:
    python scripts/rank_transcript.py [--file transcripts/02_ambiguous.txt] [--k 50]
"""

import argparse
import re
import sys
from pathlib import Path

import httpx

# Running "python scripts/rank_transcript.py" puts scripts/ on sys.path, not the project root,
# so "import app..." would fail without this.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402

# Component lines end in a real newline or a literal "\n" (chunker bug, see "Otros")
COMPONENT_RE = re.compile(r"Component: (.+?)(?:\\n|\n)")
PROJECT_RE = re.compile(r"\[Project: (.+?)\]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Full ranking of the corpus for a transcript.")
    parser.add_argument("--file", default="transcripts/02_ambiguous.txt", help="Transcript to rank against")
    parser.add_argument("--k", type=int, default=50, help="Results to request from /search")
    parser.add_argument("--api-url", default=get_settings().API_URL, help="Base URL of the API")
    args = parser.parse_args()

    transcript = Path(args.file).read_text()
    response = httpx.post(
        f"{args.api_url}/search", json={"query": transcript, "k": args.k}, timeout=60.0
    )
    response.raise_for_status()

    print(f"{'#':>3}  {'distance':<8}  {'budget_id':<12}  {'hours':>6}  {'component':<35}  project")
    for rank, result in enumerate(response.json()["results"], start=1):
        metadata = result["metadata"]
        component = COMPONENT_RE.search(result["content"])
        project = PROJECT_RE.search(result["content"])
        print(
            f"{rank:>3}  {result['distance']:<8.4f}  {metadata['budget_id']:<12}  "
            f"{metadata['estimated_hours']:>6.0f}  "
            f"{(component.group(1) if component else '?')[:35]:<35}  "
            f"{project.group(1) if project else '?'}"
        )


if __name__ == "__main__":
    main()
