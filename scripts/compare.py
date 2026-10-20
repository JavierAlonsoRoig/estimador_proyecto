"""Compare two texts by the cosine similarity of their OpenAI embeddings.

Usage:
    python scripts/compare.py --text-a "..." --text-b "..."
"""

import argparse
import math
import sys
from pathlib import Path

# Running "python scripts/compare.py" puts scripts/ on sys.path, not the project root,
# so "import app..." would fail without this.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.embedding_pipeline.embedder import OpenAIEmbedder  # noqa: E402


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"Vectors have different dimensions: {len(a)} vs {len(b)}")
    dot = math.fsum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(math.fsum(x * x for x in a))
    norm_b = math.sqrt(math.fsum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        raise ValueError("Cannot compute cosine similarity with a zero vector")
    return dot / (norm_a * norm_b)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cosine similarity between two texts.")
    parser.add_argument("--text-a", required=True, help="First text")
    parser.add_argument("--text-b", required=True, help="Second text")
    args = parser.parse_args()

    embedder = OpenAIEmbedder()
    similarity = cosine_similarity(embedder.embed_one(args.text_a), embedder.embed_one(args.text_b))

    print(f"Text A: {args.text_a}")
    print(f"Text B: {args.text_b}")
    print(f"Cosine similarity: {similarity:.4f}")


if __name__ == "__main__":
    main()
